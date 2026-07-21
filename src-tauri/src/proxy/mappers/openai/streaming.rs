// OpenAI 流式转换
use bytes::{Bytes, BytesMut};
use chrono::Utc;
use futures::{Stream, StreamExt};
use rand::Rng;
use serde_json::{json, Value};
use std::pin::Pin;
use tracing::debug;
use uuid::Uuid;

/// 保存 thoughtSignature 到会话缓存
pub fn store_thought_signature(sig: &str, session_id: &str, message_count: usize) {
    if sig.is_empty() {
        return;
    }

    // 2. [CRITICAL] 存储到 Session 隔离缓存 (对齐 Claude 协议)
    crate::proxy::SignatureCache::global().cache_session_signature(
        session_id,
        sig.to_string(),
        message_count,
    );

    tracing::debug!(
        "[ThoughtSig] 存储 Session 签名 (sid: {}, len: {}, msg_count: {})",
        session_id,
        sig.len(),
        message_count
    );
}

/// Extract and convert Gemini usageMetadata to OpenAI usage format
fn extract_usage_metadata(u: &Value) -> Option<super::models::OpenAIUsage> {
    use super::models::{OpenAIUsage, PromptTokensDetails};

    let prompt_tokens = u
        .get("promptTokenCount")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let completion_tokens = u
        .get("candidatesTokenCount")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let total_tokens = u
        .get("totalTokenCount")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;
    let cached_tokens = u
        .get("cachedContentTokenCount")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    Some(OpenAIUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        prompt_tokens_details: cached_tokens.map(|ct| PromptTokensDetails {
            cached_tokens: Some(ct),
        }),
        completion_tokens_details: None,
    })
}

fn fallback_usage_for_text(text: &str, input_tokens: u32) -> super::models::OpenAIUsage {
    let (ascii, non_ascii) = text.chars().fold((0_u32, 0_u32), |(ascii, non_ascii), ch| {
        if ch.is_ascii() { (ascii + 1, non_ascii) } else { (ascii, non_ascii + 1) }
    });
    let output_tokens = non_ascii.saturating_add(ascii.saturating_add(3) / 4).max(1);
    let input_tokens = input_tokens.max(1);
    super::models::OpenAIUsage {
        prompt_tokens: input_tokens,
        completion_tokens: output_tokens,
        total_tokens: input_tokens.saturating_add(output_tokens),
        prompt_tokens_details: Some(super::models::PromptTokensDetails { cached_tokens: Some(0) }),
        completion_tokens_details: None,
    }
}

fn parse_codex_trailing_event(
    raw_line: &str,
) -> Option<(Vec<(String, bool)>, Option<super::models::OpenAIUsage>, bool)> {
    let line = raw_line.trim();
    let json_part = line.strip_prefix("data: ")?.trim();
    if json_part == "[DONE]" {
        return Some((Vec::new(), None, true));
    }

    let mut json = serde_json::from_str::<Value>(json_part).ok()?;
    let actual_data = if let Some(inner) = json.get_mut("response").map(|value| value.take()) {
        inner
    } else {
        json
    };
    let usage = actual_data
        .get("usageMetadata")
        .and_then(extract_usage_metadata);
    let mut saw_marker = usage.is_some();
    let mut texts = Vec::new();

    if let Some(candidate) = actual_data
        .get("candidates")
        .and_then(|value| value.as_array())
        .and_then(|candidates| candidates.first())
    {
        saw_marker |= candidate.get("finishReason").is_some()
            || candidate.get("finish_reason").is_some();
        if let Some(parts) = candidate
            .get("content")
            .and_then(|content| content.get("parts"))
            .and_then(|parts| parts.as_array())
        {
            for part in parts {
                if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
                    if !text.is_empty() {
                        texts.push((
                            text.to_string(),
                            part.get("thought")
                                .and_then(|value| value.as_bool())
                                .unwrap_or(false),
                        ));
                    }
                }
            }
        }
    }

    Some((texts, usage, saw_marker))
}

/// Process a single Gemini SSE `data:` line into OpenAI chat.completion.chunk SSE bytes.
///
/// Shared by the main read loop and the EOF flush so that a terminal event which
/// arrives without a trailing newline is parsed identically instead of being dropped.
/// Sets `saw_completion_marker` when a `usageMetadata` or `finishReason` is observed.
#[allow(clippy::too_many_arguments)]
fn build_openai_chat_chunks_from_line(
    line: &str,
    stream_id: &str,
    created_ts: i64,
    model: &str,
    session_id: &str,
    message_count: usize,
    hide_thinking_output: bool,
    emitted_tool_calls: &mut std::collections::HashSet<String>,
    tool_call_index: &mut usize,
    final_usage: &mut Option<super::models::OpenAIUsage>,
    saw_completion_marker: &mut bool,
) -> Vec<Bytes> {
    let mut out = Vec::new();
    let line = line.trim();
    if line.is_empty() || !line.starts_with("data: ") {
        return out;
    }
    let json_part = line.trim_start_matches("data: ").trim();
    if json_part == "[DONE]" {
        return out;
    }
    let Ok(mut json) = serde_json::from_str::<Value>(json_part) else {
        return out;
    };
    let actual_data = if let Some(inner) = json.get_mut("response").map(|v| v.take()) {
        inner
    } else {
        json
    };
    if let Some(u) = actual_data.get("usageMetadata") {
        *final_usage = extract_usage_metadata(u);
        *saw_completion_marker = true;
    }

    let Some(candidates) = actual_data.get("candidates").and_then(|c| c.as_array()) else {
        return out;
    };
    if !candidates.is_empty() {
        tracing::debug!("[Stream-Debug] Raw Candidate: {:?}", candidates[0]);
    }
    for (idx, candidate) in candidates.iter().enumerate() {
        let parts = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array());
        let mut content_out = String::new();
        let mut thought_out = String::new();

        if let Some(parts_list) = parts {
            for part in parts_list {
                let is_thought_part = part.get("thought").and_then(|v| v.as_bool()).unwrap_or(false);
                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    if is_thought_part {
                        if !hide_thinking_output {
                            thought_out.push_str(text);
                        }
                    } else {
                        content_out.push_str(text);
                    }
                }
                if let Some(sig) = part
                    .get("thoughtSignature")
                    .or(part.get("thought_signature"))
                    .and_then(|s| s.as_str())
                {
                    store_thought_signature(sig, session_id, message_count);
                }
                if let Some(img) = part.get("inlineData") {
                    let mime_type = img.get("mimeType").and_then(|v| v.as_str()).unwrap_or("image/png");
                    let data = img.get("data").and_then(|v| v.as_str()).unwrap_or("");
                    if !data.is_empty() {
                        content_out.push_str(&format!("![image](data:{};base64,{})", mime_type, data));
                    }
                }
                if let Some(func_call) = part.get("functionCall") {
                    let call_key = serde_json::to_string(func_call).unwrap_or_default();
                    if !emitted_tool_calls.contains(&call_key) {
                        emitted_tool_calls.insert(call_key);
                        let name = func_call.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let mut args = func_call.get("args").unwrap_or(&json!({})).clone();

                        // [FIX #1575] 标准化 shell 工具参数名称
                        if name == "shell" || name == "bash" || name == "local_shell" {
                            if let Some(obj) = args.as_object_mut() {
                                if !obj.contains_key("command") {
                                    for alt_key in &["cmd", "code", "script", "shell_command"] {
                                        if let Some(val) = obj.remove(*alt_key) {
                                            obj.insert("command".to_string(), val);
                                            debug!("[OpenAI-Stream] Normalized shell arg '{}' -> 'command'", alt_key);
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        let args_str = serde_json::to_string(&args).unwrap_or_default();
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        use std::hash::{Hash, Hasher};
                        serde_json::to_string(func_call).unwrap_or_default().hash(&mut hasher);
                        let call_id = format!("call_{:x}", hasher.finish());

                        let tool_call_chunk = json!({
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model,
                            "choices": [{
                                "index": idx as u32,
                                "delta": {
                                    "role": "assistant",
                                    "tool_calls": [{
                                        "index": *tool_call_index,
                                        "id": call_id,
                                        "type": "function",
                                        "function": { "name": name, "arguments": args_str }
                                    }]
                                },
                                "finish_reason": serde_json::Value::Null
                            }]
                        });
                        *tool_call_index += 1;
                        out.push(Bytes::from(format!(
                            "data: {}\n\n",
                            serde_json::to_string(&tool_call_chunk).unwrap_or_default()
                        )));
                    }
                }
            }
        }

        if let Some(grounding) = candidate.get("groundingMetadata") {
            let mut grounding_text = String::new();
            if let Some(queries) = grounding.get("webSearchQueries").and_then(|q| q.as_array()) {
                let query_list: Vec<&str> = queries.iter().filter_map(|v| v.as_str()).collect();
                if !query_list.is_empty() {
                    grounding_text.push_str("\n\n---\n**🔍 已为您搜索：** ");
                    grounding_text.push_str(&query_list.join(", "));
                }
            }
            if let Some(chunks) = grounding.get("groundingChunks").and_then(|c| c.as_array()) {
                let mut links = Vec::new();
                for (i, chunk) in chunks.iter().enumerate() {
                    if let Some(web) = chunk.get("web") {
                        let title = web.get("title").and_then(|v| v.as_str()).unwrap_or("网页来源");
                        let uri = web.get("uri").and_then(|v| v.as_str()).unwrap_or("#");
                        links.push(format!("[{}] [{}]({})", i + 1, title, uri));
                    }
                }
                if !links.is_empty() {
                    grounding_text.push_str("\n\n**🌐 来源引文：**\n");
                    grounding_text.push_str(&links.join("\n"));
                }
            }
            if !grounding_text.is_empty() {
                content_out.push_str(&grounding_text);
            }
        }

        let gemini_finish_reason = candidate.get("finishReason").and_then(|f| f.as_str()).map(|f| match f {
            "STOP" => "stop",
            "MAX_TOKENS" => "length",
            "SAFETY" => "content_filter",
            "RECITATION" => "content_filter",
            _ => f,
        });
        if gemini_finish_reason.is_some() {
            *saw_completion_marker = true;
        }

        // [FIX #1575] 如果发射了工具调用，强制设置为 tool_calls
        let finish_reason = if !emitted_tool_calls.is_empty() && gemini_finish_reason.is_some() {
            Some("tool_calls")
        } else {
            gemini_finish_reason
        };

        if !thought_out.is_empty() {
            let reasoning_chunk = json!({
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model,
                "choices": [{
                    "index": idx as u32,
                    "delta": { "role": "assistant", "content": serde_json::Value::Null, "reasoning_content": thought_out },
                    "finish_reason": serde_json::Value::Null
                }]
            });
            out.push(Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&reasoning_chunk).unwrap_or_default()
            )));
        }

        if !content_out.is_empty() || finish_reason.is_some() {
            let mut openai_chunk = json!({
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model,
                "choices": [{
                    "index": idx as u32,
                    "delta": { "content": content_out },
                    "finish_reason": finish_reason
                }]
            });
            if finish_reason.is_some() {
                if let Some(ref usage) = final_usage {
                    openai_chunk["usage"] = serde_json::to_value(usage).unwrap();
                }
                *final_usage = None;
            }
            out.push(Bytes::from(format!(
                "data: {}\n\n",
                serde_json::to_string(&openai_chunk).unwrap_or_default()
            )));
        }
    }
    out
}

pub fn create_openai_sse_stream<S, E>(
    mut gemini_stream: Pin<Box<S>>,
    model: String,
    session_id: String,
    message_count: usize,
    hide_thinking_output: bool,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + ?Sized + 'static,
    E: std::fmt::Display + Send + 'static,
{
    let mut buffer = BytesMut::new();
    let stream_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created_ts = Utc::now().timestamp();

    let stream = async_stream::stream! {
        let mut emitted_tool_calls = std::collections::HashSet::new();
        let mut final_usage: Option<super::models::OpenAIUsage> = None;
        let mut error_occurred = false;
        let mut tool_call_index = 0;
        let mut saw_completion_marker = false;

        let mut heartbeat_interval = tokio::time::interval(std::time::Duration::from_secs(15));
        heartbeat_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                item = gemini_stream.next() => {
                    match item {
                        Some(Ok(bytes)) => {
                            buffer.extend_from_slice(&bytes);
                            while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                                let line_raw = buffer.split_to(pos + 1);
                                if let Ok(line_str) = std::str::from_utf8(&line_raw) {
                                    let line = line_str.trim();
                                    if line.is_empty() { continue; }
                                    for chunk in build_openai_chat_chunks_from_line(
                                        line,
                                        &stream_id,
                                        created_ts,
                                        &model,
                                        &session_id,
                                        message_count,
                                        hide_thinking_output,
                                        &mut emitted_tool_calls,
                                        &mut tool_call_index,
                                        &mut final_usage,
                                        &mut saw_completion_marker,
                                    ) {
                                        yield Ok::<Bytes, String>(chunk);
                                    }
                                }
                            }
                        }
                        Some(Err(e)) => {
                            use crate::proxy::mappers::error_classifier::classify_stream_error;
                            let (error_type, user_msg, i18n_key) = classify_stream_error(&e);
                            tracing::error!("OpenAI Stream Error: {}", e);
                            let error_chunk = json!({
                                "id": &stream_id, "object": "chat.completion.chunk", "created": created_ts, "model": &model, "choices": [],
                                "error": { "type": error_type, "message": user_msg, "code": "stream_error", "i18n_key": i18n_key }
                            });
                            yield Ok(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&error_chunk).unwrap_or_default())));
                            yield Ok(Bytes::from("data: [DONE]\n\n"));
                            error_occurred = true;
                            break;
                        }
                        None => break,
                    }
                }
                _ = heartbeat_interval.tick() => {
                    yield Ok::<Bytes, String>(Bytes::from(": ping\n\n"));
                }
            }
        }

        // [FIX #1732] Flush remaining buffer to prevent loss of a terminal SSE event
        // that arrived without a trailing newline (network fragmentation). Previously
        // this only debug-logged the residual bytes, dropping the last text /
        // finishReason / usage and silently truncating the response.
        if !error_occurred && !buffer.is_empty() {
            if let Ok(line_str) = std::str::from_utf8(&buffer) {
                let line = line_str.trim();
                if !line.is_empty() {
                    tracing::debug!("[OpenAI-SSE] Flushing remaining {} bytes in buffer", buffer.len());
                    for chunk in build_openai_chat_chunks_from_line(
                        line,
                        &stream_id,
                        created_ts,
                        &model,
                        &session_id,
                        message_count,
                        hide_thinking_output,
                        &mut emitted_tool_calls,
                        &mut tool_call_index,
                        &mut final_usage,
                        &mut saw_completion_marker,
                    ) {
                        yield Ok::<Bytes, String>(chunk);
                    }
                }
            }
        }
        buffer.clear();

        // Refuse to fake a successful completion. If the upstream connection ended
        // (EOF or idle timeout) without ever delivering a finishReason / usage
        // marker, the response was truncated — surface a visible error instead of a
        // bare [DONE] that clients treat as success.
        if !error_occurred && !saw_completion_marker {
            tracing::warn!("[OpenAI-Stream] Upstream ended before finish marker; refusing silent success");
            let error_chunk = json!({
                "id": &stream_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": &model,
                "choices": [],
                "error": {
                    "type": "incomplete_upstream_stream",
                    "message": "Upstream stream ended before a finish marker",
                    "code": "incomplete_upstream_stream"
                }
            });
            yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&error_chunk).unwrap_or_default())));
        }

        if !error_occurred {
            yield Ok::<Bytes, String>(Bytes::from("data: [DONE]\n\n"));
        }
    };
    Box::pin(stream)
}

pub fn create_legacy_sse_stream<S, E>(
    mut gemini_stream: Pin<Box<S>>,
    model: String,
    session_id: String,
    message_count: usize,
    hide_thinking_output: bool,
    fallback_input_tokens: u32,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + ?Sized + 'static,
    E: std::fmt::Display + Send + 'static,
{
    let mut buffer = BytesMut::new();
    let charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::thread_rng();
    let random_str: String = (0..28)
        .map(|_| {
            let idx = rng.gen_range(0..charset.len());
            charset.chars().nth(idx).unwrap()
        })
        .collect();
    let stream_id = format!("cmpl-{}", random_str);
    let created_ts = Utc::now().timestamp();

    let stream = async_stream::stream! {
        let mut final_usage: Option<super::models::OpenAIUsage> = None;
        let mut accumulated_text = String::new();
        let mut error_occurred = false;
        let mut saw_completion_marker = false;
        let mut heartbeat_interval = tokio::time::interval(std::time::Duration::from_secs(15));
        heartbeat_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                item = gemini_stream.next() => {
                    match item {
                        Some(Ok(bytes)) => {
                            buffer.extend_from_slice(&bytes);
                            while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                                let line_raw = buffer.split_to(pos + 1);
                                if let Ok(line_str) = std::str::from_utf8(&line_raw) {
                                    let line = line_str.trim();
                                    if line.is_empty() { continue; }
                                    if line.starts_with("data: ") {
                                        let json_part = line.trim_start_matches("data: ").trim();
                                        if json_part == "[DONE]" { continue; }
                                        if let Ok(mut json) = serde_json::from_str::<Value>(json_part) {
                                            let actual_data = if let Some(inner) = json.get_mut("response").map(|v| v.take()) { inner } else { json };
                                            if let Some(u) = actual_data.get("usageMetadata") {
                                                final_usage = extract_usage_metadata(u);
                                                saw_completion_marker = true;
                                            }

                                            let mut content_out = String::new();
                                            if let Some(candidates) = actual_data.get("candidates").and_then(|c| c.as_array()) {
                                                if let Some(candidate) = candidates.get(0) {
                                                    if let Some(parts) = candidate.get("content").and_then(|c| c.get("parts")).and_then(|p| p.as_array()) {
                                                        for part in parts {
                                                            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                                                let is_thought = part.get("thought").and_then(|v| v.as_bool()).unwrap_or(false);
                                                                if !is_thought || !hide_thinking_output {
                                                                    content_out.push_str(text);
                                                                }
                                                            }
                                                            if let Some(sig) = part.get("thoughtSignature").or(part.get("thought_signature")).and_then(|s| s.as_str()) {
                                                                store_thought_signature(sig, &session_id, message_count);
                                                            }
                                                        }
                                                    }
                                                }
                                            }

                                            let finish_reason = actual_data.get("candidates").and_then(|c| c.as_array()).and_then(|c| c.get(0)).and_then(|c| c.get("finishReason")).and_then(|f| f.as_str()).map(|f| match f {
                                                "STOP" => "stop", "MAX_TOKENS" => "length", "SAFETY" => "content_filter", _ => f,
                                            });
                                            saw_completion_marker |= finish_reason.is_some();

                                            let mut legacy_chunk = json!({
                                                "id": &stream_id, "object": "text_completion", "created": created_ts, "model": &model,
                                                "choices": [{ "text": content_out, "index": 0, "logprobs": null, "finish_reason": finish_reason }]
                                            });
                                            accumulated_text.push_str(&content_out);
                                            if finish_reason.is_some() {
                                                let usage = final_usage
                                                    .take()
                                                    .unwrap_or_else(|| fallback_usage_for_text(&accumulated_text, fallback_input_tokens));
                                                legacy_chunk["usage"] = serde_json::to_value(usage).unwrap();
                                            }
                                            yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&legacy_chunk).unwrap_or_default())));
                                        }
                                    }
                                }
                            }
                        }
                        Some(Err(e)) => {
                            use crate::proxy::mappers::error_classifier::classify_stream_error;
                            let (error_type, user_msg, i18n_key) = classify_stream_error(&e);
                            tracing::error!("Legacy Stream Error: {}", e);
                            let error_chunk = json!({
                                "id": &stream_id, "object": "text_completion", "created": created_ts, "model": &model, "choices": [],
                                "error": { "type": error_type, "message": user_msg, "code": "stream_error", "i18n_key": i18n_key }
                            });
                            yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&error_chunk).unwrap_or_default())));
                            yield Ok::<Bytes, String>(Bytes::from("data: [DONE]\n\n"));
                            error_occurred = true;
                            break;
                        }
                        None => break,
                    }
                }
                _ = heartbeat_interval.tick() => { yield Ok::<Bytes, String>(Bytes::from(": ping\n\n")); }
            }
        }
        // Flush a terminal SSE event that arrived without a trailing newline.
        if !buffer.is_empty() {
            if let Ok(line_str) = std::str::from_utf8(&buffer) {
                let line = line_str.trim();
                if let Some(json_part) = line.strip_prefix("data: ") {
                    if json_part.trim() != "[DONE]" {
                        if let Ok(mut json) = serde_json::from_str::<Value>(json_part.trim()) {
                            let actual_data = if let Some(inner) = json.get_mut("response").map(|v| v.take()) { inner } else { json };
                            if let Some(u) = actual_data.get("usageMetadata") {
                                final_usage = extract_usage_metadata(u);
                                saw_completion_marker = true;
                            }
                            let candidate = actual_data.get("candidates").and_then(|c| c.as_array()).and_then(|c| c.first());
                            let mut content_out = String::new();
                            if let Some(parts) = candidate.and_then(|c| c.get("content")).and_then(|c| c.get("parts")).and_then(|p| p.as_array()) {
                                for part in parts {
                                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                        if !part.get("thought").and_then(|v| v.as_bool()).unwrap_or(false) || !hide_thinking_output { content_out.push_str(text); }
                                    }
                                }
                            }
                            let finish_reason = candidate.and_then(|c| c.get("finishReason")).and_then(|f| f.as_str()).map(|f| match f { "STOP" => "stop", "MAX_TOKENS" => "length", "SAFETY" => "content_filter", _ => f });
                            saw_completion_marker |= finish_reason.is_some();
                            if finish_reason.is_some() || !content_out.is_empty() {
                                accumulated_text.push_str(&content_out);
                                let mut legacy_chunk = json!({ "id": &stream_id, "object": "text_completion", "created": created_ts, "model": &model, "choices": [{ "text": content_out, "index": 0, "logprobs": null, "finish_reason": finish_reason }] });
                                if finish_reason.is_some() {
                                    let usage = final_usage.take().unwrap_or_else(|| fallback_usage_for_text(&accumulated_text, fallback_input_tokens));
                                    legacy_chunk["usage"] = serde_json::to_value(usage).unwrap();
                                }
                                yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&legacy_chunk).unwrap_or_default())));
                            }
                        }
                    }
                }
            }
        }
        if !error_occurred && !saw_completion_marker {
            tracing::warn!("[Legacy-Stream] Upstream ended before finish marker; refusing silent success");
            let error_chunk = json!({
                "id": &stream_id,
                "object": "text_completion",
                "created": created_ts,
                "model": &model,
                "choices": [],
                "error": {
                    "type": "incomplete_upstream_stream",
                    "message": "Upstream stream ended before a finish marker",
                    "code": "incomplete_upstream_stream"
                }
            });
            yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&error_chunk).unwrap_or_default())));
            yield Ok::<Bytes, String>(Bytes::from("data: [DONE]\n\n"));
            error_occurred = true;
        }
        if !error_occurred {
            yield Ok::<Bytes, String>(Bytes::from("data: [DONE]\n\n"));
        }
    };
    Box::pin(stream)
}

pub fn create_codex_sse_stream<S, E>(
    mut gemini_stream: Pin<Box<S>>,
    _model: String,
    session_id: String,
    message_count: usize,
    hide_thinking_output: bool,
    fallback_input_tokens: u32,
) -> Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + ?Sized + 'static,
    E: std::fmt::Display + Send + 'static,
{
    let mut buffer = BytesMut::new();
    let charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::thread_rng();
    let random_str: String = (0..24)
        .map(|_| {
            let idx = rng.gen_range(0..charset.len());
            charset.chars().nth(idx).unwrap()
        })
        .collect();
    let response_id = format!("resp-{}", random_str);
    let item_id = format!("item-{}", &random_str[..16]);

    let stream = async_stream::stream! {
        // 1. response.created
        let created_ev = json!({ "type": "response.created", "response": { "id": &response_id, "object": "response", "status": "in_progress", "output": [] } });
        yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&created_ev).unwrap())));

        // 2. response.output_item.added - 告诉客户端开始一个输出项
        let output_item_added = json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": &item_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": []
            }
        });
        yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&output_item_added).unwrap())));

        // 3. response.content_part.added - 告诉客户端开始一个文本内容块
        let content_part_added = json!({
            "type": "response.content_part.added",
            "item_id": &item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": ""
            }
        });
        yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&content_part_added).unwrap())));

        let mut emitted_tool_calls = std::collections::HashSet::new();
        let mut accumulated_text = String::new();
        let mut final_usage: Option<super::models::OpenAIUsage> = None;
        let mut upstream_error: Option<String> = None;
        let mut saw_completion_marker = false;
        let mut heartbeat_interval = tokio::time::interval(std::time::Duration::from_secs(15));
        heartbeat_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                item = gemini_stream.next() => {
                    match item {
                        Some(Ok(bytes)) => {
                            buffer.extend_from_slice(&bytes);
                            while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                                let line_raw = buffer.split_to(pos + 1);
                                if let Ok(line_str) = std::str::from_utf8(&line_raw) {
                                    let line = line_str.trim();
                                    if line.is_empty() || !line.starts_with("data: ") { continue; }
                                    let json_part = line.trim_start_matches("data: ").trim();
                                    if json_part == "[DONE]" { continue; }

                                    if let Ok(mut json) = serde_json::from_str::<Value>(json_part) {
                                        let actual_data = if let Some(inner) = json.get_mut("response").map(|v| v.take()) { inner } else { json };
                                        if let Some(usage) = actual_data.get("usageMetadata") {
                                            final_usage = extract_usage_metadata(usage);
                                            saw_completion_marker = true;
                                        }
                                        if let Some(candidates) = actual_data.get("candidates").and_then(|c| c.as_array()) {
                                            if candidates.len() > 0 {
                                                tracing::debug!("[Codex-Stream-Debug] Raw Candidate: {:?}", candidates[0]);
                                            }
                                            if let Some(candidate) = candidates.get(0) {
                                                if candidate.get("finishReason").is_some()
                                                    || candidate.get("finish_reason").is_some()
                                                {
                                                    saw_completion_marker = true;
                                                }
                                                if let Some(parts) = candidate.get("content").and_then(|c| c.get("parts")).and_then(|p| p.as_array()) {
                                                    for part in parts {
                                                        let is_thought = part.get("thought").and_then(|v| v.as_bool()).unwrap_or(false);
                                                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                                            if !text.is_empty() {
                                                                if is_thought {
                                                                    if !hide_thinking_output {
                                                                        let reasoning_ev = json!({
                                                                            "type": "response.reasoning.delta",
                                                                            "item_id": &item_id,
                                                                            "output_index": 0,
                                                                            "content_index": 0,
                                                                            "delta": text
                                                                        });
                                                                        yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&reasoning_ev).unwrap())));
                                                                    }
                                                                } else {
                                                                    accumulated_text.push_str(text);
                                                                    // 4. response.output_text.delta - 文本增量
                                                                    let delta_ev = json!({
                                                                        "type": "response.output_text.delta",
                                                                        "item_id": &item_id,
                                                                        "output_index": 0,
                                                                        "content_index": 0,
                                                                        "delta": text
                                                                    });
                                                                    yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&delta_ev).unwrap())));
                                                                }
                                                            }
                                                        }
                                                        if let Some(sig) = part.get("thoughtSignature").or(part.get("thought_signature")).and_then(|s| s.as_str()) {
                                                            store_thought_signature(sig, &session_id, message_count);
                                                        }
                                                        if let Some(func_call) = part.get("functionCall") {
                                                            let call_key = serde_json::to_string(func_call).unwrap_or_default();
                                                            if !emitted_tool_calls.contains(&call_key) {
                                                                emitted_tool_calls.insert(call_key);
                                                            }
                                                        }
                                                    }

                                                }

                                                // 处理 groundingMetadata (搜索引文)
                                                if let Some(grounding) = candidate.get("groundingMetadata") {
                                                    let mut grounding_text = String::new();
                                                    if let Some(queries) = grounding.get("webSearchQueries").and_then(|q| q.as_array()) {
                                                        let query_list: Vec<&str> = queries.iter().filter_map(|v| v.as_str()).collect();
                                                        if !query_list.is_empty() {
                                                            grounding_text.push_str("\n\n---\n**🔍 已为您搜索：** ");
                                                            grounding_text.push_str(&query_list.join(", "));
                                                        }
                                                    }
                                                    if let Some(chunks) = grounding.get("groundingChunks").and_then(|c| c.as_array()) {
                                                        let mut links = Vec::new();
                                                        for (i, chunk) in chunks.iter().enumerate() {
                                                            if let Some(web) = chunk.get("web") {
                                                                let title = web.get("title").and_then(|v| v.as_str()).unwrap_or("网页来源");
                                                                let uri = web.get("uri").and_then(|v| v.as_str()).unwrap_or("#");
                                                                links.push(format!("[{}] [{}]({})", i + 1, title, uri));
                                                            }
                                                        }
                                                        if !links.is_empty() {
                                                            grounding_text.push_str("\n\n**🌐 来源引文：**\n");
                                                            grounding_text.push_str(&links.join("\n"));
                                                        }
                                                    }
                                                    if !grounding_text.is_empty() {
                                                        accumulated_text.push_str(&grounding_text);
                                                        let delta_ev = json!({
                                                            "type": "response.output_text.delta",
                                                            "item_id": &item_id,
                                                            "output_index": 0,
                                                            "content_index": 0,
                                                            "delta": grounding_text
                                                        });
                                                        yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&delta_ev).unwrap())));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Some(Err(error)) => {
                            upstream_error = Some(error.to_string());
                            break;
                        }
                        None => break,
                    }
                }
                _ = heartbeat_interval.tick() => { yield Ok::<Bytes, String>(Bytes::from(": ping\n\n")); }
            }
        }

        // The final SSE event can arrive without a trailing newline. Do not
        // discard that buffered event when the upstream connection closes.
        if !buffer.is_empty() {
            if let Ok(line) = std::str::from_utf8(&buffer) {
                if let Some((texts, usage, marker)) = parse_codex_trailing_event(line) {
                    if let Some(usage) = usage {
                        final_usage = Some(usage);
                    }
                    saw_completion_marker |= marker;
                    for (text, is_thought) in texts {
                        if is_thought {
                            if !hide_thinking_output {
                                let reasoning_ev = json!({
                                    "type": "response.reasoning.delta",
                                    "item_id": &item_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": text
                                });
                                yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&reasoning_ev).unwrap())));
                            }
                        } else {
                            accumulated_text.push_str(&text);
                            let delta_ev = json!({
                                "type": "response.output_text.delta",
                                "item_id": &item_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": text
                            });
                            yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&delta_ev).unwrap())));
                        }
                    }
                }
            }
        }

        if let Some(error) = upstream_error {
            let failed_ev = json!({
                "type": "response.failed",
                "response": {
                    "id": &response_id,
                    "object": "response",
                    "status": "failed",
                    "error": {
                        "type": "upstream_error",
                        "message": error
                    }
                }
            });
            yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&failed_ev).unwrap())));
            return;
        }

        if !saw_completion_marker {
            let incomplete_ev = json!({
                "type": "response.failed",
                "response": {
                    "id": &response_id,
                    "object": "response",
                    "status": "failed",
                    "error": {
                        "type": "incomplete_upstream_stream",
                        "message": "Upstream stream ended before a finish marker"
                    }
                }
            });
            tracing::warn!("[Codex-Stream] Upstream ended before finish marker; refusing completed response");
            yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&incomplete_ev).unwrap())));
            return;
        }

        // 5. response.output_text.done - 文本完成
        let text_done = json!({
            "type": "response.output_text.done",
            "item_id": &item_id,
            "output_index": 0,
            "content_index": 0,
            "text": &accumulated_text
        });
        yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&text_done).unwrap())));

        // 6. response.content_part.done
        let content_part_done = json!({
            "type": "response.content_part.done",
            "item_id": &item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": &accumulated_text
            }
        });
        yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&content_part_done).unwrap())));

        // 7. response.output_item.done
        let output_item_done = json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": &item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": &accumulated_text
                }]
            }
        });
        yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&output_item_done).unwrap())));

        // 8. response.completed
        let fallback_output_tokens = {
            let (ascii, non_ascii) = accumulated_text.chars().fold(
                (0_u32, 0_u32),
                |(ascii, non_ascii), ch| {
                    if ch.is_ascii() {
                        (ascii + 1, non_ascii)
                    } else {
                        (ascii, non_ascii + 1)
                    }
                },
            );
            non_ascii.saturating_add(ascii.saturating_add(3) / 4).max(1)
        };
        let mut usage = final_usage.unwrap_or(super::models::OpenAIUsage {
            prompt_tokens: fallback_input_tokens.max(1),
            completion_tokens: fallback_output_tokens,
            total_tokens: fallback_input_tokens
                .max(1)
                .saturating_add(fallback_output_tokens),
            prompt_tokens_details: Some(super::models::PromptTokensDetails {
                cached_tokens: Some(0),
            }),
            completion_tokens_details: None,
        });
        if usage.prompt_tokens == 0 {
            usage.prompt_tokens = fallback_input_tokens.max(1);
        }
        if usage.completion_tokens == 0 {
            usage.completion_tokens = fallback_output_tokens;
        }
        usage.total_tokens = usage
            .total_tokens
            .max(usage.prompt_tokens.saturating_add(usage.completion_tokens));
        let cached_tokens = usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .unwrap_or(0);
        let reasoning_tokens = usage
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens)
            .unwrap_or(0);
        let responses_usage = json!({
            "input_tokens": usage.prompt_tokens,
            "input_tokens_details": { "cached_tokens": cached_tokens },
            "output_tokens": usage.completion_tokens,
            "output_tokens_details": { "reasoning_tokens": reasoning_tokens },
            "total_tokens": usage.total_tokens
        });
        let completed_ev = json!({
            "type": "response.completed",
            "response": {
                "id": &response_id,
                "object": "response",
                "status": "completed",
                "usage": responses_usage,
                "output": [{
                    "id": &item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": &accumulated_text
                    }]
                }]
            }
        });
        yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&completed_ev).unwrap())));
        yield Ok::<Bytes, String>(Bytes::from("data: [DONE]\n\n"));
    };
    Box::pin(stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use serde_json::json;

    #[tokio::test]
    async fn test_openai_streaming_usage_only_at_end() {
        // Chunk 1: Partial content, no usage
        let chunk1_json = json!({
            "candidates": [{
                "content": {
                    "parts": [{ "text": "Hello" }]
                }
            }]
        });

        // Chunk 2: Finish reason + Usage metadata
        let chunk2_json = json!({
            "candidates": [{
                "finishReason": "STOP",
                "content": {
                    "parts": [{ "text": "" }]
                }
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7
            }
        });

        // Use a helper to create the stream items compatible with the required signature
        let items: Vec<Result<Bytes, reqwest::Error>> = vec![
            Ok(Bytes::from(format!("data: {}\n\n", chunk1_json))),
            Ok(Bytes::from(format!("data: {}\n\n", chunk2_json))),
        ];

        let gemini_stream = Box::pin(stream::iter(items));

        let mut openai_stream = create_openai_sse_stream(
            gemini_stream,
            "gemini-1.5-flash".to_string(),
            "test-session".to_string(),
            0,
            false,
        );

        let mut chunks = Vec::new();
        while let Some(result) = openai_stream.next().await {
            if let Ok(bytes) = result {
                let s = String::from_utf8_lossy(&bytes).to_string();
                for line in s.lines() {
                    if line.starts_with("data: ") && !line.contains("[DONE]") {
                        chunks.push(line.to_string());
                    }
                }
            }
        }

        let mut found_usage = false;
        let mut found_finish = false;

        for (i, chunk_str) in chunks.iter().enumerate() {
            let json_str = chunk_str.trim_start_matches("data: ").trim();
            let json: Value = serde_json::from_str(json_str).unwrap();

            if i < chunks.len() - 1 {
                assert!(
                    json.get("usage").is_none(),
                    "Usage should not be in intermediate chunks. Found in chunk {}",
                    i
                );
            } else {
                if let Some(usage) = json.get("usage") {
                    found_usage = true;
                    assert_eq!(usage["prompt_tokens"], 5);
                    assert_eq!(usage["completion_tokens"], 2);
                    assert_eq!(usage["total_tokens"], 7);
                }
                if let Some(choices) = json.get("choices") {
                    if let Some(choice) = choices.get(0) {
                        if let Some(finish_reason) = choice.get("finish_reason") {
                            if finish_reason.as_str() == Some("stop") {
                                found_finish = true;
                            }
                        }
                    }
                }
            }
        }
        assert!(found_usage, "Usage should be found in the last chunk");
        assert!(found_finish, "Finish reason should be strictly 'stop'");
    }

    // [TRUNCATION FIX] The terminal SSE event can arrive without a trailing newline.
    // The public text, finishReason and usage in that tail packet must survive, the
    // thought must still be hidden, and the stream must close with [DONE].
    #[tokio::test]
    async fn test_openai_stream_flushes_final_event_without_newline() {
        let chunk = json!({
            "candidates": [{
                "content": {"parts": [
                    {"text": "hidden reasoning", "thought": true, "thoughtSignature": "sig"},
                    {"text": "final public text"}
                ]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 4,
                "totalTokenCount": 15
            }
        });
        // Note: no trailing "\n\n" — the tail packet is unterminated.
        let items: Vec<Result<Bytes, reqwest::Error>> =
            vec![Ok(Bytes::from(format!("data: {}", chunk)))];
        let mut output = create_openai_sse_stream(
            Box::pin(stream::iter(items)),
            "gemini-3-flash".to_string(),
            "test-session".to_string(),
            0,
            true,
        );

        let mut body = String::new();
        while let Some(item) = output.next().await {
            body.push_str(&String::from_utf8_lossy(&item.unwrap()));
        }

        assert!(body.contains("final public text"), "public tail text must survive: {}", body);
        assert!(!body.contains("hidden reasoning"), "thought must stay hidden");
        assert!(body.contains("\"finish_reason\":\"stop\""), "finish reason must be present");
        assert!(body.contains("\"prompt_tokens\":11"), "usage must be present");
        assert!(body.contains("\"completion_tokens\":4"));
        assert!(body.contains("data: [DONE]"));
        assert!(!body.contains("incomplete_upstream_stream"), "a completed stream is not incomplete");
    }

    // [TRUNCATION FIX] A premature EOF without any finishReason / usage must surface a
    // visible error instead of a bare [DONE] that clients treat as success.
    #[tokio::test]
    async fn test_openai_stream_reports_incomplete_eof() {
        let chunk = json!({
            "candidates": [{"content": {"parts": [{"text": "partial answer"}]}}]
        });
        let items: Vec<Result<Bytes, reqwest::Error>> =
            vec![Ok(Bytes::from(format!("data: {}\n\n", chunk)))];
        let mut output = create_openai_sse_stream(
            Box::pin(stream::iter(items)),
            "gemini-3-flash".to_string(),
            "test-session".to_string(),
            0,
            false,
        );

        let mut body = String::new();
        while let Some(item) = output.next().await {
            body.push_str(&String::from_utf8_lossy(&item.unwrap()));
        }

        assert!(body.contains("partial answer"), "partial content should still be delivered");
        assert!(body.contains("incomplete_upstream_stream"), "premature EOF must surface an error: {}", body);
        assert!(!body.contains("\"finish_reason\":\"stop\""), "must not fake a stop finish reason");
    }

    #[tokio::test]
    async fn test_openai_stream_hides_thinking_output() {
        let chunk = json!({
            "candidates": [{
                "content": {"parts": [
                    {"text": "private reasoning", "thought": true, "thoughtSignature": "sig"},
                    {"text": "public answer"}
                ]},
                "finishReason": "STOP"
            }]
        });
        let items: Vec<Result<Bytes, reqwest::Error>> = vec![Ok(Bytes::from(format!(
            "data: {}\n\n",
            chunk
        )))];
        let mut output = create_openai_sse_stream(
            Box::pin(stream::iter(items)),
            "gemini-3-flash".to_string(),
            "test-session".to_string(),
            0,
            true,
        );

        let mut body = String::new();
        while let Some(item) = output.next().await {
            body.push_str(&String::from_utf8_lossy(&item.unwrap()));
        }

        assert!(!body.contains("private reasoning"));
        assert!(!body.contains("reasoning_content"));
        assert!(body.contains("public answer"));
    }

    #[tokio::test]
    async fn test_codex_stream_falls_back_to_non_zero_responses_usage() {
        let chunk = json!({
            "candidates": [{
                "content": {"parts": [{"text": "fallback usage"}]},
                "finishReason": "STOP"
            }]
        });
        let items: Vec<Result<Bytes, reqwest::Error>> = vec![Ok(Bytes::from(format!(
            "data: {}\n\n",
            chunk
        )))];
        let mut output = create_codex_sse_stream(
            Box::pin(stream::iter(items)),
            "gemini-3-flash".to_string(),
            "test-session".to_string(),
            0,
            true,
            123,
        );

        let mut completed = None;
        while let Some(item) = output.next().await {
            let chunk = String::from_utf8_lossy(&item.unwrap()).to_string();
            for line in chunk.lines().filter(|line| line.starts_with("data: ")) {
                if line.trim() == "data: [DONE]" {
                    continue;
                }
                let event: Value = serde_json::from_str(line.trim_start_matches("data: ")).unwrap();
                if event.get("type").and_then(Value::as_str) == Some("response.completed") {
                    completed = Some(event);
                }
            }
        }

        let usage = completed.unwrap()["response"]["usage"].clone();
        assert_eq!(usage["input_tokens"], 123);
        assert!(usage["output_tokens"].as_u64().unwrap() > 0);
        assert!(usage["total_tokens"].as_u64().unwrap() > 123);
    }

    #[tokio::test]
    async fn test_codex_stream_flushes_final_event_without_newline() {
        let chunk = json!({
            "candidates": [{
                "content": {"parts": [{"text": "tail survives"}]},
                "finishReason": "STOP"
            }]
        });
        let items: Vec<Result<Bytes, reqwest::Error>> = vec![Ok(Bytes::from(format!(
            "data: {}",
            chunk
        )))];
        let mut output = create_codex_sse_stream(
            Box::pin(stream::iter(items)),
            "gemini-3-flash".to_string(),
            "test-session".to_string(),
            0,
            true,
            10,
        );

        let mut body = String::new();
        while let Some(item) = output.next().await {
            body.push_str(&String::from_utf8_lossy(&item.unwrap()));
        }

        assert!(body.contains("tail survives"));
        assert!(body.contains("response.completed"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn test_legacy_stream_emits_usage_without_gemini_usage_metadata() {
        let chunk = json!({
            "candidates": [{
                "content": {"parts": [{"text": "OK"}]},
                "finishReason": "STOP"
            }]
        });
        let items: Vec<Result<Bytes, reqwest::Error>> = vec![Ok(Bytes::from(format!(
            "data: {}\n\n",
            chunk
        )))];
        let mut output = create_legacy_sse_stream(
            Box::pin(stream::iter(items)),
            "gemini-3.5-flash-low".to_string(),
            "test-session".to_string(),
            0,
            true,
            7,
        );
        let mut body = String::new();
        while let Some(item) = output.next().await {
            body.push_str(&String::from_utf8_lossy(&item.unwrap()));
        }
        assert!(body.contains("\"finish_reason\":\"stop\""));
        assert!(body.contains("\"prompt_tokens\":7"));
        assert!(body.contains("\"completion_tokens\":1"));
    }

    #[tokio::test]
    async fn test_legacy_stream_reports_incomplete_eof() {
        let chunk = json!({
            "candidates": [{"content": {"parts": [{"text": "partial"}]}}]
        });
        let items: Vec<Result<Bytes, reqwest::Error>> = vec![Ok(Bytes::from(format!(
            "data: {}\n\n",
            chunk
        )))];
        let mut output = create_legacy_sse_stream(
            Box::pin(stream::iter(items)),
            "gemini-3.5-flash-low".to_string(),
            "test-session".to_string(),
            0,
            true,
            7,
        );
        let mut body = String::new();
        while let Some(item) = output.next().await {
            body.push_str(&String::from_utf8_lossy(&item.unwrap()));
        }
        assert!(body.contains("partial"));
        assert!(body.contains("incomplete_upstream_stream"));
        assert!(!body.contains("\"finish_reason\":\"stop\""));
    }
}
