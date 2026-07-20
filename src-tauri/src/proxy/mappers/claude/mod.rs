// Claude mapper 模块
// 负责 Claude ↔ Gemini 协议转换

pub mod collector;
pub mod models;
pub mod request;
pub mod response;
pub mod streaming;
pub mod thinking_utils;
pub mod utils;

use crate::proxy::common::client_adapter::ClientAdapter;
pub use collector::collect_stream_to_json;
pub use models::*;
pub use request::{
    clean_cache_control_from_messages, merge_consecutive_messages, transform_claude_request_in,
};
pub use response::transform_response;
pub use streaming::{PartProcessor, StreamingState};
pub use thinking_utils::{
    close_tool_loop_for_thinking, filter_invalid_thinking_blocks_with_family,
}; // [NEW]

use bytes::Bytes;
use futures::Stream;
use std::pin::Pin;

fn emit_stream_error(message: impl Into<String>) -> Bytes {
    let event = serde_json::json!({
        "type": "error",
        "error": {
            "type": "api_error",
            "message": message.into()
        }
    });
    Bytes::from(format!("event: error\ndata: {event}\n\n"))
}

#[derive(Default)]
struct CompletionState {
    finish_reason: Option<String>,
    prompt_block_reason: Option<String>,
    usage: Option<UsageMetadata>,
}

fn emit_pending_finish(
    state: &mut StreamingState,
    completion: &CompletionState,
    trace_id: &str,
    email: &str,
) -> Result<Vec<Bytes>, String> {
    let finish_reason = completion
        .finish_reason
        .as_deref()
        .or_else(|| completion.prompt_block_reason.as_ref().map(|_| "SAFETY"))
        .ok_or_else(|| {
            "Gemini stream ended without a candidate finishReason or prompt block".to_string()
        })?;

    if let Some(usage) = &completion.usage {
        let cached_tokens = usage.cached_content_token_count.unwrap_or(0);
        let cache_info = if cached_tokens > 0 {
            format!(", Cached: {cached_tokens}")
        } else {
            String::new()
        };
        tracing::info!(
            "[{}] ✓ Stream completed | Account: {} | In: {} tokens | Out: {} tokens{}",
            trace_id,
            email,
            usage
                .prompt_token_count
                .unwrap_or(0)
                .saturating_sub(cached_tokens),
            usage.output_token_count(),
            cache_info
        );
    }

    Ok(state.emit_finish(Some(finish_reason), completion.usage.as_ref()))
}

/// 创建从 Gemini SSE 流到 Claude SSE 流的转换
pub fn create_claude_sse_stream<S, E>(
    mut gemini_stream: Pin<Box<S>>,
    trace_id: String,
    email: String,
    session_id: Option<String>, // [NEW v3.3.17] Session ID for signature caching
    scaling_enabled: bool,      // [NEW] Flag for context usage scaling
    context_limit: u32,
    estimated_prompt_tokens: Option<u32>, // [FIX] Estimated tokens for calibrator learning
    message_count: usize,                 // [NEW v4.0.0] Message count for rewind detection
    client_adapter: Option<std::sync::Arc<dyn ClientAdapter>>, // [NEW] Adapter reference
    registered_tool_names: Vec<String>,   // [FIX #MCP] Tool names for fuzzy matching
) -> Pin<Box<dyn Stream<Item = Result<Bytes, String>> + Send>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + ?Sized + 'static,
    E: std::fmt::Display + Send + 'static,
{
    use async_stream::stream;
    use bytes::BytesMut;
    use futures::StreamExt;

    Box::pin(stream! {
        let mut state = StreamingState::new();
        state.session_id = session_id; // Set session ID for signature caching
        state.message_count = message_count; // [NEW v4.0.0] Set message count
        state.scaling_enabled = scaling_enabled; // Set scaling enabled flag
        state.context_limit = context_limit;
        state.estimated_prompt_tokens = estimated_prompt_tokens; // [FIX] Pass estimated tokens
        state.set_client_adapter(client_adapter); // [NEW] Set adapter
        state.set_registered_tool_names(registered_tool_names); // [FIX #MCP] Set tool names
        let mut buffer = BytesMut::new();
        let mut stream_failed = false;
        let mut completion = CompletionState::default();

        'upstream: loop {
            // [NEW] 60秒心跳保活: 延长超时时间以增加网络抖动容错
            let next_chunk = tokio::time::timeout(
                std::time::Duration::from_secs(60),
                gemini_stream.next()
            ).await;

            match next_chunk {
                Ok(Some(chunk_result)) => {
                    match chunk_result {
                        Ok(chunk) => {
                            buffer.extend_from_slice(&chunk);

                            // Process complete lines
                            while let Some(pos) = buffer.iter().position(|&b| b == b'\n') {
                                let line_raw = buffer.split_to(pos + 1);
                                let line = match std::str::from_utf8(&line_raw) {
                                    Ok(line) => line.trim(),
                                    Err(e) => {
                                        stream_failed = true;
                                        yield Ok(emit_stream_error(format!("Invalid UTF-8 in Gemini SSE stream: {e}")));
                                        break 'upstream;
                                    }
                                };
                                if line.is_empty() { continue; }

                                match process_sse_line(
                                    line,
                                    &mut state,
                                    &mut completion,
                                    &trace_id,
                                    &email,
                                ) {
                                    Ok(Some(sse_chunks)) => {
                                        for sse_chunk in sse_chunks {
                                            yield Ok(sse_chunk);
                                        }
                                        if state.message_stop_sent {
                                            break 'upstream;
                                        }
                                    }
                                    Ok(None) => {}
                                    Err(e) => {
                                        stream_failed = true;
                                        yield Ok(emit_stream_error(e));
                                        break 'upstream;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            stream_failed = true;
                            yield Ok(emit_stream_error(format!("Gemini upstream stream error: {e}")));
                            break 'upstream;
                        }
                    }
                }
                Ok(None) => break 'upstream,
                Err(_) => {
                    // 超时，发送心跳包 (SSE Comment 格式)
                    yield Ok(Bytes::from(": ping\n\n"));
                }
            }
        }

        // [FIX #1732] Mandatory Flush remaining buffer on stream termination
        // Prevents hangs when the last SSE chunk doesn't end with a newline (network fragmentation)
        if !stream_failed && !state.message_stop_sent && !buffer.is_empty() {
            match std::str::from_utf8(&buffer) {
                Ok(line) => {
                    let line = line.trim();
                    if !line.is_empty() {
                        tracing::debug!("[{}] SSE Termination: Flushing remaining {} bytes in buffer", trace_id, buffer.len());
                        match process_sse_line(
                            line,
                            &mut state,
                            &mut completion,
                            &trace_id,
                            &email,
                        ) {
                            Ok(Some(sse_chunks)) => {
                                for sse_chunk in sse_chunks {
                                    yield Ok(sse_chunk);
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                stream_failed = true;
                                yield Ok(emit_stream_error(e));
                            }
                        }
                    }
                }
                Err(e) => {
                    stream_failed = true;
                    yield Ok(emit_stream_error(format!("Invalid UTF-8 in Gemini SSE stream: {e}")));
                }
            }
        }

        if !stream_failed && !state.message_stop_sent {
            match emit_pending_finish(&mut state, &completion, &trace_id, &email) {
                Ok(chunks) => {
                    for chunk in chunks {
                        yield Ok(chunk);
                    }
                }
                Err(e) => yield Ok(emit_stream_error(e)),
            }
        }
    })
}

/// 处理单行 SSE 数据
fn process_sse_line(
    line: &str,
    state: &mut StreamingState,
    completion: &mut CompletionState,
    trace_id: &str,
    email: &str,
) -> Result<Option<Vec<Bytes>>, String> {
    let Some(data_str) = line.strip_prefix("data:") else {
        return Ok(None);
    };
    let data_str = data_str.trim();
    if data_str.is_empty() {
        return Ok(None);
    }

    if data_str == "[DONE]" {
        if state.message_stop_sent {
            return Ok(None);
        }
        return emit_pending_finish(state, completion, trace_id, email).map(|chunks| {
            if chunks.is_empty() {
                None
            } else {
                Some(chunks)
            }
        });
    }

    // 解析 JSON
    let json_value: serde_json::Value = match serde_json::from_str(data_str) {
        Ok(v) => v,
        Err(e) => return Err(format!("Invalid Gemini SSE JSON: {e}")),
    };

    if let Some(error) = json_value.get("error").filter(|error| !error.is_null()) {
        return Err(format!("Gemini upstream error: {error}"));
    }

    let mut chunks = Vec::new();

    // 解包 response 字段 (如果存在)
    let raw_json = json_value.get("response").unwrap_or(&json_value);
    if let Some(error) = raw_json.get("error").filter(|error| !error.is_null()) {
        return Err(format!("Gemini upstream error: {error}"));
    }

    if let Some(usage) = raw_json
        .get("usageMetadata")
        .or_else(|| json_value.get("usageMetadata"))
        .and_then(|usage| serde_json::from_value::<UsageMetadata>(usage.clone()).ok())
    {
        if let Some(current) = completion.usage.as_mut() {
            current.merge_cumulative(&usage);
        } else {
            completion.usage = Some(usage);
        }
    }

    if let Some(block_reason) = raw_json
        .get("promptFeedback")
        .and_then(|feedback| feedback.get("blockReason"))
        .and_then(serde_json::Value::as_str)
        .filter(|reason| !reason.is_empty())
    {
        completion.prompt_block_reason = Some(block_reason.to_string());
    }

    // 发送 message_start
    if !state.message_start_sent {
        chunks.push(state.emit_message_start(raw_json));
    }

    // 捕获 groundingMetadata (Web Search)
    if let Some(candidate) = raw_json.get("candidates").and_then(|c| c.get(0)) {
        if let Some(grounding) = candidate.get("groundingMetadata") {
            // 提取搜索词
            if let Some(query) = grounding
                .get("webSearchQueries")
                .and_then(|v| v.as_array())
                .and_then(|arr| arr.get(0))
                .and_then(|v| v.as_str())
            {
                state.web_search_query = Some(query.to_string());
            }

            // 提取结果块
            if let Some(chunks_arr) = grounding.get("groundingChunks").and_then(|v| v.as_array()) {
                state.grounding_chunks = Some(chunks_arr.clone());
            } else if let Some(chunks_arr) = grounding
                .get("grounding_metadata")
                .and_then(|m| m.get("groundingChunks"))
                .and_then(|v| v.as_array())
            {
                state.grounding_chunks = Some(chunks_arr.clone());
            }
        }
    }

    // 处理所有 parts
    if let Some(parts) = raw_json
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|cand| cand.get("content"))
        .and_then(|content| content.get("parts"))
        .and_then(|p| p.as_array())
    {
        for part_value in parts {
            if let Ok(part) = serde_json::from_value::<GeminiPart>(part_value.clone()) {
                let mut processor = PartProcessor::new(state);
                chunks.extend(processor.process(&part));
            }
        }
    }

    // Process grounding metadata (googleSearch results) and append as citations
    // [DISABLED] Temporarily disabled to fix Cherry Studio compatibility
    // Cherry Studio doesn't recognize "web_search_tool_result" type, causing validation errors
    // Search results are still displayed via Markdown text block in streaming.rs (lines 341-381)

    /*
    if let Some(grounding) = raw_json
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|cand| cand.get("groundingMetadata"))
    {
        if let Some(citation_chunks) = process_grounding_metadata(grounding, state) {
            chunks.extend(citation_chunks);
        }
    }
    */

    // 检查是否结束
    if let Some(finish_reason) = raw_json
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|cand| cand.get("finishReason"))
        .and_then(|f| f.as_str())
        .filter(|reason| !reason.is_empty())
    {
        if completion.finish_reason.is_none() {
            completion.finish_reason = Some(finish_reason.to_string());
        }
    }

    if chunks.is_empty() {
        Ok(None)
    } else {
        Ok(Some(chunks))
    }
}

/// Process grounding metadata from Gemini's googleSearch and emit as Claude web_search blocks
#[allow(dead_code)] // Temporarily disabled for Cherry Studio compatibility, kept for future use
fn process_grounding_metadata(
    metadata: &serde_json::Value,
    state: &mut StreamingState,
) -> Option<Vec<Bytes>> {
    use serde_json::json;

    // Extract search queries and grounding chunks
    let search_queries = metadata
        .get("webSearchQueries")
        .and_then(|q| q.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
        .unwrap_or_default();

    let grounding_chunks = metadata.get("groundingChunks").and_then(|c| c.as_array())?;

    if grounding_chunks.is_empty() {
        return None;
    }

    // Generate a unique tool_use_id
    let tool_use_id = format!(
        "srvtoolu_{}",
        crate::proxy::common::utils::generate_random_id()
    );

    // Build search results array
    let mut search_results = Vec::new();
    for chunk in grounding_chunks.iter() {
        if let Some(web) = chunk.get("web") {
            let title = web
                .get("title")
                .and_then(|t| t.as_str())
                .unwrap_or("Source");
            let uri = web.get("uri").and_then(|u| u.as_str()).unwrap_or("");
            if !uri.is_empty() {
                search_results.push(json!({
                    "url": uri,
                    "title": title,
                    "encrypted_content": "", // Gemini doesn't provide this
                    "page_age": null
                }));
            }
        }
    }

    if search_results.is_empty() {
        return None;
    }

    let search_query = search_queries
        .first()
        .map(|s| s.to_string())
        .unwrap_or_default();

    tracing::debug!(
        "[Grounding] Emitting {} search results for query: {}",
        search_results.len(),
        search_query
    );

    let mut chunks = Vec::new();

    // 1. Emit server_tool_use block (start)
    let server_tool_use_start = json!({
        "type": "content_block_start",
        "index": state.block_index,
        "content_block": {
            "type": "server_tool_use",
            "id": tool_use_id,
            "name": "web_search",
            "input": {
                "query": search_query
            }
        }
    });
    chunks.push(Bytes::from(format!(
        "event: content_block_start\ndata: {}\n\n",
        server_tool_use_start
    )));

    // server_tool_use block stop
    let server_tool_use_stop = json!({
        "type": "content_block_stop",
        "index": state.block_index
    });
    chunks.push(Bytes::from(format!(
        "event: content_block_stop\ndata: {}\n\n",
        server_tool_use_stop
    )));
    state.block_index += 1;

    // 2. Emit web_search_tool_result block (start)
    let tool_result_start = json!({
        "type": "content_block_start",
        "index": state.block_index,
        "content_block": {
            "type": "web_search_tool_result",
            "tool_use_id": tool_use_id,
            "content": search_results
        }
    });
    chunks.push(Bytes::from(format!(
        "event: content_block_start\ndata: {}\n\n",
        tool_result_start
    )));

    // web_search_tool_result block stop
    let tool_result_stop = json!({
        "type": "content_block_stop",
        "index": state.block_index
    });
    chunks.push(Bytes::from(format!(
        "event: content_block_stop\ndata: {}\n\n",
        tool_result_stop
    )));
    state.block_index += 1;

    Some(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{stream, StreamExt};

    async fn collect_output<S>(stream: S) -> String
    where
        S: Stream<Item = Result<Bytes, String>> + Send + 'static,
    {
        let mut claude_stream = create_claude_sse_stream(
            Box::pin(stream),
            "trace_test".to_string(),
            "test@example.com".to_string(),
            None,
            false,
            1_000,
            None,
            1,
            None,
            Vec::new(),
        );

        let mut output = String::new();
        while let Some(result) = claude_stream.next().await {
            output.push_str(&String::from_utf8(result.expect("mapper stream item").to_vec()).unwrap());
        }
        output
    }

    #[test]
    fn done_without_finish_reason_is_not_completion() {
        let mut state = StreamingState::new();
        let mut completion = CompletionState::default();
        let error = process_sse_line(
            "data: [DONE]",
            &mut state,
            &mut completion,
            "test_id",
            "test@example.com",
        )
        .expect_err("DONE is only framing");
        assert!(error.contains("without a candidate finishReason"));
        assert!(!state.message_stop_sent);
    }

    #[test]
    fn text_without_finish_reason_does_not_stop_message() {
        let mut state = StreamingState::new();
        let mut completion = CompletionState::default();
        let test_data = r#"data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}],"usageMetadata":{},"modelVersion":"test","responseId":"123"}"#;

        let chunks = process_sse_line(
            test_data,
            &mut state,
            &mut completion,
            "test_id",
            "test@example.com",
        )
        .expect("valid JSON")
        .expect("content events");
        let all_text: String = chunks
            .iter()
            .map(|b| String::from_utf8(b.to_vec()).unwrap_or_default())
            .collect();
        assert!(all_text.contains("message_start"));
        assert!(all_text.contains("content_block_start"));
        assert!(all_text.contains("Hello"));
        assert!(!all_text.contains("message_stop"));
        assert!(!state.message_stop_sent);
    }

    #[tokio::test]
    async fn abrupt_eof_after_thinking_emits_error_without_fake_completion() {
        let thinking = serde_json::json!({
            "candidates": [{
                "content": { "parts": [{ "text": "Thinking...", "thought": true }] }
            }],
            "modelVersion": "gemini-2.0-flash-thinking",
            "responseId": "msg_interrupted"
        });
        let chunks = vec![Ok(Bytes::from(format!("data: {thinking}\n\n")))];
        let output = collect_output(stream::iter(chunks)).await;

        assert!(output.contains("Thinking..."));
        assert!(output.contains("event: error"));
        assert!(output.contains("without a candidate finishReason"));
        assert!(!output.contains("Recovered by Antigravity"));
        assert!(!output.contains("\"output_tokens\":100"));
        assert!(!output.contains("message_stop"));
    }

    #[tokio::test]
    async fn split_chunks_and_unterminated_tail_preserve_max_tokens() {
        let chunks = vec![
            Ok(Bytes::from_static(
                b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hel",
            )),
            Ok(Bytes::from_static(
                b"lo\"}]},\"finishReason\":\"MAX_TOKENS\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":1}}",
            )),
        ];
        let output = collect_output(stream::iter(chunks)).await;

        assert!(output.contains("hello"));
        assert!(output.contains("\"stop_reason\":\"max_tokens\""));
        assert!(output.contains("message_stop"));
        assert!(!output.contains("event: error"));
    }

    #[tokio::test]
    async fn upstream_error_after_content_never_sends_message_stop() {
        let chunks = vec![
            Ok(Bytes::from_static(
                b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n",
            )),
            Err("connection reset".to_string()),
        ];
        let output = collect_output(stream::iter(chunks)).await;

        assert!(output.contains("partial"));
        assert!(output.contains("event: error"));
        assert!(output.contains("connection reset"));
        assert!(!output.contains("message_stop"));
    }

    #[tokio::test]
    async fn finish_then_usage_only_chunk_keeps_final_usage() {
        let chunks = vec![
            Ok(Bytes::from_static(
                b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"done\"}]},\"finishReason\":\"STOP\"}]}\n\n",
            )),
            Ok(Bytes::from_static(
                b"data: {\"usageMetadata\":{\"promptTokenCount\":11,\"candidatesTokenCount\":7,\"totalTokenCount\":18}}\n\n",
            )),
            Ok(Bytes::from_static(b"data: [DONE]\n\n")),
        ];
        let output = collect_output(stream::iter(chunks)).await;

        assert!(output.contains("done"));
        assert!(output.contains("\"stop_reason\":\"end_turn\""));
        assert!(output.contains("\"output_tokens\":7"));
        assert_eq!(output.matches("event: message_stop").count(), 1);
        assert!(!output.contains("event: error"));
    }

    #[tokio::test]
    async fn prompt_feedback_block_finishes_as_refusal() {
        let chunks = vec![Ok(Bytes::from_static(
            b"data: {\"promptFeedback\":{\"blockReason\":\"SAFETY\"},\"usageMetadata\":{\"promptTokenCount\":4,\"totalTokenCount\":4}}\n\ndata: [DONE]\n\n",
        ))];
        let output = collect_output(stream::iter(chunks)).await;

        assert!(output.contains("\"stop_reason\":\"refusal\""));
        assert!(output.contains("event: message_stop"));
        assert!(!output.contains("event: error"));
    }

    #[tokio::test]
    async fn later_zero_usage_snapshot_does_not_erase_real_counts() {
        let chunks = vec![Ok(Bytes::from_static(
            b"data: {\"usageMetadata\":{\"promptTokenCount\":12,\"candidatesTokenCount\":7,\"totalTokenCount\":19}}\n\ndata: {\"candidates\":[{\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":0,\"candidatesTokenCount\":0,\"totalTokenCount\":0}}\n\ndata: [DONE]\n\n",
        ))];
        let output = collect_output(stream::iter(chunks)).await;

        assert!(output.contains("\"input_tokens\":12"));
        assert!(output.contains("\"output_tokens\":7"));
        assert!(!output.contains("event: error"));
    }
}
