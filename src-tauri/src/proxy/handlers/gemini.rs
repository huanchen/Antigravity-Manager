// Gemini Handler
use axum::{
    extract::State,
    extract::{Json, Path},
    http::StatusCode,
    response::IntoResponse,
};
use bytes::Bytes;
use serde_json::{json, Value};
use tracing::{debug, error, info};

use crate::proxy::common::client_adapter::CLIENT_ADAPTERS;
use crate::proxy::debug_logger;
use crate::proxy::handlers::common::{
    account_rotation_attempts, acquire_global_image_permit, acquire_image_account_permit,
    apply_retry_strategy, determine_retry_strategy, image_max_attempts, is_retryable_image_error,
    should_rotate_account,
};
use crate::proxy::mappers::gemini::{unwrap_response, wrap_request, wrap_request_v2};
use crate::proxy::server::AppState;
use crate::proxy::session_manager::SessionManager;
use crate::proxy::upstream::client::mask_email;
use axum::http::HeaderMap;

#[derive(Default)]
struct NativeStreamState {
    saw_finish_reason: bool,
    saw_prompt_block: bool,
    saw_done: bool,
    usage_metadata: Option<Value>,
}

/// Remove only visible thought text from a Gemini response. Signatures,
/// function calls, inline data, usage metadata, and finish markers remain
/// untouched so downstream tool continuations and billing stay valid.
fn hide_gemini_thought_text(value: &mut Value) {
    let target = if value.get("response").is_some() {
        value.get_mut("response").expect("response exists")
    } else {
        value
    };
    let Some(candidates) = target.get_mut("candidates").and_then(Value::as_array_mut) else {
        return;
    };
    for candidate in candidates {
        let Some(parts) = candidate
            .get_mut("content")
            .and_then(|content| content.get_mut("parts"))
            .and_then(Value::as_array_mut)
        else {
            continue;
        };
        for part in parts {
            if part.get("thought").and_then(Value::as_bool) == Some(true) {
                if let Some(obj) = part.as_object_mut() {
                    obj.remove("text");
                }
            }
        }
    }
}

fn hide_native_sse_line(line: &str) -> String {
    let trimmed = line.trim();
    let Some(data) = trimmed.strip_prefix("data:").map(str::trim) else {
        return line.to_string();
    };
    if data.is_empty() || data == "[DONE]" {
        return line.to_string();
    }
    let Ok(mut payload) = serde_json::from_str::<Value>(data) else {
        return line.to_string();
    };
    hide_gemini_thought_text(&mut payload);
    format!("data: {}\n\n", serde_json::to_string(&payload).unwrap_or_else(|_| data.to_string()))
}

#[derive(Debug)]
enum NativeLineAction {
    Ignore,
    Forward(String),
    Done(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativePeekDisposition {
    Ignore,
    Meaningful,
    Error,
}

fn classify_native_peek_payload(payload: &Value) -> NativePeekDisposition {
    let response = payload.get("response").unwrap_or(payload);
    if payload.get("error").is_some_and(|error| !error.is_null())
        || response.get("error").is_some_and(|error| !error.is_null())
    {
        return NativePeekDisposition::Error;
    }
    if response
        .get("promptFeedback")
        .and_then(|feedback| feedback.get("blockReason"))
        .and_then(Value::as_str)
        .is_some_and(|reason| !reason.is_empty())
    {
        return NativePeekDisposition::Meaningful;
    }

    let meaningful = response
        .get("candidates")
        .and_then(Value::as_array)
        .is_some_and(|candidates| {
            candidates.iter().any(|candidate| {
                candidate
                    .get("finishReason")
                    .and_then(Value::as_str)
                    .is_some_and(|reason| !reason.is_empty())
                    || candidate
                        .get("content")
                        .and_then(|content| content.get("parts"))
                        .and_then(Value::as_array)
                        .is_some_and(|parts| {
                            parts.iter().any(|part| {
                                part.get("text")
                                    .and_then(Value::as_str)
                                    .is_some_and(|text| !text.is_empty())
                                    || part.get("functionCall").is_some()
                                    || part.get("inlineData").is_some()
                            })
                        })
            })
        });
    if meaningful {
        NativePeekDisposition::Meaningful
    } else {
        NativePeekDisposition::Ignore
    }
}

/// Inspect all complete frames accumulated during peek. A trailing partial JSON
/// frame is left pending until more transport bytes arrive.
fn classify_native_peek_buffer(buffer: &[u8]) -> Result<NativePeekDisposition, String> {
    let mut meaningful = false;
    let mut offset = 0;
    while let Some(relative) = buffer[offset..].iter().position(|byte| *byte == b'\n') {
        let end = offset + relative + 1;
        let line = std::str::from_utf8(&buffer[offset..end])
            .map_err(|error| format!("Invalid UTF-8 in Gemini SSE peek: {error}"))?;
        offset = end;
        let line = line.trim();
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            return Ok(if meaningful {
                NativePeekDisposition::Meaningful
            } else {
                NativePeekDisposition::Error
            });
        }
        let payload: Value = serde_json::from_str(data)
            .map_err(|error| format!("Invalid Gemini SSE JSON during peek: {error}"))?;
        match classify_native_peek_payload(&payload) {
            NativePeekDisposition::Error => return Ok(NativePeekDisposition::Error),
            NativePeekDisposition::Meaningful => meaningful = true,
            NativePeekDisposition::Ignore => {}
        }
    }

    let tail = &buffer[offset..];
    if !tail.is_empty() {
        match std::str::from_utf8(tail) {
            Ok(line) => {
                if let Some(data) = line.trim().strip_prefix("data:").map(str::trim) {
                    if data == "[DONE]" {
                        return Ok(if meaningful {
                            NativePeekDisposition::Meaningful
                        } else {
                            NativePeekDisposition::Error
                        });
                    }
                    if !data.is_empty() {
                        match serde_json::from_str::<Value>(data) {
                            Ok(payload) => match classify_native_peek_payload(&payload) {
                                NativePeekDisposition::Error => {
                                    return Ok(NativePeekDisposition::Error)
                                }
                                NativePeekDisposition::Meaningful => meaningful = true,
                                NativePeekDisposition::Ignore => {}
                            },
                            Err(error) if error.is_eof() => {}
                            Err(error) => {
                                return Err(format!(
                                    "Invalid Gemini SSE JSON during peek: {error}"
                                ))
                            }
                        }
                    }
                }
            }
            Err(error) if error.error_len().is_none() => {}
            Err(error) => return Err(format!("Invalid UTF-8 in Gemini SSE peek: {error}")),
        }
    }

    Ok(if meaningful {
        NativePeekDisposition::Meaningful
    } else {
        NativePeekDisposition::Ignore
    })
}

fn native_error_sse(message: impl Into<String>) -> Bytes {
    let error = json!({
        "error": {
            "code": 502,
            "message": message.into(),
            "status": "UPSTREAM_ERROR"
        }
    });
    Bytes::from(format!("data: {}\n\n", error))
}

/// Validates and unwraps one complete upstream SSE line.
///
/// `[DONE]` only terminates framing. A Gemini response is complete only after a
/// candidate supplied a non-empty `finishReason` (including `MAX_TOKENS`).
fn process_native_sse_line(
    line: &str,
    state: &mut NativeStreamState,
    session_id: &str,
    model_name: &str,
) -> Result<NativeLineAction, String> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(NativeLineAction::Ignore);
    }

    let Some(data) = line.strip_prefix("data:") else {
        return Ok(NativeLineAction::Forward(format!("{line}\n\n")));
    };
    let data = data.trim();
    if data.is_empty() {
        return Ok(NativeLineAction::Ignore);
    }

    if data == "[DONE]" {
        if state.saw_done {
            return Ok(NativeLineAction::Ignore);
        }
        if !state.saw_finish_reason && !state.saw_prompt_block {
            return Err(
                "Gemini stream sent [DONE] without a candidate finishReason or prompt block"
                    .to_string(),
            );
        }
        state.saw_done = true;
        return Ok(NativeLineAction::Done("data: [DONE]\n\n".to_string()));
    }

    let mut envelope: Value = serde_json::from_str(data)
        .map_err(|e| format!("Invalid Gemini SSE JSON: {e}"))?;
    if let Some(error) = envelope.get("error").filter(|error| !error.is_null()) {
        return Err(format!("Gemini upstream error: {error}"));
    }

    let envelope_usage = envelope.get("usageMetadata").cloned();
    let response = envelope.get("response").unwrap_or(&envelope);
    if let Some(error) = response.get("error").filter(|error| !error.is_null()) {
        return Err(format!("Gemini upstream error: {error}"));
    }

    if let Some(usage) = response
        .get("usageMetadata")
        .cloned()
        .or_else(|| envelope_usage.clone())
    {
        crate::proxy::mappers::gemini::collector::merge_usage_value(
            &mut state.usage_metadata,
            usage,
        );
    }
    if response
        .get("promptFeedback")
        .and_then(|feedback| feedback.get("blockReason"))
        .and_then(Value::as_str)
        .is_some_and(|reason| !reason.is_empty())
    {
        state.saw_prompt_block = true;
    }

    if let Some(candidates) = response.get("candidates").and_then(Value::as_array) {
        for candidate in candidates {
            if candidate
                .get("finishReason")
                .and_then(Value::as_str)
                .is_some_and(|reason| !reason.is_empty())
            {
                state.saw_finish_reason = true;
            }

            if let Some(parts) = candidate
                .get("content")
                .and_then(|content| content.get("parts"))
                .and_then(Value::as_array)
            {
                for part in parts {
                    if let Some(signature) = part
                        .get("thoughtSignature")
                        .or_else(|| part.get("thought_signature"))
                        .and_then(Value::as_str)
                    {
                        crate::proxy::SignatureCache::global().cache_session_signature(
                            session_id,
                            signature.to_string(),
                            1,
                        );
                        debug!(
                            "[Gemini-SSE] Cached signature (len: {}) for session: {}",
                            signature.len(),
                            session_id
                        );
                    }
                }
            }
        }
    }

    crate::proxy::mappers::gemini::wrapper::inject_ids_to_response(&mut envelope, model_name);
    let mut output = if let Some(mut inner) = envelope.get_mut("response").map(Value::take) {
        if inner.get("usageMetadata").is_none() {
            if let Some(usage) = envelope_usage {
                inner["usageMetadata"] = usage;
            }
        }
        inner
    } else {
        envelope
    };
    if let Some(usage) = state.usage_metadata.clone() {
        output["usageMetadata"] = usage;
    }
    let output = serde_json::to_string(&output)
        .map_err(|e| format!("Failed to serialize Gemini SSE response: {e}"))?;
    Ok(NativeLineAction::Forward(format!("data: {output}\n\n")))
}

/// Drains complete SSE lines from a byte buffer. When `flush_tail` is true,
/// the final unterminated line is processed as well.
fn drain_native_sse_buffer(
    buffer: &mut bytes::BytesMut,
    state: &mut NativeStreamState,
    session_id: &str,
    model_name: &str,
    flush_tail: bool,
) -> Result<(Vec<String>, bool), String> {
    drain_native_sse_buffer_with_options(
        buffer,
        state,
        session_id,
        model_name,
        flush_tail,
        false,
    )
}

fn drain_native_sse_buffer_with_options(
    buffer: &mut bytes::BytesMut,
    state: &mut NativeStreamState,
    session_id: &str,
    model_name: &str,
    flush_tail: bool,
    hide_thinking_output: bool,
) -> Result<(Vec<String>, bool), String> {
    let mut output = Vec::new();

    loop {
        let line = if let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
            Some(buffer.split_to(newline + 1))
        } else if flush_tail && !buffer.is_empty() {
            Some(buffer.split_to(buffer.len()))
        } else {
            None
        };
        let Some(line) = line else {
            break;
        };
        let line = std::str::from_utf8(&line)
            .map_err(|e| format!("Invalid UTF-8 in Gemini SSE stream: {e}"))?;

        match process_native_sse_line(line, state, session_id, model_name)? {
            NativeLineAction::Ignore => {}
            NativeLineAction::Forward(line) => output.push(if hide_thinking_output {
                hide_native_sse_line(&line)
            } else {
                line
            }),
            NativeLineAction::Done(line) => {
                output.push(line);
                return Ok((output, true));
            }
        }
    }

    Ok((output, false))
}

/// 处理 generateContent 和 streamGenerateContent
/// 路径参数: model_name, method (e.g. "gemini-pro", "generateContent")
pub async fn handle_generate(
    State(state): State<AppState>,
    Path(model_action): Path<String>,
    headers: HeaderMap,          // [NEW] Extract headers for adapter detection
    Json(mut body): Json<Value>, // 改为 mut 以支持修复提示词注入
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // 解析 model:method
    let (model_name, method) = if let Some((m, action)) = model_action.rsplit_once(':') {
        (m.to_string(), action.to_string())
    } else {
        (model_action, "generateContent".to_string())
    };

    crate::modules::logger::log_info(&format!(
        "Received Gemini request: {}/{}",
        model_name, method
    ));
    let trace_id = format!("req_{}", chrono::Utc::now().timestamp_subsec_millis());
    let debug_cfg = state.debug_logging.read().await.clone();
    let experimental = state.experimental.read().await.clone();
    let direct_non_stream = experimental.direct_non_stream;
    let hide_thinking_output = experimental.hide_thinking_output;

    // [NEW] Detect Client Adapter
    let client_adapter = CLIENT_ADAPTERS
        .iter()
        .find(|a| a.matches(&headers))
        .cloned();
    if client_adapter.is_some() {
        debug!("[{}] Client Adapter detected", trace_id);
    }

    // 1. 验证方法
    if method != "generateContent" && method != "streamGenerateContent" {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Unsupported method: {}", method),
        ));
    }
    if debug_logger::is_enabled(&debug_cfg) {
        let original_payload = json!({
            "kind": "original_request",
            "protocol": "gemini",
            "trace_id": trace_id,
            "original_model": model_name,
            "method": method,
            "request": body.clone(),
        });
        debug_logger::write_debug_payload(
            &debug_cfg,
            Some(&trace_id),
            "original_request",
            &original_payload,
        )
        .await;
    }
    let client_wants_stream = method == "streamGenerateContent";
    // [AUTO-CONVERSION] 强制内部流式化
    let force_stream_internally = !client_wants_stream && !direct_non_stream;
    let is_stream = client_wants_stream || force_stream_internally;

    if force_stream_internally {
        // debug!("[AutoConverter] Converting non-stream request to stream");
    }

    // 2. 获取 UpstreamClient 和 TokenManager
    let upstream = state.upstream.clone();
    let token_manager = state.token_manager;
    let pool_size = token_manager.len();

    let mut last_error = String::new();
    let mut last_email: Option<String> = None;
    let mut force_rotate = false;

    // Resolve request-level routing once before entering the retry loop.  This
    // keeps the retry budget and image concurrency guards in scope for every
    // attempt, while the concrete account model is still resolved below per
    // account (some accounts expose different image checkpoints).
    let base_mapped_model = crate::proxy::common::model_mapping::resolve_model_route(
        &model_name,
        &*state.custom_mapping.read().await,
    );
    let tools_val: Option<Vec<Value>> = body
        .get("tools")
        .and_then(|t| t.as_array())
        .map(|arr| {
            let mut flattened = Vec::new();
            for tool_entry in arr {
                if let Some(decls) = tool_entry
                    .get("functionDeclarations")
                    .and_then(|v| v.as_array())
                {
                    flattened.extend(decls.iter().cloned());
                } else {
                    flattened.push(tool_entry.clone());
                }
            }
            flattened
        });
    let config = crate::proxy::mappers::common_utils::resolve_request_config(
        &model_name,
        &base_mapped_model,
        &tools_val,
        None,        // size (not applicable for Gemini native protocol)
        None,        // quality
        None,        // image_size
        Some(&body), // [NEW] Pass request body for imageConfig parsing
    );
    let is_image_request = config.request_type == "image_gen";
    let max_attempts = if is_image_request {
        image_max_attempts(pool_size)
    } else {
        account_rotation_attempts(pool_size)
    };

    // Hold one global image permit across the complete request, including
    // retries. The per-account permit below is acquired after each account is
    // selected, allowing different accounts to make progress independently.
    let _global_image_permit = if is_image_request {
        match acquire_global_image_permit().await {
            Ok(permit) => Some(permit),
            Err(error) => {
                return Ok((
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(json!({
                        "error": {
                            "code": 429,
                            "message": error,
                            "status": "RESOURCE_EXHAUSTED"
                        }
                    })),
                )
                    .into_response())
            }
        }
    } else {
        None
    };

    for attempt in 0..max_attempts {
        // 3. Resolve the account-specific concrete model after token selection.
        // The request-level config above supplies stable routing/request type.

        // 4. 获取 Token (使用准确的 request_type)
        // 提取 SessionId (粘性指纹)
        let session_id = SessionManager::extract_gemini_session_id(&body, &model_name);

        // 关键：根据 force_rotate 标志决定是否轮换账号（支持 Grace Retry 原地重试）
        let (access_token, project_id, email, account_id, _wait_ms) = match token_manager
            .get_token(
                &config.request_type,
                force_rotate,
                Some(&session_id),
                &config.final_model,
            )
            .await
        {
            Ok(t) => t,
            Err(e) => {
                return Err((
                    StatusCode::SERVICE_UNAVAILABLE,
                    format!("Token error: {}", e),
                ));
            }
        };

        let mapped_model = token_manager
            .resolve_dynamic_model_for_account(&account_id, &base_mapped_model)
            .await;

        last_email = Some(email.clone());
        info!("✓ Using account: {} (type: {})", email, config.request_type);

        let _image_account_permit = if is_image_request {
            Some(acquire_image_account_permit(&account_id).await)
        } else {
            None
        };

        // 5. 包装请求 (project injection)
        // [FIX #765] Pass session_id to wrap_request for signature injection
        // [NEW] 获取完整 Token 对象以注入动态规格 (dynamic > static default > 65535)
        let token_obj = token_manager.get_token_by_id(&account_id);
        let wrapped_body = wrap_request_v2(
            &body,
            &project_id,
            &mapped_model,
            Some(account_id.as_str()),
            Some(&session_id),
            token_obj.as_ref(),
            Some(&token_manager),
        );

        if debug_logger::is_enabled(&debug_cfg) {
            let payload = json!({
                "kind": "v1internal_request",
                "protocol": "gemini",
                "trace_id": trace_id,
                "original_model": model_name,
                "mapped_model": mapped_model,
                "request_type": config.request_type,
                "attempt": attempt,
                "v1internal_request": wrapped_body.clone(),
            });
            debug_logger::write_debug_payload(
                &debug_cfg,
                Some(&trace_id),
                "v1internal_request",
                &payload,
            )
            .await;
        }

        // 5. 上游调用
        let query_string = if is_stream { Some("alt=sse") } else { None };
        let upstream_method = if is_stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        };

        // [FIX #1522] Inject Anthropic Beta Headers for Claude models
        let mut extra_headers = std::collections::HashMap::new();
        if mapped_model.to_lowercase().contains("claude") {
            extra_headers.insert("anthropic-beta".to_string(), "claude-code-20250219,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14".to_string());
            tracing::debug!(
                "[Gemini] Injected Anthropic beta headers for Claude model: {}",
                mapped_model
            );
        }

        let call_result = match upstream
            .call_v1_internal_with_headers(
                upstream_method,
                &access_token,
                wrapped_body,
                query_string,
                extra_headers.clone(),
                Some(account_id.as_str()),
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                last_error = e.clone();
                debug!(
                    "Gemini Request failed on attempt {}/{}: {}",
                    attempt + 1,
                    max_attempts,
                    e
                );
                force_rotate = true;
                continue;
            }
        };

        // [NEW] 记录端点降级日志到 debug 文件
        if !call_result.fallback_attempts.is_empty() && debug_logger::is_enabled(&debug_cfg) {
            let fallback_entries: Vec<serde_json::Value> = call_result
                .fallback_attempts
                .iter()
                .map(|a| {
                    json!({
                        "endpoint_url": a.endpoint_url,
                        "status": a.status,
                        "error": a.error,
                    })
                })
                .collect();
            let payload = json!({
                "kind": "endpoint_fallback",
                "protocol": "gemini",
                "trace_id": trace_id,
                "original_model": model_name,
                "mapped_model": mapped_model,
                "attempt": attempt,
                "account": mask_email(&email),
                "fallback_attempts": fallback_entries,
            });
            debug_logger::write_debug_payload(
                &debug_cfg,
                Some(&trace_id),
                "endpoint_fallback",
                &payload,
            )
            .await;
        }

        let response = call_result.response;
        // [NEW] 提取实际请求的上游端点 URL，用于日志记录和排查
        let upstream_url = response.url().to_string();
        let status = response.status();

        // [NEW] 提取官方 TraceID
        let cloud_code_trace_id = response
            .headers()
            .get("x-cloudaicompanion-trace-id")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string());

        if status.is_success() {
            // 6. 响应处理
            if is_stream {
                use axum::body::Body;
                use axum::response::Response;
                use bytes::BytesMut;
                use futures::StreamExt;

                let meta = json!({
                    "protocol": "gemini",
                    "trace_id": trace_id,
                    "original_model": model_name,
                    "mapped_model": mapped_model,
                    "request_type": config.request_type,
                    "attempt": attempt,
                    "status": status.as_u16(),
                    "upstream_url": upstream_url,
                });
                let mut response_stream = debug_logger::wrap_stream_with_debug(
                    Box::pin(response.bytes_stream()),
                    debug_cfg.clone(),
                    trace_id.clone(),
                    "upstream_response",
                    meta,
                );
                let mut buffer = BytesMut::new();
                let s_id = session_id.clone(); // Clone for stream closure

                // [FIX #859] Implement peek logic for Gemini stream to prevent 0-token 200 OK
                let mut first_chunk = None;
                let mut retry_gemini = false;
                let mut peek_buffer = bytes::BytesMut::new();

                // [NEW] 实施双阶段超时：第一阶段为 FirstChunkTimeout (300s / 5min)
                // 这精准对齐了官方 Worker 在模型冷启动（Initialization）阶段的极度耐心
                loop {
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(300),
                        response_stream.next(),
                    )
                    .await
                    {
                        Ok(Some(Ok(bytes))) => {
                            if bytes.is_empty() {
                                continue;
                            }
                            peek_buffer.extend_from_slice(&bytes);
                            match classify_native_peek_buffer(&peek_buffer) {
                                Ok(NativePeekDisposition::Ignore) => continue,
                                Ok(NativePeekDisposition::Meaningful) => {
                                    first_chunk = Some(peek_buffer.split().freeze());
                                    break;
                                }
                                Ok(NativePeekDisposition::Error) => {
                                    last_error =
                                        "Gemini error or premature terminal during peek".to_string();
                                    retry_gemini = true;
                                    break;
                                }
                                Err(error) => {
                                    last_error = error;
                                    retry_gemini = true;
                                    break;
                                }
                            }
                        }
                        Ok(Some(Err(e))) => {
                            tracing::warn!("[Gemini] Stream error during peek: {}, retrying...", e);
                            last_error = format!("Stream error: {}", e);
                            retry_gemini = true;
                            break;
                        }
                        Ok(None) => {
                            tracing::warn!("[Gemini] Stream ended during peek, retrying...");
                            last_error = "Empty or lifecycle-only response".to_string();
                            retry_gemini = true;
                            break;
                        }
                        Err(_) => {
                            tracing::warn!("[Gemini] First meaningful chunk timeout after 300s, retrying...");
                            last_error = "First meaningful chunk timeout".to_string();
                            retry_gemini = true;
                            break;
                        }
                    }
                }

                if retry_gemini {
                    token_manager.record_failure(&account_id);
                    force_rotate = true;
                    continue;
                }

                let s_id_for_stream = s_id.clone();
                let model_name_for_stream = mapped_model.clone();
                let stream = async_stream::stream! {
                    let mut first_data = first_chunk;
                    let mut meta_sent = false;
                    let mut native_state = NativeStreamState::default();
                    let mut stream_failed = false;

                    'upstream: loop {
                        // [NEW] 阶段 6.2: 补全 __cloudCodeMeta 响应元数据透传
                        // 官方 Worker 会将 TraceID 作为 SSE 流的第 0 个数据包下发
                        if !meta_sent {
                            if let Some(tid) = &cloud_code_trace_id {
                                let meta_pkg = serde_json::json!({
                                    "__cloudCodeMeta": {
                                        "traceId": tid
                                    }
                                });
                                yield Ok::<Bytes, String>(Bytes::from(format!("data: {}\n\n", serde_json::to_string(&meta_pkg).unwrap())));
                            }
                            meta_sent = true;
                        }

                        let item = if let Some(fd) = first_data.take() {
                            Some(Ok(fd))
                        } else {
                            // [NEW] 第二阶段为 StreamIdleTimeout (300s / 5min)
                            match tokio::time::timeout(std::time::Duration::from_secs(300), response_stream.next()).await {
                                Ok(next_item) => next_item,
                                Err(_) => {
                                    error!("[Gemini-SSE] Idle timeout after 300s, terminating stream");
                                    stream_failed = true;
                                    yield Ok::<Bytes, String>(native_error_sse(
                                        "Gemini upstream stream idle timeout before completion"
                                    ));
                                    break 'upstream;
                                }
                            }
                        };

                        let bytes = match item {
                            Some(Ok(b)) => b,
                            Some(Err(e)) => {
                                error!("[Gemini-SSE] Stream error: {}", e);
                                stream_failed = true;
                                yield Ok::<Bytes, String>(native_error_sse(format!(
                                    "Gemini upstream stream error: {e}"
                                )));
                                break 'upstream;
                            }
                            None => break 'upstream,
                        };

                        debug!("[Gemini-SSE] Received chunk: {} bytes", bytes.len());
                        buffer.extend_from_slice(&bytes);
                        match drain_native_sse_buffer_with_options(
                            &mut buffer,
                            &mut native_state,
                            &s_id_for_stream,
                            &model_name_for_stream,
                            false,
                            hide_thinking_output,
                        ) {
                            Ok((output, done)) => {
                                for line in output {
                                    yield Ok::<Bytes, String>(Bytes::from(line));
                                }
                                if done {
                                    break 'upstream;
                                }
                            }
                            Err(e) => {
                                stream_failed = true;
                                yield Ok::<Bytes, String>(native_error_sse(e));
                                break 'upstream;
                            }
                        }
                    }

                    // A final SSE data line may legally arrive without a trailing newline.
                    if !stream_failed && !native_state.saw_done && !buffer.is_empty() {
                        match drain_native_sse_buffer_with_options(
                            &mut buffer,
                            &mut native_state,
                            &s_id_for_stream,
                            &model_name_for_stream,
                            true,
                            hide_thinking_output,
                        ) {
                            Ok((output, _)) => {
                                for line in output {
                                    yield Ok::<Bytes, String>(Bytes::from(line));
                                }
                            }
                            Err(e) => {
                                stream_failed = true;
                                yield Ok::<Bytes, String>(native_error_sse(e));
                            }
                        }
                    }

                    if !stream_failed {
                        if !native_state.saw_finish_reason && !native_state.saw_prompt_block {
                            yield Ok::<Bytes, String>(native_error_sse(
                                "Gemini stream ended without a candidate finishReason or prompt block"
                            ));
                        } else if !native_state.saw_done {
                            // A few upstream/proxy paths close cleanly after the
                            // terminal candidate but omit the framing sentinel.
                            // Emit it only after validating finishReason so native
                            // clients can reliably release the response stream.
                            yield Ok::<Bytes, String>(Bytes::from("data: [DONE]\n\n"));
                        }
                    }
                };

                if client_wants_stream {
                    let body = Body::from_stream(stream);
                    return Ok(Response::builder()
                        .header("Content-Type", "text/event-stream")
                        .header("Cache-Control", "no-cache, no-transform")
                        .header("Connection", "keep-alive")
                        .header("X-Accel-Buffering", "no")
                        .header("X-Account-Email", &email)
                        .header("X-Mapped-Model", &mapped_model)
                        .body(body)
                        .unwrap()
                        .into_response());
                } else {
                    // Collect to JSON
                    use crate::proxy::mappers::gemini::collector::collect_stream_to_json;
                    match collect_stream_to_json(Box::pin(stream), &s_id).await {
                        Ok(gemini_resp) => {
                            info!(
                                "[{}] ✓ Stream collected and converted to JSON (Gemini)",
                                session_id
                            );
                            let unwrapped = unwrap_response(&gemini_resp);
                            return Ok((
                                StatusCode::OK,
                                [
                                    ("X-Account-Email", email.as_str()),
                                    ("X-Mapped-Model", mapped_model.as_str()),
                                ],
                                Json(unwrapped),
                            )
                                .into_response());
                        }
                        Err(e) => {
                            error!("Stream collection error: {}", e);
                            last_error = format!("Stream collection error: {e}");
                            token_manager.record_failure(&account_id);
                            force_rotate = true;
                            continue;
                        }
                    }
                }
            }

            let mut gemini_resp: Value = match response.json().await {
                Ok(value) => value,
                Err(error) => {
                    last_error = format!("Failed to parse Gemini response: {error}");
                    token_manager.record_failure(&account_id);
                    force_rotate = true;
                    continue;
                }
            };
            if let Err(error) =
                super::common::validate_gemini_terminal_response(&gemini_resp)
            {
                last_error = error;
                token_manager.record_failure(&account_id);
                force_rotate = true;
                continue;
            }

            // [FIX #1522] Inject Tool ID into Non-streaming Response
            crate::proxy::mappers::gemini::wrapper::inject_ids_to_response(
                &mut gemini_resp,
                &mapped_model,
            );

            // [FIX #765] Extract thoughtSignature from non-streaming response
            let inner_val = if gemini_resp.get("response").is_some() {
                gemini_resp.get("response")
            } else {
                Some(&gemini_resp)
            };

            if let Some(resp) = inner_val {
                if let Some(candidates) = resp.get("candidates").and_then(|c| c.as_array()) {
                    for cand in candidates {
                        if let Some(parts) = cand
                            .get("content")
                            .and_then(|c| c.get("parts"))
                            .and_then(|p| p.as_array())
                        {
                            for part in parts {
                                if let Some(sig) = part
                                    .get("thoughtSignature")
                                    .or_else(|| part.get("thought_signature"))
                                    .and_then(|s| s.as_str())
                                {
                                    crate::proxy::SignatureCache::global().cache_session_signature(
                                        &session_id,
                                        sig.to_string(),
                                        1,
                                    );
                                    debug!("[Gemini-Response] Cached signature (len: {}) for session: {}", sig.len(), session_id);
                                }
                            }
                        }
                    }
                }
            }

            if hide_thinking_output {
                hide_gemini_thought_text(&mut gemini_resp);
            }

            let unwrapped = unwrap_response(&gemini_resp);
            return Ok((
                StatusCode::OK,
                [
                    ("X-Account-Email", email.as_str()),
                    ("X-Mapped-Model", mapped_model.as_str()),
                ],
                Json(unwrapped),
            )
                .into_response());
        }

        // 处理错误并重试
        let status_code = status.as_u16();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| format!("HTTP {}", status_code));
        last_error = format!("HTTP {}: {}", status_code, error_text);
        if debug_logger::is_enabled(&debug_cfg) {
            let payload = json!({
                "kind": "upstream_response_error",
                "protocol": "gemini",
                "trace_id": trace_id,
                "original_model": model_name,
                "mapped_model": mapped_model,
                "request_type": config.request_type,
                "attempt": attempt,
                "status": status_code,
                "upstream_url": upstream_url,
                "account": mask_email(&email),
                "error_text": error_text,
            });
            debug_logger::write_debug_payload(
                &debug_cfg,
                Some(&trace_id),
                "upstream_response_error",
                &payload,
            )
            .await;
        }

        if is_image_request {
            if is_retryable_image_error(status_code, &error_text) {
                tracing::warn!(
                    "[Gemini-Images] Account {} returned {}, fast-cooling and rotating",
                    email,
                    status_code
                );
                token_manager.mark_image_error_fast_with_body(
                    &account_id,
                    status_code,
                    Some(&mapped_model),
                    Some(&error_text),
                );
                force_rotate = true;
                if attempt + 1 < max_attempts {
                    continue;
                }
            }

            // 5xx is an upstream capacity issue, not evidence that every account
            // is bad. Fail fast instead of fanning the same outage across the pool.
            if matches!(status_code, 500 | 503 | 529) {
                token_manager.mark_image_error_fast(
                    &account_id,
                    status_code,
                    Some(&mapped_model),
                );
                return Ok((
                    status,
                    [
                        ("X-Account-Email", email.as_str()),
                        ("X-Mapped-Model", mapped_model.as_str()),
                    ],
                    Json(json!({
                        "error": {
                            "code": status_code,
                            "message": error_text,
                            "status": "UPSTREAM_ERROR"
                        }
                    })),
                )
                    .into_response());
            }
        }

        // 确定重试策略
        let strategy = determine_retry_strategy(status_code, &error_text, false);
        let trace_id = format!("gemini_{}", session_id);

        // 执行退避
        if apply_retry_strategy(
            strategy.clone(),
            attempt,
            max_attempts,
            status_code,
            &trace_id,
        )
        .await
        {
            // [NEW] Apply Client Adapter "let_it_crash" strategy
            if let Some(adapter) = &client_adapter {
                if adapter.let_it_crash() && attempt > 0 {
                    tracing::warn!(
                        "[Gemini] let_it_crash active: Aborting retries after attempt {}",
                        attempt
                    );
                    break;
                }
            }

            // 判断是否需要轮换账号
            if !should_rotate_account(status_code, Some(&strategy)) {
                debug!(
                "[{}] Keeping same account for status {} (Gemini server-side issue or Grace Retry)",
                trace_id, status_code
            );
                force_rotate = false;
            } else {
                force_rotate = true;
            }

            continue;
        }

        // [NEW] 处理 400 错误 (Thinking 签名失效)
        if status_code == 400
            && (error_text.contains("Invalid `signature`")
                || error_text.contains("thinking.signature")
                || error_text.contains("Invalid signature")
                || error_text.contains("Corrupted thought signature"))
        {
            tracing::warn!(
                "[Gemini] Signature error detected on account {}, retrying without thinking",
                email
            );

            // 追加修复提示词到请求体的最后一条内容
            if let Some(contents) = body.get_mut("contents").and_then(|v| v.as_array_mut()) {
                if let Some(last_content) = contents.last_mut() {
                    if let Some(parts) =
                        last_content.get_mut("parts").and_then(|v| v.as_array_mut())
                    {
                        parts.push(json!({
                            "text": "\n\n[System Recovery] Your previous output contained an invalid signature. Please regenerate the response without the corrupted signature block."
                        }));
                        tracing::debug!("[Gemini] Appended repair prompt to last content");
                    }
                }
            }

            continue; // 重试
        }

        // 404 等由于模型配置或路径错误的 HTTP 异常，直接报错，不进行无效轮换
        error!(
            "Gemini Upstream non-retryable error {}: {}",
            status_code, error_text
        );
        return Ok((
            status,
            [
                ("X-Account-Email", email.as_str()),
                ("X-Mapped-Model", mapped_model.as_str()),
            ],
            // [FIX] Return JSON error
            Json(json!({
                "error": {
                    "code": status_code,
                    "message": error_text,
                    "status": "UPSTREAM_ERROR"
                }
            })),
        )
            .into_response());
    }

    if let Some(email) = last_email {
        Ok((
            StatusCode::TOO_MANY_REQUESTS,
            [("X-Account-Email", email)],
            format!("All accounts exhausted. Last error: {}", last_error),
        )
            .into_response())
    } else {
        Ok((
            StatusCode::TOO_MANY_REQUESTS,
            format!("All accounts exhausted. Last error: {}", last_error),
        )
            .into_response())
    }
}

pub async fn handle_list_models(
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    use crate::proxy::common::model_mapping::get_all_dynamic_models;

    // 获取所有动态模型列表（与 /v1/models 一致）
    let model_ids = get_all_dynamic_models(&state.custom_mapping, Some(&state.token_manager)).await;

    // 转换为 Gemini API 格式
    let models: Vec<_> = model_ids
        .into_iter()
        .map(|id| {
            json!({
                "name": format!("models/{}", id),
                "version": "001",
                "displayName": id.clone(),
                "description": "",
                "inputTokenLimit": 128000,
                "outputTokenLimit": 8192,
                "supportedGenerationMethods": ["generateContent", "countTokens"],
                "temperature": 1.0,
                "topP": 0.95,
                "topK": 64
            })
        })
        .collect();

    Ok(Json(json!({ "models": models })))
}

pub async fn handle_get_model(Path(model_name): Path<String>) -> impl IntoResponse {
    Json(json!({
        "name": format!("models/{}", model_name),
        "displayName": model_name
    }))
}

pub async fn handle_count_tokens(
    State(state): State<AppState>,
    Path(_model_name): Path<String>,
    Json(_body): Json<Value>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let model_group = "gemini";
    let (_access_token, _project_id, _, _, _wait_ms) = state
        .token_manager
        .get_token(model_group, false, None, "gemini")
        .await
        .map_err(|e| {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                format!("Token error: {}", e),
            )
        })?;

    Ok(Json(json!({"totalTokens": 0})))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_peek_waits_through_fragmented_and_usage_only_frames() {
        let partial = br#"data: {"usageMetadata":{"promptTokenCount":2}}
data: {"candidates":[{"content":{"parts":[{"text":"hel"#;
        assert_eq!(
            classify_native_peek_buffer(partial).expect("partial frame"),
            NativePeekDisposition::Ignore
        );

        let complete = br#"data: {"usageMetadata":{"promptTokenCount":2}}
data: {"candidates":[{"content":{"parts":[{"text":"hello"}]}}]}

"#;
        assert_eq!(
            classify_native_peek_buffer(complete).expect("complete frame"),
            NativePeekDisposition::Meaningful
        );
    }

    #[test]
    fn native_peek_rejects_error_before_content() {
        let error = br#"data: {"error":{"code":500,"message":"failed"}}

"#;
        assert_eq!(
            classify_native_peek_buffer(error).expect("error frame"),
            NativePeekDisposition::Error
        );
    }

    #[test]
    fn native_line_requires_finish_reason_before_done() {
        let mut state = NativeStreamState::default();
        let error = process_native_sse_line(
            "data: [DONE]",
            &mut state,
            "native-test",
            "gemini-test",
        )
        .expect_err("DONE alone must not complete a Gemini response");

        assert!(error.contains("without a candidate finishReason"));
        assert!(!state.saw_done);
    }

    #[test]
    fn native_line_preserves_max_tokens_and_allows_done() {
        let mut state = NativeStreamState::default();
        let line = r#"data: {"response":{"candidates":[{"content":{"parts":[{"text":"partial"}]},"finishReason":"MAX_TOKENS"}]}}"#;

        let output = process_native_sse_line(
            line,
            &mut state,
            "native-test",
            "gemini-test",
        )
        .expect("valid finish event");
        let NativeLineAction::Forward(output) = output else {
            panic!("expected forwarded response");
        };
        assert!(output.contains("\"finishReason\":\"MAX_TOKENS\""));
        assert!(state.saw_finish_reason);

        let done = process_native_sse_line(
            "data: [DONE]",
            &mut state,
            "native-test",
            "gemini-test",
        )
        .expect("DONE after finishReason is valid");
        assert!(matches!(done, NativeLineAction::Done(_)));
        assert!(state.saw_done);
    }

    #[test]
    fn native_line_rejects_embedded_upstream_error() {
        let mut state = NativeStreamState::default();
        let error = process_native_sse_line(
            r#"data: {"error":{"code":500,"message":"failed"}}"#,
            &mut state,
            "native-test",
            "gemini-test",
        )
        .expect_err("upstream error must terminate the stream");

        assert!(error.contains("Gemini upstream error"));
        assert!(!state.saw_finish_reason);
    }

    #[test]
    fn native_prompt_feedback_block_allows_done() {
        let mut state = NativeStreamState::default();
        process_native_sse_line(
            r#"data: {"promptFeedback":{"blockReason":"SAFETY"}}"#,
            &mut state,
            "native-test",
            "gemini-test",
        )
        .expect("prompt block response");
        assert!(state.saw_prompt_block);

        let done = process_native_sse_line(
            "data: [DONE]",
            &mut state,
            "native-test",
            "gemini-test",
        )
        .expect("DONE after prompt block is valid");
        assert!(matches!(done, NativeLineAction::Done(_)));
    }

    #[test]
    fn native_later_zero_usage_does_not_erase_real_counts() {
        let mut state = NativeStreamState::default();
        process_native_sse_line(
            r#"data: {"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":7,"totalTokenCount":19}}"#,
            &mut state,
            "native-test",
            "gemini-test",
        )
        .expect("usage frame");
        let output = process_native_sse_line(
            r#"data: {"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":0,"candidatesTokenCount":0,"totalTokenCount":0}}"#,
            &mut state,
            "native-test",
            "gemini-test",
        )
        .expect("terminal frame");
        let NativeLineAction::Forward(output) = output else {
            panic!("expected forwarded response");
        };
        assert!(output.contains("\"promptTokenCount\":12"));
        assert!(output.contains("\"candidatesTokenCount\":7"));
        assert!(output.contains("\"totalTokenCount\":19"));
    }

    #[test]
    fn native_buffer_handles_split_json_and_unterminated_tail() {
        let mut state = NativeStreamState::default();
        let mut buffer = bytes::BytesMut::new();
        buffer.extend_from_slice(
            b"data: {\"response\":{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hel",
        );
        let (first_output, first_done) = drain_native_sse_buffer(
            &mut buffer,
            &mut state,
            "native-test",
            "gemini-test",
            false,
        )
        .expect("partial line is buffered");
        assert!(first_output.is_empty());
        assert!(!first_done);

        buffer.extend_from_slice(
            b"lo\"}]},\"finishReason\":\"STOP\"}]}}",
        );
        let (output, done) = drain_native_sse_buffer(
            &mut buffer,
            &mut state,
            "native-test",
            "gemini-test",
            true,
        )
        .expect("unterminated tail is processed at EOF");

        assert_eq!(output.len(), 1);
        assert!(output[0].contains("hello"));
        assert!(!done);
        assert!(state.saw_finish_reason);
        assert!(buffer.is_empty());
    }

    #[test]
    fn native_thought_filter_hides_text_but_keeps_signature_and_tools() {
        let mut payload = json!({
            "response": {
                "candidates": [{
                    "content": {"parts": [
                        {"thought": true, "text": "private", "thoughtSignature": "sig"},
                        {"text": "answer"},
                        {"thought": true, "functionCall": {"name": "lookup", "args": {}}}
                    ]},
                    "finishReason": "STOP"
                }],
                "usageMetadata": {"totalTokenCount": 3}
            }
        });
        hide_gemini_thought_text(&mut payload);
        let parts = payload["response"]["candidates"][0]["content"]["parts"]
            .as_array()
            .expect("parts");
        assert!(parts[0].get("text").is_none());
        assert_eq!(parts[0]["thoughtSignature"], "sig");
        assert_eq!(parts[1]["text"], "answer");
        assert!(parts[2].get("functionCall").is_some());
        assert_eq!(payload["response"]["usageMetadata"]["totalTokenCount"], 3);
    }

    #[test]
    fn native_thought_filter_is_opt_in_at_sse_drain_boundary() {
        let mut state = NativeStreamState::default();
        let mut hidden_buffer = bytes::BytesMut::from(
            &b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"thought\":true,\"text\":\"private\"},{\"text\":\"answer\"}]},\"finishReason\":\"STOP\"}]}\n\n"[..],
        );
        let (hidden, _) = drain_native_sse_buffer_with_options(
            &mut hidden_buffer,
            &mut state,
            "native-test",
            "gemini-test",
            false,
            true,
        )
        .expect("hidden stream");
        assert!(!hidden.join("").contains("private"));
        assert!(hidden.join("").contains("answer"));

        let mut visible_state = NativeStreamState::default();
        let mut visible_buffer = bytes::BytesMut::from(
            &b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"thought\":true,\"text\":\"private\"},{\"text\":\"answer\"}]},\"finishReason\":\"STOP\"}]}\n\n"[..],
        );
        let (visible, _) = drain_native_sse_buffer_with_options(
            &mut visible_buffer,
            &mut visible_state,
            "native-test",
            "gemini-test",
            false,
            false,
        )
        .expect("visible stream");
        assert!(visible.join("").contains("private"));
    }
}
