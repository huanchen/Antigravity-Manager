// Gemini Stream Collector
// Used for auto-converting streaming responses to JSON for non-streaming requests

use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use serde_json::{json, Value};
use tracing::debug;

use crate::proxy::SignatureCache;

#[derive(Default)]
struct CollectionState {
    content_parts: Vec<Value>,
    usage_metadata: Option<Value>,
    finish_reason: Option<String>,
    prompt_feedback: Option<Value>,
    prompt_block_reason: Option<String>,
}

pub(crate) fn merge_usage_value(current: &mut Option<Value>, incoming: Value) {
    fn merge_object(current: &mut serde_json::Map<String, Value>, incoming: serde_json::Map<String, Value>) {
        for (key, incoming_value) in incoming {
            match (current.get_mut(&key), incoming_value) {
                (Some(Value::Number(current_number)), Value::Number(incoming_number)) => {
                    let current_value = current_number.as_u64().unwrap_or(0);
                    let incoming_value = incoming_number.as_u64().unwrap_or(0);
                    if incoming_value > current_value {
                        *current_number = serde_json::Number::from(incoming_value);
                    }
                }
                (Some(Value::Object(current_object)), Value::Object(incoming_object)) => {
                    merge_object(current_object, incoming_object);
                }
                (Some(existing), incoming_value) => {
                    if existing.is_null() && !incoming_value.is_null() {
                        *existing = incoming_value;
                    }
                }
                (None, incoming_value) => {
                    current.insert(key, incoming_value);
                }
            }
        }
    }

    if current.is_none() {
        *current = Some(incoming);
        return;
    }

    match (current.as_mut().expect("usage slot initialized"), incoming) {
        (Value::Object(current), Value::Object(incoming)) => merge_object(current, incoming),
        (existing, incoming) if existing.is_null() && !incoming.is_null() => *existing = incoming,
        _ => {}
    }
}

impl CollectionState {
    /// Returns true when an explicit SSE terminator was received.
    fn process_line(&mut self, line: &str, session_id: &str) -> Result<bool, String> {
        let line = line.trim();
        let Some(data) = line.strip_prefix("data:") else {
            return Ok(false);
        };
        let data = data.trim();
        if data.is_empty() {
            return Ok(false);
        }
        if data == "[DONE]" {
            return Ok(true);
        }

        let mut envelope: Value = serde_json::from_str(data)
            .map_err(|e| format!("Invalid Gemini SSE JSON: {e}"))?;

        if let Some(error) = envelope.get("error").filter(|error| !error.is_null()) {
            return Err(format!("Gemini upstream error: {error}"));
        }

        let envelope_usage = envelope.get("usageMetadata").cloned();
        let actual_data = if let Some(inner) = envelope.get_mut("response").map(Value::take) {
            inner
        } else {
            envelope
        };

        if let Some(error) = actual_data.get("error").filter(|error| !error.is_null()) {
            return Err(format!("Gemini upstream error: {error}"));
        }

        if let Some(usage) = actual_data
            .get("usageMetadata")
            .cloned()
            .or(envelope_usage)
        {
            merge_usage_value(&mut self.usage_metadata, usage);
        }

        if let Some(feedback) = actual_data.get("promptFeedback") {
            self.prompt_feedback = Some(feedback.clone());
            if let Some(reason) = feedback
                .get("blockReason")
                .and_then(Value::as_str)
                .filter(|reason| !reason.is_empty())
            {
                self.prompt_block_reason = Some(reason.to_string());
            }
        }

        if let Some(candidate) = actual_data
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
        {
            if let Some(reason) = candidate
                .get("finishReason")
                .and_then(Value::as_str)
                .filter(|reason| !reason.is_empty())
            {
                self.finish_reason = Some(reason.to_string());
            }

            if let Some(parts) = candidate
                .get("content")
                .and_then(|content| content.get("parts"))
                .and_then(Value::as_array)
            {
                for part in parts {
                    if let Some(signature) =
                        part.get("thoughtSignature").and_then(Value::as_str)
                    {
                        SignatureCache::global().cache_session_signature(
                            session_id,
                            signature.to_string(),
                            1,
                        );
                        debug!(
                            "[Gemini-AutoConverter] Cached signature (len: {}) for session: {}",
                            signature.len(),
                            session_id
                        );
                    }

                    let plain_text = part
                        .as_object()
                        .is_some_and(|object| object.len() == 1 && object.contains_key("text"));
                    if plain_text {
                        if let (Some(text), Some(last)) =
                            (part.get("text").and_then(Value::as_str), self.content_parts.last_mut())
                        {
                            let last_is_plain_text = last
                                .as_object()
                                .is_some_and(|object| object.len() == 1 && object.contains_key("text"));
                            if last_is_plain_text {
                                if let Some(last_text) = last.get("text").and_then(Value::as_str) {
                                    let merged = format!("{last_text}{text}");
                                    *last = json!({ "text": merged });
                                    continue;
                                }
                            }
                        }
                    }
                    self.content_parts.push(part.clone());
                }
            }
        }

        Ok(false)
    }

    fn into_response(self) -> Result<Value, String> {
        let mut response = if let Some(finish_reason) = self.finish_reason {
            json!({
                "candidates": [{
                    "content": {
                        "parts": self.content_parts,
                        "role": "model"
                    },
                    "finishReason": finish_reason,
                    "index": 0
                }]
            })
        } else if self.prompt_block_reason.is_some() {
            json!({ "candidates": [] })
        } else {
            return Err(
                "Gemini stream ended without a candidate finishReason or prompt block; refusing to return a partial response"
                    .to_string(),
            );
        };
        if let Some(feedback) = self.prompt_feedback {
            response["promptFeedback"] = feedback;
        }
        if let Some(usage) = self.usage_metadata {
            response["usageMetadata"] = usage;
        }
        Ok(response)
    }
}

/// Collects a Gemini SSE stream into a complete Gemini response.
///
/// A response is complete only when Gemini supplied a candidate `finishReason`.
/// Transport EOF and `[DONE]` are framing signals, not model completion signals.
pub async fn collect_stream_to_json<S, E>(mut stream: S, session_id: &str) -> Result<Value, String>
where
    S: futures::Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    let mut state = CollectionState::default();
    let mut buffer = BytesMut::new();
    let mut terminated = false;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|e| format!("Stream error: {e}"))?;
        buffer.extend_from_slice(&chunk);

        while let Some(newline) = buffer.iter().position(|byte| *byte == b'\n') {
            let line = buffer.split_to(newline + 1);
            let line = std::str::from_utf8(&line)
                .map_err(|e| format!("Invalid UTF-8 in Gemini SSE stream: {e}"))?;
            if state.process_line(line, session_id)? {
                terminated = true;
                break;
            }
        }

        if terminated {
            break;
        }
    }

    // SSE producers do not always terminate the final data line with a newline.
    if !terminated && !buffer.is_empty() {
        let tail = std::str::from_utf8(&buffer)
            .map_err(|e| format!("Invalid UTF-8 in Gemini SSE stream: {e}"))?;
        state.process_line(tail, session_id)?;
    }

    state.into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[tokio::test]
    async fn collects_json_split_across_chunks_without_trailing_newline() {
        let chunks: Vec<Result<Bytes, String>> = vec![
            Ok(Bytes::from_static(b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hel")),
            Ok(Bytes::from_static(b"lo\"}]}}]}\n\ndata: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" world\"}]},")),
            Ok(Bytes::from_static(b"\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"candidatesTokenCount\":2}}")),
        ];

        let response = collect_stream_to_json(stream::iter(chunks), "collector-test")
            .await
            .expect("complete response");

        assert_eq!(
            response["candidates"][0]["content"]["parts"][0]["text"],
            "hello world"
        );
        assert_eq!(response["candidates"][0]["finishReason"], "STOP");
        assert_eq!(response["usageMetadata"]["candidatesTokenCount"], 2);
    }

    #[tokio::test]
    async fn preserves_max_tokens_finish_reason() {
        let chunks: Vec<Result<Bytes, String>> = vec![Ok(Bytes::from_static(
            b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]},\"finishReason\":\"MAX_TOKENS\"}]}\n\n",
        ))];

        let response = collect_stream_to_json(stream::iter(chunks), "collector-test")
            .await
            .expect("MAX_TOKENS is a real finish reason");
        assert_eq!(response["candidates"][0]["finishReason"], "MAX_TOKENS");
    }

    #[tokio::test]
    async fn rejects_done_or_eof_without_finish_reason() {
        for final_chunk in [
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\ndata: [DONE]\n\n",
        ] {
            let chunks: Vec<Result<Bytes, String>> =
                vec![Ok(Bytes::copy_from_slice(final_chunk.as_bytes()))];
            let error = collect_stream_to_json(stream::iter(chunks), "collector-test")
                .await
                .expect_err("partial response must not look complete");
            assert!(error.contains("without a candidate finishReason"));
        }
    }

    #[tokio::test]
    async fn propagates_upstream_error() {
        let chunks: Vec<Result<Bytes, String>> = vec![Ok(Bytes::from_static(
            b"data: {\"error\":{\"code\":500,\"message\":\"failed\"}}\n\n",
        ))];
        let error = collect_stream_to_json(stream::iter(chunks), "collector-test")
            .await
            .expect_err("upstream error must not become a response");
        assert!(error.contains("Gemini upstream error"));
    }

    #[tokio::test]
    async fn prompt_feedback_block_is_a_valid_terminal_response() {
        let chunks: Vec<Result<Bytes, String>> = vec![Ok(Bytes::from_static(
            b"data: {\"promptFeedback\":{\"blockReason\":\"SAFETY\"},\"usageMetadata\":{\"promptTokenCount\":4,\"totalTokenCount\":4}}\n\ndata: [DONE]\n\n",
        ))];

        let response = collect_stream_to_json(stream::iter(chunks), "collector-test")
            .await
            .expect("prompt-level block is a complete provider response");
        assert_eq!(response["promptFeedback"]["blockReason"], "SAFETY");
        assert_eq!(response["candidates"], json!([]));
    }

    #[tokio::test]
    async fn later_zero_usage_snapshot_does_not_erase_real_counts() {
        let chunks: Vec<Result<Bytes, String>> = vec![Ok(Bytes::from_static(
            b"data: {\"usageMetadata\":{\"promptTokenCount\":12,\"candidatesTokenCount\":7,\"totalTokenCount\":19}}\n\ndata: {\"candidates\":[{\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":0,\"candidatesTokenCount\":0,\"totalTokenCount\":0}}\n\n",
        ))];

        let response = collect_stream_to_json(stream::iter(chunks), "collector-test")
            .await
            .expect("complete response");
        assert_eq!(response["usageMetadata"]["promptTokenCount"], 12);
        assert_eq!(response["usageMetadata"]["candidatesTokenCount"], 7);
        assert_eq!(response["usageMetadata"]["totalTokenCount"], 19);
    }
}
