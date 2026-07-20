// OpenAI Stream Collector
// Used for auto-converting streaming responses to JSON for non-streaming requests

use super::models::*;
use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use serde_json::Value;
use std::collections::HashMap;

struct CollectorState {
    response: OpenAIResponse,
    role: Option<String>,
    content_parts: Vec<String>,
    reasoning_parts: Vec<String>,
    finish_reason: Option<String>,
    tool_calls_map: HashMap<u32, (String, String, String, Vec<String>)>,
}

impl CollectorState {
    fn new() -> Self {
        Self {
            response: OpenAIResponse {
                id: "chatcmpl-unknown".to_string(),
                object: "chat.completion".to_string(),
                created: chrono::Utc::now().timestamp() as u64,
                model: "unknown".to_string(),
                choices: Vec::new(),
                usage: None,
            },
            role: None,
            content_parts: Vec::new(),
            reasoning_parts: Vec::new(),
            finish_reason: None,
            tool_calls_map: HashMap::new(),
        }
    }

    /// Process one complete SSE line. Returns true for the framing `[DONE]` line.
    fn process_line(&mut self, line: &str) -> Result<bool, String> {
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

        let json: Value = serde_json::from_str(data)
            .map_err(|error| format!("Invalid OpenAI SSE JSON: {error}"))?;
        if json.get("error").is_some_and(|error| !error.is_null())
            || matches!(json.get("type").and_then(Value::as_str), Some("error" | "response.failed"))
        {
            return Err(format!("OpenAI upstream stream error: {}", json));
        }

        if let Some(id) = json.get("id").and_then(Value::as_str) {
            self.response.id = id.to_string();
        }
        if let Some(model) = json.get("model").and_then(Value::as_str) {
            self.response.model = model.to_string();
        }
        if let Some(created) = json.get("created").and_then(Value::as_u64) {
            self.response.created = created;
        }
        if let Some(usage) = json.get("usage") {
            let incoming = serde_json::from_value::<OpenAIUsage>(usage.clone())
                .map_err(|error| format!("Invalid OpenAI usage object: {error}"))?;
            if let Some(current) = self.response.usage.as_mut() {
                current.merge_max(incoming);
            } else {
                self.response.usage = Some(incoming);
            }
        }

        if let Some(choice) = json
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        {
            if let Some(delta) = choice.get("delta") {
                if let Some(role) = delta.get("role").and_then(Value::as_str) {
                    self.role = Some(role.to_string());
                }
                if let Some(content) = delta.get("content").and_then(Value::as_str) {
                    self.content_parts.push(content.to_string());
                }
                if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
                    self.reasoning_parts.push(reasoning.to_string());
                }

                if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for tool_call in tool_calls {
                        let raw_index = tool_call
                            .get("index")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as u32;
                        let new_id = tool_call
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        let index = if !new_id.is_empty() {
                            if let Some(existing) = self.tool_calls_map.get(&raw_index) {
                                if !existing.0.is_empty() && existing.0 != new_id {
                                    let mut next = raw_index + 1;
                                    while self.tool_calls_map.contains_key(&next) {
                                        next += 1;
                                    }
                                    next
                                } else {
                                    raw_index
                                }
                            } else {
                                raw_index
                            }
                        } else {
                            raw_index
                        };
                        let entry = self.tool_calls_map.entry(index).or_insert_with(|| {
                            (
                                String::new(),
                                "function".to_string(),
                                String::new(),
                                Vec::new(),
                            )
                        });
                        if !new_id.is_empty() {
                            entry.0 = new_id.to_string();
                        }
                        if let Some(tool_type) = tool_call.get("type").and_then(Value::as_str) {
                            if !tool_type.is_empty() {
                                entry.1 = tool_type.to_string();
                            }
                        }
                        if let Some(function) = tool_call.get("function") {
                            if let Some(name) = function.get("name").and_then(Value::as_str) {
                                if !name.is_empty() {
                                    entry.2 = name.to_string();
                                }
                            }
                            if let Some(arguments) =
                                function.get("arguments").and_then(Value::as_str)
                            {
                                entry.3.push(arguments.to_string());
                            }
                        }
                    }
                }
            }
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                if !reason.is_empty() {
                    self.finish_reason = Some(reason.to_string());
                }
            }
        }

        Ok(false)
    }

    fn into_response(mut self) -> Result<OpenAIResponse, String> {
        let finish_reason = self.finish_reason.ok_or_else(|| {
            "OpenAI stream ended without finish_reason; refusing to return a partial response"
                .to_string()
        })?;
        let final_tool_calls = if self.tool_calls_map.is_empty() {
            None
        } else {
            let mut calls: Vec<(u32, ToolCall)> = self
                .tool_calls_map
                .into_iter()
                .map(|(index, (id, tool_type, name, arguments))| {
                    (
                        index,
                        ToolCall {
                            id,
                            r#type: tool_type,
                            function: Some(ToolFunction {
                                name,
                                arguments: arguments.join(""),
                            }),
                            status: None,
                            call_id: None,
                            operation: None,
                        },
                    )
                })
                .collect();
            calls.sort_by_key(|(index, _)| *index);
            Some(calls.into_iter().map(|(_, call)| call).collect())
        };
        let content = self.content_parts.join("");
        let reasoning = if self.reasoning_parts.is_empty() {
            None
        } else {
            Some(self.reasoning_parts.join(""))
        };
        self.response.choices.push(Choice {
            index: 0,
            message: OpenAIMessage {
                role: self.role.unwrap_or_else(|| "assistant".to_string()),
                content: Some(OpenAIContent::String(content)),
                reasoning_content: reasoning,
                tool_calls: final_tool_calls,
                tool_call_id: None,
                name: None,
                refusal: None,
            },
            finish_reason: Some(finish_reason),
        });
        Ok(self.response)
    }
}

/// Collects an OpenAI SSE stream into a complete OpenAI response.
///
/// A transport EOF or `[DONE]` is only framing. A real `finish_reason` is
/// required before a non-streaming response is returned to the caller.
pub async fn collect_stream_to_json<S, E>(mut stream: S) -> Result<OpenAIResponse, String>
where
    S: futures::Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::fmt::Display,
{
    let mut state = CollectorState::new();
    let mut buffer = BytesMut::new();
    let mut saw_done = false;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.map_err(|error| format!("Stream error: {error}"))?;
        buffer.extend_from_slice(&chunk);
        while let Some(position) = buffer.iter().position(|byte| *byte == b'\n') {
            let line = buffer.split_to(position + 1);
            let line = std::str::from_utf8(&line)
                .map_err(|error| format!("Invalid UTF-8 in OpenAI SSE stream: {error}"))?;
            if state.process_line(line)? {
                saw_done = true;
                break;
            }
        }
        if saw_done {
            break;
        }
    }

    // A final SSE data frame is allowed to omit its newline. Process it rather
    // than silently dropping the last visible text/finish marker.
    if !saw_done && !buffer.is_empty() {
        let tail = std::str::from_utf8(&buffer)
            .map_err(|error| format!("Invalid UTF-8 in OpenAI SSE stream: {error}"))?;
        state.process_line(tail)?;
    }

    state.into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[tokio::test]
    async fn preserves_split_frames_and_requires_finish_reason() {
        let chunks = vec![
            Ok::<Bytes, String>(Bytes::from_static(
                b"data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hel",
            )),
            Ok(Bytes::from_static(
                b"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":3,\"total_tokens\":5}}\n\ndata: [DONE]",
            )),
        ];
        let response = collect_stream_to_json(stream::iter(chunks))
            .await
            .expect("complete stream should collect");
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            response.choices[0].message.content,
            Some(OpenAIContent::String("hello".to_string()))
        );
    }

    #[tokio::test]
    async fn eof_without_finish_is_not_a_successful_response() {
        let chunks = vec![Ok::<Bytes, String>(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\ndata: [DONE]\n\n",
        ))];
        let error = collect_stream_to_json(stream::iter(chunks))
            .await
            .expect_err("missing finish reason must fail collection");
        assert!(error.contains("finish_reason"));
    }

    #[tokio::test]
    async fn upstream_error_is_not_converted_to_stop() {
        let chunks = vec![Ok::<Bytes, String>(Bytes::from_static(
            b"data: {\"error\":{\"message\":\"boom\"}}\n\n",
        ))];
        let error = collect_stream_to_json(stream::iter(chunks))
            .await
            .expect_err("error event must fail collection");
        assert!(error.contains("boom"));
    }
}
