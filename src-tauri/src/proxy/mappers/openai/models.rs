// OpenAI 数据模型

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<OpenAIMessage>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub n: Option<u32>, // [NEW] 支持多候选结果数量
    #[serde(
        rename = "max_tokens",
        alias = "max_completion_tokens",
        alias = "max_output_tokens"
    )]
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    #[serde(rename = "top_p")]
    pub top_p: Option<f64>,
    #[serde(rename = "presence_penalty")]
    pub presence_penalty: Option<f64>,
    #[serde(rename = "frequency_penalty")]
    pub frequency_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub stop: Option<Value>,
    pub response_format: Option<ResponseFormat>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(rename = "tool_choice")]
    pub tool_choice: Option<Value>,
    #[serde(rename = "parallel_tool_calls")]
    pub parallel_tool_calls: Option<bool>,
    // Codex proprietary fields
    pub instructions: Option<String>,
    pub input: Option<Value>,
    // [NEW] Image generation parameters (for Chat API compatibility)
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub quality: Option<String>,
    #[serde(default, rename = "personGeneration")]
    pub person_generation: Option<String>,
    // [NEW] Thinking/Extended Thinking 支持 (兼容 Anthropic/Claude 协议)
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    // [NEW] Direct imageSize support (for Gemini native parameter)
    #[serde(default, rename = "imageSize")]
    pub image_size: Option<String>,
}

/// Thinking 配置 (兼容 Anthropic 和 OpenAI 扩展协议)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: Option<String>, // "enabled", "disabled", or "adaptive"
    #[serde(rename = "budget_tokens", alias = "budgetTokens")]
    pub budget_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>, // "low", "high", or "max"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFormat {
    pub r#type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum OpenAIContent {
    String(String),
    Array(Vec<OpenAIContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum OpenAIContentBlock {
    #[serde(rename = "text", alias = "input_text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: OpenAIImageUrl },
    #[serde(rename = "audio_url")]
    AudioUrl { audio_url: AudioUrlContent },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAIImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AudioUrlContent {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolFunction>,

    // [NEW] Fields for apply_patch_call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation: Option<ApplyPatchOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyPatchOperation {
    pub r#type: String,
    pub diff: String,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<OpenAIUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: u32,
    pub message: OpenAIMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    #[serde(skip)]
    pub input_tokens_by_modality: Option<Value>,
    #[serde(skip)]
    pub raw_output_tokens: Option<u32>,
    #[serde(skip)]
    pub total_thought_tokens: Option<u32>,
    #[serde(skip)]
    pub total_tool_use_tokens: Option<u32>,
    #[serde(skip)]
    pub gemini_total_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}

impl OpenAIUsage {
    /// Merge cumulative usage snapshots. Gemini occasionally emits a final
    /// all-zero placeholder after an earlier informative snapshot, so replacing
    /// the slot wholesale would recreate the downstream 0/0 accounting bug.
    pub fn merge_max(&mut self, incoming: OpenAIUsage) {
        self.prompt_tokens = self.prompt_tokens.max(incoming.prompt_tokens);
        self.completion_tokens = self.completion_tokens.max(incoming.completion_tokens);
        self.total_tokens = self.total_tokens.max(incoming.total_tokens);

        let incoming_cached = incoming
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens);
        if let Some(incoming_cached) = incoming_cached {
            let details = self
                .prompt_tokens_details
                .get_or_insert(PromptTokensDetails { cached_tokens: None });
            details.cached_tokens = Some(details.cached_tokens.unwrap_or(0).max(incoming_cached));
        }

        let incoming_reasoning = incoming
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens);
        if let Some(incoming_reasoning) = incoming_reasoning {
            let details = self
                .completion_tokens_details
                .get_or_insert(CompletionTokensDetails {
                    reasoning_tokens: None,
                });
            details.reasoning_tokens = Some(
                details
                    .reasoning_tokens
                    .unwrap_or(0)
                    .max(incoming_reasoning),
            );
        }

        fn merge_optional_max(current: &mut Option<u32>, incoming: Option<u32>) {
            if let Some(incoming) = incoming {
                *current = Some(current.unwrap_or(0).max(incoming));
            }
        }
        merge_optional_max(&mut self.raw_output_tokens, incoming.raw_output_tokens);
        merge_optional_max(
            &mut self.total_thought_tokens,
            incoming.total_thought_tokens,
        );
        merge_optional_max(
            &mut self.total_tool_use_tokens,
            incoming.total_tool_use_tokens,
        );
        merge_optional_max(&mut self.gemini_total_tokens, incoming.gemini_total_tokens);
        if self.input_tokens_by_modality.is_none() {
            self.input_tokens_by_modality = incoming.input_tokens_by_modality;
        }

        self.total_tokens = self
            .total_tokens
            .max(self.prompt_tokens.saturating_add(self.completion_tokens));
    }

    /// Converts Gemini usage metadata while keeping OpenAI's token equation
    /// (`input + output == total`) consistent.
    ///
    /// Gemini variants disagree on whether candidate/output tokens already
    /// include hidden thoughts. `totalTokenCount` is authoritative when both it
    /// and the prompt count are present; otherwise the separate counters are
    /// summed. This avoids both dropping thought-only output and double-counting
    /// thoughts on models that report an inclusive candidate count.
    pub fn from_gemini_metadata(metadata: &Value) -> Self {
        fn token_count(metadata: &Value, keys: &[&str]) -> Option<u32> {
            keys.iter()
                .find_map(|key| metadata.get(*key).and_then(Value::as_u64))
                .map(|value| u32::try_from(value).unwrap_or(u32::MAX))
        }

        let prompt_tokens_raw = token_count(
            metadata,
            &["total_input_tokens", "promptTokenCount"],
        );
        let candidate_tokens_raw = token_count(
            metadata,
            &["total_output_tokens", "candidatesTokenCount"],
        );
        let total_tokens_raw =
            token_count(metadata, &["total_tokens", "totalTokenCount"]);
        let cached_tokens = token_count(
            metadata,
            &[
                "total_cached_tokens",
                "cachedContentTokenCount",
                "cachedTokens",
            ],
        );
        let reasoning_tokens = token_count(
            metadata,
            &[
                "total_thought_tokens",
                "totalThoughtTokens",
                "thoughtsTokenCount",
            ],
        );
        let tool_use_tokens = token_count(metadata, &["total_tool_use_tokens"]);

        let prompt_tokens = prompt_tokens_raw.unwrap_or(0);
        let separated_output = candidate_tokens_raw
            .unwrap_or(0)
            .saturating_add(reasoning_tokens.unwrap_or(0))
            .saturating_add(tool_use_tokens.unwrap_or(0));
        let output_from_total = prompt_tokens_raw
            .zip(total_tokens_raw)
            .and_then(|(prompt, total)| total.checked_sub(prompt));
        // Some upstream frames carry placeholder zeroes for prompt/total while
        // still reporting a real candidate/thought count. Prefer the total
        // equation when it is compatible with the component counters, but do
        // not let a zero placeholder erase non-zero output.
        let completion_tokens = match output_from_total {
            Some(total_output)
                if candidate_tokens_raw
                    .map(|candidate| candidate <= total_output)
                    .unwrap_or(true)
                && reasoning_tokens
                    .map(|reasoning| reasoning <= total_output)
                    .unwrap_or(true) => total_output,
            _ => separated_output,
        };

        // A few Gemini 3 responses omit candidatesTokenCount entirely. Preserve
        // the provider's raw value when present; otherwise derive visible output
        // after subtracting the explicitly reported hidden/tool tokens.
        let visible_output_tokens = candidate_tokens_raw.unwrap_or_else(|| {
            completion_tokens
                .saturating_sub(reasoning_tokens.unwrap_or(0))
                .saturating_sub(tool_use_tokens.unwrap_or(0))
        });
        let total_tokens = prompt_tokens.saturating_add(completion_tokens);

        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
            prompt_tokens_details: cached_tokens.map(|cached| PromptTokensDetails {
                cached_tokens: Some(cached),
            }),
            completion_tokens_details: reasoning_tokens.map(|reasoning| {
                CompletionTokensDetails {
                    reasoning_tokens: Some(reasoning),
                }
            }),
            input_tokens_by_modality: metadata.get("input_tokens_by_modality").cloned(),
            raw_output_tokens: Some(visible_output_tokens),
            total_thought_tokens: reasoning_tokens,
            total_tool_use_tokens: tool_use_tokens,
            gemini_total_tokens: total_tokens_raw,
        }
    }

    pub fn to_responses_usage_value(&self) -> Value {
        let cached_tokens = self
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens)
            .unwrap_or(0);
        let reasoning_tokens = self
            .completion_tokens_details
            .as_ref()
            .and_then(|details| details.reasoning_tokens)
            .unwrap_or(0);

        json!({
            "input_tokens": self.prompt_tokens,
            "input_tokens_details": {
                "cached_tokens": cached_tokens
            },
            "output_tokens": self.completion_tokens,
            "output_tokens_details": {
                "reasoning_tokens": reasoning_tokens
            },
            "total_tokens": self.total_tokens
        })
    }
}

#[cfg(test)]
mod usage_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn openai_output_limit_aliases_are_preserved() {
        for field in ["max_tokens", "max_completion_tokens", "max_output_tokens"] {
            let mut value = json!({
                "model": "gemini-3-flash",
                "messages": []
            });
            value[field] = json!(321);
            let request: OpenAIRequest = serde_json::from_value(value)
            .expect("request alias");
            assert_eq!(request.max_tokens, Some(321), "field={field}");
        }
    }

    #[test]
    fn cumulative_usage_merge_ignores_later_zero_placeholder() {
        let mut usage = OpenAIUsage::from_gemini_metadata(&json!({
            "promptTokenCount": 12,
            "candidatesTokenCount": 7,
            "thoughtsTokenCount": 5,
            "totalTokenCount": 24
        }));
        usage.merge_max(OpenAIUsage::from_gemini_metadata(&json!({
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "thoughtsTokenCount": 0,
            "totalTokenCount": 0
        })));

        assert_eq!(usage.prompt_tokens, 12);
        assert_eq!(usage.completion_tokens, 12);
        assert_eq!(usage.total_tokens, 24);
        assert_eq!(
            usage
                .completion_tokens_details
                .and_then(|details| details.reasoning_tokens),
            Some(5)
        );
    }

    #[test]
    fn gemini_separate_thoughts_are_included_in_output() {
        let usage = OpenAIUsage::from_gemini_metadata(&json!({
            "promptTokenCount": 105,
            "candidatesTokenCount": 179,
            "thoughtsTokenCount": 237,
            "totalTokenCount": 521
        }));

        assert_eq!(usage.prompt_tokens, 105);
        assert_eq!(usage.completion_tokens, 416);
        assert_eq!(usage.total_tokens, 521);
        assert_eq!(usage.raw_output_tokens, Some(179));
        assert_eq!(usage.total_thought_tokens, Some(237));
    }

    #[test]
    fn inclusive_candidate_count_does_not_double_count_thoughts() {
        let usage = OpenAIUsage::from_gemini_metadata(&json!({
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "thoughtsTokenCount": 20,
            "totalTokenCount": 150
        }));

        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.total_thought_tokens, Some(20));
    }

    #[test]
    fn thought_only_gemini_output_is_derived_from_total() {
        let usage = OpenAIUsage::from_gemini_metadata(&json!({
            "promptTokenCount": 105,
            "thoughtsTokenCount": 125,
            "totalTokenCount": 230
        }));

        assert_eq!(usage.completion_tokens, 125);
        assert_eq!(usage.raw_output_tokens, Some(0));
        assert_eq!(usage.total_tokens, 230);
        assert_eq!(usage.total_thought_tokens, Some(125));
    }

    #[test]
    fn missing_total_falls_back_to_separate_counters() {
        let usage = OpenAIUsage::from_gemini_metadata(&json!({
            "promptTokenCount": 100,
            "candidatesTokenCount": 10,
            "thoughtsTokenCount": 20
        }));

        assert_eq!(usage.completion_tokens, 30);
        assert_eq!(usage.total_tokens, 130);
    }

    #[test]
    fn zero_total_placeholder_does_not_hide_candidate_output() {
        let usage = OpenAIUsage::from_gemini_metadata(&json!({
            "promptTokenCount": 0,
            "candidatesTokenCount": 10,
            "totalTokenCount": 0
        }));

        assert_eq!(usage.completion_tokens, 10);
        assert_eq!(usage.total_tokens, 10);
    }

    #[test]
    fn zero_total_placeholder_does_not_hide_thought_output() {
        let usage = OpenAIUsage::from_gemini_metadata(&json!({
            "promptTokenCount": 0,
            "thoughtsTokenCount": 286,
            "totalTokenCount": 0
        }));

        assert_eq!(usage.completion_tokens, 286);
        assert_eq!(usage.total_tokens, 286);
    }
}
