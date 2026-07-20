// Claude 数据模型
// Claude 协议相关数据模型

use serde::{Deserialize, Serialize};

/// Claude API 请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    /// Output configuration for effort level (Claude API v2.0.67+)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    // [NEW] Image generation parameters (for Anthropic protocol compatibility)
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub quality: Option<String>,
}

/// Thinking 配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub type_: String, // "enabled" or "adaptive"
    #[serde(alias = "budgetTokens")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>, // "low", "high", or "max"
}

/// System Prompt
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    String(String),
    Array(Vec<SystemBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub text: String,
}

/// Message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    String(String),
    Array(Vec<ContentBlock>),
}

/// Content Block (Claude)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },

    #[serde(rename = "image")]
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },

    #[serde(rename = "document")]
    Document {
        source: DocumentSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },

    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<serde_json::Value>,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: serde_json::Value, // Changed from String to Value to support Array of Blocks
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },

    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult {
        tool_use_id: String,
        content: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSource {
    #[serde(rename = "type")]
    pub source_type: String, // "base64"
    pub media_type: String, // e.g. "application/pdf"
    pub data: String,       // base64 data
}

/// Tool - supports both client tools (with input_schema) and server tools (like web_search)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Tool type - for server tools like "web_search_20250305"
    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub type_: Option<String>,
    /// Tool name - "web_search" for server tools, custom name for client tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Input schema - required for client tools, absent for server tools
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
}

impl Tool {
    /// Check if this is the web_search server tool
    pub fn is_web_search(&self) -> bool {
        // Check by type (preferred for server tools)
        if let Some(ref t) = self.type_ {
            if t.starts_with("web_search") {
                return true;
            }
        }
        // Check by name (fallback)
        if let Some(ref n) = self.name {
            if n == "web_search" {
                return true;
            }
        }
        false
    }

    /// Get the effective tool name
    #[allow(dead_code)]
    pub fn get_name(&self) -> String {
        self.name.clone().unwrap_or_else(|| {
            // For server tools, derive name from type
            if let Some(ref t) = self.type_ {
                if t.starts_with("web_search") {
                    return "web_search".to_string();
                }
            }
            "unknown".to_string()
        })
    }
}

/// Metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// Output Configuration (Claude API v2.0.67+)
/// Controls effort level for model reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Effort level: "high", "medium", "low"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
}

/// Claude API 响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub role: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

/// Usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<serde_json::Value>,
}

impl Usage {
    /// Merge cumulative usage frames without allowing a later placeholder
    /// (for example `output_tokens: 0`) to erase a real terminal count.
    pub fn merge_cumulative(&mut self, incoming: &Self) {
        fn max_optional(slot: &mut Option<u32>, value: Option<u32>) {
            if let Some(value) = value {
                if slot.map(|current| value > current).unwrap_or(true) {
                    *slot = Some(value);
                }
            }
        }

        if incoming.input_tokens > self.input_tokens {
            self.input_tokens = incoming.input_tokens;
        }
        if incoming.output_tokens > self.output_tokens {
            self.output_tokens = incoming.output_tokens;
        }
        max_optional(
            &mut self.cache_read_input_tokens,
            incoming.cache_read_input_tokens,
        );
        max_optional(
            &mut self.cache_creation_input_tokens,
            incoming.cache_creation_input_tokens,
        );
        if self.server_tool_use.is_none() && incoming.server_tool_use.is_some() {
            self.server_tool_use = incoming.server_tool_use.clone();
        }
    }
}

// ========== Gemini 数据模型 ==========

/// Gemini Content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiContent {
    pub role: String,
    pub parts: Vec<GeminiPart>,
}

/// Gemini Part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "thoughtSignature")]
    #[serde(alias = "thought_signature")]
    pub thought_signature: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "functionCall")]
    pub function_call: Option<FunctionCall>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "functionResponse")]
    pub function_response: Option<FunctionResponse>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "inlineData")]
    pub inline_data: Option<InlineData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    pub name: String,
    pub response: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineData {
    #[serde(rename = "mimeType")]
    pub mime_type: String,
    pub data: String,
}

/// Gemini 完整响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Vec<Candidate>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<UsageMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "modelVersion")]
    pub model_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "responseId")]
    pub response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "promptFeedback")]
    pub prompt_feedback: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "groundingMetadata")]
    pub grounding_metadata: Option<GroundingMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "promptTokenCount")]
    pub prompt_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(
        rename = "thoughtsTokenCount",
        alias = "totalThoughtTokens",
        alias = "total_thought_tokens"
    )]
    pub thoughts_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "totalTokenCount")]
    pub total_token_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "cachedContentTokenCount")]
    pub cached_content_token_count: Option<u32>,
}

impl UsageMetadata {
    /// Usage counters in Gemini SSE frames are cumulative. Keep the greatest
    /// observed value for each field so a final zero/empty placeholder cannot
    /// turn a successful request into `0/0` in downstream monitors.
    pub fn merge_cumulative(&mut self, incoming: &Self) {
        fn max_optional(slot: &mut Option<u32>, value: Option<u32>) {
            if let Some(value) = value {
                if slot.map(|current| value > current).unwrap_or(true) {
                    *slot = Some(value);
                }
            }
        }

        max_optional(&mut self.prompt_token_count, incoming.prompt_token_count);
        max_optional(
            &mut self.candidates_token_count,
            incoming.candidates_token_count,
        );
        max_optional(&mut self.thoughts_token_count, incoming.thoughts_token_count);
        max_optional(&mut self.total_token_count, incoming.total_token_count);
        max_optional(
            &mut self.cached_content_token_count,
            incoming.cached_content_token_count,
        );
    }

    pub fn output_token_count(&self) -> u32 {
        let separated_output = self
            .candidates_token_count
            .unwrap_or(0)
            .saturating_add(self.thoughts_token_count.unwrap_or(0));

        // `totalTokenCount` is normally authoritative, but account metadata
        // occasionally arrives as a placeholder pair such as prompt=0/total=0
        // while candidate/thought counters are already populated. Do not let
        // that zero erase real output. Conversely, when candidate tokens are
        // already inclusive of thoughts, use the total equation to avoid
        // double-counting them.
        match self
            .prompt_token_count
            .zip(self.total_token_count)
            .and_then(|(prompt, total)| total.checked_sub(prompt))
        {
            Some(total_output)
                if total_output > 0
                    || separated_output == 0
                    || (self.candidates_token_count.unwrap_or(0) == 0
                        && self.thoughts_token_count.unwrap_or(0) == 0) =>
            {
                let candidate_fits = self
                    .candidates_token_count
                    .map(|candidate| candidate <= total_output)
                    .unwrap_or(true);
                let thoughts_fit = self
                    .thoughts_token_count
                    .map(|thoughts| thoughts <= total_output)
                    .unwrap_or(true);
                if candidate_fits && thoughts_fit {
                    total_output
                } else {
                    separated_output
                }
            }
            _ => separated_output,
        }
    }
}

// ========== Grounding Metadata (for googleSearch results) ==========

/// Gemini Grounding Metadata - contains search results from googleSearch tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingMetadata {
    #[serde(rename = "webSearchQueries")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_queries: Option<Vec<String>>,

    #[serde(rename = "groundingChunks")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding_chunks: Option<Vec<GroundingChunk>>,

    #[serde(rename = "groundingSupports")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding_supports: Option<Vec<GroundingSupport>>,

    #[serde(rename = "searchEntryPoint")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_entry_point: Option<SearchEntryPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingChunk {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web: Option<WebSource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSource {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingSupport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment: Option<TextSegment>,
    #[serde(rename = "groundingChunkIndices")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grounding_chunk_indices: Option<Vec<i32>>,
    #[serde(rename = "confidenceScores")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_scores: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSegment {
    #[serde(rename = "startIndex")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_index: Option<i32>,
    #[serde(rename = "endIndex")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_index: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchEntryPoint {
    #[serde(rename = "renderedContent")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rendered_content: Option<String>,
}
