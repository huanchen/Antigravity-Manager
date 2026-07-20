use crate::proxy::token_manager::ProxyToken;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub max_output_tokens: Option<u64>,
    pub thinking_budget: Option<u64>,
    pub is_thinking: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpecsConfig {
    models: HashMap<String, ModelSpec>,
    aliases: HashMap<String, String>,
}

static SPECS: Lazy<SpecsConfig> = Lazy::new(|| {
    let json_str = include_str!("../../resources/model_specs.json");
    serde_json::from_str(json_str).expect("Failed to parse model_specs.json")
});

// Account quota payloads occasionally contain `max_output_tokens: 0` (or a
// tiny placeholder) while the model metadata is still loading. Treating that
// value as a hard generation cap turns an otherwise normal request into a
// visibly truncated 0/10-token response. A real Gemini/Claude output limit is
// always substantially larger than this floor; invalid dynamic values fall
// back to the trusted static specification below.
const MIN_VALID_DYNAMIC_OUTPUT_TOKENS: u64 = 1024;

fn valid_dynamic_output_limit(model_id: &str, limit: u64) -> Option<u64> {
    if limit >= MIN_VALID_DYNAMIC_OUTPUT_TOKENS {
        Some(limit)
    } else {
        tracing::warn!(
            "[ModelSpecs] Ignoring suspicious dynamic max_output_tokens={} for {}; using static limit",
            limit,
            model_id
        );
        None
    }
}

/// 获取归一化后的模型 ID (基于别名)
pub fn resolve_alias(model_id: &str) -> String {
    // Resolve chained compatibility aliases (for example the legacy
    // claude-3-7-sonnet -> claude-sonnet-4-6-thinking -> claude-sonnet-4-6
    // path) while bounding traversal so a malformed future config cannot loop.
    let mut resolved = model_id.to_string();
    for _ in 0..8 {
        let Some(next) = SPECS.aliases.get(&resolved) else {
            break;
        };
        if next == &resolved {
            break;
        }
        resolved = next.clone();
    }
    resolved
}

/// 获取模型输出 Token 限额 (动态优先)
pub fn get_max_output_tokens(model_id: &str, token: Option<&ProxyToken>) -> u64 {
    let std_id = resolve_alias(model_id);

    // 1. 尝试从账号动态数据中读取
    if let Some(t) = token {
        if let Some(&limit) = t.model_limits.get(&std_id) {
            if let Some(limit) = valid_dynamic_output_limit(&std_id, limit) {
                return limit;
            }
        }
        // 如果原始 ID 没找到，尝试用归一化后的 ID 找
        if let Some(&limit) = t.model_limits.get(model_id) {
            if let Some(limit) = valid_dynamic_output_limit(model_id, limit) {
                return limit;
            }
        }
    }

    // 2. 回退到静态 JSON
    if let Some(spec) = SPECS.models.get(&std_id) {
        if let Some(limit) = spec.max_output_tokens {
            return limit;
        }
    }

    // 3. 全局兜底
    65535
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_output_limit_floor_rejects_loading_placeholders() {
        assert_eq!(valid_dynamic_output_limit("gemini-3-flash", 0), None);
        assert_eq!(valid_dynamic_output_limit("gemini-3-flash", 10), None);
        assert_eq!(valid_dynamic_output_limit("gemini-3-flash", 1024), Some(1024));
    }

    #[test]
    fn aliases_resolve_through_legacy_claude_chain() {
        assert_eq!(resolve_alias("claude-3-7-sonnet"), "claude-sonnet-4-6");
    }
}

/// 获取思维链预算 (动态优先)
pub fn get_thinking_budget(model_id: &str, _token: Option<&ProxyToken>) -> u64 {
    let std_id = resolve_alias(model_id);

    // 1. 优先尝试从 token 的 quota 信息中推断 (如果以后 quota 返回了具体 budget)
    // 目前 ProxyToken 结构体暂未直接缓存每个模型的 thinking_budget，
    // 但可以通过 model_limits 比例或直接从 JSON 补全。

    // 2. 静态 JSON 配置
    if let Some(spec) = SPECS.models.get(&std_id) {
        if let Some(budget) = spec.thinking_budget {
            return budget;
        }
    }

    // 3. 默认安全限额
    if std_id.contains("claude") {
        16000
    } else {
        24576
    }
}

/// 判断是否为思维模型
#[allow(dead_code)]
pub fn is_thinking_model(model_id: &str) -> bool {
    let std_id = resolve_alias(model_id);
    if let Some(spec) = SPECS.models.get(&std_id) {
        return spec.is_thinking.unwrap_or(false);
    }
    model_id.contains("-thinking") || model_id.contains("thinking")
}
