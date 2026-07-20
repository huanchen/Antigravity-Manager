// 模型名称映射
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};

// 动态官方废弃模型转发表 (old_model_id -> new_model_id)
pub static DYNAMIC_MODEL_FORWARDING_RULES: Lazy<DashMap<String, String>> =
    Lazy::new(|| DashMap::new());

pub fn update_dynamic_forwarding_rules(old_model: String, new_model: String) {
    if !DYNAMIC_MODEL_FORWARDING_RULES.contains_key(&old_model) {
        crate::modules::logger::log_info(&format!(
            "[Mapping] Registered automatic forwarding rule: {} -> {}",
            old_model, new_model
        ));
    }
    DYNAMIC_MODEL_FORWARDING_RULES.insert(old_model, new_model);
}

static CLAUDE_TO_GEMINI: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();

    // 直接支持的模型
    m.insert("claude-sonnet-4-6", "claude-sonnet-4-6");
    // The upstream Claude Sonnet checkpoint is shared by the thinking and
    // non-thinking Anthropic aliases; keep the synthetic suffix out of the
    // v1internal model id while preserving thinking config separately.
    m.insert("claude-sonnet-4-6-thinking", "claude-sonnet-4-6");

    // [Redirect] Sonnet 4.5 -> Sonnet 4.6
    m.insert("claude-sonnet-4-5", "claude-sonnet-4-6");
    m.insert("claude-sonnet-4-5-thinking", "claude-sonnet-4-6");

    // 别名映射
    m.insert("claude-sonnet-4-5-20250929", "claude-sonnet-4-6");
    m.insert("claude-3-5-sonnet-20241022", "claude-sonnet-4-6");
    m.insert("claude-3-5-sonnet-20240620", "claude-sonnet-4-6");
    // [Redirect] Opus 4.5 -> Opus 4.6 (Issue #1743)
    m.insert("claude-opus-4", "claude-opus-4-6-thinking");
    m.insert("claude-opus-4-5-thinking", "claude-opus-4-6-thinking");
    m.insert("claude-opus-4-5-20251101", "claude-opus-4-6-thinking");

    // Claude Opus 4.6
    m.insert("claude-opus-4-6-thinking", "claude-opus-4-6-thinking");
    m.insert("claude-opus-4-6", "claude-opus-4-6-thinking");
    m.insert("claude-opus-4.6-thinking", "claude-opus-4-6-thinking");
    m.insert("claude-opus-4.6", "claude-opus-4-6-thinking");
    m.insert("claude-opus-4-6-20260201", "claude-opus-4-6-thinking");

    m.insert("claude-haiku-4", "claude-sonnet-4-6");
    m.insert("claude-3-haiku-20240307", "claude-sonnet-4-6");
    m.insert("claude-haiku-4-5-20251001", "claude-sonnet-4-6");
    // OpenAI 协议映射表
    m.insert("gpt-4", "gemini-2.5-flash");
    m.insert("gpt-4-turbo", "gemini-2.5-flash");
    m.insert("gpt-4-turbo-preview", "gemini-2.5-flash");
    m.insert("gpt-4-0125-preview", "gemini-2.5-flash");
    m.insert("gpt-4-1106-preview", "gemini-2.5-flash");
    m.insert("gpt-4-0613", "gemini-2.5-flash");

    m.insert("gpt-4o", "gemini-2.5-flash");
    m.insert("gpt-4o-2024-05-13", "gemini-2.5-flash");
    m.insert("gpt-4o-2024-08-06", "gemini-2.5-flash");

    m.insert("gpt-4o-mini", "gemini-2.5-flash");
    m.insert("gpt-4o-mini-2024-07-18", "gemini-2.5-flash");

    m.insert("gpt-3.5-turbo", "gemini-2.5-flash");
    m.insert("gpt-3.5-turbo-16k", "gemini-2.5-flash");
    m.insert("gpt-3.5-turbo-0125", "gemini-2.5-flash");
    m.insert("gpt-3.5-turbo-1106", "gemini-2.5-flash");
    m.insert("gpt-3.5-turbo-0613", "gemini-2.5-flash");

    // Gemini 协议映射表
    m.insert("gemini-2.5-flash-lite", "gemini-2.5-flash");
    m.insert("gemini-2.5-flash-thinking", "gemini-2.5-flash-thinking");
    // Gemini Pro family. The old preview entrypoints now return a generic
    // INVALID_ARGUMENT response. Keep explicit tiers stable and route legacy /
    // generic names to the conservative, verified low checkpoint.
    m.insert("gemini-3.1-pro-low", "gemini-3.1-pro-low");
    m.insert("gemini-3.1-pro-high", "gemini-pro-agent");
    m.insert("gemini-3.1-pro-preview", "gemini-3.1-pro-low");
    m.insert("gemini-3.1-pro", "gemini-3.1-pro-low");
    m.insert("gemini-3-pro-low", "gemini-3-pro-low");
    m.insert("gemini-3-pro-high", "gemini-pro-agent");
    m.insert("gemini-3-pro-preview", "gemini-3.1-pro-low");
    m.insert("gemini-3-pro", "gemini-3.1-pro-low");
    m.insert("gemini-2.5-flash", "gemini-2.5-flash");
    m.insert("gemini-3-flash", "gemini-3-flash");
    m.insert("gemini-3-pro-image", "gemini-3-pro-image");

    // [New] Unified Virtual ID for Background Tasks (Title, Summary, etc.)
    // Allows users to override all background tasks via custom_mapping
    m.insert("internal-background-task", "gemini-2.5-flash");

    m
});

/// Map Claude model names to Gemini model names
///
/// # 映射策略
/// 1. **精确匹配**: 检查 CLAUDE_TO_GEMINI 映射表
/// 2. **已知前缀透传**: gemini-* 和 *-thinking 模型直接透传
/// 3. **[NEW] 直接透传**: 未知模型 ID 直接传递给 Google API (支持体验未发布模型)
///
/// # 参数
/// - `input`: 原始模型名称
///
/// # 返回
/// 映射后的目标模型名称
///
/// # 示例
/// ```ignore
/// use antigravity_tools_lib::proxy::common::model_mapping::map_claude_model_to_gemini;
/// // 精确匹配
/// assert_eq!(map_claude_model_to_gemini("claude-opus-4"), "claude-opus-4-5-thinking");
///
/// // Gemini 模型透传
/// assert_eq!(map_claude_model_to_gemini("gemini-2.5-flash"), "gemini-2.5-flash");
///
/// // 直接透传未知模型 (NEW!)
/// assert_eq!(map_claude_model_to_gemini("claude-opus-4-6"), "claude-opus-4-6");
/// assert_eq!(map_claude_model_to_gemini("claude-sonnet-5"), "claude-sonnet-5");
/// ```
pub fn map_claude_model_to_gemini(input: &str) -> String {
    // 1. Check exact match in map
    if let Some(mapped) = CLAUDE_TO_GEMINI.get(input) {
        return mapped.to_string();
    }

    // 2. Pass-through known prefixes (gemini-, -thinking) to support dynamic suffixes
    if input.starts_with("gemini-") || input.contains("thinking") {
        return input.to_string();
    }

    // 3. [ENHANCED] 直接透传未知模型 ID,而不是强制 fallback
    // 这允许用户通过自定义映射体验未发布的模型 (如 claude-opus-4-6)
    // Google API 会自动处理无效模型并返回错误,用户可以根据错误调整映射
    input.to_string()
}

/// 获取所有内置支持的模型列表关键字
pub fn get_supported_models() -> Vec<String> {
    CLAUDE_TO_GEMINI.keys().map(|s| s.to_string()).collect()
}

/// 动态获取所有可用模型列表 (包含内置与用户自定义与官方端点动态下发)
pub async fn get_all_dynamic_models(
    custom_mapping: &tokio::sync::RwLock<std::collections::HashMap<String, String>>,
    token_manager: Option<&crate::proxy::token_manager::TokenManager>,
) -> Vec<String> {
    use std::collections::HashSet;
    let mut model_ids = HashSet::new();

    // 1. 获取所有内置映射模型
    for m in get_supported_models() {
        model_ids.insert(m);
    }

    // 2. 获取所有自定义映射模型 (Custom)
    {
        let mapping = custom_mapping.read().await;
        for key in mapping.keys() {
            model_ids.insert(key.clone());
        }
    }

    // 3. [NEW] 获取所有账号从官方接口汇聚而来的动态模型
    if let Some(tm) = token_manager {
        for dynamic_model in tm.get_all_collected_models() {
            model_ids.insert(dynamic_model);
        }
    }

    // 5. 确保包含常用的 Gemini/画画模型 ID
    model_ids.insert("gemini-3.1-pro-low".to_string());

    // [NEW] Issue #247: Dynamically generate all Image Gen Combinations
    let base = "gemini-3-pro-image";
    let resolutions = vec!["", "-2k", "-4k"];
    let ratios = vec!["", "-1x1", "-4x3", "-3x4", "-16x9", "-9x16", "-21x9"];

    for res in resolutions {
        for ratio in ratios.iter() {
            let mut id = base.to_string();
            id.push_str(res);
            id.push_str(ratio);
            model_ids.insert(id);
        }
    }

    model_ids.insert("gemini-2.0-flash-exp".to_string());
    model_ids.insert("gemini-2.5-flash".to_string());
    // gemini-2.5-pro removed
    model_ids.insert("gemini-3-flash".to_string());
    model_ids.insert("gemini-3.1-pro-high".to_string());
    model_ids.insert("gemini-3.1-pro-low".to_string());

    let mut sorted_ids: Vec<_> = model_ids.into_iter().collect();
    sorted_ids.sort();
    sorted_ids
}

/// Wildcard matching - supports multiple wildcards
///
/// **Note**: Matching is **case-sensitive**. Pattern `GPT-4*` will NOT match `gpt-4-turbo`.
///
/// Examples:
/// - `gpt-4*` matches `gpt-4`, `gpt-4-turbo` ✓
/// - `claude-*-sonnet-*` matches `claude-3-5-sonnet-20241022` ✓
/// - `*-thinking` matches `claude-opus-4-5-thinking` ✓
/// - `a*b*c` matches `a123b456c` ✓
fn wildcard_match(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();

    // No wildcard - exact match
    if parts.len() == 1 {
        return pattern == text;
    }

    let mut text_pos = 0;

    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue; // Skip empty segments from consecutive wildcards
        }

        if i == 0 {
            // First segment must match start
            if !text[text_pos..].starts_with(part) {
                return false;
            }
            text_pos += part.len();
        } else if i == parts.len() - 1 {
            // Last segment must match end
            return text[text_pos..].ends_with(part);
        } else {
            // Middle segments - find next occurrence
            if let Some(pos) = text[text_pos..].find(part) {
                text_pos += pos + part.len();
            } else {
                return false;
            }
        }
    }

    true
}

/// 核心模型路由解析引擎
/// 优先级：精确匹配 > 通配符匹配 > 系统默认映射
///
/// # 参数
/// - `original_model`: 原始模型名称
/// - `custom_mapping`: 用户自定义映射表
///
/// # 返回
/// 映射后的目标模型名称
pub fn resolve_model_route(
    original_model: &str,
    custom_mapping: &std::collections::HashMap<String, String>,
) -> String {
    // 0. API 热更新废弃模型转发 (最高物理优先级，强制纠正)
    // 如果用户非要用已经被移除的模型，并且官方下发了 fallback path，我们在此拦截并纠正
    if let Some(forwarded) = DYNAMIC_MODEL_FORWARDING_RULES.get(original_model) {
        crate::modules::logger::log_info(&format!(
            "[Router] 官方淘汰重定向: {} -> {}",
            original_model,
            forwarded.value()
        ));
        return forwarded.value().clone();
    }

    // 1. 精确匹配 (次高优先级)
    if let Some(target) = custom_mapping.get(original_model) {
        crate::modules::logger::log_info(&format!(
            "[Router] 精确映射: {} -> {}",
            original_model, target
        ));
        return target.clone();
    }

    // 2. Wildcard match - most specific (highest non-wildcard chars) wins
    // Note: When multiple patterns have the SAME specificity, HashMap iteration order
    // determines the result (non-deterministic). Users can avoid this by making patterns
    // more specific. Future improvement: use IndexMap + frontend sorting for full control.
    let mut best_match: Option<(&str, &str, usize)> = None;

    for (pattern, target) in custom_mapping.iter() {
        if pattern.contains('*') && wildcard_match(pattern, original_model) {
            let specificity = pattern.chars().count() - pattern.matches('*').count();
            if best_match.is_none() || specificity > best_match.unwrap().2 {
                best_match = Some((pattern.as_str(), target.as_str(), specificity));
            }
        }
    }

    if let Some((pattern, target, _)) = best_match {
        crate::modules::logger::log_info(&format!(
            "[Router] Wildcard match: {} -> {} (rule: {})",
            original_model, target, pattern
        ));
        return target.to_string();
    }

    // 3. 系统默认映射
    let result = map_claude_model_to_gemini(original_model);
    if result != original_model {
        crate::modules::logger::log_info(&format!(
            "[Router] 系统默认映射: {} -> {}",
            original_model, result
        ));
    }
    result
}

pub const SUPPORTED_CLAUDE_MODELS: [&str; 2] =
    ["claude-sonnet-4-6", "claude-opus-4-6-thinking"];

/// Canonicalize Claude aliases only when they map to a model exposed by the
/// upstream quota catalog. Keeping Sonnet and Opus in separate buckets prevents
/// an Opus capacity lock from accidentally suppressing Sonnet.
pub fn canonical_supported_claude_model(model_name: &str) -> Option<&'static str> {
    let lower = model_name
        .trim()
        .trim_start_matches("models/")
        .to_ascii_lowercase();
    match lower.as_str() {
        "claude-sonnet-4-6" | "claude-sonnet-4-6-thinking" => Some("claude-sonnet-4-6"),
        "claude-opus-4-6" | "claude-opus-4-6-thinking" => Some("claude-opus-4-6-thinking"),
        _ => None,
    }
}

pub fn is_supported_claude_model(model_name: &str) -> bool {
    canonical_supported_claude_model(model_name).is_some()
}

pub fn is_legacy_claude_group(model_name: &str) -> bool {
    matches!(
        model_name.trim().to_ascii_lowercase().as_str(),
        "claude" | "claude-sonnet" | "claude-opus"
    )
}

pub fn expand_legacy_claude_group(model_name: &str) -> Vec<String> {
    match model_name.trim().to_ascii_lowercase().as_str() {
        "claude" => SUPPORTED_CLAUDE_MODELS
            .iter()
            .map(|model| (*model).to_string())
            .collect(),
        "claude-sonnet" => vec!["claude-sonnet-4-6".to_string()],
        "claude-opus" => vec!["claude-opus-4-6-thinking".to_string()],
        _ => vec![model_name.to_string()],
    }
}

/// Resolve a configured quota-protection selector to the exact buckets used by
/// scheduling. This accepts legacy family names and old versioned Claude names
/// without coupling Sonnet and Opus again.
pub fn quota_protection_targets(model_name: &str) -> Vec<String> {
    let lower = model_name.trim().to_ascii_lowercase();
    match lower.as_str() {
        "claude" => SUPPORTED_CLAUDE_MODELS
            .iter()
            .map(|model| (*model).to_string())
            .collect(),
        "claude-sonnet" => vec!["claude-sonnet-4-6".to_string()],
        "claude-opus" => vec!["claude-opus-4-6-thinking".to_string()],
        _ if lower.starts_with("claude-") && lower.contains("sonnet") => {
            vec!["claude-sonnet-4-6".to_string()]
        }
        _ if lower.starts_with("claude-") && lower.contains("opus") => {
            vec!["claude-opus-4-6-thinking".to_string()]
        }
        _ => vec![normalize_to_standard_id(&lower).unwrap_or(lower)],
    }
}

fn managed_quota_protection_targets(marker: &str) -> Option<Vec<String>> {
    let lower = marker.trim().to_ascii_lowercase();
    let is_managed_family = lower == "claude"
        || lower == "claude-sonnet"
        || lower == "claude-opus"
        || lower.starts_with("claude-")
        || lower.starts_with("gemini-");
    if !is_managed_family {
        return None;
    }

    let targets = quota_protection_targets(&lower);
    let all_known = targets.iter().all(|target| {
        matches!(
            target.as_str(),
            "gemini-3-flash"
                | "gemini-3-pro-high"
                | "gemini-3-pro-image"
                | "gemini-3.1-flash-image"
                | "claude-sonnet-4-6"
                | "claude-opus-4-6-thinking"
        )
    });
    all_known.then_some(targets)
}

/// Drop stale system-managed protection markers after the monitored-model
/// configuration changes. Unknown/custom markers are retained verbatim.
pub fn reconcile_quota_protection_markers(
    existing: &HashSet<String>,
    configured_models: &[String],
) -> HashSet<String> {
    let monitored: HashSet<String> = configured_models
        .iter()
        .flat_map(|model| quota_protection_targets(model))
        .collect();
    let mut reconciled = HashSet::new();

    for marker in existing {
        if let Some(targets) = managed_quota_protection_targets(marker) {
            for target in targets {
                if monitored.contains(&target) {
                    reconciled.insert(target);
                }
            }
        } else {
            reconciled.insert(marker.clone());
        }
    }

    reconciled
}

fn claude_protection_bucket(model_name: &str) -> Option<&'static str> {
    canonical_supported_claude_model(model_name).or_else(|| {
        match model_name.trim().to_ascii_lowercase().as_str() {
            "claude-sonnet" => Some("claude-sonnet-4-6"),
            "claude-opus" => Some("claude-opus-4-6-thinking"),
            _ => None,
        }
    })
}

pub fn protected_marker_matches(marker: &str, target: &str) -> bool {
    if marker == target {
        return true;
    }

    let target_bucket = claude_protection_bucket(target);
    if marker.trim().eq_ignore_ascii_case("claude") {
        return target_bucket.is_some();
    }

    claude_protection_bucket(marker).is_some_and(|marker_bucket| {
        target_bucket.is_some_and(|target_bucket| marker_bucket == target_bucket)
    })
}

/// Backward-compatible matching for old account files that stored one `claude`
/// protection marker for both supported Claude models.
pub fn protected_model_matches(
    protected_models: &std::collections::HashSet<String>,
    target: &str,
) -> bool {
    protected_models
        .iter()
        .any(|marker| protected_marker_matches(marker, target))
}

pub fn is_image_generation_model(model_name: &str) -> bool {
    let lower = model_name.trim().to_ascii_lowercase();
    lower.contains("image") || lower.contains("imagen")
}

/// Merge one physical model's quota into its normalized scheduling/protection
/// bucket. Variants in the same bucket are interchangeable, so the bucket stays
/// usable while at least one variant has quota.
pub fn merge_standard_quota_max(
    grouped: &mut HashMap<String, i32>,
    model_name: &str,
    percentage: i32,
) {
    if let Some(standard_id) = normalize_to_standard_id(model_name) {
        grouped
            .entry(standard_id)
            .and_modify(|current| *current = (*current).max(percentage))
            .or_insert(percentage);
    }
}

/// Normalize any physical model name to a stable quota/rate-limit bucket.
/// Gemini image/Pro/Flash families remain grouped as before; supported Claude
/// models intentionally use separate exact buckets.
///
/// Standard IDs:
/// - `gemini-3-flash`: All Flash variants (1.5-flash, 2.5-flash, 3-flash, etc.)
/// - `gemini-3.1-flash-image`: Flash image generation/edit quota.
/// - `gemini-3-pro-high`: All Pro variants (1.5-pro, 2.5-pro, etc.)
/// - `gemini-3-pro-image`: Pro image generation quota.
/// - `claude-sonnet-4-5`: All Claude Sonnet variants (3-5-sonnet, sonnet-4-5, etc.)
///
/// Returns `None` if the model doesn't match any of the 3 protected categories.
pub fn normalize_to_standard_id(model_name: &str) -> Option<String> {
    let lower = model_name.to_lowercase();

    // 1. Image resources must keep Flash image and Pro image in separate quota buckets.
    // The quota API exposes `gemini-3.1-flash-image` separately, so grouping it under
    // `gemini-3-pro-image` makes available Flash image quota look exhausted.
    if is_image_generation_model(&lower) {
        if lower.contains("flash") {
            return Some("gemini-3.1-flash-image".to_string());
        }
        return Some("gemini-3-pro-image".to_string());
    }

    // 2. gemini-3-flash (包含所有 flash 变体)
    if lower.contains("flash") {
        return Some("gemini-3-flash".to_string());
    }

    // 3. gemini-3-pro-high (包含 pro 变体)
    if lower.contains("pro") && !lower.contains("image") {
        return Some("gemini-3-pro-high".to_string());
    }

    // 4. Claude: exact supported buckets. Keep a generic legacy bucket for old
    // aliases/configuration values; request routing canonicalizes supported
    // 4.5/4.6 aliases before scheduling, while this compatibility path keeps
    // existing protected_models data readable.
    if let Some(canonical) = canonical_supported_claude_model(&lower) {
        return Some(canonical.to_string());
    }
    if lower.contains("claude")
        || lower.contains("opus")
        || lower.contains("sonnet")
        || lower.contains("haiku")
    {
        return Some("claude".to_string());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_model_mapping() {
        assert_eq!(
            map_claude_model_to_gemini("claude-3-5-sonnet-20241022"),
            "claude-sonnet-4-6"
        );
        // [Redirect] Sonnet 4.5 -> Sonnet 4.6
        assert_eq!(
            map_claude_model_to_gemini("claude-sonnet-4-5"),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            map_claude_model_to_gemini("claude-sonnet-4-5-thinking"),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            map_claude_model_to_gemini("claude-opus-4"),
            "claude-opus-4-6-thinking"
        );
        // Test gemini pass-through (should not be caught by "mini" rule)
        assert_eq!(
            map_claude_model_to_gemini("gemini-2.5-flash-mini-test"),
            "gemini-2.5-flash-mini-test"
        );
        assert_eq!(map_claude_model_to_gemini("unknown-model"), "unknown-model");
        // Gemini Pro concrete IDs should pass through unchanged.
        assert_eq!(
            map_claude_model_to_gemini("gemini-3-pro-high"),
            "gemini-pro-agent"
        );
        assert_eq!(
            map_claude_model_to_gemini("gemini-3-pro-low"),
            "gemini-3-pro-low"
        );
        assert_eq!(
            map_claude_model_to_gemini("gemini-3.1-pro-high"),
            "gemini-pro-agent"
        );
        assert_eq!(
            map_claude_model_to_gemini("gemini-3.1-pro-low"),
            "gemini-3.1-pro-low"
        );
        // Generic/preview aliases must avoid retired preview entrypoints.
        assert_eq!(
            map_claude_model_to_gemini("gemini-3-pro"),
            "gemini-3.1-pro-low"
        );
        assert_eq!(
            map_claude_model_to_gemini("gemini-3.1-pro"),
            "gemini-3.1-pro-low"
        );
        assert_eq!(
            map_claude_model_to_gemini("gemini-3.1-pro-preview"),
            "gemini-3.1-pro-low"
        );

        // Claude models use separate quota buckets so Opus capacity does not lock Sonnet.
        assert_eq!(
            normalize_to_standard_id("claude-opus-4-6-thinking"),
            Some("claude-opus-4-6-thinking".to_string())
        );
        assert_eq!(
            normalize_to_standard_id("claude-sonnet-4-6-thinking"),
            Some("claude-sonnet-4-6".to_string())
        );

        // [Regression] gemini-3-pro-image must NOT be grouped with gemini-3-pro-high
        assert_eq!(
            normalize_to_standard_id("gemini-3-pro-image"),
            Some("gemini-3-pro-image".to_string())
        );
        assert_eq!(
            normalize_to_standard_id("gemini-3-pro-high"),
            Some("gemini-3-pro-high".to_string())
        );

        // [FIX #1955] Test normalization with image suffixes
        assert_eq!(
            normalize_to_standard_id("gemini-3-pro-image-4k"),
            Some("gemini-3-pro-image".to_string())
        );
        assert_eq!(
            normalize_to_standard_id("gemini-3-pro-image-16x9"),
            Some("gemini-3-pro-image".to_string())
        );
        assert_eq!(
            normalize_to_standard_id("gemini-3-pro-image-4k-16x9"),
            Some("gemini-3-pro-image".to_string())
        );
        assert_eq!(
            normalize_to_standard_id("gemini-3.1-flash-image"),
            Some("gemini-3.1-flash-image".to_string())
        );
        assert_eq!(
            normalize_to_standard_id("gemini-3.1-flash-image-4k"),
            Some("gemini-3.1-flash-image".to_string())
        );
    }

    #[test]
    fn legacy_claude_groups_keep_sonnet_and_opus_independent() {
        assert_eq!(
            expand_legacy_claude_group("claude"),
            vec![
                "claude-sonnet-4-6".to_string(),
                "claude-opus-4-6-thinking".to_string(),
            ]
        );
        assert_eq!(
            expand_legacy_claude_group("claude-sonnet"),
            vec!["claude-sonnet-4-6".to_string()]
        );
        assert_eq!(
            expand_legacy_claude_group("claude-opus"),
            vec!["claude-opus-4-6-thinking".to_string()]
        );
    }

    #[test]
    fn legacy_claude_protection_markers_match_only_their_family() {
        let generic = HashSet::from(["claude".to_string()]);
        assert!(protected_model_matches(&generic, "claude-sonnet-4-6"));
        assert!(protected_model_matches(
            &generic,
            "claude-opus-4-6-thinking"
        ));

        let sonnet = HashSet::from(["claude-sonnet".to_string()]);
        assert!(protected_model_matches(&sonnet, "claude-sonnet-4-6"));
        assert!(!protected_model_matches(
            &sonnet,
            "claude-opus-4-6-thinking"
        ));

        let opus = HashSet::from(["claude-opus".to_string()]);
        assert!(!protected_model_matches(&opus, "claude-sonnet-4-6"));
        assert!(protected_model_matches(
            &opus,
            "claude-opus-4-6-thinking"
        ));
    }

    #[test]
    fn normalized_quota_group_uses_best_interchangeable_variant() {
        let mut grouped = HashMap::new();
        merge_standard_quota_max(&mut grouped, "gemini-3.1-pro-low", 0);
        merge_standard_quota_max(&mut grouped, "gemini-3.1-pro-high", 100);
        assert_eq!(grouped.get("gemini-3-pro-high"), Some(&100));
        assert!(!grouped
            .get("gemini-3-pro-high")
            .is_some_and(|quota| *quota < 10));
    }

    #[test]
    fn removed_monitored_models_drop_only_managed_protection_markers() {
        let existing = HashSet::from([
            "claude".to_string(),
            "gemini-3-flash".to_string(),
            "custom-provider-model".to_string(),
        ]);
        let configured = vec!["claude-sonnet".to_string()];

        let reconciled = reconcile_quota_protection_markers(&existing, &configured);
        assert!(reconciled.contains("claude-sonnet-4-6"));
        assert!(!reconciled.contains("claude-opus-4-6-thinking"));
        assert!(!reconciled.contains("gemini-3-flash"));
        assert!(reconciled.contains("custom-provider-model"));
    }

    #[test]
    fn test_wildcard_priority() {
        let mut custom = HashMap::new();
        custom.insert("gpt*".to_string(), "fallback".to_string());
        custom.insert("gpt-4*".to_string(), "specific".to_string());
        custom.insert("claude-opus-*".to_string(), "opus-default".to_string());
        custom.insert(
            "claude-opus*thinking".to_string(),
            "opus-thinking".to_string(),
        );

        // More specific pattern wins
        assert_eq!(resolve_model_route("gpt-4-turbo", &custom), "specific");
        assert_eq!(resolve_model_route("gpt-3.5", &custom), "fallback");
        // Suffix constraint is more specific than prefix-only
        assert_eq!(
            resolve_model_route("claude-opus-4-5-thinking", &custom),
            "opus-thinking"
        );
        assert_eq!(
            resolve_model_route("claude-opus-4", &custom),
            "opus-default"
        );
    }

    #[test]
    fn test_multi_wildcard_support() {
        let mut custom = HashMap::new();
        custom.insert(
            "claude-*-sonnet-*".to_string(),
            "sonnet-versioned".to_string(),
        );
        custom.insert("gpt-*-*".to_string(), "gpt-multi".to_string());
        custom.insert("*thinking*".to_string(), "has-thinking".to_string());

        // Multi-wildcard patterns should work
        assert_eq!(
            resolve_model_route("claude-3-5-sonnet-20241022", &custom),
            "sonnet-versioned"
        );
        assert_eq!(
            resolve_model_route("gpt-4-turbo-preview", &custom),
            "gpt-multi"
        );
        assert_eq!(
            resolve_model_route("claude-thinking-extended", &custom),
            "has-thinking"
        );

        // Negative case: *thinking* should NOT match models without "thinking"
        assert_eq!(
            resolve_model_route("random-model-name", &custom),
            "random-model-name" // Falls back to system default (pass-through)
        );
    }

    #[test]
    fn test_wildcard_edge_cases() {
        let mut custom = HashMap::new();
        custom.insert("prefix*".to_string(), "prefix-match".to_string());
        custom.insert("*".to_string(), "catch-all".to_string());
        custom.insert("a*b*c".to_string(), "multi-wild".to_string());

        // Specificity: "prefix*" (6) > "*" (0)
        assert_eq!(
            resolve_model_route("prefix-anything", &custom),
            "prefix-match"
        );
        // Catch-all has lowest specificity
        assert_eq!(resolve_model_route("random-model", &custom), "catch-all");
        // Multi-wildcard: "a*b*c" (3)
        assert_eq!(resolve_model_route("a-test-b-foo-c", &custom), "multi-wild");
    }
}
