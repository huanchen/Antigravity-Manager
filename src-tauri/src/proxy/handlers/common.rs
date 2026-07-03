use crate::proxy::server::AppState;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};
use tokio::time::{sleep, timeout, Duration};
use tracing::{debug, info};

// ===== Image generation account controls =====

pub const IMAGE_PER_ACCOUNT_CONCURRENCY: usize = 2;
const DEFAULT_IMAGE_GLOBAL_CONCURRENCY: usize = 48;
const DEFAULT_IMAGE_GLOBAL_QUEUE_WAIT_MS: u64 = 300_000;
static IMAGE_ACCOUNT_PERMITS: std::sync::OnceLock<
    dashmap::DashMap<String, std::sync::Arc<tokio::sync::Semaphore>>,
> = std::sync::OnceLock::new();
static IMAGE_GLOBAL_PERMITS: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>> =
    std::sync::OnceLock::new();

pub fn image_max_attempts(pool_size: usize) -> usize {
    // Image models are not uniformly exposed across accounts. Try the whole pool so
    // unsupported accounts do not cause an early failure after the generic 3 attempts.
    pool_size.max(1)
}

pub fn is_retryable_image_error(status_code: u16, error_text: &str) -> bool {
    match status_code {
        // Account/model-specific failures should rotate accounts. Upstream 5xx means
        // Google capacity is unhealthy and should fail fast instead of fanning out.
        401 | 403 | 404 | 429 => true,
        400 => {
            let lower = error_text.to_lowercase();
            lower.contains("not supported")
                || lower.contains("unsupported")
                || lower.contains("does not support")
                || lower.contains("not available")
                || lower.contains("model not found")
                || lower.contains("model is not found")
        }
        _ => false,
    }
}

fn image_global_concurrency_limit() -> usize {
    std::env::var("ABV_IMAGE_GLOBAL_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_IMAGE_GLOBAL_CONCURRENCY)
}

fn image_global_queue_wait_ms() -> u64 {
    std::env::var("ABV_IMAGE_GLOBAL_QUEUE_WAIT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_IMAGE_GLOBAL_QUEUE_WAIT_MS)
}

pub async fn acquire_global_image_permit() -> Result<tokio::sync::OwnedSemaphorePermit, String> {
    let semaphore = IMAGE_GLOBAL_PERMITS
        .get_or_init(|| {
            let limit = image_global_concurrency_limit();
            info!(
                "[Images] Global image concurrency limit: {}, queue_wait={}ms",
                limit,
                image_global_queue_wait_ms()
            );
            std::sync::Arc::new(tokio::sync::Semaphore::new(limit))
        })
        .clone();
    let wait_ms = image_global_queue_wait_ms();

    match timeout(Duration::from_millis(wait_ms), semaphore.acquire_owned()).await
    {
        Ok(Ok(permit)) => Ok(permit),
        Ok(Err(_)) => Err("Image concurrency limiter closed".to_string()),
        Err(_) => Err(format!(
            "Image concurrency queue timeout (limit={}, waited={}ms)",
            image_global_concurrency_limit(),
            wait_ms
        )),
    }
}

pub async fn acquire_image_account_permit(
    account_id: &str,
) -> tokio::sync::OwnedSemaphorePermit {
    let semaphore = {
        let entry = IMAGE_ACCOUNT_PERMITS
            .get_or_init(dashmap::DashMap::new)
            .entry(account_id.to_string())
            .or_insert_with(|| {
                std::sync::Arc::new(tokio::sync::Semaphore::new(
                    IMAGE_PER_ACCOUNT_CONCURRENCY,
                ))
            });
        entry.value().clone()
    };

    semaphore
        .acquire_owned()
        .await
        .expect("image account semaphore closed unexpectedly")
}

// ===== 统一重试与退避策略 =====

/// 重试策略枚举
#[derive(Debug, Clone)]
pub enum RetryStrategy {
    /// 不重试，直接返回错误
    NoRetry,
    /// 固定延迟
    FixedDelay(Duration),
    /// 线性退避：base_ms * (attempt + 1)
    LinearBackoff { base_ms: u64 },
    /// 指数退避：base_ms * 2^attempt，上限 max_ms
    ExponentialBackoff { base_ms: u64, max_ms: u64 },
    /// [NEW] 原地重试 (Grace Retry)：在当前账号上小窗口等待后直接重试，不计入常规切换
    GraceRetry(Duration),
}

/// 根据错误状态码和错误信息确定重试策略
pub fn determine_retry_strategy(
    status_code: u16,
    error_text: &str,
    retried_without_thinking: bool,
) -> RetryStrategy {
    match status_code {
        // 400 错误：仅在特定 Thinking 签名失败时重试一次
        400 if !retried_without_thinking
            && (error_text.contains("Invalid `signature`")
                || error_text.contains("thinking.signature")
                || error_text.contains("thinking.thinking")
                || error_text.contains("Corrupted thought signature")) =>
        {
            RetryStrategy::FixedDelay(Duration::from_millis(200))
        }

        // 429 限流错误
        429 => {
            // 优先使用服务端返回的 Retry-After / quotaResetDelay
            if let Some(delay_ms) = crate::proxy::upstream::retry::parse_retry_delay(error_text) {
                // [NEW] 如果延迟在 2s 内，执行 Grace Retry (原地重试)
                if crate::proxy::upstream::retry::should_grace_retry(delay_ms) {
                    let actual_delay = delay_ms.saturating_add(100); // 增加 100ms 安全缓冲
                    tracing::info!(
                        "Grace Retry Triggered: Delay {}ms is within window, using same account",
                        actual_delay
                    );
                    RetryStrategy::GraceRetry(Duration::from_millis(actual_delay))
                } else {
                    let actual_delay = delay_ms.saturating_add(200).min(30_000);
                    RetryStrategy::FixedDelay(Duration::from_millis(actual_delay))
                }
            } else {
                // 否则使用线性退避：起始 5s，逐步增加
                RetryStrategy::LinearBackoff { base_ms: 5000 }
            }
        }

        // 503 服务不可用 / 529 服务器过载
        503 | 529 => {
            // 指数退避：起始 10s，上限 60s (针对 Google 边缘节点过载)
            RetryStrategy::ExponentialBackoff {
                base_ms: 10000,
                max_ms: 60000,
            }
        }

        // 500 服务器内部错误
        500 => {
            // 线性退避：起始 3s
            RetryStrategy::LinearBackoff { base_ms: 3000 }
        }

        // 401/403 认证/权限错误：切换账号前给予极短缓冲
        401 | 403 => RetryStrategy::FixedDelay(Duration::from_millis(200)),

        // 404 资源未找到：Google Cloud Code API 的 404 通常是账号级别的间歇性问题
        // (灰度发布、账号权限不同步等)，轮换账号往往能解决
        404 => RetryStrategy::FixedDelay(Duration::from_millis(300)),

        // 其他错误：不重试
        _ => RetryStrategy::NoRetry,
    }
}

/// 执行退避策略并返回是否应该继续重试
pub async fn apply_retry_strategy(
    strategy: RetryStrategy,
    attempt: usize,
    max_attempts: usize,
    status_code: u16,
    trace_id: &str,
) -> bool {
    match strategy {
        RetryStrategy::NoRetry => {
            debug!(
                "[{}] Non-retryable error {}, stopping",
                trace_id, status_code
            );
            false
        }

        RetryStrategy::FixedDelay(duration) => {
            let base_ms = duration.as_millis() as u64;
            info!(
                "[{}] ⏱️ Retry with fixed delay: status={}, attempt={}/{}, delay={}ms",
                trace_id,
                status_code,
                attempt + 1,
                max_attempts,
                base_ms
            );
            sleep(duration).await;
            true
        }

        RetryStrategy::LinearBackoff { base_ms } => {
            let calculated_ms = base_ms * (attempt as u64 + 1);
            info!(
                "[{}] ⏱️ Retry with linear backoff: status={}, attempt={}/{}, delay={}ms",
                trace_id,
                status_code,
                attempt + 1,
                max_attempts,
                calculated_ms
            );
            sleep(Duration::from_millis(calculated_ms)).await;
            true
        }

        RetryStrategy::ExponentialBackoff { base_ms, max_ms } => {
            let calculated_ms = (base_ms * 2_u64.pow(attempt as u32)).min(max_ms);
            info!(
                "[{}] ⏱️ Retry with exponential backoff: status={}, attempt={}/{}, delay={}ms",
                trace_id,
                status_code,
                attempt + 1,
                max_attempts,
                calculated_ms
            );
            sleep(Duration::from_millis(calculated_ms)).await;
            true
        }

        RetryStrategy::GraceRetry(duration) => {
            info!(
                "[{}] ⚡ Grace Retry: Performing micro-wait ({}ms) on current account...",
                trace_id,
                duration.as_millis()
            );
            sleep(duration).await;
            true // 原地重试在 handlers 层面通过 should_rotate_account 判断是否切换
        }
    }
}

/// 判断是否应该轮换账号
pub fn should_rotate_account(status_code: u16, strategy: Option<&RetryStrategy>) -> bool {
    // [NEW] 如果识别为 Grace Retry，则显式要求不轮换账号
    if let Some(RetryStrategy::GraceRetry(_)) = strategy {
        return false;
    }

    match status_code {
        // 这些错误是账号级别或特定节点配额的，需要轮换
        429 | 401 | 403 | 404 | 500 => true,
        // 503/529 通常是后端过载，切号效果有限，暂不轮换
        503 | 529 => false,
        _ => false,
    }
}

/// Detects model capabilities and configuration
/// POST /v1/models/detect
pub async fn handle_detect_model(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Response {
    let model_name = body.get("model").and_then(|v| v.as_str()).unwrap_or("");

    if model_name.is_empty() {
        return (StatusCode::BAD_REQUEST, "Missing 'model' field").into_response();
    }

    // 1. Resolve mapping
    let mapped_model = crate::proxy::common::model_mapping::resolve_model_route(
        model_name,
        &*state.custom_mapping.read().await,
    );

    // 2. Resolve capabilities
    let config = crate::proxy::mappers::common_utils::resolve_request_config(
        model_name,
        &mapped_model,
        &None, // We don't check tools for static capability detection
        None,  // size
        None,  // quality
        None,  // image_size
        None,  // body (not needed for static detection)
    );

    // 3. Construct response
    let mut response = json!({
        "model": model_name,
        "mapped_model": mapped_model,
        "type": config.request_type,
        "features": {
            "has_web_search": config.inject_google_search,
            "is_image_gen": config.request_type == "image_gen"
        }
    });

    if let Some(img_conf) = config.image_config {
        if let Some(obj) = response.as_object_mut() {
            obj.insert("config".to_string(), img_conf);
        }
    }

    Json(response).into_response()
}
