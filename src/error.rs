//! 错误类型。
//!
//! 统一定义 SDK 对外错误枚举。

use std::time::Duration;

use thiserror::Error;

/// SDK 对外统一暴露的错误类型。
///
/// 该枚举覆盖了调用 LLM Provider 时常见的错误来源，包括鉴权失败、限流、
/// 网络异常、响应解析失败以及 Provider 不支持某项能力等场景。
#[derive(Debug, Error)]
pub enum LlmError {
    /// Provider 返回了非预期的业务错误。
    ///
    /// 该变体用于保留 HTTP 状态码、错误消息以及 Provider 名称，便于上层统一记录。
    #[error("调用 {provider} 接口失败（状态码：{status_code}）：{message}")]
    ApiError {
        /// HTTP 状态码。
        status_code: u16,
        /// Provider 返回的错误消息。
        message: String,
        /// 产生错误的 Provider 名称。
        provider: String,
    },

    /// 鉴权失败。
    ///
    /// 该错误通常对应 HTTP `401 Unauthorized`，表示 API Key 无效、缺失或已过期。
    #[error("鉴权失败，请检查 API Key 是否正确或是否已过期")]
    AuthError,

    /// 触发了 Provider 的速率限制。
    ///
    /// 当返回 `429 Too Many Requests` 时，SDK 可以尝试从响应头中提取建议重试时间。
    #[error("请求频率超限，请稍后重试")]
    RateLimitError {
        /// 建议的重试等待时间。
        retry_after: Option<Duration>,
    },

    /// 发送 HTTP 请求或接收响应时出现网络错误。
    #[error("网络请求失败：{0}")]
    NetworkError(#[from] reqwest::Error),

    /// 解析 JSON 数据时发生错误。
    #[error("解析响应数据失败：{0}")]
    ParseError(#[from] serde_json::Error),

    /// 处理流式响应时发生错误。
    #[error("处理流式响应失败：{0}")]
    StreamError(String),

    /// 本地构建请求或输入校验失败。
    #[error("参数校验失败：{0}")]
    ValidationError(String),

    /// 当前 Provider 不支持调用方请求的能力。
    #[error("Provider {provider} 不支持功能：{feature}")]
    UnsupportedFeature {
        /// Provider 名称。
        provider: String,
        /// 不支持的功能名称。
        feature: String,
    },
}
