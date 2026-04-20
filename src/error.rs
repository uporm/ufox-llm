//! 错误类型模块。
//!
//! 该模块负责定义 SDK 对外统一暴露的错误枚举 [`LlmError`]，将不同 Provider、
//! HTTP 客户端、序列化与流式处理阶段产生的问题收敛为稳定的公共接口。
//!
//! 设计上优先保证：
//! 1. 调用方可以基于错误变体做精确分支处理；
//! 2. 错误消息保持中文，便于在国内团队项目中直接记录与排障；
//! 3. 保留底层错误源，方便调试与日志采集。
//!
//! 该模块依赖 `thiserror` 生成错误实现，并引用 `reqwest`、`serde_json` 与
//! 标准库中的 [`std::time::Duration`] 表达底层错误和重试等待时间。

use std::time::Duration;

use thiserror::Error;

/// SDK 对外统一暴露的错误类型。
///
/// 该枚举覆盖了调用 LLM Provider 时常见的错误来源，包括鉴权失败、限流、
/// 网络异常、响应解析失败以及 Provider 不支持某项能力等场景。
///
/// # 示例
/// ```rust
/// use ufox_llm::LlmError;
///
/// let error = LlmError::UnsupportedFeature {
///     provider: "Qwen".to_string(),
///     feature: "工具调用".to_string(),
/// };
///
/// assert!(error.to_string().contains("Qwen"));
/// ```
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

    /// 当前 Provider 不支持调用方请求的能力。
    #[error("Provider {provider} 不支持功能：{feature}")]
    UnsupportedFeature {
        /// Provider 名称。
        provider: String,
        /// 不支持的功能名称。
        feature: String,
    },
}
