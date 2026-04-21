//! Client 配置类型。
//!
//! 该模块承载 `Client` 相关的内部配置，以及请求级选项类型。

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::{Provider, ToolChoice, types::response::ReasoningEffort};

/// Provider 级别的客户端运行时配置。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProviderConfig {
    pub provider: Provider,
    pub api_key: String,
    pub base_url: Option<String>,
    pub organization: Option<String>,
    pub default_model: Option<String>,
    pub timeout_secs: Option<u64>,
    pub extra_headers: HashMap<String, String>,
}

/// 单次聊天请求的附加选项。
///
/// 该类型承载由 `ChatRequestBuilder` 构建的请求级配置，例如采样参数、思考模式、
/// 思考预算与推理强度。普通调用方通常无需直接构造它，而是通过
/// `ChatRequest::new(...)` 返回的构建器链式设置。
#[derive(Debug, Clone, Default)]
pub struct RequestOptions {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) provider_options: Map<String, Value>,
    pub(crate) thinking: bool,
    pub(crate) thinking_budget: Option<u32>,
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
    pub(crate) tool_choice: Option<ToolChoice>,
    pub(crate) parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ThinkingCapability {
    pub(crate) supports_thinking: bool,
    pub(crate) supports_thinking_budget: bool,
    pub(crate) supports_reasoning_effort: bool,
}
