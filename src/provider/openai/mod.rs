//! OpenAI 系 provider adapter 工厂。
//!
//! 本模块是 OpenAI 兼容协议实现的顶层入口，负责：
//! - 声明子模块并控制可见性
//! - 定义两套协议共用的常量与类型别名
//! - 提供 [`build`] 工厂函数，根据 [`ApiProtocol`] 选择具体 adapter 实现
//!
//! ## 子模块职责
//!
//! | 模块 | 职责 |
//! |------|------|
//! | `chat_completions` | Chat Completions 协议 adapter（含流式状态机） |
//! | `responses` | Responses 协议 adapter（含流式状态机） |
//! | `http` | HTTP 构造 trait、错误映射、SSE 解析、token 用量解析 |
//! | `media` | 媒体资源解析（URL / Base64 / 本地文件 → 字节 / 图片 URL） |
//! | `audio` | 语音识别与语音合成（两套协议共用） |
//! | `embedding` | 文本向量化（两套协议共用） |
//! | `image` | 图片生成（两套协议共用） |

mod audio;
mod chat_completions;
mod embedding;
mod http;
mod image;
mod media;
mod responses;

#[cfg(test)]
mod tests;

/// OpenAI Chat Completions 接口路径。
const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
/// OpenAI Responses 接口路径。
const RESPONSES_PATH: &str = "/responses";

use std::pin::Pin;

use futures::Stream;

use crate::{
    error::LlmError,
    middleware::Transport,
    provider::ApiProtocol,
    types::content::Role,
    types::response::ChatChunk,
};

use super::ProviderAdapter;
use chat_completions::ChatCompletionsAdapter;
use http::HttpContext;
use responses::ResponsesAdapter;

/// 两套协议共用的流式 chunk 输出类型别名。
pub(super) type ChatChunkStream =
    Pin<Box<dyn Stream<Item = Result<ChatChunk, LlmError>> + Send>>;

fn unsupported_multimodal_error(http_context: &HttpContext, role: Role) -> LlmError {
    LlmError::UnsupportedCapability {
        provider: Some(http_context.provider_name().into()),
        capability: match role {
            Role::User => "user_multimodal_content",
            Role::System => "system_multimodal_content",
            Role::Assistant => "assistant_multimodal_content",
            Role::Tool => "tool_multimodal_content",
        }
        .into(),
    }
}

/// 构造 OpenAI 兼容 provider adapter。
///
/// 根据 `protocol` 选择实现：
/// - [`ApiProtocol::ChatCompletions`] → [`ChatCompletionsAdapter`]（`POST /chat/completions`）
/// - [`ApiProtocol::Responses`] → [`ResponsesAdapter`]（`POST /responses`）
pub(crate) fn build(
    provider_name: &'static str,
    protocol: ApiProtocol,
    api_key: &str,
    base_url: &str,
    transport: &Transport,
) -> Result<Box<dyn ProviderAdapter>, LlmError> {
    let http_context = HttpContext::new(provider_name, api_key, base_url, transport.clone());
    match protocol {
        ApiProtocol::ChatCompletions => Ok(Box::new(ChatCompletionsAdapter::new(http_context))),
        ApiProtocol::Responses => Ok(Box::new(ResponsesAdapter::new(http_context))),
    }
}
