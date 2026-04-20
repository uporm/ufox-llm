//! `ufox-llm` 库入口。
//!
//! 该文件负责整理 SDK 最终对外暴露的公共 API 面，统一收口以下能力：
//! 1. [`client`]：客户端主体、构建器与运行时配置；
//! 2. [`provider`]：Provider 抽象、官方适配器与兼容协议适配器；
//! 3. [`types`]：消息、工具调用、响应等公共数据模型；
//! 4. [`error`]：统一错误类型。
//!
//! 设计上采用“模块保留 + 根级常用类型再导出”的方式：
//! 1. 保留模块路径，方便高级用法按需访问底层能力；
//! 2. 将高频类型直接提升到 crate 根级，减少普通调用方的导入路径长度；
//! 3. 对 Provider 适配器与流式解析器同样提供根级再导出，便于扩展和测试。
//!
//! # 示例
//! ```rust
//! use ufox_llm::{Client, Message, Provider};
//!
//! let client = Client::builder()
//!     .provider(Provider::OpenAI)
//!     .api_key("sk-demo")
//!     .model("gpt-4o")
//!     .build()
//!     .expect("应构建成功");
//! let messages = vec![Message::user("你好")];
//!
//! let _ = (client, messages);
//! ```

pub mod error;
pub mod client;
pub mod provider;
pub mod types;

pub use client::{ChatRequestBuilder, ChatStream, ChatStreamRequestBuilder, Client};
pub use client::builder::{
    ApiKeySet, ApiKeyUnset, ClientBuilder, ClientConfig, CompatibleConfig, OpenAiConfig,
    ProviderConfig, ProviderSet, ProviderUnset, QwenConfig,
};
pub use error::LlmError;
pub use provider::{Provider, ProviderAdapter};
pub use provider::compatible::{CompatibleAdapter, CompatibleStreamParser};
pub use provider::openai::{OpenAiAdapter, OpenAiStreamParser};
pub use provider::qwen::{QwenAdapter, QwenStreamParser};
pub use types::{
    ChatResponse, Content, ContentPart, DeltaType, FinishReason, ImageFile, ImageSource,
    JsonType, Message, MessageBuilder, ReasoningEffort, Role, StreamChunk, Tool, ToolBuilder,
    ToolCall, ToolChoice, ToolKind, ToolParameter, ToolResult, Usage,
};
