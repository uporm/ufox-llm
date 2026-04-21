//! `ufox-llm` 库入口。
//!
//! 统一导出 `client`、`provider`、`types` 与 `error` 的公共 API。

pub mod client;
pub mod error;
pub mod provider;
pub mod types;

pub use client::{ChatRequest, ChatRequestBuilder, ChatStream, Client, ClientBuilder};
pub use error::LlmError;
pub use provider::compatible::{CompatibleAdapter, CompatibleStreamParser};
pub use provider::openai::{OpenAiAdapter, OpenAiStreamParser};
pub use provider::qwen::{QwenAdapter, QwenStreamParser};
pub use provider::{Provider, ProviderAdapter};
pub use types::{
    AudioFile, AudioSource, ChatResponse, Content, ContentPart, DeltaType, FinishReason, ImageFile,
    ImageSource, JsonType, Message, MessageBuilder, ReasoningEffort, Role, StreamChunk, Tool,
    ToolBuilder, ToolCall, ToolChoice, ToolParameter, ToolResult, Usage, VideoFile, VideoSource,
};
