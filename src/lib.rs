//! `ufox-llm` 库入口。
//!
//! 统一导出根级公共 API，并将具体实现模块保留在 crate 内部。

mod client;
mod error;
mod provider;
mod types;

pub use client::{ChatRequest, ChatRequestBuilder, ChatStream, Client, ClientBuilder};
pub use error::LlmError;
pub use provider::compatible::{CompatibleAdapter, CompatibleStreamParser};
pub use provider::openai::{OpenAiAdapter, OpenAiStreamParser};
pub use provider::qwen::{QwenAdapter, QwenStreamParser};
pub use provider::{Provider, ProviderAdapter};
pub use types::{
    AudioFile, AudioSource, ChatResponse, Content, ContentPart, DeltaKind, DeltaType,
    FinishReason, ImageFile, ImageSource, JsonType, Message, MessageBuilder, ReasoningEffort,
    Role, StreamChunk, Tool, ToolBuilder, ToolCall, ToolChoice, ToolParameter, ToolResult, Usage,
    VideoFile, VideoSource,
};
