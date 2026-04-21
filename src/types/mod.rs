//! 类型系统模块。
//!
//! 该模块负责聚合 SDK 对外暴露的核心数据模型，包括消息模型、工具调用模型与响应模型，
//! 为 `client` 层和各个 `provider` 适配层提供统一的数据交换边界。
//!
//! 设计上采用“按语义拆分、在根模块聚合”的方式：
//! 1. `message` 聚焦对话输入与多模态内容表达；
//! 2. `tool` 聚焦工具声明、工具调用与工具执行结果；
//! 3. `response` 聚焦非流式与流式输出的统一响应建模。
//!
//! 这种拆分方式的好处是各子模块边界清晰，后续在 Provider 私有协议与公共类型之间做
//! 转换时，可以减少循环依赖和重复定义。

pub mod message;
pub mod response;
pub mod tool;

pub use message::{
    AudioFile, AudioSource, Content, ContentPart, ImageFile, ImageSource, Message, MessageBuilder,
    Role, VideoFile, VideoSource,
};
pub use response::{ChatResponse, DeltaType, FinishReason, ReasoningEffort, StreamChunk, Usage};
pub use tool::{JsonType, Tool, ToolBuilder, ToolCall, ToolChoice, ToolParameter, ToolResult};
