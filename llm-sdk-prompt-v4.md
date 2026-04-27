# ufox-llm SDK 设计规范 v4

> 适用于 AI 编程工具（Cursor、Trae 等）。本文件是唯一设计权威来源，实现时严格遵循章节顺序，依赖项先于使用方出现。

---

## 目录

1. [角色与目标](#1-角色与目标)
2. [技术选型与约束](#2-技术选型与约束)
3. [Crate 结构](#3-crate-结构)
4. [错误类型 — `src/error.rs`](#4-错误类型)
5. [核心类型 — `src/types/`](#5-核心类型)
   - 5.1 [内容类型 — `content/`](#51-内容类型)
   - 5.2 [请求类型 — `request.rs`](#52-请求类型)
   - 5.3 [响应类型 — `response.rs`](#53-响应类型)
6. [Provider 层 — `src/provider/`](#6-provider-层)
   - 6.1 [内部 Adapter Trait](#61-内部-adapter-trait)
   - 6.2 [Provider 枚举（兼工厂）](#62-provider-枚举兼工厂)
   - 6.3 [单个 Adapter 实现模板](#63-单个-adapter-实现模板)
   - 6.4 [协议映射参考表](#64-协议映射参考表)
   - 6.5 [工具调用闭环规范](#65-工具调用闭环规范)
7. [SDK 门面 — `src/client.rs`](#7-sdk-门面)
8. [中间件 — `src/middleware.rs`](#8-中间件)
9. [扩展指南：新增 Provider](#9-扩展指南新增-provider)
10. [测试规范](#10-测试规范)
11. [注释与编码规范](#11-注释与编码规范)
12. [完整使用示例](#12-完整使用示例)

---

## 1. 角色与目标

你是一名资深 Rust 系统工程师，正在设计并实现一个生产级、多 Provider、多模态的 LLM SDK（**ufox-llm**）。

**核心设计目标：**

1. **对外**：提供统一、稳定、符合人体工学的 Rust API，调用方无需感知底层协议差异。
2. **对内**：不同 Provider 走不同协议（OpenAI 兼容、Anthropic Messages、Doubao、Qwen、Gemini、自定义网关），每个 Provider 独立实现，互不耦合。
3. **多模态 + 工具调用一等公民**：围绕 `Text`、`Image`、`Speech`、`Video`、`Embedding`、`Tool` 六类核心能力建模，全部强类型表达。工具能力必须覆盖**定义、选择、调用、流式聚合、结果回传**的完整闭环。
4. **可扩展**：新增 Provider 只需新增枚举变体 + `match` 臂 + 实现文件，`ClientBuilder` 和所有对外 API 零改动；新增模态只需扩展枚举，不破坏现有接口。
5. **生产级质量**：完整错误处理、流式输出、重试/限流中间件、可观测性、零 unsafe（除非有充分文档说明）。

---

## 2. 技术选型与约束

```toml
# Cargo.toml 关键依赖（具体版本以最新稳定版为准）
[dependencies]
tokio        = { version = "1",    features = ["full"] }
reqwest      = { version = "0.12", features = ["json", "stream", "multipart"] }
serde        = { version = "1",    features = ["derive"] }
serde_json   = "1"
futures      = "0.3"
tower        = "0.5"
tower-http   = "0.6"
bytes        = "1"
thiserror    = "2"
tracing      = "0.1"
async-trait  = "0.1"
mime_guess   = "2"        # MediaSource::File 的 MIME 推断
base64       = "0.22"     # MediaSource::File 的 base64 编码
tokio-util   = { version = "0.7", features = ["io"] }  # 流式字节处理
```

**强制约束：**

- 所有异步 trait 方法使用 `async-trait` 宏（待 AFIT 稳定后迁移）。
- `ProviderAdapter::chat_stream` 内部返回 `Pin<Box<dyn Stream<Item = Result<ChatChunk, LlmError>> + Send>>`；`Client::chat_stream` 对外暴露 `impl Stream<...> + Send`（通过 `-> impl Stream` 语法隐藏装箱细节）。
- 禁止在 `lib.rs` 根层 re-export 三方库类型，防止版本锁定。
- 所有公共类型必须实现 `Debug`、`Clone`（流类型除外）。
- `LlmError` 必须是 `Send + Sync + 'static`。
- 零 unsafe，如有例外必须附 `// SAFETY:` 注释。
- `Client` 不实现 `Clone`；跨任务共享请使用 `Arc<Client>`。

---

## 3. Crate 结构

单 crate，按模块职责分目录，无 workspace 拆分。**各节代码按此结构一一对应，实现顺序遵循依赖顺序（被依赖项先实现）。**

```
ufox-llm/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # pub use 统一对外暴露（见本节末尾）  ← 最后实现
│   ├── error.rs                # LlmError                            ← 最先实现
│   ├── types/
│   │   ├── mod.rs              # re-export content / request / response
│   │   ├── content/
│   │   │   ├── mod.rs          # 统一 re-export                       ← 依赖 media/tool/message
│   │   │   ├── media.rs        # MediaSource / Text / Image / Audio / Video
│   │   │   ├── tool.rs         # Tool / ToolChoice / ToolCall / ToolResult
│   │   │   └── message.rs      # ContentPart / Role / Message
│   │   ├── request.rs          # ChatRequest / EmbeddingRequest 等    ← 依赖 content
│   │   └── response.rs         # ChatResponse / ChatChunk 等          ← 依赖 content
│   ├── provider/
│   │   ├── mod.rs              # ProviderAdapter trait + Provider 枚举 ← 依赖 types + error
│   │   ├── openai.rs           # OpenAI & Compatible 协议实现
│   │   ├── anthropic.rs        # Anthropic Messages 协议实现
│   │   ├── doubao.rs           # 豆包协议实现
│   │   ├── qwen.rs             # 通义千问协议实现
│   │   └── gemini.rs           # Gemini 协议实现
│   ├── middleware.rs            # Tower retry / rate-limit / logging   ← 依赖 error
│   └── client.rs               # Client + ClientBuilder               ← 依赖 provider + middleware
└── examples/
    ├── chat_basic.rs
    ├── streaming.rs
    ├── multimodal_image.rs
    ├── embed_and_search.rs
    ├── speech_to_text.rs
    ├── text_to_speech.rs
    ├── tool_calling.rs
    └── compatible_provider.rs
```

**`src/types/mod.rs` 内容：**

```rust
pub mod content;
pub mod request;
pub mod response;
```

**`src/lib.rs` 公共导出树（完整）：**

```rust
// 错误类型
pub use error::LlmError;

// 内容原语
pub use types::content::{
    // media
    AudioFormat, ImageFidelity, MediaSource, VideoFormat,
    Audio, Image, Text, Video,
    // tool
    Tool, ToolCall, ToolChoice, ToolResult, ToolResultPayload,
    // message
    ContentPart, Message, Role,
};

// 请求类型
pub use types::request::{
    ChatRequest, ChatRequestBuilder,
    EmbeddingRequest,
    ImageGenRequest,
    SpeechToTextRequest,
    TextToSpeechRequest,
    VideoGenRequest,
};

// 响应类型
pub use types::response::{
    ChatChunk, ChatResponse,
    EmbeddingResponse,
    FinishReason,
    GeneratedImage, ImageGenResponse,
    SpeechToTextResponse,
    TaskStatus, TextToSpeechResponse,
    Usage,
    VideoGenResponse,
};

// SDK 门面
pub use client::{Client, ClientBuilder};
pub use provider::Provider;

// 模块声明（不对外暴露内部结构）
mod error;
mod types;
mod provider;
mod middleware;
mod client;
```

**模块边界（强制）：**

- `src/types/` 不引用 `src/provider/` 下任何模块。
- `src/provider/xxx.rs` 只引用 `src/types/`、`src/error.rs`，provider 文件间不互相引用。
- `src/client.rs` 只依赖 `src/provider/mod.rs` 暴露的 `Provider` 与 `ProviderAdapter` 抽象，不直接依赖具体 `provider/xxx.rs`。
- 每个 provider 的协议私有类型定义在对应 `provider/xxx.rs` 内部，标记 `pub(crate)` 或直接私有，不进入 `lib.rs` 的 `pub use` 树。

---

## 4. 错误类型

**文件：`src/error.rs`**

```rust
#[derive(thiserror::Error, Debug)]
pub enum LlmError {
    // ── 构建期错误 ─────────────────────────────────────────────────────────
    #[error("缺少必填配置项：{field}")]
    MissingConfig { field: &'static str },

    #[error("配置不合法：{message}")]
    InvalidConfig { message: String },

    // ── 远端调用错误 ───────────────────────────────────────────────────────
    #[error("HTTP 状态错误 [{provider}]：status={status}，body={body}")]
    HttpStatus { provider: String, status: u16, body: String },

    #[error("Provider 响应错误 [{provider}]：code={code:?}，message={message}")]
    ProviderResponse { provider: String, code: Option<String>, message: String },

    #[error("认证错误：{message}")]
    Authentication { message: String },

    #[error("触发限流：retry_after={retry_after_secs:?}s")]
    RateLimit { retry_after_secs: Option<u64> },

    #[error("请求超时：{timeout_ms}ms")]
    RequestTimeout { timeout_ms: u64 },

    #[error("网络传输错误：{0}")]
    Transport(#[from] reqwest::Error),

    // ── 协议与编解码错误 ───────────────────────────────────────────────────
    #[error("JSON 编解码错误：{0}")]
    JsonCodec(#[from] serde_json::Error),

    #[error("流式协议错误 [{provider}]：{message}")]
    StreamProtocol { provider: String, message: String },

    #[error("工具协议错误：{message}")]
    ToolProtocol { message: String },

    // ── 能力边界错误 ───────────────────────────────────────────────────────
    // provider 为 None 表示能力在 SDK 层就不支持（与具体 provider 无关）
    #[error("Provider [{provider:?}] 不支持该能力：{capability}")]
    UnsupportedCapability { provider: Option<String>, capability: String },

    // ── 多模态输入错误 ─────────────────────────────────────────────────────
    // 文件读取、MIME 推断、尺寸/格式校验等
    #[error("多模态输入错误：{message}")]
    MediaInput { message: String },
}
```

**命名原则：**

- 变体名优先体现**失败发生的位置**，而不是笼统结果；例如 `HttpStatus` 明确表示已收到 HTTP 响应但状态码异常，而 `Transport` 表示请求尚处于传输层失败。
- 对容易与其他层重名的概念补充语义后缀；例如 `ProviderResponse`、`JsonCodec`、`MediaInput`，避免出现 `Provider`、`Json`、`Media` 这类过宽命名。
- 构建期 / 能力边界错误保留短而稳定的名词短语；如 `MissingConfig`、`InvalidConfig`、`UnsupportedCapability`。
- 不引入 `anyhow`，也不保留公共兜底变体；所有错误必须在 adapter 或 SDK 边界被映射到明确语义的 `LlmError` 变体。

**错误分层原则：**

| 错误变体 | 触发阶段 | 是否可重试 |
|---|---|---|
| `MissingConfig` / `InvalidConfig` | 构建期 | 否，直接返回 |
| `UnsupportedCapability` | 能力路由层 | 否，直接返回 |
| `Authentication` | 认证阶段 | 默认否（除非 provider 文档明确说明 token 可瞬时刷新） |
| `RateLimit` / `RequestTimeout` / `HttpStatus`（命中 retryable_status） | 网络调用层 | 是，进入中间件重试 |
| `ProviderResponse` | 远端响应解析 | 由 adapter 映射决定，只有明确可重试的错误码才重试 |
| `JsonCodec` / `StreamProtocol` / `ToolProtocol` | 协议解析层 | 否，直接返回 |
| `MediaInput` | 多模态预处理 | 否，直接返回 |

---

## 5. 核心类型

> `src/types/` 内的所有类型**不引用** `src/provider/` 下任何内容。

### 5.1 内容类型

**目录：`src/types/content/`**

拆分原则：

- `media.rs` 只承载媒体与格式类型
- `tool.rs` 只承载工具定义、调用与回填结果类型
- `message.rs` 负责把媒体与工具类型组合为 `ContentPart` / `Message`
- `mod.rs` 只做统一 re-export，对外仍保留 `crate::types::content::{...}` 的稳定引用路径

**文件：`src/types/content/mod.rs`**

```rust
mod media;
mod tool;
mod message;

pub use media::*;
pub use tool::*;
pub use message::*;
```

**文件：`src/types/content/media.rs`**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageFidelity { Auto, Low, High }

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat { Mp3, Wav, Flac, Opus, Aac, Pcm }

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VideoFormat { Mp4, Webm, Avi, Mov }

/// 多模态内容来源。
///
/// `File` 变体由 adapter 层在组装请求体时异步读取并转为 `Base64`，
/// 调用方无需手动编码。读取失败返回 `LlmError::MediaInput`。
///
/// 对不支持 `Url` 变体的 provider，adapter 内部自动下载并转为 `Base64`；
/// 若下载失败同样返回 `LlmError::MediaInput`。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MediaSource {
    /// 内联 base64 编码内容
    Base64 { data: String, mime_type: String },
    /// 远程 URL
    Url { url: String },
    /// 本地文件路径（SDK 在 adapter 层异步读取、编码、推断 MIME）
    File { path: std::path::PathBuf },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Text { pub text: String }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Image {
    pub source: MediaSource,
    /// OpenAI 图像输入保真度参数，其他 provider 忽略此字段
    pub fidelity: Option<ImageFidelity>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Audio {
    pub source: MediaSource,
    pub format: AudioFormat,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Video {
    pub source: MediaSource,
    pub format: VideoFormat,
    /// 采样帧数，None 表示由 provider 决定
    pub sample_frames: Option<u32>,
}
```

**文件：`src/types/content/tool.rs`**

```rust
/// 注册给模型的可用工具。
///
/// `input_schema` 字段名与 Anthropic 协议对齐；OpenAI 系 adapter 在序列化时
/// 将其映射为 `parameters` 字段，调用方无需感知这一差异。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// 工具输入的 JSON Schema；公共接口直接使用 `serde_json::Value`。
    pub input_schema: serde_json::Value,
}

impl Tool {
    /// 用 JSON Schema 描述参数构造工具定义。
    ///
    /// # Examples
    ///
    /// ```
    /// use ufox_llm::Tool;
    ///
    /// let tool = Tool::function(
    ///     "get_weather",
    ///     "查询城市实时天气",
    ///     serde_json::json!({
    ///         "type": "object",
    ///         "properties": { "city": { "type": "string" } },
    ///         "required": ["city"]
    ///     }),
    /// );
    /// ```
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema: parameters,
        }
    }
}

/// 工具选择策略。
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// 模型自行决定是否调用工具（默认）
    #[default]
    Auto,
    /// 强制不调用任何工具；adapter 应按协议最精确的方式表达"禁止调用"语义，
    /// 而非仅靠省略 tools 字段（省略与禁止在部分 provider 语义不同）。
    None,
    /// 强制必须调用至少一个工具，由模型自行选择
    Required,
    /// 强制调用指定名称的工具
    Specific(String),
}

/// assistant 发出的单次工具调用（已完整解析，不含增量片段）。
///
/// 在流式路径中，SDK 内部完成所有分片聚合，只有当 `id`、`tool_name`、
/// `arguments` 全部就绪且 JSON 可解析后，才会写入 `ChatChunk.tool_calls`。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    /// provider 分配的唯一 ID，用于匹配后续 `ToolResult`，adapter 不得重写
    pub id: String,
    pub tool_name: String,
    /// 始终为已解析的 JSON 值，禁止以字符串形式存储。
    pub arguments: serde_json::Value,
}

/// tool role 回传的工具执行结果。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolResult {
    /// 对应 `ToolCall.id`
    pub tool_call_id: String,
    /// 部分 provider（如 Doubao、Qwen）在回传结果时需要重复携带工具名；
    /// 若 provider 不需要此字段，adapter 在序列化时忽略 `None` 值。
    pub tool_name: Option<String>,
    pub payload: ToolResultPayload,
    /// 标记执行是否出错（部分 provider 使用此字段区分成功/失败结果）
    pub is_error: bool,
}

/// 工具执行结果载荷，支持纯文本或结构化 JSON，不强行降级为字符串。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum ToolResultPayload {
    Text(String),
    Json(serde_json::Value),
}

impl ToolResultPayload {
    pub fn text(s: impl Into<String>) -> Self { Self::Text(s.into()) }
    pub fn json(v: serde_json::Value) -> Self { Self::Json(v) }
}
```

**文件：`src/types/content/message.rs`**

```rust
use super::{Audio, Image, MediaSource, Text, ToolCall, ToolResult, ToolResultPayload, Video};

/// 消息内容的最小单元，覆盖所有模态和工具交互。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text(Text),
    Image(Image),
    Audio(Audio),
    Video(Video),
    /// assistant 发出的工具调用（已完整解析）
    ToolCall(ToolCall),
    /// tool role 回传的工具执行结果
    ToolResult(ToolResult),
}

impl ContentPart {
    pub fn text(s: impl Into<String>) -> Self {
        ContentPart::Text(Text { text: s.into() })
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        ContentPart::Image(Image {
            source: MediaSource::Url { url: url.into() },
            fidelity: None,
        })
    }

    pub fn image_file(path: impl Into<std::path::PathBuf>) -> Self {
        ContentPart::Image(Image {
            source: MediaSource::File { path: path.into() },
            fidelity: None,
        })
    }

    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        ContentPart::ToolCall(ToolCall {
            id: id.into(),
            tool_name: name.into(),
            arguments,
        })
    }

    /// 构造文本型工具结果。
    ///
    /// 若需携带 `tool_name`（Doubao / Qwen 等 provider 要求），请直接构造
    /// `ContentPart::ToolResult(ToolResult { tool_name: Some(...), ... })`。
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        ContentPart::ToolResult(ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: None,
            payload: ToolResultPayload::text(content),
            is_error: false,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role { System, User, Assistant, Tool }

/// 对话历史中的单条消息。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
    /// provider-specific 元数据，透传不解析
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    /// 按顺序拼接消息中所有文本片段。
    pub fn text(&self) -> String {
        self.content.iter().filter_map(|p| {
            if let ContentPart::Text(t) = p { Some(t.text.as_str()) } else { None }
        }).collect::<Vec<_>>().join("")
    }

    /// 返回消息中所有工具调用的副本（保持原顺序），便于上层 agent loop 驱动执行。
    pub fn tool_calls(&self) -> Vec<ToolCall> {
        self.content.iter().filter_map(|p| {
            if let ContentPart::ToolCall(c) = p { Some(c.clone()) } else { None }
        }).collect()
    }
}
```

### 5.2 请求类型

**文件：`src/types/request.rs`**

> `model` 字段**不在**任何请求类型中，由 `Client` 持有，调用 `ProviderAdapter` 时传入。

```rust
use crate::types::content::{ContentPart, Message, Role, Tool, ToolChoice};

// ── Chat ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub tools: Vec<Tool>,
    pub tool_choice: ToolChoice,
    /// 透传给 provider 的差异化参数（如 Anthropic 的 `thinking`）。
    ///
    /// adapter 将整个 map **浅层 merge** 到请求体顶层；key 冲突时 extensions 优先。
    /// 未被当前 provider 识别的 key 会静默忽略（由上层协议决定）。
    /// key 命名建议加 provider 前缀以避免冲突，例如 `"anthropic_thinking"`。
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

impl ChatRequest {
    pub fn builder() -> ChatRequestBuilder {
        ChatRequestBuilder { inner: Self::default() }
    }
}

pub struct ChatRequestBuilder {
    inner: ChatRequest,
}

impl ChatRequestBuilder {
    /// 覆盖整个消息列表。
    ///
    /// 若在此之后再调用 `system()` 或 `user_text()`，消息将追加到已有列表中。
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.inner.messages = messages; self
    }

    /// 在消息列表**头部**插入一条 system 消息。
    ///
    /// 若已存在 system 消息（`messages[0].role == Role::System`），则**替换**而非重复插入，
    /// 避免多次调用时产生多条 system 消息。
    pub fn system(mut self, text: impl Into<String>) -> Self {
        let msg = Message {
            role: Role::System,
            content: vec![ContentPart::text(text)],
            name: None,
        };
        if self.inner.messages.first().map(|m| m.role) == Some(Role::System) {
            self.inner.messages[0] = msg;
        } else {
            self.inner.messages.insert(0, msg);
        }
        self
    }

    pub fn user_text(mut self, text: impl Into<String>) -> Self {
        self.inner.messages.push(Message {
            role: Role::User,
            content: vec![ContentPart::text(text)],
            name: None,
        });
        self
    }

    pub fn user(mut self, parts: Vec<ContentPart>) -> Self {
        self.inner.messages.push(Message { role: Role::User, content: parts, name: None });
        self
    }

    pub fn max_tokens(mut self, n: u32) -> Self { self.inner.max_tokens = Some(n); self }
    pub fn temperature(mut self, t: f32) -> Self { self.inner.temperature = Some(t); self }
    pub fn top_p(mut self, p: f32) -> Self { self.inner.top_p = Some(p); self }
    pub fn tools(mut self, tools: Vec<Tool>) -> Self { self.inner.tools = tools; self }
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self { self.inner.tool_choice = choice; self }

    pub fn extension(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.inner.extensions.insert(key.into(), value); self
    }

    pub fn build(self) -> ChatRequest { self.inner }
}

// ── Embedding ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// 待向量化的文本列表；单条输入使用长度为 1 的 Vec 表示。
    pub inputs: Vec<String>,
    /// 期望的向量维度，None 由 model 决定
    pub dimensions: Option<usize>,
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

// ── Speech ────────────────────────────────────────────────────────────────

use crate::types::content::{AudioFormat, MediaSource};

#[derive(Debug, Clone)]
pub struct SpeechToTextRequest {
    /// 音频来源。
    ///
    /// 若 provider 不支持 `MediaSource::Url` 变体，adapter 内部自动下载转为
    /// `Base64`；若下载失败，返回 `LlmError::MediaInput`。
    pub source: MediaSource,
    pub format: AudioFormat,
    /// 语言提示（如 "zh"、"en"），None 由 provider 自动检测
    pub language: Option<String>,
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct TextToSpeechRequest {
    pub text: String,
    pub voice: Option<String>,
    pub output_format: AudioFormat,
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

// ── Image Generation ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ImageGenRequest {
    pub prompt: String,
    pub n: Option<u32>,
    pub size: Option<String>,
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

// ── Video Generation ───────────────────────────────────────────────────────

use crate::types::content::VideoFormat;

#[derive(Debug, Clone)]
pub struct VideoGenRequest {
    pub prompt: String,
    pub duration_secs: Option<u32>,
    pub output_format: Option<VideoFormat>,
    pub extensions: serde_json::Map<String, serde_json::Value>,
}
```

### 5.3 响应类型

**文件：`src/types/response.rs`**

```rust
use crate::types::content::{AudioFormat, ContentPart, Message, Role, ToolCall};

// ── Chat 非流式响应 ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    /// 累积的完整文本（对应流式路径中所有 `text_delta` 拼接结果）
    pub text: String,
    /// 累积的推理文本（仅支持 extended thinking 的 provider/model 有值）
    pub thinking: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    /// 原始响应体，调试用。
    ///
    /// 默认由 adapter 填充（`Some(...)`）；若需在生产环境关闭以节省内存，
    /// 可在 `ChatRequest.extensions` 中传入 `{"_sdk_omit_raw": true}`，
    /// adapter 检测到该 key 时将此字段设为 `None`。
    pub raw: Option<serde_json::Value>,
}

impl ChatResponse {
    /// 将响应转为可追加到历史对话的 assistant 消息。
    ///
    /// 文本和工具调用按输出顺序排列；thinking 不计入标准历史，
    /// 如需保留可通过 `ChatRequest.extensions` 传递给支持它的 provider。
    pub fn into_message(self) -> Message {
        let mut parts = vec![];
        if !self.text.is_empty() {
            parts.push(ContentPart::text(self.text));
        }
        for call in self.tool_calls {
            parts.push(ContentPart::ToolCall(call));
        }
        Message { role: Role::Assistant, content: parts, name: None }
    }
}

// ── Chat 流式 chunk ────────────────────────────────────────────────────────

/// 流式 chat 的语义输出单元。
///
/// 公共 API 不暴露 provider 的原始 chunk、参数增量或未完成的工具调用分片，
/// 这些均属 adapter 内部细节。一个 `ChatChunk` 表示一个有序的可消费单元：
///
/// - 若 provider 同一帧同时含内容增量和结束信号，聚合到同一 `ChatChunk`。
/// - 若 provider 同一帧出现 `text → tool_call → text` 交错，拆成多个 `ChatChunk`
///   保序输出；中间 chunk 的 `finish_reason` 和 `usage` 均为 `None`，
///   仅最后一个 chunk 携带这两个字段。
#[derive(Debug, Clone, Default)]
pub struct ChatChunk {
    pub text_delta: Option<String>,
    /// 推理过程增量（仅支持 extended thinking 的 provider/model 有值）
    pub thinking_delta: Option<String>,
    /// 完整工具调用（仅当 id/name/arguments 全部就绪且 JSON 可解析后才写入）
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<FinishReason>,
    /// 通常仅在最后一个 chunk 中出现
    pub usage: Option<Usage>,
}

impl ChatChunk {
    pub fn is_finished(&self) -> bool { self.finish_reason.is_some() }
}

// ── 公共枚举与结构 ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Completed,
    MaxOutputTokens,
    ToolCalls,
    ContentFilter,
    Failed,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Embedding 响应 ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub usage: Option<Usage>,
}

// ── Speech 响应 ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SpeechToTextResponse {
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: Option<f32>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone)]
pub struct TextToSpeechResponse {
    pub audio_data: bytes::Bytes,
    pub format: AudioFormat,
    pub duration_secs: Option<f32>,
}

// ── Image / Video 响应 ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ImageGenResponse {
    pub images: Vec<GeneratedImage>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub url: Option<String>,
    pub base64: Option<String>,
    pub revised_prompt: Option<String>,
}

/// 视频生成响应。
///
/// 视频生成通常为异步任务；拿到 `task_id` 后，调用方应轮询
/// `Client::poll_video_task(task_id)` 直至 `status` 变为
/// `TaskStatus::Succeeded` 或 `TaskStatus::Failed`。
#[derive(Debug, Clone)]
pub struct VideoGenResponse {
    pub task_id: String,
    pub status: TaskStatus,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus { Pending, Processing, Succeeded, Failed }
```

---

## 6. Provider 层

**目录：`src/provider/`**

### 6.1 内部 Adapter Trait

**文件：`src/provider/mod.rs`（前半部分）**

此 trait 是 `pub(crate)` 的内部接口，外部不可见。所有签名中的 `model` 参数由 `Client` 持有并在调用时传入。

```rust
use std::pin::Pin;
use futures::Stream;
use async_trait::async_trait;
use crate::{
    error::LlmError,
    types::{
        request::{
            ChatRequest, EmbeddingRequest, ImageGenRequest,
            SpeechToTextRequest, TextToSpeechRequest, VideoGenRequest,
        },
        response::{
            ChatChunk, ChatResponse, EmbeddingResponse, ImageGenResponse,
            SpeechToTextResponse, TextToSpeechResponse, VideoGenResponse,
        },
    },
};

#[async_trait]
pub(crate) trait ProviderAdapter: Send + Sync {
    /// 返回稳定的 provider 名称，用于错误字段、日志与指标标签。
    fn name(&self) -> &'static str;

    async fn chat(&self, model: &str, req: ChatRequest) -> Result<ChatResponse, LlmError>;

    async fn chat_stream(
        &self,
        model: &str,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, LlmError>> + Send>>, LlmError>;

    // 以下能力有默认实现，返回 UnsupportedCapability。
    // 各 adapter 按需覆盖，无需实现全部能力。

    async fn embed(
        &self, _model: &str, _req: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            provider: Some(self.name().into()),
            capability: "embed".into(),
        })
    }

    async fn speech_to_text(
        &self, _model: &str, _req: SpeechToTextRequest,
    ) -> Result<SpeechToTextResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            provider: Some(self.name().into()),
            capability: "speech_to_text".into(),
        })
    }

    async fn text_to_speech(
        &self, _model: &str, _req: TextToSpeechRequest,
    ) -> Result<TextToSpeechResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            provider: Some(self.name().into()),
            capability: "text_to_speech".into(),
        })
    }

    async fn generate_image(
        &self, _model: &str, _req: ImageGenRequest,
    ) -> Result<ImageGenResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            provider: Some(self.name().into()),
            capability: "generate_image".into(),
        })
    }

    async fn generate_video(
        &self, _model: &str, _req: VideoGenRequest,
    ) -> Result<VideoGenResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            provider: Some(self.name().into()),
            capability: "generate_video".into(),
        })
    }

    /// 查询视频生成任务状态。
    ///
    /// 默认返回 `UnsupportedCapability`；实现了 `generate_video` 的 adapter
    /// 必须同时覆盖此方法。
    async fn poll_video_task(
        &self, _task_id: &str,
    ) -> Result<VideoGenResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            provider: Some(self.name().into()),
            capability: "poll_video_task".into(),
        })
    }
}
```

### 6.2 Provider 枚举（兼工厂）

**文件：`src/provider/mod.rs`（后半部分）**

`Provider` 枚举承担两个职责：协议名称（变体语义）与 adapter 创建（`into_adapter()` 方法）。`base_url`、超时、重试、限流等连接配置由 `ClientBuilder` 持有并在构建时解析。

```rust
mod openai;
mod anthropic;
mod doubao;
mod qwen;
mod gemini;

/// 协议名称与 adapter 工厂，二者合一。
#[derive(Debug, Clone)]
pub enum Provider {
    /// OpenAI 兼容协议，必须由 ClientBuilder 显式提供 base_url。
    /// 适用于 DeepSeek、Moonshot、01.AI、本地 Ollama、私有网关等。
    Compatible,

    /// 标准 OpenAI 协议，base_url 固定
    OpenAI,

    /// Anthropic Messages 协议
    Anthropic,

    /// 豆包（火山引擎）协议
    Doubao,

    /// 通义千问协议
    Qwen,

    /// Google Gemini 协议
    Gemini,
}

impl Provider {
    /// 返回稳定的 provider 名称字符串。
    pub fn name(&self) -> &'static str {
        match self {
            Provider::Compatible  => "compatible",
            Provider::OpenAI      => "openai",
            Provider::Anthropic   => "anthropic",
            Provider::Doubao      => "doubao",
            Provider::Qwen        => "qwen",
            Provider::Gemini      => "gemini",
        }
    }

    /// 根据稳定名称解析 provider。
    ///
    /// 该方法与 `name()` 共同构成双向稳定映射；
    /// 所有字符串到枚举的解析逻辑统一收口到 `Provider`，避免在 `Client::from_env()`
    /// 等上层入口重复维护同一份映射表。
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "compatible" => Some(Provider::Compatible),
            "openai"     => Some(Provider::OpenAI),
            "anthropic"  => Some(Provider::Anthropic),
            "doubao"     => Some(Provider::Doubao),
            "qwen"       => Some(Provider::Qwen),
            "gemini"     => Some(Provider::Gemini),
            _ => None,
        }
    }

    /// 返回该 provider 的默认 base URL。
    ///
    /// `Compatible` 没有默认值，必须由 `ClientBuilder::base_url(...)` 提供。
    pub fn default_base_url(&self) -> Option<&'static str> {
        match self {
            Provider::OpenAI     => Some("https://api.openai.com/v1"),
            Provider::Anthropic  => Some("https://api.anthropic.com"),
            Provider::Doubao     => Some("https://ark.cn-beijing.volces.com/api/v3"),
            Provider::Qwen       => Some("https://dashscope.aliyuncs.com/compatible-mode/v1"),
            Provider::Gemini     => Some("https://generativelanguage.googleapis.com/v1beta"),
            Provider::Compatible => None,
        }
    }

    /// adapter 创建的唯一入口。
    ///
    /// 只做稳定路由：每新增变体，只在此处加一个 match 臂。
    pub(crate) fn into_adapter(
        &self,
        api_key: &str,
        base_url: &str,
        transport: &crate::middleware::Transport,
    ) -> Result<Box<dyn ProviderAdapter>, LlmError> {
        match self {
            // Compatible 复用 OpenAI adapter；provider_name 传入 "compatible"
            // 使错误信息和日志能区分两者。
            Provider::Compatible =>
                openai::build("compatible", api_key, base_url, transport),

            Provider::OpenAI =>
                openai::build("openai", api_key, base_url, transport),

            Provider::Anthropic =>
                anthropic::build(api_key, base_url, transport),

            Provider::Doubao =>
                doubao::build(api_key, base_url, transport),

            Provider::Qwen =>
                qwen::build(api_key, base_url, transport),

            Provider::Gemini =>
                gemini::build(api_key, base_url, transport),
        }
    }
}
```

### 6.3 单个 Adapter 实现模板

以 **Anthropic** 为例。除 `OpenAI` / `Compatible` 这种**复用型 adapter** 外，其余独立 provider 的文件结构大体相同。

**复用型 adapter（`openai.rs`）的 `build` 签名：**

```rust
// OpenAI / Compatible 共享此签名，额外接收 provider_name 以支持日志和错误字段区分。
pub(crate) fn build(
    provider_name: &'static str,
    api_key: &str,
    base_url: &str,
    transport: &Transport,
) -> Result<Box<dyn ProviderAdapter>, LlmError>
```

**独立 provider 的统一 `build` 签名：**

```rust
// anthropic / doubao / qwen / gemini 及所有未来新增 provider 均采用此签名。
pub(crate) fn build(
    api_key: &str,
    base_url: &str,
    transport: &Transport,
) -> Result<Box<dyn ProviderAdapter>, LlmError>
```

**文件：`src/provider/anthropic.rs`**

```rust
use std::pin::Pin;
use futures::Stream;
use async_trait::async_trait;
use crate::{error::LlmError, middleware::Transport, types::{request::*, response::*}};
use super::ProviderAdapter;

// ── adapter 结构体 ─────────────────────────────────────────────────────────

pub(crate) struct AnthropicAdapter {
    transport: Transport,
    api_key: String,
    base_url: String,
    // Anthropic 要求显式传 API 版本；如后续升级，在此处集中调整
    api_version: &'static str,
}

impl AnthropicAdapter {
    fn new(api_key: &str, base_url: &str, transport: Transport) -> Self {
        Self {
            transport,
            api_key: api_key.to_owned(),
            base_url: base_url.to_owned(),
            api_version: "2023-06-01",
        }
    }

    /// 将 `MediaSource::File` 异步读取并转为 `Base64`。
    ///
    /// 必须在 `async` 上下文中调用；调用方（`to_request_body_async`）负责 await。
    async fn resolve_media_source(
        source: &crate::types::content::MediaSource,
    ) -> Result<(String, String), LlmError> {
        use crate::types::content::MediaSource;
        match source {
            MediaSource::Base64 { data, mime_type } => {
                Ok((data.clone(), mime_type.clone()))
            }
            MediaSource::Url { url } => {
                // 对不支持 URL 的场景：下载并转码
                // 此处返回 URL 字符串；若 provider 不支持 URL，调用方应先调用此函数下载
                Ok((url.clone(), "".to_owned()))
            }
            MediaSource::File { path } => {
                let data = tokio::fs::read(path).await.map_err(|e| {
                    LlmError::MediaInput {
                        message: format!("读取文件失败 {:?}: {}", path, e),
                    }
                })?;
                let mime = mime_guess::from_path(path)
                    .first_raw()
                    .unwrap_or("application/octet-stream")
                    .to_owned();
                Ok((base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD, &data,
                ), mime))
            }
        }
    }

    /// 异步版请求体构造（处理 File/Url → Base64 转换）。
    async fn to_request_body_async(
        &self, model: &str, req: &ChatRequest, stream: bool,
    ) -> Result<serde_json::Value, LlmError> {
        // 检查工具能力
        if !req.tools.is_empty() {
            // Anthropic 支持工具；若未来某 adapter 不支持，在此处返回：
            // return Err(LlmError::UnsupportedCapability {
            //     provider: Some(self.name().into()),
            //     capability: "tools".into(),
            // });
        }

        // extensions 浅层 merge：先构造基础 body，再 merge extensions
        let omit_raw = req.extensions.get("_sdk_omit_raw")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let _ = omit_raw; // 此处标记，parse_success_body 中使用

        // ... 协议转换实现（消息、系统提示、工具、工具选择）...
        // extensions 浅层 merge 示例：
        // let mut body = serde_json::json!({ "model": model, "stream": stream, ... });
        // if let Some(obj) = body.as_object_mut() {
        //     for (k, v) in &req.extensions {
        //         obj.insert(k.clone(), v.clone());
        //     }
        // }
        todo!()
    }

    fn map_error_response(&self, status: u16, body_text: &str) -> LlmError {
        match status {
            401 | 403 => LlmError::Authentication {
                message: format!("[{}] {}", self.name(), body_text),
            },
            429 => LlmError::RateLimit { retry_after_secs: None },
            _ => LlmError::HttpStatus {
                provider: self.name().into(),
                status,
                body: body_text.to_owned(),
            },
        }
    }

    fn parse_success_body(
        &self, body_text: &str, omit_raw: bool,
    ) -> Result<ChatResponse, LlmError> {
        let raw: serde_json::Value = serde_json::from_str(body_text)
            .map_err(LlmError::JsonCodec)?;
        let raw_field = if omit_raw { None } else { Some(raw.clone()) };
        self.parse_response(raw, raw_field)
    }

    fn parse_response(
        &self, raw: serde_json::Value, raw_field: Option<serde_json::Value>,
    ) -> Result<ChatResponse, LlmError> {
        // ... 解析实现 ...
        todo!()
    }

    /// 解析单个 Anthropic SSE 事件帧，返回 `None` 表示该帧可跳过（如 ping）。
    fn parse_sse_event(&self, event_type: &str, data: &str) -> Result<Option<StreamEvent>, LlmError> {
        // ... 解析实现 ...
        todo!()
    }
}

// ── 内部 StreamEvent（私有，不进入公共类型） ───────────────────────────────

enum StreamEvent {
    TextDelta(String),
    ThinkingDelta(String),
    /// 仅在工具调用的 id/name/arguments 全部聚合完整后才发出。
    ToolCall(crate::types::content::ToolCall),
    Usage(crate::types::response::Usage),
    Done(crate::types::response::FinishReason),
}

// ── ProviderAdapter 实现 ───────────────────────────────────────────────────

#[async_trait]
impl ProviderAdapter for AnthropicAdapter {
    fn name(&self) -> &'static str { "anthropic" }

    async fn chat(&self, model: &str, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        let omit_raw = req.extensions.get("_sdk_omit_raw")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let body = self.to_request_body_async(model, &req, false).await?;

        let response = self.transport.send(
            self.transport.client()
                .post(format!("{}/v1/messages", self.base_url))
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", self.api_version)
                .json(&body),
        ).await?;

        // transport.send() 已处理重试；到达此处的响应 status 可能仍为 4xx/5xx
        // （非 retryable 的），需要在 adapter 层统一映射错误。
        let status = response.status().as_u16();
        let body_text = response.text().await.map_err(LlmError::Transport)?;

        if status >= 400 {
            return Err(self.map_error_response(status, &body_text));
        }

        self.parse_success_body(&body_text, omit_raw)
    }

    async fn chat_stream(
        &self,
        model: &str,
        req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatChunk, LlmError>> + Send>>, LlmError> {
        // 外层 Result 捕获"建立流之前"的错误（请求体构造、HTTP 连接建立失败）。
        // HTTP 连接建立后，SSE 数据接收过程中的错误通过 Stream::Item = Err(...) 传出，
        // 流遇到首个 Err 后立即终止，不再产出任何 Item。
        //
        // 实现步骤：
        // 1. to_request_body_async() → 构造失败时从外层 Err 返回
        // 2. transport.send() → 连接失败或非 2xx 时从外层 Err 返回
        // 3. 将 response.bytes_stream() 转为逐行 SSE 解析流
        // 4. 在流内部用 parse_sse_event() 解析，聚合 tool_call 分片
        // 5. 产出 ChatChunk；遇到解析错误立即 yield Some(Err(...)) 后终止
        todo!()
    }
}

// ── 模块门面函数 ───────────────────────────────────────────────────────────

pub(crate) fn build(
    api_key: &str,
    base_url: &str,
    transport: &Transport,
) -> Result<Box<dyn ProviderAdapter>, LlmError> {
    Ok(Box::new(AnthropicAdapter::new(api_key, base_url, transport.clone())))
}
```

**协议私有类型约束：**

- `StreamEvent`、pending tool call assembler、provider 原始 SSE/chunk 结构体等必须保持私有。
- 这些类型只服务于 adapter 内部解析与聚合，禁止进入 `src/types/` 模块。

### 6.4 协议映射参考表

adapter 实现时参考此表完成字段映射。

| 统一字段 | OpenAI / Compatible | Anthropic | Doubao | Qwen | Gemini |
|---|---|---|---|---|---|
| `messages[].role=system` | messages 中 `role: "system"` | 顶层 `system` 字段（字符串或 content block） | messages 中 `role: "system"` | messages 中 `role: "system"` | 顶层 `systemInstruction` |
| `ContentPart::Image` | `image_url` object | `image` source block | `image_url` object | `image_url` object | `inlineData` part |
| `ContentPart::ToolCall` | assistant message 的 `tool_calls[]` | `tool_use` content block | assistant message 的 `tool_calls[]` | assistant message 的 `tool_calls[]` | `functionCall` part |
| `ContentPart::ToolResult` | `role: "tool"` message + `tool_call_id` | `tool_result` content block | `role: "tool"` + `tool_call_id` + `tool_name`（必填） | `role: "tool"` + `tool_call_id` + `tool_name`（必填） | `functionResponse` part |
| `Tool.input_schema` | `tools[].function.parameters` | `tools[].input_schema` | `tools[].function.parameters` | `tools[].function.parameters` | `tools[].functionDeclarations[].parameters` |
| `ToolChoice::Auto` | `tool_choice: "auto"` | 默认（不传 `tool_choice`） | `tool_choice: "auto"` | `tool_choice: "auto"` | 默认 |
| `ToolChoice::None` | `tool_choice: "none"` | `tool_choice: {type:"none"}` | `tool_choice: "none"` | `tool_choice: "none"` | 省略 `tools` 字段 |
| `ToolChoice::Required` | `tool_choice: "required"` | `tool_choice: {type:"any"}` | `tool_choice: "required"` | `tool_choice: "required"` | `toolConfig.mode: "ANY"` |
| `ToolChoice::Specific(n)` | `tool_choice: {type:"function", function:{name:n}}` | `tool_choice: {type:"tool", name:n}` | 同 OpenAI | 同 OpenAI | `toolConfig.mode: "ANY"` + `allowedFunctionNames` filter |
| 流式模式 | `stream: true` | `stream: true` | `stream: true` | `stream: true` | `streamGenerateContent` endpoint |
| `usage` | 响应体 `usage` object | 响应体 `usage` object | 响应体 `usage` object | 响应体 `usage` object | 响应体 `usageMetadata` |

**Doubao / Qwen 的 `tool_name` 要求：** 这两个 provider 在 `tool` role 消息中要求携带 `tool_name`。adapter 在序列化 `ToolResult` 时必须补全此字段；若调用方通过 `ContentPart::tool_result()` 便捷方法构造了 `tool_name: None` 的消息，adapter 应从对应 `ToolCall` 中反查并填充，或在文档中提示调用方使用完整构造器。

**Doubao 与 Qwen 保留独立 adapter 的原因：** 两者 HTTP 接口与 OpenAI 高度兼容，但各自有差异化鉴权头、错误码映射、特有扩展字段，以及上述 `tool_name` 要求。若后续确认某一 provider 完全兼容，可改为在 `into_adapter()` 中路由到 `openai::build`，无需修改任何公共 API。

### 6.5 工具调用闭环规范

工具调用是一条完整状态机，生命周期如下：

```
1. 调用方通过 ChatRequest.tools 注册 Tool
        ↓
2. 模型返回 ContentPart::ToolCall（非流式）
   或 ChatChunk.tool_calls（流式，SDK 内部完成聚合）
        ↓
3. 上层应用根据 tool_name + arguments 执行本地函数
        ↓
4. 调用方构造 Message { role: Role::Tool, content: [ContentPart::ToolResult(...)] }
        ↓
5. 再次调用 Client::chat / chat_stream，直到模型返回普通 assistant 消息
```

**adapter 实现强制要求：**

- 必须保留 provider 生成的 `tool_call_id`，禁止自行重写（否则后续 `ToolResult` 无法关联）。
- 一个 assistant 消息可包含多个 `ToolCall`，必须全部保序输出，不得只取第一个。
- 流式场景下，`ToolCall` 的 `id`、`tool_name`、`arguments` 三者全部就绪且 JSON 完整可解析后，才写入 `ChatChunk.tool_calls`；参数增量分片、`arguments_delta` 等属于 adapter 内部状态，不进入公共接口。
- 流式场景必须支持同一响应中多个工具调用的并行分片归并（按 provider 分配的 index 区分）。
- provider 不支持工具时，若请求携带 `tools`，应在 adapter 层检测并返回 `LlmError::UnsupportedCapability { capability: "tools".into(), ... }`。
- `FinishReason::ToolCalls` 是正常流程结束，不是错误。
- SDK 只提供调用/回传闭环，不负责执行本地函数；工具执行由上层应用决定。

---

## 7. SDK 门面

**文件：`src/client.rs`**

`Client` 是唯一对外入口。`model` 由 `Client` 持有，不放入请求类型。

配置分层约束：

- `Provider`：只表达协议类型，不携带运行时连接参数。
- `ClientBuilder`：持有连接级配置，如 `api_key`、`base_url`、`timeout_secs`、`max_retries`、`rate_limit_rpm`、`model`。
- `ChatRequest` / `EmbeddingRequest` 等请求类型：只表达单次调用的语义参数，如 `messages`、`tools`、`temperature`、`extensions`。

```rust
use crate::{
    error::LlmError,
    middleware::{RateLimitConfig, RetryConfig, Transport, TransportConfig},
    provider::{Provider, ProviderAdapter},
    types::{request::*, response::*},
};
use futures::Stream;

// ── ClientBuilder ──────────────────────────────────────────────────────────

pub struct ClientBuilder {
    provider: Option<Provider>,
    base_url: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    timeout_secs: u64,
    max_retries: u32,
    rate_limit_rpm: Option<u32>,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self {
            provider: None,
            base_url: None,
            api_key: None,
            model: None,
            timeout_secs: 30,
            max_retries: 3,
            rate_limit_rpm: None,
        }
    }
}

impl ClientBuilder {
    pub fn provider(mut self, p: Provider) -> Self { self.provider = Some(p); self }
    pub fn base_url(mut self, url: impl Into<String>) -> Self { self.base_url = Some(url.into()); self }
    pub fn api_key(mut self, key: impl Into<String>) -> Self { self.api_key = Some(key.into()); self }
    pub fn model(mut self, m: impl Into<String>) -> Self { self.model = Some(m.into()); self }
    pub fn timeout_secs(mut self, secs: u64) -> Self { self.timeout_secs = secs; self }
    pub fn max_retries(mut self, n: u32) -> Self { self.max_retries = n; self }
    pub fn rate_limit_rpm(mut self, rpm: u32) -> Self { self.rate_limit_rpm = Some(rpm); self }

    /// 构建 `Client`。
    ///
    /// 默认值：`timeout_secs = 30`，`max_retries = 3`，`rate_limit_rpm = None`。
    ///
    /// # Errors
    ///
    /// - 缺少 `provider`、`api_key`、`model` 任一必填项：`LlmError::MissingConfig`
    /// - `provider = Compatible` 且未提供 `base_url`：`LlmError::MissingConfig { field: "base_url" }`
    pub fn build(self) -> Result<Client, LlmError> {
        let provider  = self.provider.ok_or(LlmError::MissingConfig { field: "provider" })?;
        let api_key   = self.api_key.ok_or(LlmError::MissingConfig  { field: "api_key" })?;
        let model     = self.model.ok_or(LlmError::MissingConfig    { field: "model" })?;
        let base_url  = match self.base_url {
            Some(url) => url,
            None => provider.default_base_url()
                .ok_or(LlmError::MissingConfig { field: "base_url" })?
                .to_owned(),
        };
        let transport_config = TransportConfig {
            timeout_secs: self.timeout_secs,
            retry: RetryConfig {
                max_retries: self.max_retries,
                ..RetryConfig::default()
            },
            rate_limit: self.rate_limit_rpm.map(|rpm| RateLimitConfig {
                requests_per_minute: rpm,
            }),
        };
        let transport = Transport::new(transport_config);
        let adapter = provider.into_adapter(&api_key, &base_url, &transport)?;

        Ok(Client {
            adapter,
            model,
            provider,
            base_url,
            timeout_secs: self.timeout_secs,
            max_retries: self.max_retries,
            rate_limit_rpm: self.rate_limit_rpm,
        })
    }
}

// ── Client ─────────────────────────────────────────────────────────────────

/// SDK 唯一对外入口。
///
/// `Client` 不实现 `Clone`（内部 adapter 为 trait object，不保证可 Clone）；
/// 跨多个异步任务共享同一 `Client` 时，请使用 `Arc<Client>`。
pub struct Client {
    adapter: Box<dyn ProviderAdapter>,
    model: String,
    provider: Provider,
    base_url: String,
    timeout_secs: u64,
    max_retries: u32,
    rate_limit_rpm: Option<u32>,
}

impl Client {
    pub fn builder() -> ClientBuilder { ClientBuilder::default() }

    /// 从环境变量快速构建。
    ///
    /// 必填：`LLM_PROVIDER`、`LLM_API_KEY`、`LLM_MODEL`
    /// 可选：`LLM_BASE_URL`（`compatible` 时必填；其他 provider 可用于覆盖默认地址）
    ///
    /// 支持的 `LLM_PROVIDER` 值：`openai`、`compatible`、`anthropic`、
    /// `doubao`、`qwen`、`gemini`。
    pub fn from_env() -> Result<Self, LlmError> {
        let raw_provider = std::env::var("LLM_PROVIDER")
            .map_err(|_| LlmError::MissingConfig { field: "LLM_PROVIDER" })?;
        let api_key = std::env::var("LLM_API_KEY")
            .map_err(|_| LlmError::MissingConfig { field: "LLM_API_KEY" })?;
        let model = std::env::var("LLM_MODEL")
            .map_err(|_| LlmError::MissingConfig { field: "LLM_MODEL" })?;

        let provider = Provider::from_name(&raw_provider).ok_or(LlmError::InvalidConfig {
            message: format!("不支持的 LLM_PROVIDER 值：{raw_provider}"),
        })?;

        let mut builder = Client::builder()
            .provider(provider)
            .api_key(api_key)
            .model(model);
        if let Ok(base_url) = std::env::var("LLM_BASE_URL") {
            builder = builder.base_url(base_url);
        }
        builder.build()
    }

    // ── Chat ───────────────────────────────────────────────────────────────

    pub async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.adapter.chat(&self.model, req).await
    }

    /// 返回流式 chat 响应。
    ///
    /// 外层 `Result` 表示"建立流之前"的错误（请求体构造失败、HTTP 连接失败）。
    /// 流建立后，接收过程中的错误通过 `Stream::Item = Err(...)` 传出；
    /// 流遇到首个 `Err` 后立即终止，调用方不应在错误后继续 poll。
    pub async fn chat_stream(
        &self,
        req: ChatRequest,
    ) -> Result<impl Stream<Item = Result<ChatChunk, LlmError>> + Send, LlmError> {
        self.adapter.chat_stream(&self.model, req).await
    }

    // ── Embedding ──────────────────────────────────────────────────────────

    pub async fn embed(&self, req: EmbeddingRequest) -> Result<EmbeddingResponse, LlmError> {
        self.adapter.embed(&self.model, req).await
    }

    // ── Speech ─────────────────────────────────────────────────────────────

    pub async fn speech_to_text(
        &self, req: SpeechToTextRequest,
    ) -> Result<SpeechToTextResponse, LlmError> {
        self.adapter.speech_to_text(&self.model, req).await
    }

    pub async fn text_to_speech(
        &self, req: TextToSpeechRequest,
    ) -> Result<TextToSpeechResponse, LlmError> {
        self.adapter.text_to_speech(&self.model, req).await
    }

    // ── Image / Video ───────────────────────────────────────────────────────

    pub async fn generate_image(
        &self, req: ImageGenRequest,
    ) -> Result<ImageGenResponse, LlmError> {
        self.adapter.generate_image(&self.model, req).await
    }

    pub async fn generate_video(
        &self, req: VideoGenRequest,
    ) -> Result<VideoGenResponse, LlmError> {
        self.adapter.generate_video(&self.model, req).await
    }

    /// 查询视频生成任务状态。
    ///
    /// 调用 `generate_video` 后，若 `VideoGenResponse.status` 不是 `Succeeded`，
    /// 应轮询此方法直至任务完成或失败。
    pub async fn poll_video_task(&self, task_id: &str) -> Result<VideoGenResponse, LlmError> {
        self.adapter.poll_video_task(task_id).await
    }

    // ── 元信息 ─────────────────────────────────────────────────────────────

    pub fn provider(&self) -> &Provider { &self.provider }
    pub fn model(&self) -> &str { &self.model }
    pub fn base_url(&self) -> &str { &self.base_url }
    pub fn timeout_secs(&self) -> u64 { self.timeout_secs }
    pub fn max_retries(&self) -> u32 { self.max_retries }
    pub fn rate_limit_rpm(&self) -> Option<u32> { self.rate_limit_rpm }
    // 出于安全考虑，不提供 api_key getter。
}
```

---

## 8. 中间件

**文件：`src/middleware.rs`**

本节术语统一如下：
- `TransportConfig`：传输配置载体，只在 `ClientBuilder → middleware` 交界处存在。
- `Transport`：传输层抽象，供 adapter 使用。
- "中间件"：`Transport` 内部消费的 retry / rate-limit / tracing 等横切策略。

**重试与响应 body 的关系：** `Transport::send()` 负责重试，但 HTTP 响应的 body 只能被消费一次。因此 `send()` 的重试判断**只依赖 HTTP status code**，不读取 body；4xx/5xx 的 body 解析由 adapter 在拿到 `Response` 后负责。可重试的 status（429、500、502、503）由 `RetryConfig::retryable_status` 定义。对于这些 status，`send()` 内部重试，不将中间响应暴露给 adapter；超出最大重试次数后，将最后一次响应的 status 和 body 文本包装为 `LlmError::HttpStatus` 返回（需消费 body）。非 retryable 的 4xx/5xx（如 400、401、403）由 `send()` 直接返回 `Ok(Response)`，由 adapter 读取 body 并映射错误。

```rust
// pub(crate)，外部不可见

#[derive(Debug, Clone)]
pub(crate) struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    /// 触发重试的 HTTP status code 列表，默认 [429, 500, 502, 503]
    pub retryable_status: Vec<u16>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 500,
            retryable_status: vec![429, 500, 502, 503],
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RateLimitConfig {
    pub requests_per_minute: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct TransportConfig {
    pub timeout_secs: u64,
    pub retry: RetryConfig,
    pub rate_limit: Option<RateLimitConfig>,
}

/// 传输层抽象：adapter 只感知 `Transport`，不直接感知 `TransportConfig`。
#[derive(Clone)]
pub(crate) struct Transport {
    http: reqwest::Client,
    retry: RetryConfig,
    rate_limit: Option<RateLimitConfig>,
}

impl Transport {
    pub(crate) fn new(config: TransportConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .build()
            .expect("reqwest client 构造不应失败");
        Self { http, retry: config.retry, rate_limit: config.rate_limit }
    }

    pub(crate) fn client(&self) -> &reqwest::Client { &self.http }

    /// 统一发送入口，内部消费 retry / rate-limit / tracing 策略。
    ///
    /// **重试策略：**
    /// - 仅对 `retryable_status` 中的 HTTP status code 重试（body 不读取）。
    /// - Transport 层错误（超时、连接中断）也会重试。
    /// - 超出最大重试次数后，retryable 错误以 `LlmError::HttpStatus` 返回
    ///   （最后一次响应的 body 已被读取）。
    /// - 非 retryable 的 4xx/5xx 以 `Ok(Response)` 返回，由 adapter 处理。
    ///
    /// **执行顺序：** rate-limit → retry/backoff → tracing → send
    pub(crate) async fn send(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response, LlmError> {
        use std::time::Duration;

        if let Some(rl) = &self.rate_limit {
            // TODO(maintainer): 接入 token bucket 或 leaky bucket 实现
            let _rpm = rl.requests_per_minute;
        }

        let mut attempt: u32 = 0;
        loop {
            let req = request.try_clone().ok_or_else(|| LlmError::InvalidConfig {
                message: "请求体不可克隆（含流式 body），无法安全重试".into(),
            })?;

            let result = req.send().await;

            match result {
                Err(err) if err.is_timeout() => {
                    if attempt >= self.retry.max_retries {
                        return Err(LlmError::RequestTimeout {
                            timeout_ms: 0, // 实际值由 reqwest 超时配置决定
                        });
                    }
                }
                Err(err) if err.is_connect() && attempt < self.retry.max_retries => {
                    // 连接层错误可重试
                }
                Err(err) => return Err(LlmError::Transport(err)),

                Ok(resp) => {
                    let status = resp.status().as_u16();
                    if self.retry.retryable_status.contains(&status)
                        && attempt < self.retry.max_retries
                    {
                        // retryable status：不读取 body，直接进入退避重试
                        tracing::warn!(
                            attempt, status,
                            "retryable status，进入退避重试"
                        );
                    } else {
                        // 非 retryable（包括成功的 2xx 和非 retryable 的 4xx）
                        // 直接返回，由 adapter 负责读取 body 和映射错误
                        return Ok(resp);
                    }
                }
            }

            let backoff_ms = self.retry.initial_backoff_ms
                .saturating_mul(1_u64 << attempt.min(10));
            tracing::debug!(attempt, backoff_ms, "退避等待");
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            attempt += 1;
        }
    }
}
```

**错误处理策略（中间件层）：**

| 错误类型 | 处理方式 |
|---|---|
| `MissingConfig` / `InvalidConfig` / `UnsupportedCapability` | 构建期/路由期直接返回，禁止重试 |
| `Authentication` | 直接返回，不重试 |
| `RateLimit` / `RequestTimeout` / retryable `HttpStatus` | 指数退避重试 |
| `JsonCodec` / `StreamProtocol` / `ToolProtocol` / `MediaInput` | 直接返回，不重试 |
| 流式场景错误 | 立即终止 stream，先 yield `Some(Err(...))` 再终止，不静默吞错 |

---

## 9. 扩展指南：新增 Provider

以新增 `Mistral` 为例。

**需要修改的位置（共 9 处）：**

1. 新建 `src/provider/mistral.rs`，实现：
   - `pub(crate) struct MistralAdapter`
   - 协议转换私有函数：`to_request_body_async`、`parse_response`、`parse_sse_event`
   - `#[async_trait] impl ProviderAdapter for MistralAdapter`（至少覆盖 `chat` 和 `chat_stream`）
   - 统一门面函数（独立 provider 签名）：
     ```rust
     pub(crate) fn build(
         api_key: &str, base_url: &str, transport: &Transport,
     ) -> Result<Box<dyn ProviderAdapter>, LlmError> {
         Ok(Box::new(MistralAdapter::new(api_key, base_url, transport.clone())))
     }
     ```

2. 在 `src/provider/mod.rs` 顶部添加 `mod mistral;`。

3. 在 `Provider` 枚举中添加变体 `Mistral`。

4. 在 `Provider::name()` 中添加 `Provider::Mistral => "mistral"`。

5. 在 `Provider::from_name()` 中添加 `"mistral" => Some(Provider::Mistral)`。

6. 在 `Provider::default_base_url()` 中添加对应 URL。

7. 在 `Provider::into_adapter()` 中添加 `Provider::Mistral => mistral::build(api_key, base_url, transport)`。

8. `Client::from_env()` 无需新增 `match` 分支；它统一调用 `Provider::from_name()` 完成解析。

9. 添加 `examples/mistral_chat.rs` 和对应的 wiremock 集成测试。

**禁止：** 在 `src/types/`、`src/error.rs`、`Client` 的公开方法签名中写入任何 Mistral 协议细节。

---

## 10. 测试规范

| 测试层 | 位置 | 工具 | 说明 |
|---|---|---|---|
| 类型单元测试 | `src/types/` 各文件内 `#[cfg(test)]` | 标准 `#[test]` | 覆盖 builder 逻辑、`system()` 幂等行为、类型转换、`Message::text()`、`into_message()` |
| Adapter 集成测试 | `src/provider/xxx.rs` 内 `#[cfg(test)]` | `wiremock` | 不依赖真实 API Key；覆盖普通 chat、流式 chat（含工具调用聚合）、错误映射（401/429/500）、`MediaSource::File` 读取、extensions merge |
| 端到端测试 | `tests/` | 标准 `#[tokio::test]` | 通过 `INTEGRATION_TEST=1` 环境变量控制是否运行，依赖真实 API Key |

---

## 11. 注释与编码规范

### 注释原则

只注释**为什么**，不注释**是什么**。一个熟悉 Rust 但不了解本项目的人能在 10 秒内看懂的代码，不要加注释。

**必须写注释的场景：**

- `pub` 类型、函数、trait、枚举变体：使用 `///` 文档注释
- 所有 `unsafe` 块：必须有 `// SAFETY:` 说明
- 协议约束、魔法数字、兼容性分支、性能取舍

**不要写注释的场景：**

- 变量赋值、简单分支、字面上能看懂的返回值
- 与代码逐字复述的注释

### 文档注释 `///`

- 第一行一句话概括，简洁。
- `# Errors`：仅当错误触发条件无法从签名直接推断时才写。
- `# Panics`：存在非预期 panic 路径时写（优先重构为 `Result`）。
- `# Safety`：`unsafe fn` 强制要求。
- 示例代码应可通过 `cargo test --doc`。

### 行内注释 `//`

- 解释原因，不翻译代码。
- 优先写在语句上方；同行注释前留两个空格。
- 统一使用中文。

### 标记注释

```
TODO(name):   待实现功能，说明背景或优先级
FIXME(name):  已知问题，说明限制和临时规避
HACK:         临时方案，说明为何如此以及未来替换方向
SAFETY:       unsafe 块强制使用
```

### Clippy

CI 强制通过 `cargo clippy -- -D warnings`。

---

## 12. 完整使用示例

### 基础对话

```rust
// examples/chat_basic.rs
use ufox_llm::{Client, Provider, ChatRequest};

#[tokio::main]
async fn main() -> Result<(), ufox_llm::LlmError> {
    let client = Client::builder()
        .provider(Provider::OpenAI)
        .api_key("sk-xxx")
        .model("gpt-4o")
        .build()?;

    let output = client.chat(
        ChatRequest::builder()
            .user_text("解释一下 Rust 的所有权模型")
            .max_tokens(1024)
            .build()
    ).await?;

    println!("{}", output.text);
    Ok(())
}
```

### OpenAI 兼容协议（DeepSeek）

```rust
// examples/compatible_provider.rs
let client = Client::builder()
    .provider(Provider::Compatible)
    .base_url("https://api.deepseek.com/v1")
    .api_key("sk-xxx")
    .model("deepseek-chat")
    .build()?;

assert_eq!(client.base_url(), "https://api.deepseek.com/v1");
```

### 流式输出

```rust
// examples/streaming.rs
use futures::StreamExt;

let mut stream = client.chat_stream(
    ChatRequest::builder()
        .user_text("写一首关于 Rust 的诗")
        .build()
).await?;

while let Some(chunk) = stream.next().await {
    let chunk = chunk?;  // 流中的 Err 会在此处冒泡并终止循环
    if let Some(t) = &chunk.text_delta {
        print!("{t}");
    }
    if let Some(t) = &chunk.thinking_delta {
        eprint!("[thinking]{t}");
    }
    if chunk.is_finished() { break; }
}
```

### 多模态图片理解

```rust
// examples/multimodal_image.rs
use ufox_llm::{ContentPart, ChatRequest};

let req = ChatRequest::builder()
    .user(vec![
        ContentPart::image_url("https://example.com/chart.png"),
        ContentPart::text("这张图表说明了什么趋势？"),
    ])
    .max_tokens(512)
    .build();

let output = client.chat(req).await?;
```

### 工具调用闭环

```rust
// examples/tool_calling.rs
use ufox_llm::{
    Tool, ToolChoice, ToolResult, ToolResultPayload,
    Message, Role, ContentPart, ChatRequest,
};

let weather_tool = Tool::function(
    "get_weather",
    "查询指定城市的实时天气",
    serde_json::json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"]
    }),
);

let mut messages = vec![
    Message {
        role: Role::User,
        content: vec![ContentPart::text("帮我查询杭州天气，并给出穿衣建议")],
        name: None,
    }
];

loop {
    let output = client.chat(
        ChatRequest::builder()
            .messages(messages.clone())
            .tools(vec![weather_tool.clone()])
            .tool_choice(ToolChoice::Auto)
            .build()
    ).await?;

    let tool_calls = output.tool_calls.clone();
    messages.push(output.into_message());

    if tool_calls.is_empty() {
        if let Some(last) = messages.last() {
            println!("{}", last.text());
        }
        break;
    }

    for call in &tool_calls {
        // run_local_tool 属于业务层，不属于 SDK
        let result = run_local_tool(&call.tool_name, &call.arguments)?;
        messages.push(Message {
            role: Role::Tool,
            content: vec![ContentPart::ToolResult(ToolResult {
                tool_call_id: call.id.clone(),
                tool_name: Some(call.tool_name.clone()),  // Doubao/Qwen 需要此字段
                payload: ToolResultPayload::json(result),
                is_error: false,
            })],
            name: None,
        });
    }
}
```

### 语音转文本

```rust
// examples/speech_to_text.rs
use ufox_llm::{Client, Provider, MediaSource, AudioFormat, SpeechToTextRequest};

let client = Client::builder()
    .provider(Provider::OpenAI)
    .api_key("sk-xxx")
    .model("gpt-4o-mini-transcribe")
    .build()?;

let output = client.speech_to_text(SpeechToTextRequest {
    source: MediaSource::File { path: "sample.wav".into() },
    format: AudioFormat::Wav,
    language: Some("zh".into()),
    extensions: Default::default(),
}).await?;

println!("{}", output.text);
```

### 文本转语音

```rust
// examples/text_to_speech.rs
use ufox_llm::{Client, Provider, AudioFormat, TextToSpeechRequest};

let client = Client::builder()
    .provider(Provider::OpenAI)
    .api_key("sk-xxx")
    .model("gpt-4o-mini-tts")
    .build()?;

let output = client.text_to_speech(TextToSpeechRequest {
    text: "你好，欢迎使用 ufox-llm。".into(),
    voice: Some("alloy".into()),
    output_format: AudioFormat::Mp3,
    extensions: Default::default(),
}).await?;

std::fs::write("speech.mp3", output.audio_data)?;
```

### 视频生成与状态轮询

```rust
// examples/video_gen.rs
use ufox_llm::{Client, Provider, VideoGenRequest, TaskStatus};

let client = Client::builder()
    .provider(Provider::Doubao)   // 以支持视频生成的 provider 为例
    .api_key("xxx")
    .model("video-gen-v1")
    .build()?;

let resp = client.generate_video(VideoGenRequest {
    prompt: "夕阳下奔跑的马群".into(),
    duration_secs: Some(5),
    output_format: None,
    extensions: Default::default(),
}).await?;

// 轮询直至完成
let final_resp = loop {
    let status_resp = client.poll_video_task(&resp.task_id).await?;
    match status_resp.status {
        TaskStatus::Succeeded | TaskStatus::Failed => break status_resp,
        _ => tokio::time::sleep(std::time::Duration::from_secs(3)).await,
    }
};

if let Some(url) = final_resp.url {
    println!("视频地址：{url}");
}
```

### 从环境变量构建

```rust
// 必填：LLM_PROVIDER, LLM_API_KEY, LLM_MODEL
// compatible 时必须提供：LLM_BASE_URL
// 其他 provider 可用 LLM_BASE_URL 覆盖默认地址
let client = Client::from_env()?;
```

### 跨任务共享 Client

```rust
use std::sync::Arc;

let client = Arc::new(Client::builder()
    .provider(Provider::OpenAI)
    .api_key("sk-xxx")
    .model("gpt-4o")
    .build()?);

let c1 = Arc::clone(&client);
let c2 = Arc::clone(&client);

tokio::join!(
    async move { c1.chat(ChatRequest::builder().user_text("任务 1").build()).await },
    async move { c2.chat(ChatRequest::builder().user_text("任务 2").build()).await },
);
```

---

*版本：v3.1 | Rust 2024 Edition | tokio 1.x | reqwest 0.12.x*
