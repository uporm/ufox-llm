//! 消息模型模块。
//!
//! 该模块负责定义对话消息的统一内部表示，包括角色、文本内容与多模态内容片段。
//! 它是 `client` 发起请求与 `provider` 序列化请求体时共享的核心基础类型。
//!
//! 设计上采用“`Provider` 无关的中间层模型”：
//! 1. `Message` 只表达语义，不直接绑定 `OpenAI` 或 `Qwen` 的私有字段；
//! 2. 多模态内容使用顺序化片段表示，确保调用方追加内容时不会丢失先后关系；
//! 3. 本地图片文件仅保存路径与推断出的 `MIME` 类型，延迟到 `Provider` 层再执行文件读取，
//!    这样可以避免在构建消息时引入阻塞 `I/O`，并保持 `build()` 为纯内存操作。
//!
//! 该模块依赖 `serde` 进行序列化，依赖 `mime_guess` 推断本地图片文件的 `MIME` 类型。

use std::path::{Path, PathBuf};

use mime_guess::MimeGuess;
use serde::{Deserialize, Serialize};

use super::tool::ToolCall;

/// 对话消息的角色。
///
/// 该枚举对应主流 `Chat API` 中的消息来源语义，不直接耦合任一 `Provider` 的实现细节。
///
/// # 示例
/// ```rust
/// use ufox_llm::Role;
///
/// assert_eq!(Role::User.as_str(), "user");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// 系统消息，用于设定模型行为。
    System,
    /// 用户消息，用于表达提问或输入。
    User,
    /// 助手消息，用于承载模型回复。
    Assistant,
    /// 工具消息，用于回传工具执行结果。
    Tool,
}

impl Role {
    /// 返回角色的稳定字符串表示。
    ///
    /// # Returns
    /// 与主流 `Chat API` 兼容的小写角色名称。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Role;
    ///
    /// assert_eq!(Role::Assistant.as_str(), "assistant");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// 消息内容。
///
/// 单纯文本消息使用 [`Content::Text`]，多模态消息使用 [`Content::Parts`] 以保留内容片段顺序。
///
/// # 示例
/// ```rust
/// use ufox_llm::Content;
///
/// let content = Content::text("你好");
/// assert_eq!(content.as_text(), Some("你好"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// 纯文本内容。
    Text(String),
    /// 顺序化的多模态内容片段。
    Parts(Vec<ContentPart>),
}

impl Content {
    /// 创建纯文本内容。
    ///
    /// # Arguments
    /// * `text` - 文本内容
    ///
    /// # Returns
    /// 包含纯文本的内容对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Content;
    ///
    /// let content = Content::text("请总结这段日志");
    /// assert_eq!(content.as_text(), Some("请总结这段日志"));
    /// ```
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// 创建多模态内容。
    ///
    /// # Arguments
    /// * `parts` - 按出现顺序排列的内容片段
    ///
    /// # Returns
    /// 包含多个内容片段的内容对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{Content, ContentPart};
    ///
    /// let content = Content::parts(vec![ContentPart::text("请描述图片")]);
    /// assert!(content.is_multimodal());
    /// ```
    #[must_use]
    pub fn parts(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }

    /// 尝试以纯文本形式读取内容。
    ///
    /// 当内容为多模态片段时返回 `None`，因为这类内容无法无损降级为单一字符串。
    ///
    /// # Returns
    /// 纯文本内容的只读视图；若为多模态内容则返回 `None`。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Content;
    ///
    /// let content = Content::text("你好");
    /// assert_eq!(content.as_text(), Some("你好"));
    /// ```
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text.as_str()),
            Self::Parts(_) => None,
        }
    }

    /// 返回内容是否为多模态片段集合。
    ///
    /// # Returns
    /// 如果内容为 [`Content::Parts`]，则返回 `true`。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{Content, ContentPart};
    ///
    /// let content = Content::parts(vec![ContentPart::text("请描述这张图")]);
    /// assert!(content.is_multimodal());
    /// ```
    #[must_use]
    pub const fn is_multimodal(&self) -> bool {
        matches!(self, Self::Parts(_))
    }
}

/// 多模态消息中的单个内容片段。
///
/// 片段采用顺序列表而不是映射结构，以确保序列化时可以严格保留调用方输入顺序。
///
/// # 示例
/// ```rust
/// use ufox_llm::ContentPart;
///
/// let part = ContentPart::text("先分析文本，再分析图片");
/// assert!(matches!(part, ContentPart::Text { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContentPart {
    /// 文本片段。
    Text {
        /// 文本内容。
        text: String,
    },
    /// 图片片段。
    Image {
        /// 图片来源描述。
        source: ImageSource,
    },
}

impl ContentPart {
    /// 创建文本片段。
    ///
    /// # Arguments
    /// * `text` - 文本内容
    ///
    /// # Returns
    /// 文本内容片段。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ContentPart;
    ///
    /// let part = ContentPart::text("请结合图片回答");
    /// assert!(matches!(part, ContentPart::Text { .. }));
    /// ```
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// 创建图片 URL 片段。
    ///
    /// # Arguments
    /// * `url` - 图片 URL
    ///
    /// # Returns
    /// 以远程 URL 表示的图片片段。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ContentPart;
    ///
    /// let part = ContentPart::image_url("https://example.com/photo.jpg");
    /// assert!(matches!(part, ContentPart::Image { .. }));
    /// ```
    #[must_use]
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: ImageSource::url(url),
        }
    }

    /// 创建本地图片文件片段。
    ///
    /// 该方法只记录路径与推断出的 `MIME` 类型，不会立即读取文件内容。
    ///
    /// # Arguments
    /// * `path` - 本地图片路径
    ///
    /// # Returns
    /// 以本地文件表示的图片片段。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ContentPart;
    ///
    /// let part = ContentPart::image_file("./assets/logo.png");
    /// assert!(matches!(part, ContentPart::Image { .. }));
    /// ```
    #[must_use]
    pub fn image_file(path: impl Into<PathBuf>) -> Self {
        Self::Image {
            source: ImageSource::file(path),
        }
    }
}

/// 图片来源。
///
/// 统一抽象远程 `URL` 与本地文件两类来源，便于 `Provider` 层按各自协议进行转换。
///
/// # 示例
/// ```rust
/// use ufox_llm::ImageSource;
///
/// let source = ImageSource::url("https://example.com/photo.jpg");
/// assert!(matches!(source, ImageSource::Url { .. }));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// 远程图片 URL。
    Url {
        /// 图片 URL。
        url: String,
    },
    /// 本地图片文件。
    File(ImageFile),
}

impl ImageSource {
    /// 创建远程图片 URL 来源。
    ///
    /// # Arguments
    /// * `url` - 图片 URL
    ///
    /// # Returns
    /// 远程 URL 图片来源。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ImageSource;
    ///
    /// let source = ImageSource::url("https://example.com/photo.jpg");
    /// assert!(matches!(source, ImageSource::Url { .. }));
    /// ```
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url { url: url.into() }
    }

    /// 创建本地图片文件来源。
    ///
    /// 该方法会基于文件扩展名推断 `MIME` 类型，但不会立即检查文件是否存在。
    ///
    /// # Arguments
    /// * `path` - 本地图片路径
    ///
    /// # Returns
    /// 本地文件图片来源。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ImageSource;
    ///
    /// let source = ImageSource::file("./assets/photo.png");
    /// assert!(matches!(source, ImageSource::File(_)));
    /// ```
    #[must_use]
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::File(ImageFile::new(path))
    }
}

/// 本地图片文件描述。
///
/// 该结构体仅保存路径与 `MIME` 类型元数据，不在构造阶段执行 `I/O`。
///
/// # 示例
/// ```rust
/// use ufox_llm::ImageFile;
///
/// let file = ImageFile::new("./assets/photo.png");
/// assert_eq!(file.path().to_string_lossy(), "./assets/photo.png");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageFile {
    path: PathBuf,
    mime_type: Option<String>,
}

impl ImageFile {
    /// 创建本地图片文件描述。
    ///
    /// # Arguments
    /// * `path` - 本地图片路径
    ///
    /// # Returns
    /// 包含路径与推断 `MIME` 类型的文件描述对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ImageFile;
    ///
    /// let file = ImageFile::new("./assets/avatar.jpg");
    /// assert!(file.mime_type().is_some());
    /// ```
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let mime_type = guess_mime_type(&path);

        Self { path, mime_type }
    }

    /// 返回本地图片路径。
    ///
    /// # Returns
    /// 图片文件路径的只读引用。
    ///
    /// # 示例
    /// ```rust
    /// use std::path::Path;
    ///
    /// use ufox_llm::ImageFile;
    ///
    /// let file = ImageFile::new("./assets/avatar.jpg");
    /// assert_eq!(file.path(), Path::new("./assets/avatar.jpg"));
    /// ```
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 返回推断得到的 `MIME` 类型。
    ///
    /// # Returns
    /// 若能从扩展名推断出类型，则返回 `MIME` 字符串。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ImageFile;
    ///
    /// let file = ImageFile::new("./assets/avatar.jpg");
    /// assert_eq!(file.mime_type(), Some("image/jpeg"));
    /// ```
    #[must_use]
    pub fn mime_type(&self) -> Option<&str> {
        self.mime_type.as_deref()
    }
}

/// 对话消息。
///
/// 该结构体是 `SDK` 内部与对外共享的统一消息模型，后续会被各 `Provider` 转换为各自请求格式。
/// 除基础角色和内容外，它还可以承载工具调用元数据，用于表达：
/// 1. 助手返回的工具调用请求；
/// 2. 工具执行结果回填时关联的 `tool_call_id`。
///
/// # 示例
/// ```rust
/// use ufox_llm::Message;
///
/// let message = Message::user("请解释这段 Rust 代码");
/// assert_eq!(message.role().as_str(), "user");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    role: Role,
    content: Content,
    name: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
    tool_call_id: Option<String>,
}

impl Message {
    /// 创建一条消息。
    ///
    /// # Arguments
    /// * `role` - 消息角色
    /// * `content` - 消息内容
    ///
    /// # Returns
    /// 新的消息对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{Content, Message, Role};
    ///
    /// let message = Message::new(Role::System, Content::text("你是审查助手"));
    /// assert_eq!(message.role(), Role::System);
    /// ```
    #[must_use]
    pub fn new(role: Role, content: Content) -> Self {
        Self {
            role,
            content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// 创建系统文本消息。
    ///
    /// # Arguments
    /// * `text` - 系统提示词
    ///
    /// # Returns
    /// 角色为系统的文本消息。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Message;
    ///
    /// let message = Message::system("你是专业的代码审查助手");
    /// assert_eq!(message.role().as_str(), "system");
    /// ```
    #[must_use]
    pub fn system(text: impl Into<String>) -> Self {
        Self::new(Role::System, Content::text(text))
    }

    /// 创建用户文本消息。
    ///
    /// # Arguments
    /// * `text` - 用户输入文本
    ///
    /// # Returns
    /// 角色为用户的文本消息。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Message;
    ///
    /// let message = Message::user("请审查这段代码");
    /// assert_eq!(message.role().as_str(), "user");
    /// ```
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::User, Content::text(text))
    }

    /// 创建助手文本消息。
    ///
    /// # Arguments
    /// * `text` - 助手回复文本
    ///
    /// # Returns
    /// 角色为助手的文本消息。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Message;
    ///
    /// let message = Message::assistant("这段代码存在资源泄漏风险");
    /// assert_eq!(message.role().as_str(), "assistant");
    /// ```
    #[must_use]
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new(Role::Assistant, Content::text(text))
    }

    /// 创建携带工具调用列表的助手消息。
    ///
    /// 该构造器用于“模型请求工具调用”这一中间状态。默认文本内容为空，这是因为大多数
    /// Provider 在这类消息中只关心 `tool_calls` 元数据；真正需要文本说明时，调用方
    /// 仍可先使用 [`Message::assistant`] 自行构造普通回复消息。
    ///
    /// # Arguments
    /// * `tool_calls` - 模型返回的工具调用列表
    ///
    /// # Returns
    /// 角色为助手、携带工具调用元数据的消息对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{Message, ToolCall};
    ///
    /// let calls = vec![ToolCall::new("call_1", "get_weather", r#"{"city":"杭州"}"#)];
    /// let message = Message::assistant_with_tool_calls(&calls);
    ///
    /// assert_eq!(message.role().as_str(), "assistant");
    /// assert_eq!(message.tool_calls().expect("应包含工具调用").len(), 1);
    /// ```
    #[must_use]
    pub fn assistant_with_tool_calls(tool_calls: &[ToolCall]) -> Self {
        Self {
            role: Role::Assistant,
            content: Content::text(""),
            name: None,
            tool_calls: Some(tool_calls.to_vec()),
            tool_call_id: None,
        }
    }

    /// 创建工具结果消息。
    ///
    /// 该构造器用于把本地工具执行结果回填给模型。`tool_call_id` 会与之前助手消息中的
    /// 工具调用请求关联起来，确保模型可以把结果对应到正确的工具调用。
    ///
    /// # Arguments
    /// * `tool_call_id` - 对应的工具调用唯一标识
    /// * `content` - 工具执行结果文本
    ///
    /// # Returns
    /// 角色为工具的结果消息。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Message;
    ///
    /// let message = Message::tool_result("call_1", r#"{"temp":26}"#);
    /// assert_eq!(message.role().as_str(), "tool");
    /// assert_eq!(message.tool_call_id(), Some("call_1"));
    /// ```
    #[must_use]
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: Content::text(content),
            name: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }

    /// 为指定角色创建消息构建器。
    ///
    /// 由于 Rust 不支持同名关联函数的参数重载，多模态构建入口单独提供为构建器。
    ///
    /// # Arguments
    /// * `role` - 消息角色
    ///
    /// # Returns
    /// 可链式追加内容片段的消息构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{Message, Role};
    ///
    /// let message = Message::builder(Role::User)
    ///     .text("描述这张图片")
    ///     .image_url("https://example.com/photo.jpg")
    ///     .build();
    ///
    /// assert!(message.content().is_multimodal());
    /// ```
    #[must_use]
    pub fn builder(role: Role) -> MessageBuilder {
        MessageBuilder::new(role)
    }

    /// 返回消息角色。
    ///
    /// # Returns
    /// 当前消息的角色。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{Message, Role};
    ///
    /// let message = Message::assistant("你好");
    /// assert_eq!(message.role(), Role::Assistant);
    /// ```
    #[must_use]
    pub const fn role(&self) -> Role {
        self.role
    }

    /// 返回消息内容。
    ///
    /// # Returns
    /// 当前消息内容的只读引用。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Message;
    ///
    /// let message = Message::user("你好");
    /// assert_eq!(message.content().as_text(), Some("你好"));
    /// ```
    #[must_use]
    pub const fn content(&self) -> &Content {
        &self.content
    }

    /// 返回可选的消息名称。
    ///
    /// # Returns
    /// 若调用方显式设置了名称，则返回该名称。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Message;
    ///
    /// let message = Message::assistant("你好").with_name("reviewer");
    /// assert_eq!(message.name(), Some("reviewer"));
    /// ```
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// 返回可选的工具调用列表。
    ///
    /// # Returns
    /// 若该消息用于表达助手请求工具调用，则返回工具调用切片。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{Message, ToolCall};
    ///
    /// let calls = vec![ToolCall::new("call_1", "get_weather", "{}")];
    /// let message = Message::assistant_with_tool_calls(&calls);
    ///
    /// assert_eq!(message.tool_calls().expect("应包含工具调用")[0].name(), "get_weather");
    /// ```
    #[must_use]
    pub fn tool_calls(&self) -> Option<&[ToolCall]> {
        self.tool_calls.as_deref()
    }

    /// 返回可选的工具调用关联标识。
    ///
    /// # Returns
    /// 若该消息是工具结果消息，则返回对应的 `tool_call_id`。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Message;
    ///
    /// let message = Message::tool_result("call_1", "ok");
    /// assert_eq!(message.tool_call_id(), Some("call_1"));
    /// ```
    #[must_use]
    pub fn tool_call_id(&self) -> Option<&str> {
        self.tool_call_id.as_deref()
    }

    /// 设置消息名称。
    ///
    /// # Arguments
    /// * `name` - 消息名称
    ///
    /// # Returns
    /// 设置名称后的消息对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Message;
    ///
    /// let message = Message::system("你是测试助手").with_name("system-profile");
    /// assert_eq!(message.name(), Some("system-profile"));
    /// ```
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// 消息构建器。
///
/// 该构建器主要用于多模态场景，支持按顺序追加文本和图片片段。
///
/// # 示例
/// ```rust
/// use ufox_llm::MessageBuilder;
///
/// let message = MessageBuilder::user()
///     .text("请描述这张图片")
///     .image_url("https://example.com/photo.jpg")
///     .build();
///
/// assert!(message.content().is_multimodal());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageBuilder {
    role: Role,
    parts: Vec<ContentPart>,
    name: Option<String>,
}

impl MessageBuilder {
    /// 为指定角色创建消息构建器。
    ///
    /// # Arguments
    /// * `role` - 消息角色
    ///
    /// # Returns
    /// 空的消息构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{MessageBuilder, Role};
    ///
    /// let builder = MessageBuilder::new(Role::User);
    /// assert_eq!(builder.role(), Role::User);
    /// ```
    #[must_use]
    pub fn new(role: Role) -> Self {
        Self {
            role,
            parts: Vec::new(),
            name: None,
        }
    }

    /// 创建用户消息构建器。
    ///
    /// # Returns
    /// 角色为用户的消息构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::MessageBuilder;
    ///
    /// let builder = MessageBuilder::user();
    /// assert_eq!(builder.role().as_str(), "user");
    /// ```
    #[must_use]
    pub fn user() -> Self {
        Self::new(Role::User)
    }

    /// 创建系统消息构建器。
    ///
    /// # Returns
    /// 角色为系统的消息构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::MessageBuilder;
    ///
    /// let builder = MessageBuilder::system();
    /// assert_eq!(builder.role().as_str(), "system");
    /// ```
    #[must_use]
    pub fn system() -> Self {
        Self::new(Role::System)
    }

    /// 创建助手消息构建器。
    ///
    /// # Returns
    /// 角色为助手的消息构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::MessageBuilder;
    ///
    /// let builder = MessageBuilder::assistant();
    /// assert_eq!(builder.role().as_str(), "assistant");
    /// ```
    #[must_use]
    pub fn assistant() -> Self {
        Self::new(Role::Assistant)
    }

    /// 返回构建器绑定的角色。
    ///
    /// # Returns
    /// 当前构建器的消息角色。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{MessageBuilder, Role};
    ///
    /// let builder = MessageBuilder::new(Role::Tool);
    /// assert_eq!(builder.role(), Role::Tool);
    /// ```
    #[must_use]
    pub const fn role(&self) -> Role {
        self.role
    }

    /// 追加文本片段。
    ///
    /// # Arguments
    /// * `text` - 要追加的文本内容
    ///
    /// # Returns
    /// 追加文本后的构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::MessageBuilder;
    ///
    /// let message = MessageBuilder::user().text("先看文字").build();
    /// assert_eq!(message.content().as_text(), Some("先看文字"));
    /// ```
    #[must_use]
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.parts.push(ContentPart::text(text));
        self
    }

    /// 追加远程图片 URL 片段。
    ///
    /// # Arguments
    /// * `url` - 远程图片 URL
    ///
    /// # Returns
    /// 追加图片后的构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::MessageBuilder;
    ///
    /// let message = MessageBuilder::user()
    ///     .image_url("https://example.com/photo.jpg")
    ///     .build();
    ///
    /// assert!(message.content().is_multimodal());
    /// ```
    #[must_use]
    pub fn image_url(mut self, url: impl Into<String>) -> Self {
        self.parts.push(ContentPart::image_url(url));
        self
    }

    /// 追加本地图片文件片段。
    ///
    /// # Arguments
    /// * `path` - 本地图片文件路径
    ///
    /// # Returns
    /// 追加图片后的构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::MessageBuilder;
    ///
    /// let message = MessageBuilder::user().image_file("./assets/photo.png").build();
    /// assert!(message.content().is_multimodal());
    /// ```
    #[must_use]
    pub fn image_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.parts.push(ContentPart::image_file(path));
        self
    }

    /// 设置消息名称。
    ///
    /// # Arguments
    /// * `name` - 消息名称
    ///
    /// # Returns
    /// 设置名称后的构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::MessageBuilder;
    ///
    /// let message = MessageBuilder::assistant()
    ///     .text("你好")
    ///     .name("reviewer")
    ///     .build();
    ///
    /// assert_eq!(message.name(), Some("reviewer"));
    /// ```
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// 构建消息对象。
    ///
    /// 当构建器中仅包含一个文本片段时，SDK 会将其折叠为 [`Content::Text`]。
    /// 这样做的原因是纯文本消息是最常见场景，直接使用字符串可以减少后续 `Provider`
    /// 适配层的分支判断与序列化体积。
    ///
    /// # Returns
    /// 最终消息对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::MessageBuilder;
    ///
    /// let message = MessageBuilder::user().text("你好").build();
    /// assert_eq!(message.content().as_text(), Some("你好"));
    /// ```
    #[must_use]
    pub fn build(self) -> Message {
        let content = match self.parts.as_slice() {
            [ContentPart::Text { text }] => Content::Text(text.clone()),
            _ => Content::Parts(self.parts),
        };

        Message {
            role: self.role,
            content,
            name: self.name,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

fn guess_mime_type(path: &Path) -> Option<String> {
    let guess = MimeGuess::from_path(path).first_raw()?;
    Some(guess.to_string())
}

#[cfg(test)]
mod tests {
    use super::{Content, ContentPart, ImageSource, Message, MessageBuilder, Role};
    use crate::ToolCall;

    #[test]
    fn message_test() {
        let message = MessageBuilder::user().text("你好").build();

        assert_eq!(message.content().as_text(), Some("你好"));
        assert!(!message.content().is_multimodal());
    }

    #[test]
    fn message_test_2() {
        let message = Message::builder(Role::User)
            .text("先看文字")
            .image_url("https://example.com/photo.jpg")
            .text("再输出结论")
            .build();

        assert!(message.content().is_multimodal());

        let Content::Parts(parts) = message.content() else {
            panic!("预期为多模态片段内容");
        };

        assert!(matches!(parts[0], ContentPart::Text { .. }));
        assert!(matches!(parts[1], ContentPart::Image { .. }));
        assert!(matches!(parts[2], ContentPart::Text { .. }));
    }

    #[test]
    fn mime_type_2() {
        let source = ImageSource::file("./fixtures/avatar.png");

        let ImageSource::File(file) = source else {
            panic!("预期为文件来源");
        };

        assert_eq!(file.mime_type(), Some("image/png"));
    }

    #[test]
    fn message_test_3() {
        let calls = vec![ToolCall::new("call_1", "get_weather", r#"{"city":"杭州"}"#)];
        let message = Message::assistant_with_tool_calls(&calls);

        assert_eq!(message.role(), Role::Assistant);
        assert_eq!(
            message.tool_calls().expect("应包含工具调用")[0].id(),
            "call_1"
        );
        assert_eq!(message.content().as_text(), Some(""));
    }

    #[test]
    fn tool_call_id_2() {
        let message = Message::tool_result("call_1", r#"{"temp":26}"#);

        assert_eq!(message.role(), Role::Tool);
        assert_eq!(message.tool_call_id(), Some("call_1"));
        assert_eq!(message.content().as_text(), Some(r#"{"temp":26}"#));
    }
}
