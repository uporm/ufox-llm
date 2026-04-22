//! 消息模型。
//!
//! 定义与 Provider 无关的消息结构，覆盖文本、多模态和工具相关字段。

use std::path::{Path, PathBuf};

use mime_guess::MimeGuess;
use serde::{Deserialize, Serialize};

use super::tool::ToolCall;

/// 对话消息的角色。
///
/// 该枚举对应主流 `Chat API` 中的消息来源语义，不直接耦合任一 `Provider` 的实现细节。
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    /// 纯文本内容。
    Text(String),
    /// 顺序化的多模态内容片段。
    Parts(Vec<ContentPart>),
}

impl Content {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    pub fn parts(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }

    /// 尝试以纯文本形式读取内容。
    ///
    /// 当内容为多模态片段时返回 `None`，因为这类内容无法无损降级为单一字符串。
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text.as_str()),
            Self::Parts(_) => None,
        }
    }

    pub const fn is_multimodal(&self) -> bool {
        matches!(self, Self::Parts(_))
    }
}

/// 多模态消息中的单个内容片段。
///
/// 片段采用顺序列表而不是映射结构，以确保序列化时可以严格保留调用方输入顺序。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContentPart {
    /// 文本片段。
    Text { text: String },
    /// 图片片段。
    Image { source: ImageSource },
    /// 音频片段。
    Audio { source: AudioSource },
    /// 视频片段。
    Video { source: VideoSource },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: ImageSource::url(url),
        }
    }

    pub fn image_file(path: impl Into<PathBuf>) -> Self {
        Self::Image {
            source: ImageSource::file(path),
        }
    }

    pub fn audio_url(url: impl Into<String>) -> Self {
        Self::Audio {
            source: AudioSource::url(url),
        }
    }

    pub fn audio_file(path: impl Into<PathBuf>) -> Self {
        Self::Audio {
            source: AudioSource::file(path),
        }
    }

    pub fn video_url(url: impl Into<String>) -> Self {
        Self::Video {
            source: VideoSource::url(url),
        }
    }

    pub fn video_file(path: impl Into<PathBuf>) -> Self {
        Self::Video {
            source: VideoSource::file(path),
        }
    }
}

/// 图片来源。
///
/// 统一抽象远程 `URL` 与本地文件两类来源，便于 `Provider` 层按各自协议进行转换。
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
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url { url: url.into() }
    }

    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::File(ImageFile::new(path))
    }
}

/// 音频来源。
///
/// 统一抽象远程 `URL` 与本地文件两类来源，便于 `Provider` 层按各自协议进行转换。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AudioSource {
    /// 远程音频 URL。
    Url {
        /// 音频 URL。
        url: String,
    },
    /// 本地音频文件。
    File(AudioFile),
}

impl AudioSource {
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url { url: url.into() }
    }

    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::File(AudioFile::new(path))
    }
}

/// 视频来源。
///
/// 统一抽象远程 `URL` 与本地文件两类来源，便于 `Provider` 层按各自协议进行转换。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VideoSource {
    /// 远程视频 URL。
    Url {
        /// 视频 URL。
        url: String,
    },
    /// 本地视频文件。
    File(VideoFile),
}

impl VideoSource {
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url { url: url.into() }
    }

    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::File(VideoFile::new(path))
    }
}

/// 本地图片文件描述。
///
/// 该结构体仅保存路径与 `MIME` 类型元数据，不在构造阶段执行 `I/O`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageFile {
    pub path: PathBuf,
    pub mime_type: Option<String>,
}

impl ImageFile {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let mime_type = guess_mime_type(&path);

        Self { path, mime_type }
    }
}

/// 本地音频文件描述。
///
/// 该结构体仅保存路径与 `MIME` 类型元数据，不在构造阶段执行 `I/O`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioFile {
    pub path: PathBuf,
    pub mime_type: Option<String>,
}

impl AudioFile {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let mime_type = guess_mime_type(&path);

        Self { path, mime_type }
    }
}

/// 本地视频文件描述。
///
/// 该结构体仅保存路径与 `MIME` 类型元数据，不在构造阶段执行 `I/O`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VideoFile {
    pub path: PathBuf,
    pub mime_type: Option<String>,
}

impl VideoFile {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let mime_type = guess_mime_type(&path);

        Self { path, mime_type }
    }
}

/// 对话消息。
///
/// 该结构体是 `SDK` 内部与对外共享的统一消息模型，后续会被各 `Provider` 转换为各自请求格式。
/// 除基础角色和内容外，它还可以承载工具调用元数据，用于表达：
/// 1. 助手返回的工具调用请求；
/// 2. 工具执行结果回填时关联的 `tool_call_id`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Content,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: Content) -> Self {
        Self {
            role,
            content,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self::new(Role::System, Content::text(text))
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::User, Content::text(text))
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new(Role::Assistant, Content::text(text))
    }

    pub fn assistant_with_tool_calls(tool_calls: &[ToolCall]) -> Self {
        Self {
            role: Role::Assistant,
            content: Content::text(""),
            name: None,
            tool_calls: Some(tool_calls.to_vec()),
            tool_call_id: None,
        }
    }

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
    pub fn builder(role: Role) -> MessageBuilder {
        MessageBuilder::new(role)
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// 消息构建器。
///
/// 该构建器主要用于多模态场景，支持按顺序追加文本和图片片段。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageBuilder {
    role: Role,
    parts: Vec<ContentPart>,
    name: Option<String>,
}

impl MessageBuilder {
    /// 为指定角色创建消息构建器。

    pub fn new(role: Role) -> Self {
        Self {
            role,
            parts: Vec::new(),
            name: None,
        }
    }

    pub fn user() -> Self {
        Self::new(Role::User)
    }

    pub fn system() -> Self {
        Self::new(Role::System)
    }

    pub fn assistant() -> Self {
        Self::new(Role::Assistant)
    }

    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.parts.push(ContentPart::text(text));
        self
    }

    pub fn image_url(mut self, url: impl Into<String>) -> Self {
        self.parts.push(ContentPart::image_url(url));
        self
    }

    pub fn image_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.parts.push(ContentPart::image_file(path));
        self
    }

    pub fn audio_url(mut self, url: impl Into<String>) -> Self {
        self.parts.push(ContentPart::audio_url(url));
        self
    }

    pub fn audio_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.parts.push(ContentPart::audio_file(path));
        self
    }

    pub fn video_url(mut self, url: impl Into<String>) -> Self {
        self.parts.push(ContentPart::video_url(url));
        self
    }

    pub fn video_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.parts.push(ContentPart::video_file(path));
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// 构建消息对象。
    ///
    /// 当构建器中仅包含一个文本片段时，SDK 会将其折叠为 [`Content::Text`]。
    /// 这样做的原因是纯文本消息是最常见场景，直接使用字符串可以减少后续 `Provider`
    /// 适配层的分支判断与序列化体积。

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
    use super::{
        AudioSource, Content, ContentPart, ImageSource, Message, MessageBuilder, Role, VideoSource,
    };
    use crate::ToolCall;

    #[test]
    fn message_test() {
        let message = MessageBuilder::user().text("你好").build();

        assert_eq!(message.content.as_text(), Some("你好"));
        assert!(!message.content.is_multimodal());
    }

    #[test]
    fn message_test_2() {
        let message = Message::builder(Role::User)
            .text("先看文字")
            .image_url("https://example.com/photo.jpg")
            .audio_url("https://example.com/audio.mp3")
            .video_url("https://example.com/video.mp4")
            .text("再输出结论")
            .build();

        assert!(message.content.is_multimodal());

        let Content::Parts(parts) = &message.content else {
            panic!("预期为多模态片段内容");
        };

        assert!(matches!(parts[0], ContentPart::Text { .. }));
        assert!(matches!(parts[1], ContentPart::Image { .. }));
        assert!(matches!(parts[2], ContentPart::Audio { .. }));
        assert!(matches!(parts[3], ContentPart::Video { .. }));
        assert!(matches!(parts[4], ContentPart::Text { .. }));
    }

    #[test]
    fn mime_type_2() {
        let source = ImageSource::file("./fixtures/avatar.png");

        let ImageSource::File(file) = source else {
            panic!("预期为文件来源");
        };

        assert_eq!(file.mime_type.as_deref(), Some("image/png"));
    }

    #[test]
    fn audio_source_2() {
        let source = AudioSource::file("./fixtures/voice.mp3");

        let AudioSource::File(file) = source else {
            panic!("预期为文件来源");
        };

        assert_eq!(file.mime_type.as_deref(), Some("audio/mpeg"));
    }

    #[test]
    fn video_source_2() {
        let source = VideoSource::file("./fixtures/demo.mp4");

        let VideoSource::File(file) = source else {
            panic!("预期为文件来源");
        };

        assert_eq!(file.mime_type.as_deref(), Some("video/mp4"));
    }

    #[test]
    fn message_test_3() {
        let calls = vec![ToolCall::new("call_1", "get_weather", r#"{"city":"杭州"}"#)];
        let message = Message::assistant_with_tool_calls(&calls);

        assert_eq!(message.role, Role::Assistant);
        assert_eq!(
            message.tool_calls.as_ref().expect("应包含工具调用")[0].id,
            "call_1"
        );
        assert_eq!(message.content.as_text(), Some(""));
    }

    #[test]
    fn tool_call_id_2() {
        let message = Message::tool_result("call_1", r#"{"temp":26}"#);

        assert_eq!(message.role, Role::Tool);
        assert_eq!(message.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(message.content.as_text(), Some(r#"{"temp":26}"#));
    }
}
