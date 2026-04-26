use crate::types::content::{
    AudioFormat, ContentPart, MediaSource, Message, Role, Tool, ToolChoice, VideoFormat,
};

/// 推理强度等级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    /// 返回协议层使用的字符串值。
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// 单次聊天请求。
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub tools: Vec<Tool>,
    pub tool_choice: ToolChoice,
    /// 是否启用 thinking 模式。
    pub thinking: bool,
    /// thinking 模式下允许使用的预算。
    pub thinking_budget: Option<u32>,
    /// 推理强度等级。
    pub reasoning_effort: Option<ReasoningEffort>,
    /// 是否允许并行发起多个工具调用。
    pub parallel_tool_calls: Option<bool>,
    /// 透传给 provider 的差异化参数。
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: Vec::new(),
            tool_choice: Default::default(),
            // 默认开启 thinking，避免调用方遗漏该能力。
            thinking: true,
            thinking_budget: None,
            reasoning_effort: None,
            parallel_tool_calls: None,
            extensions: serde_json::Map::new(),
        }
    }
}

impl ChatRequest {
    pub fn builder() -> ChatRequestBuilder {
        ChatRequestBuilder {
            inner: Self::default(),
        }
    }
}

/// `ChatRequest` 的链式构造器。
pub struct ChatRequestBuilder {
    inner: ChatRequest,
}

impl ChatRequestBuilder {
    /// 覆盖整个消息列表。
    pub fn messages(mut self, messages: Vec<Message>) -> Self {
        self.inner.messages = messages;
        self
    }

    /// 在消息列表头部插入一条 system 消息，若已存在则替换。
    pub fn system(mut self, text: impl Into<String>) -> Self {
        let msg = Message {
            role: Role::System,
            content: vec![ContentPart::text(text)],
            name: None,
        };
        if self.inner.messages.first().map(|message| message.role) == Some(Role::System) {
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
        self.inner.messages.push(Message {
            role: Role::User,
            content: parts,
            name: None,
        });
        self
    }

    pub fn max_tokens(mut self, n: u32) -> Self {
        self.inner.max_tokens = Some(n);
        self
    }

    pub fn temperature(mut self, t: f32) -> Self {
        self.inner.temperature = Some(t);
        self
    }

    pub fn top_p(mut self, p: f32) -> Self {
        self.inner.top_p = Some(p);
        self
    }

    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.inner.tools = tools;
        self
    }

    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.inner.tool_choice = choice;
        self
    }

    /// 设置是否启用 thinking 模式。
    pub fn thinking(mut self, enabled: bool) -> Self {
        self.inner.thinking = enabled;
        self
    }

    /// 设置 thinking 模式下允许使用的预算。
    pub fn thinking_budget(mut self, budget: u32) -> Self {
        self.inner.thinking_budget = Some(budget);
        self
    }

    /// 设置推理强度等级。
    pub fn reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.inner.reasoning_effort = Some(effort);
        self
    }

    /// 设置是否允许并行发起多个工具调用。
    pub fn parallel_tool_calls(mut self, enabled: bool) -> Self {
        self.inner.parallel_tool_calls = Some(enabled);
        self
    }

    pub fn extension(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.inner.extensions.insert(key.into(), value);
        self
    }

    pub fn build(self) -> ChatRequest {
        self.inner
    }
}

/// 向量化请求。
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    pub inputs: Vec<String>,
    pub dimensions: Option<usize>,
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

/// 语音转文本请求。
#[derive(Debug, Clone)]
pub struct SpeechToTextRequest {
    pub source: MediaSource,
    pub format: AudioFormat,
    pub language: Option<String>,
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

/// 文本转语音请求。
#[derive(Debug, Clone)]
pub struct TextToSpeechRequest {
    pub text: String,
    pub voice: Option<String>,
    pub output_format: AudioFormat,
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

/// 图片生成请求。
#[derive(Debug, Clone)]
pub struct ImageGenRequest {
    pub prompt: String,
    pub n: Option<u32>,
    pub size: Option<String>,
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

/// 视频生成请求。
#[derive(Debug, Clone)]
pub struct VideoGenRequest {
    pub prompt: String,
    pub duration_secs: Option<u32>,
    pub output_format: Option<VideoFormat>,
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use crate::types::content::{Message, Role, ToolChoice};

    use super::{ChatRequest, ReasoningEffort};

    #[test]
    fn system_message_is_inserted_at_head() {
        let request = ChatRequest::builder()
            .user_text("hello")
            .system("system")
            .build();

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, Role::System);
        assert_eq!(request.messages[0].text(), "system");
    }

    #[test]
    fn system_message_is_replaced_instead_of_duplicated() {
        let request = ChatRequest::builder()
            .system("first")
            .system("second")
            .user_text("hello")
            .build();

        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].text(), "second");
    }

    #[test]
    fn messages_setter_allows_follow_up_appends() {
        let request = ChatRequest::builder()
            .messages(vec![Message {
                role: Role::User,
                content: vec![],
                name: None,
            }])
            .user_text("hello")
            .tool_choice(ToolChoice::Required)
            .build();

        assert_eq!(request.messages.len(), 2);
        assert!(matches!(request.tool_choice, ToolChoice::Required));
    }

    #[test]
    fn builder_sets_thinking_related_fields() {
        let request = ChatRequest::builder()
            .thinking(true)
            .thinking_budget(2048)
            .reasoning_effort(ReasoningEffort::High)
            .parallel_tool_calls(false)
            .build();

        assert!(request.thinking);
        assert_eq!(request.thinking_budget, Some(2048));
        assert_eq!(request.reasoning_effort, Some(ReasoningEffort::High));
        assert_eq!(request.parallel_tool_calls, Some(false));
    }

    #[test]
    fn chat_request_defaults_to_thinking_enabled() {
        let request = ChatRequest::default();

        assert!(request.thinking);
        assert_eq!(request.thinking_budget, None);
        assert_eq!(request.reasoning_effort, None);
    }
}
