use crate::types::content::{
    AudioFormat, ContentPart, MediaSource, Message, Role, Tool, ToolChoice, VideoFormat,
};

/// 推理强度等级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// 低强度推理，适合延迟敏感场景。
    Low,
    /// 中等强度推理，在质量与延迟之间折中。
    Medium,
    /// 高强度推理，适合复杂任务或更长思考链路。
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
    /// 按顺序发送给模型的消息列表。
    pub messages: Vec<Message>,
    /// 限制本次生成允许返回的最大输出 token 数。
    pub max_tokens: Option<u32>,
    /// 控制采样随机性，值越高结果通常越发散。
    pub temperature: Option<f32>,
    /// 控制核采样的累积概率阈值，用于裁剪候选 token 分布。
    pub top_p: Option<f32>,
    /// 提供给模型的可调用工具定义列表。
    pub tools: Vec<Tool>,
    /// 指定模型在本次请求中的工具选择策略。
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
            // 默认关闭 thinking，避免不同兼容后端对字段形态要求不一致时直接失败。
            thinking: false,
            thinking_budget: None,
            reasoning_effort: None,
            parallel_tool_calls: None,
            extensions: serde_json::Map::new(),
        }
    }
}

impl ChatRequest {
    /// 返回 `ChatRequest` 的链式构造器。
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

    /// 追加一条仅包含文本内容的 user 消息。
    pub fn user_text(mut self, text: impl Into<String>) -> Self {
        self.inner.messages.push(Message {
            role: Role::User,
            content: vec![ContentPart::text(text)],
            name: None,
        });
        self
    }

    /// 追加一条由多个内容片段组成的 user 消息。
    pub fn user(mut self, parts: Vec<ContentPart>) -> Self {
        self.inner.messages.push(Message {
            role: Role::User,
            content: parts,
            name: None,
        });
        self
    }

    /// 设置本次生成允许返回的最大输出 token 数。
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.inner.max_tokens = Some(n);
        self
    }

    /// 设置采样温度。
    pub fn temperature(mut self, t: f32) -> Self {
        self.inner.temperature = Some(t);
        self
    }

    /// 设置核采样的累积概率阈值。
    pub fn top_p(mut self, p: f32) -> Self {
        self.inner.top_p = Some(p);
        self
    }

    /// 覆盖可供模型调用的工具列表。
    pub fn tools(mut self, tools: Vec<Tool>) -> Self {
        self.inner.tools = tools;
        self
    }

    /// 设置模型在本次请求中的工具选择策略。
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

    /// 追加一项透传给 provider 的扩展参数。
    pub fn extension(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.inner.extensions.insert(key.into(), value);
        self
    }

    /// 构建最终的聊天请求对象。
    pub fn build(self) -> ChatRequest {
        self.inner
    }
}

/// 向量化请求。
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// 需要生成向量的原始输入文本列表。
    pub inputs: Vec<String>,
    /// 期望返回的向量维度，未指定时由 provider 决定。
    pub dimensions: Option<usize>,
    /// 透传给 provider 的差异化参数。
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

/// 语音转文本请求。
#[derive(Debug, Clone)]
pub struct SpeechToTextRequest {
    /// 待识别的音频媒体来源。
    pub source: MediaSource,
    /// 输入音频的封装格式。
    pub format: AudioFormat,
    /// 提示 provider 优先按该语言进行识别。
    pub language: Option<String>,
    /// 透传给 provider 的差异化参数。
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

/// 文本转语音请求。
#[derive(Debug, Clone)]
pub struct TextToSpeechRequest {
    /// 需要合成语音的原始文本。
    pub text: String,
    /// 指定 provider 使用的音色或说话人。
    pub voice: Option<String>,
    /// 输出音频的封装格式。
    pub output_format: AudioFormat,
    /// 透传给 provider 的差异化参数。
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

/// 图片生成请求。
#[derive(Debug, Clone)]
pub struct ImageGenRequest {
    /// 用于生成图片的文本提示词。
    pub prompt: String,
    /// 期望生成的图片数量。
    pub n: Option<u32>,
    /// 期望输出的图片尺寸。
    pub size: Option<String>,
    /// 透传给 provider 的差异化参数。
    pub extensions: serde_json::Map<String, serde_json::Value>,
}

/// 视频生成请求。
#[derive(Debug, Clone)]
pub struct VideoGenRequest {
    /// 用于生成视频的文本提示词。
    pub prompt: String,
    /// 期望输出的视频时长，单位为秒。
    pub duration_secs: Option<u32>,
    /// 期望输出的视频封装格式。
    pub output_format: Option<VideoFormat>,
    /// 透传给 provider 的差异化参数。
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
    fn chat_request_defaults_to_thinking_disabled() {
        let request = ChatRequest::default();

        assert!(!request.thinking);
        assert_eq!(request.thinking_budget, None);
        assert_eq!(request.reasoning_effort, None);
    }
}
