use crate::types::content::{AudioFormat, ContentPart, Message, Role, ToolCall};

/// chat 非流式响应。
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    /// 累积的完整文本。
    pub text: String,
    /// 累积的推理文本。
    pub thinking: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
    /// 原始响应体，调试用。
    pub raw: Option<serde_json::Value>,
}

impl ChatResponse {
    /// 将响应转为可追加到历史对话的 assistant 消息。
    pub fn into_message(self) -> Message {
        let mut parts = Vec::new();
        if !self.text.is_empty() {
            parts.push(ContentPart::text(self.text));
        }
        for call in self.tool_calls {
            parts.push(ContentPart::ToolCall(call));
        }
        Message {
            role: Role::Assistant,
            content: parts,
            name: None,
        }
    }
}

/// 流式 chat 的语义输出单元。
#[derive(Debug, Clone, Default)]
pub struct ChatChunk {
    pub text_delta: Option<String>,
    /// 推理过程增量。
    pub thinking_delta: Option<String>,
    /// 完整工具调用。
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<FinishReason>,
    /// 通常仅在最后一个 chunk 中出现。
    pub usage: Option<Usage>,
}

impl ChatChunk {
    pub fn is_finished(&self) -> bool {
        self.finish_reason.is_some()
    }
}

/// 统一结束原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Completed,
    MaxOutputTokens,
    ToolCalls,
    ContentFilter,
    Failed,
}

/// 统一 token 统计。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// 向量化响应。
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub usage: Option<Usage>,
}

/// 语音转文本响应。
#[derive(Debug, Clone)]
pub struct SpeechToTextResponse {
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: Option<f32>,
    pub usage: Option<Usage>,
}

/// 文本转语音响应。
#[derive(Debug, Clone)]
pub struct TextToSpeechResponse {
    pub audio_data: bytes::Bytes,
    pub format: AudioFormat,
    pub duration_secs: Option<f32>,
}

/// 图片生成响应。
#[derive(Debug, Clone)]
pub struct ImageGenResponse {
    pub images: Vec<GeneratedImage>,
    pub usage: Option<Usage>,
}

/// 单张生成图片。
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub url: Option<String>,
    pub base64: Option<String>,
    pub revised_prompt: Option<String>,
}

/// 视频生成响应。
#[derive(Debug, Clone)]
pub struct VideoGenResponse {
    pub task_id: String,
    pub status: TaskStatus,
    pub url: Option<String>,
}

/// 视频任务状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Processing,
    Succeeded,
    Failed,
}

#[cfg(test)]
mod tests {
    use crate::types::content::ToolCall;

    use super::{ChatResponse, FinishReason};

    #[test]
    fn into_message_preserves_text_and_tool_call_order() {
        let response = ChatResponse {
            id: "resp_1".into(),
            model: "test-model".into(),
            text: "hello".into(),
            thinking: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                tool_name: "lookup".into(),
                arguments: serde_json::json!({ "city": "hangzhou" }),
            }],
            finish_reason: Some(FinishReason::ToolCalls),
            usage: None,
            raw: None,
        };

        let message = response.into_message();
        assert_eq!(message.text(), "hello");
        assert_eq!(message.content.len(), 2);
    }
}
