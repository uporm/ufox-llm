mod anthropic;
mod doubao;
mod gemini;
mod openai;
mod qwen;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::{
    error::LlmError,
    types::{
        request::{
            ChatRequest, EmbeddingRequest, ImageGenRequest, SpeechToTextRequest,
            TextToSpeechRequest, VideoGenRequest,
        },
        response::{
            ChatChunk, ChatResponse, EmbeddingResponse, ImageGenResponse, SpeechToTextResponse,
            TextToSpeechResponse, VideoGenResponse,
        },
    },
};

/// 选择与 OpenAI 兼容 endpoint 通信时使用的线路协议。
///
/// 对于 `Provider::OpenAI`，默认使用 `Responses`；
/// 对于其他 provider，默认使用 `ChatCompletions`。
/// 可通过 `ClientBuilder::api_protocol` 显式覆盖。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiProtocol {
    /// OpenAI Chat Completions 协议（`POST /chat/completions`）。
    ChatCompletions,
    /// OpenAI Responses 协议（`POST /responses`）。
    Responses,
}

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

    async fn embed(
        &self,
        _model: &str,
        _req: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            provider: Some(self.name().into()),
            capability: "embed".into(),
        })
    }

    async fn speech_to_text(
        &self,
        _model: &str,
        _req: SpeechToTextRequest,
    ) -> Result<SpeechToTextResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            provider: Some(self.name().into()),
            capability: "speech_to_text".into(),
        })
    }

    async fn text_to_speech(
        &self,
        _model: &str,
        _req: TextToSpeechRequest,
    ) -> Result<TextToSpeechResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            provider: Some(self.name().into()),
            capability: "text_to_speech".into(),
        })
    }

    async fn generate_image(
        &self,
        _model: &str,
        _req: ImageGenRequest,
    ) -> Result<ImageGenResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            provider: Some(self.name().into()),
            capability: "generate_image".into(),
        })
    }

    async fn generate_video(
        &self,
        _model: &str,
        _req: VideoGenRequest,
    ) -> Result<VideoGenResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            provider: Some(self.name().into()),
            capability: "generate_video".into(),
        })
    }

    async fn poll_video_task(&self, _task_id: &str) -> Result<VideoGenResponse, LlmError> {
        Err(LlmError::UnsupportedCapability {
            provider: Some(self.name().into()),
            capability: "poll_video_task".into(),
        })
    }
}

/// 协议名称与 adapter 工厂，二者合一。
#[derive(Debug, Clone)]
pub enum Provider {
    /// OpenAI 兼容协议，必须由 ClientBuilder 显式提供 base_url。
    Compatible,
    /// 标准 OpenAI 协议。
    OpenAI,
    /// Anthropic Messages 协议。
    Anthropic,
    /// 豆包协议。
    Doubao,
    /// 通义千问协议。
    Qwen,
    /// Google Gemini 协议。
    Gemini,
}

impl Provider {
    /// 返回稳定的 provider 名称字符串。
    pub fn name(&self) -> &'static str {
        match self {
            Provider::Compatible => "compatible",
            Provider::OpenAI => "openai",
            Provider::Anthropic => "anthropic",
            Provider::Doubao => "doubao",
            Provider::Qwen => "qwen",
            Provider::Gemini => "gemini",
        }
    }

    /// 根据稳定名称解析 provider。
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "compatible" => Some(Provider::Compatible),
            "openai" => Some(Provider::OpenAI),
            "anthropic" => Some(Provider::Anthropic),
            "doubao" => Some(Provider::Doubao),
            "qwen" => Some(Provider::Qwen),
            "gemini" => Some(Provider::Gemini),
            _ => None,
        }
    }

    /// 返回该 provider 的默认 base URL。
    pub fn default_base_url(&self) -> Option<&'static str> {
        match self {
            Provider::OpenAI => Some("https://api.openai.com/v1"),
            Provider::Anthropic => Some("https://api.anthropic.com"),
            Provider::Doubao => Some("https://ark.cn-beijing.volces.com/api/v3"),
            Provider::Qwen => Some("https://dashscope.aliyuncs.com/compatible-mode/v1"),
            Provider::Gemini => Some("https://generativelanguage.googleapis.com/v1beta"),
            Provider::Compatible => None,
        }
    }

    /// 返回该 provider 的默认协议。
    pub fn default_protocol(&self) -> ApiProtocol {
        match self {
            Provider::OpenAI => ApiProtocol::Responses,
            _ => ApiProtocol::ChatCompletions,
        }
    }

    /// adapter 创建的唯一入口。
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn into_adapter(
        &self,
        protocol: ApiProtocol,
        api_key: &str,
        base_url: &str,
        transport: &crate::middleware::Transport,
    ) -> Result<Box<dyn ProviderAdapter>, LlmError> {
        match self {
            Provider::Compatible => {
                openai::build("compatible", protocol, api_key, base_url, transport)
            }
            Provider::OpenAI => openai::build("openai", protocol, api_key, base_url, transport),
            Provider::Anthropic => anthropic::build(api_key, base_url, transport),
            Provider::Doubao => doubao::build(api_key, base_url, transport),
            Provider::Qwen => qwen::build(api_key, base_url, transport),
            Provider::Gemini => gemini::build(api_key, base_url, transport),
        }
    }
}
