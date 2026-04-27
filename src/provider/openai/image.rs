use crate::{
    error::LlmError,
    types::{
        request::{ImageGenRequest, VideoGenRequest},
        response::{ImageGenResponse, VideoGenResponse},
    },
};

use super::http::{OpenAiRequestBuilder, parse_usage, send_json_request};

pub(super) async fn execute_generate_image<A: OpenAiRequestBuilder>(
    adapter: &A,
    model: &str,
    req: ImageGenRequest,
) -> Result<ImageGenResponse, LlmError> {
    let mut body = serde_json::Map::new();
    body.insert("model".into(), serde_json::Value::String(model.to_owned()));
    body.insert("prompt".into(), serde_json::Value::String(req.prompt));
    if let Some(n) = req.n {
        body.insert("n".into(), serde_json::json!(n));
    }
    if let Some(size) = req.size {
        body.insert("size".into(), serde_json::Value::String(size));
    }
    for (key, value) in req.extensions {
        body.insert(key, value);
    }

    let raw = send_json_request(
        adapter,
        adapter
            .post_json("/images/generations")
            .json(&serde_json::Value::Object(body)),
    )
    .await?;
    let images = raw
        .get("data")
        .and_then(|value| value.as_array())
        .ok_or_else(|| LlmError::ProviderResponse {
            provider: adapter.provider_name().into(),
            code: None,
            message: "图片生成响应缺少 data".into(),
        })?
        .iter()
        .map(|item| crate::types::response::GeneratedImage {
            url: item
                .get("url")
                .and_then(|value| value.as_str())
                .map(str::to_owned),
            base64: item
                .get("b64_json")
                .and_then(|value| value.as_str())
                .map(str::to_owned),
            revised_prompt: item
                .get("revised_prompt")
                .and_then(|value| value.as_str())
                .map(str::to_owned),
        })
        .collect();

    Ok(ImageGenResponse {
        images,
        usage: parse_usage(raw.get("usage")),
    })
}

pub(super) async fn execute_generate_video<A: OpenAiRequestBuilder>(
    adapter: &A,
    _model: &str,
    _req: VideoGenRequest,
) -> Result<VideoGenResponse, LlmError> {
    // TODO(video): Sora 等视频生成 API 尚未稳定公开，待可用后实现
    Err(LlmError::UnsupportedCapability {
        provider: Some(adapter.provider_name().into()),
        capability: "generate_video".into(),
    })
}
