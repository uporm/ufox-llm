use crate::{
    error::LlmError,
    types::{request::EmbeddingRequest, response::EmbeddingResponse},
};

use super::http::{OpenAiRequestBuilder, parse_usage, send_json_request};

pub(super) async fn execute_embed<A: OpenAiRequestBuilder>(
    adapter: &A,
    model: &str,
    req: EmbeddingRequest,
) -> Result<EmbeddingResponse, LlmError> {
    let mut body = serde_json::Map::new();
    body.insert("model".into(), serde_json::Value::String(model.to_owned()));
    body.insert(
        "input".into(),
        serde_json::Value::Array(
            req.inputs
                .into_iter()
                .map(serde_json::Value::String)
                .collect(),
        ),
    );
    if let Some(dimensions) = req.dimensions {
        body.insert("dimensions".into(), serde_json::json!(dimensions));
    }
    for (key, value) in req.extensions {
        body.insert(key, value);
    }

    let raw = send_json_request(
        adapter,
        adapter
            .post_json("/embeddings")
            .json(&serde_json::Value::Object(body)),
    )
    .await?;
    let embeddings =
        raw.get("data")
            .and_then(|value| value.as_array())
            .ok_or_else(|| LlmError::ProviderResponse {
                provider: adapter.provider_name().into(),
                code: None,
                message: "embedding 响应缺少 data".into(),
            })?
            .iter()
            .map(|item| {
                item.get("embedding")
                    .and_then(|value| value.as_array())
                    .ok_or_else(|| LlmError::ProviderResponse {
                        provider: adapter.provider_name().into(),
                        code: None,
                        message: "embedding 缺少 embedding 数组".into(),
                    })?
                    .iter()
                    .map(|value| {
                        value.as_f64().map(|n| n as f32).ok_or_else(|| {
                            LlmError::ProviderResponse {
                                provider: adapter.provider_name().into(),
                                code: None,
                                message: "embedding 含非数字元素".into(),
                            }
                        })
                    })
                    .collect::<Result<Vec<_>, LlmError>>()
            })
            .collect::<Result<Vec<_>, LlmError>>()?;

    Ok(EmbeddingResponse {
        embeddings,
        model: raw
            .get("model")
            .and_then(|value| value.as_str())
            .unwrap_or(model)
            .to_owned(),
        usage: parse_usage(raw.get("usage")),
    })
}
