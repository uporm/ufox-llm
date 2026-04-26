use reqwest::multipart::{Form, Part};

use crate::{
    error::LlmError,
    types::{
        request::{SpeechToTextRequest, TextToSpeechRequest},
        response::{SpeechToTextResponse, TextToSpeechResponse},
    },
};

use super::http::{OpenAiRequestBuilder, parse_usage, send_bytes_request, send_json_request};
use super::media::{audio_extension, audio_mime, resolve_media_source_bytes};

pub(super) async fn execute_speech_to_text<A: OpenAiRequestBuilder>(
    adapter: &A,
    model: &str,
    req: SpeechToTextRequest,
) -> Result<SpeechToTextResponse, LlmError> {
    let fallback_filename = format!("audio.{}", audio_extension(req.format));
    let (bytes, mime_type, filename) = resolve_media_source_bytes(
        adapter.transport(),
        &req.source,
        &fallback_filename,
        Some(audio_mime(req.format)),
    )
    .await?;

    let part = Part::bytes(bytes)
        .file_name(filename)
        .mime_str(&mime_type)
        .map_err(|err| LlmError::MediaInput {
            message: format!("音频 MIME 不合法: {err}"),
        })?;
    let mut form = Form::new()
        .part("file", part)
        .text("model", model.to_owned());
    if let Some(language) = req.language {
        form = form.text("language", language);
    }
    for (key, value) in req.extensions {
        form = form.text(
            key,
            value
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| value.to_string()),
        );
    }

    let raw = send_json_request(
        adapter,
        adapter.post_multipart("/audio/transcriptions").multipart(form),
    )
    .await?;
    Ok(SpeechToTextResponse {
        text: raw
            .get("text")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_owned(),
        language: raw
            .get("language")
            .and_then(|value| value.as_str())
            .map(str::to_owned),
        duration_secs: raw
            .get("duration")
            .or_else(|| raw.get("duration_secs"))
            .and_then(|value| value.as_f64())
            .map(|value| value as f32),
        usage: parse_usage(raw.get("usage")),
    })
}

pub(super) async fn execute_text_to_speech<A: OpenAiRequestBuilder>(
    adapter: &A,
    model: &str,
    req: TextToSpeechRequest,
) -> Result<TextToSpeechResponse, LlmError> {
    let mut body = serde_json::Map::new();
    body.insert("model".into(), serde_json::Value::String(model.to_owned()));
    body.insert("input".into(), serde_json::Value::String(req.text));
    body.insert(
        "voice".into(),
        serde_json::Value::String(req.voice.unwrap_or_else(|| "alloy".into())),
    );
    body.insert(
        "response_format".into(),
        serde_json::Value::String(audio_extension(req.output_format).into()),
    );
    for (key, value) in req.extensions {
        body.insert(key, value);
    }

    Ok(TextToSpeechResponse {
        audio_data: send_bytes_request(
            adapter,
            adapter
                .post_json("/audio/speech")
                .json(&serde_json::Value::Object(body)),
            "读取音频响应",
        )
        .await?,
        format: req.output_format,
        duration_secs: None,
    })
}
