//! 媒体资源处理工具。
//!
//! 提供将 [`MediaSource`]（URL / Base64 / 本地文件）解析为字节序列或
//! OpenAI 图片 URL 对象的辅助函数，供 `audio`、`image` 等模块调用。
//!
//! 同时包含音频格式与 MIME 类型 / 文件扩展名的映射。

use base64::Engine as _;

use crate::{
    error::LlmError,
    middleware::Transport,
    types::content::{AudioFormat, MediaSource},
};

const DEFAULT_MIME: &str = "application/octet-stream";

async fn read_local_file(path: &std::path::Path) -> Result<Vec<u8>, LlmError> {
    tokio::fs::read(path).await.map_err(|err| LlmError::MediaInput {
        message: format!("读取文件失败 {:?}: {}", path, err),
    })
}

fn guess_path_mime(path: &std::path::Path, fallback: Option<&str>) -> String {
    mime_guess::from_path(path)
        .first_raw()
        .or(fallback)
        .unwrap_or(DEFAULT_MIME)
        .to_owned()
}

fn filename_from_path(path: &std::path::Path, fallback_filename: &str) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| fallback_filename.to_owned())
}

// ── 图片 URL 解析 ─────────────────────────────────────────────────────────────

/// 将 [`MediaSource`] 转换为 OpenAI `image_url` 对象（`{ "url": "..." }`）。
///
/// - `Url` → 直接包装
/// - `Base64` → 构造 `data:<mime>;base64,<data>` URL
/// - `File` → 读取文件并编码为 Base64 data URL
pub(super) async fn resolve_media_source_to_image_url(
    source: &MediaSource,
) -> Result<serde_json::Value, LlmError> {
    match source {
        MediaSource::Url { url } => Ok(serde_json::json!({ "url": url })),
        MediaSource::Base64 { data, mime_type } => Ok(serde_json::json!({
            "url": format!("data:{mime_type};base64,{data}")
        })),
        MediaSource::File { path } => {
            let data = read_local_file(path).await?;
            let mime_type = guess_path_mime(path, None);
            let data =
                base64::Engine::encode(&base64::engine::general_purpose::STANDARD, data);
            Ok(serde_json::json!({
                "url": format!("data:{mime_type};base64,{data}")
            }))
        }
    }
}

// ── 媒体字节解析 ──────────────────────────────────────────────────────────────

/// 将 [`MediaSource`] 解析为 `(bytes, mime_type, filename)` 三元组。
///
/// 用于 multipart 上传场景（如语音识别），返回值直接传入 `reqwest::Part`。
///
/// - `Base64` → 解码并返回
/// - `Url` → 下载资源，从响应头推断 MIME
/// - `File` → 读取文件，从路径推断 MIME
pub(super) async fn resolve_media_source_bytes(
    transport: &Transport,
    source: &MediaSource,
    fallback_filename: &str,
    default_mime: Option<&str>,
) -> Result<(Vec<u8>, String, String), LlmError> {
    // TODO(memory): 当前将完整媒体文件读入内存，适用于语音转文字等较小文件（OpenAI 限制 25MB）。
    // 若未来支持超大文件，需改为流式读取，并留意目标 API 是否支持 Chunked Multipart Part（部分 API 强制要求 Content-Length）。
    match source {
        MediaSource::Base64 { data, mime_type } => {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|err| LlmError::MediaInput {
                    message: format!("base64 解码失败: {err}"),
                })?;
            Ok((bytes, mime_type.clone(), fallback_filename.to_owned()))
        }
        MediaSource::Url { url } => {
            let response = transport
                .client()
                .get(url)
                .send()
                .await
                .map_err(|err| LlmError::transport("下载媒体资源", err))?;
            if !response.status().is_success() {
                return Err(LlmError::MediaInput {
                    message: format!(
                        "下载媒体资源失败：status={} url={url}",
                        response.status().as_u16()
                    ),
                });
            }
            let mime_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned)
                .or_else(|| default_mime.map(str::to_owned))
                .unwrap_or_else(|| DEFAULT_MIME.to_owned());
            let bytes = response
                .bytes()
                .await
                .map_err(|err| LlmError::transport("读取媒体响应", err))?
                .to_vec();
            let filename = url
                .split('/')
                .next_back()
                .filter(|segment| !segment.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| fallback_filename.to_owned());
            Ok((bytes, mime_type, filename))
        }
        MediaSource::File { path } => {
            let bytes = read_local_file(path).await?;
            let mime_type = guess_path_mime(path, default_mime);
            let filename = filename_from_path(path, fallback_filename);
            Ok((bytes, mime_type, filename))
        }
    }
}

// ── 音频格式辅助 ──────────────────────────────────────────────────────────────

/// 返回音频格式对应的 MIME 类型字符串。
pub(super) fn audio_mime(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Mp3 => "audio/mpeg",
        AudioFormat::Wav => "audio/wav",
        AudioFormat::Flac => "audio/flac",
        AudioFormat::Opus => "audio/ogg",
        AudioFormat::Aac => "audio/aac",
        AudioFormat::Pcm => "audio/pcm",
    }
}

/// 返回音频格式对应的文件扩展名（不含前缀点）。
pub(super) fn audio_extension(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Mp3 => "mp3",
        AudioFormat::Wav => "wav",
        AudioFormat::Flac => "flac",
        AudioFormat::Opus => "opus",
        AudioFormat::Aac => "aac",
        AudioFormat::Pcm => "pcm",
    }
}
