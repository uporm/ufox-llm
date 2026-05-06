use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use ufox_llm::{Audio, AudioFormat, ContentPart, Image, MediaSource, Video, VideoFormat};
use uuid::Uuid;

use crate::error::ArcError;

/// 会话可附加内容的类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttachmentKind {
    Text,
    Image,
    Audio,
    Video,
    /// 结构化文档（PDF、Office、纯文本等），提取后转为文本片段。
    Document,
}

/// 已附加内容在会话中的唯一引用。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AttachmentRef(pub Uuid);

impl AttachmentRef {
    /// 生成新的附件引用。
    pub fn new() -> Self {
        AttachmentRef(Uuid::new_v4())
    }
}

impl Default for AttachmentRef {
    fn default() -> Self {
        Self::new()
    }
}

/// 提取后的内容需要保留来源元信息，后续可接入 thread 级上下文或外部记忆系统。
#[derive(Debug, Clone)]
pub(crate) struct ExtractedContent {
    pub parts: Vec<ContentPart>,
    #[allow(dead_code)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// 提取器只服务于线程输入链路，因此保持 crate 内部可见，避免过早暴露扩展面。
#[async_trait]
pub(crate) trait MediaExtractor: Send + Sync {
    async fn extract(
        &self,
        source: MediaSource,
        kind: AttachmentKind,
    ) -> Result<ExtractedContent, ArcError>;
}

/// 默认提取器：按模态路由。
///
/// - Image/Audio/Video：直接封装为对应 `ContentPart`，避免在接入层读取大块二进制。
/// - Text/Document：读取文本内容后转成 `ContentPart::Text`，让推理链路统一消费。
pub(crate) struct DefaultExtractor;

#[async_trait]
impl MediaExtractor for DefaultExtractor {
    async fn extract(
        &self,
        source: MediaSource,
        kind: AttachmentKind,
    ) -> Result<ExtractedContent, ArcError> {
        let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();

        match &source {
            MediaSource::File { path } => {
                metadata.insert(
                    "file_path".into(),
                    serde_json::Value::String(path.display().to_string()),
                );
            }
            MediaSource::Url { url } => {
                metadata.insert("url".into(), serde_json::Value::String(url.clone()));
            }
            MediaSource::Base64 { mime_type, .. } => {
                metadata.insert(
                    "mime_type".into(),
                    serde_json::Value::String(mime_type.clone()),
                );
            }
        }

        let parts = match kind {
            AttachmentKind::Image => vec![ContentPart::Image(Image {
                source: source.clone(),
                fidelity: None,
            })],
            AttachmentKind::Audio => vec![ContentPart::Audio(Audio {
                format: guess_audio_format(&source),
                source: source.clone(),
            })],
            AttachmentKind::Video => vec![ContentPart::Video(Video {
                format: guess_video_format(&source),
                source: source.clone(),
                sample_frames: None,
            })],
            AttachmentKind::Text | AttachmentKind::Document => {
                let text = read_as_text(&source).await?;
                vec![ContentPart::text(text)]
            }
        };

        Ok(ExtractedContent { parts, metadata })
    }
}

async fn read_as_text(source: &MediaSource) -> Result<String, ArcError> {
    match source {
        MediaSource::File { path } => tokio::fs::read_to_string(path)
            .await
            .map_err(|e| ArcError::Thread(format!("failed to read {}: {e}", path.display()))),
        MediaSource::Url { url } => {
            let text = reqwest::get(url.as_str())
                .await
                .map_err(|e| ArcError::Thread(format!("fetch document URL '{url}': {e}")))?
                .error_for_status()
                .map_err(|e| {
                    ArcError::Thread(format!("document URL '{url}' returned error: {e}"))
                })?
                .text()
                .await
                .map_err(|e| ArcError::Thread(format!("read document URL '{url}' body: {e}")))?;
            Ok(text)
        }
        MediaSource::Base64 { mime_type, .. } => Err(ArcError::Thread(format!(
            "base64 text extraction not supported for mime_type '{mime_type}'; provide a File or URL source"
        ))),
    }
}

fn guess_audio_format(source: &MediaSource) -> AudioFormat {
    let s = path_or_url_str(source).to_lowercase();
    if s.ends_with(".wav") {
        AudioFormat::Wav
    } else if s.ends_with(".flac") {
        AudioFormat::Flac
    } else if s.ends_with(".opus") {
        AudioFormat::Opus
    } else if s.ends_with(".aac") {
        AudioFormat::Aac
    } else {
        AudioFormat::Mp3
    }
}

fn guess_video_format(source: &MediaSource) -> VideoFormat {
    let s = path_or_url_str(source).to_lowercase();
    if s.ends_with(".webm") {
        VideoFormat::Webm
    } else if s.ends_with(".avi") {
        VideoFormat::Avi
    } else if s.ends_with(".mov") {
        VideoFormat::Mov
    } else {
        VideoFormat::Mp4
    }
}

fn path_or_url_str(source: &MediaSource) -> String {
    match source {
        MediaSource::File { path } => path.display().to_string(),
        MediaSource::Url { url } => url.clone(),
        MediaSource::Base64 { .. } => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    #[tokio::test]
    async fn extract_image_url_produces_image_part() {
        let extractor = DefaultExtractor;
        let source = MediaSource::Url {
            url: "https://example.com/photo.jpg".into(),
        };
        let result = extractor
            .extract(source, AttachmentKind::Image)
            .await
            .unwrap();
        assert_eq!(result.parts.len(), 1);
        assert!(matches!(result.parts[0], ContentPart::Image(_)));
    }

    #[tokio::test]
    async fn extract_text_file_produces_text_part() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "Hello, multimodal!").unwrap();

        let extractor = DefaultExtractor;
        let source = MediaSource::File {
            path: tmp.path().to_path_buf(),
        };
        let result = extractor
            .extract(source, AttachmentKind::Text)
            .await
            .unwrap();
        assert_eq!(result.parts.len(), 1);
        if let ContentPart::Text(t) = &result.parts[0] {
            assert!(t.text.contains("Hello, multimodal!"));
        } else {
            panic!("expected Text part");
        }
    }

    #[tokio::test]
    async fn extract_document_records_file_path_in_metadata() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        writeln!(tmp, "doc content").unwrap();

        let extractor = DefaultExtractor;
        let source = MediaSource::File {
            path: tmp.path().to_path_buf(),
        };
        let result = extractor
            .extract(source.clone(), AttachmentKind::Document)
            .await
            .unwrap();
        assert!(result.metadata.contains_key("file_path"));
    }

    #[tokio::test]
    async fn guess_audio_format_by_extension() {
        let wav = MediaSource::File {
            path: "/audio/clip.wav".into(),
        };
        assert_eq!(guess_audio_format(&wav), AudioFormat::Wav);

        let mp3 = MediaSource::File {
            path: "/audio/clip.mp3".into(),
        };
        assert_eq!(guess_audio_format(&mp3), AudioFormat::Mp3);
    }

    #[tokio::test]
    async fn extract_document_url_fetches_content() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/doc.txt"))
            .respond_with(
                ResponseTemplate::new(200)
                    .append_header("content-type", "text/plain")
                    .set_body_string("fetched document content"),
            )
            .mount(&server)
            .await;

        let url = format!("{}/doc.txt", server.uri());
        let extractor = DefaultExtractor;
        let source = MediaSource::Url { url };
        let result = extractor
            .extract(source, AttachmentKind::Document)
            .await
            .unwrap();
        if let ContentPart::Text(t) = &result.parts[0] {
            assert!(t.text.contains("fetched document content"));
        } else {
            panic!("expected Text part");
        }
    }
}
