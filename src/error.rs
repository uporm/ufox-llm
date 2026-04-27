#[derive(thiserror::Error, Debug)]
pub enum LlmError {
    /// 本地文件系统读写失败。
    #[error("I/O 错误（{action}）：{source}")]
    Io {
        action: &'static str,
        #[source]
        source: std::io::Error,
    },

    #[error("缺少必填配置项：{field}")]
    MissingConfig { field: &'static str },

    #[error("配置不合法：{message}")]
    InvalidConfig { message: String },

    #[error("HTTP 状态错误 [{provider}]：status={status}，body={body}")]
    HttpStatus {
        provider: String,
        status: u16,
        body: String,
    },

    #[error("Provider 响应错误 [{provider}]：code={code:?}，message={message}")]
    ProviderResponse {
        provider: String,
        code: Option<String>,
        message: String,
    },

    #[error("认证错误：{message}")]
    Authentication { message: String },

    #[error("触发限流：retry_after={retry_after_secs:?}s")]
    RateLimit { retry_after_secs: Option<u64> },

    #[error("请求超时（{stage}，{timeout_ms}ms）：{message}")]
    RequestTimeout {
        stage: &'static str,
        timeout_ms: u64,
        message: String,
    },

    #[error("网络传输错误（{stage}）：{message}；底层错误：{source}")]
    Transport {
        stage: &'static str,
        message: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("JSON 编解码错误：{0}")]
    JsonCodec(#[from] serde_json::Error),

    #[error("流式协议错误 [{provider}]：{message}")]
    StreamProtocol { provider: String, message: String },

    #[error("工具协议错误：{message}")]
    ToolProtocol { message: String },

    #[error("Provider [{provider:?}] 不支持该能力：{capability}")]
    UnsupportedCapability {
        provider: Option<String>,
        capability: String,
    },

    #[error("多模态输入错误：{message}")]
    MediaInput { message: String },
}

impl LlmError {
    pub(crate) fn io(action: &'static str, source: std::io::Error) -> Self {
        Self::Io { action, source }
    }

    pub(crate) fn request_timeout(
        stage: &'static str,
        timeout_ms: u64,
        message: impl Into<String>,
    ) -> Self {
        Self::RequestTimeout {
            stage,
            timeout_ms,
            message: message.into(),
        }
    }

    pub(crate) fn transport(stage: &'static str, source: reqwest::Error) -> Self {
        let message = if source.is_connect() {
            "无法建立连接，请检查 base_url、网络连通性、代理或 TLS 配置".into()
        } else if source.is_timeout() {
            "请求在超时窗口内未完成，请检查网络状况或适当调大超时配置".into()
        } else if source.is_decode() {
            "响应体读取或解码失败，请检查 provider 是否返回了预期格式的数据".into()
        } else if source.is_body() {
            "响应体读取失败，连接可能已被服务端中断或网络不稳定".into()
        } else if source.is_request() {
            "请求构造或发送失败，请检查 URL、请求体和底层 HTTP 配置".into()
        } else {
            "底层 HTTP 传输失败".into()
        };

        Self::Transport {
            stage,
            message,
            source,
        }
    }
}

impl From<reqwest::Error> for LlmError {
    fn from(source: reqwest::Error) -> Self {
        Self::transport("处理 HTTP 响应", source)
    }
}

#[cfg(test)]
mod tests {
    use super::LlmError;

    #[test]
    fn request_timeout_keeps_stage_and_message() {
        let error = LlmError::request_timeout("读取流式响应", 60_000, "60 秒内未收到新数据");

        assert!(matches!(
            error,
            LlmError::RequestTimeout {
                stage,
                timeout_ms: 60_000,
                ref message,
            } if stage == "读取流式响应" && message == "60 秒内未收到新数据"
        ));
    }
}
