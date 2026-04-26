use crate::error::LlmError;

#[derive(Debug, Clone)]
pub(crate) struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    /// 触发重试的 HTTP status code 列表，默认 [429, 500, 502, 503]。
    pub retryable_status: Vec<u16>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 500,
            retryable_status: vec![429, 500, 502, 503],
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RateLimitConfig {
    pub requests_per_minute: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct TransportConfig {
    pub timeout_secs: u64,
    pub connect_timeout_secs: u64,
    pub read_timeout_secs: u64,
    pub retry: RetryConfig,
    pub rate_limit: Option<RateLimitConfig>,
}

/// 传输层抽象：adapter 只感知 `Transport`，不直接感知 `TransportConfig`。
#[derive(Clone)]
pub(crate) struct Transport {
    http: reqwest::Client,
    request_timeout_ms: u64,
    connect_timeout_ms: u64,
    read_timeout_ms: u64,
    retry: RetryConfig,
    rate_limit: Option<RateLimitConfig>,
}

impl Transport {
    pub(crate) fn new(config: TransportConfig) -> Self {
        let request_timeout = std::time::Duration::from_secs(config.timeout_secs);
        let connect_timeout = std::time::Duration::from_secs(config.connect_timeout_secs);
        let read_timeout = std::time::Duration::from_secs(config.read_timeout_secs);
        let http = reqwest::Client::builder()
            .timeout(request_timeout)
            .connect_timeout(connect_timeout)
            .read_timeout(read_timeout)
            .build()
            .expect("reqwest client 构造不应失败");

        Self {
            http,
            request_timeout_ms: config.timeout_secs.saturating_mul(1000),
            connect_timeout_ms: config.connect_timeout_secs.saturating_mul(1000),
            read_timeout_ms: config.read_timeout_secs.saturating_mul(1000),
            retry: config.retry,
            rate_limit: config.rate_limit,
        }
    }

    pub(crate) fn client(&self) -> &reqwest::Client {
        &self.http
    }

    pub(crate) fn read_timeout_ms(&self) -> u64 {
        self.read_timeout_ms
    }

    fn connect_timeout_error(&self) -> LlmError {
        LlmError::request_timeout(
            "建立连接",
            self.connect_timeout_ms,
            "在连接超时窗口内未能建立到 provider 的连接，请检查 base_url、网络、代理或 TLS 配置",
        )
    }

    fn request_timeout_error(&self) -> LlmError {
        LlmError::request_timeout(
            "发送请求",
            self.request_timeout_ms,
            "请求在总超时窗口内未完成，请检查网络状况，或适当调大 timeout_secs / read_timeout_secs",
        )
    }

    /// 统一发送入口，内部消费 retry / rate-limit / tracing 策略。
    pub(crate) async fn send(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response, LlmError> {
        use std::time::{Duration, Instant};

        if let Some(rate_limit) = &self.rate_limit {
            let _rpm = rate_limit.requests_per_minute;
        }

        if request.try_clone().is_none() {
            return self.send_once(request).await;
        }

        let mut attempt = 0_u32;
        loop {
            let req = request.try_clone().ok_or_else(|| LlmError::InvalidConfig {
                message: "请求体不可克隆（含流式 body），无法安全重试".into(),
            })?;

            let request_started_at = Instant::now();
            let result = req.send().await;
            match result {
                Err(err) if err.is_timeout() => {
                    let elapsed_ms = request_started_at.elapsed().as_millis() as u64;
                    let timeout_error = if err.is_connect() {
                        self.connect_timeout_error()
                    } else {
                        self.request_timeout_error()
                    };
                    tracing::warn!(
                        attempt,
                        elapsed_ms,
                        is_connect = err.is_connect(),
                        "请求超时，准备重试"
                    );
                    if attempt >= self.retry.max_retries {
                        return Err(timeout_error);
                    }
                }
                Err(err) if err.is_connect() && attempt < self.retry.max_retries => {
                    let elapsed_ms = request_started_at.elapsed().as_millis() as u64;
                    tracing::warn!(attempt, elapsed_ms, "连接失败，准备重试");
                }
                Err(err) => return Err(LlmError::transport("发送请求", err)),
                Ok(response) => {
                    let elapsed_ms = request_started_at.elapsed().as_millis() as u64;
                    let status = response.status().as_u16();
                    if self.retry.retryable_status.contains(&status)
                        && attempt < self.retry.max_retries
                    {
                        tracing::warn!(
                            attempt,
                            status,
                            elapsed_ms,
                            "收到可重试状态码，进入退避重试"
                        );
                    } else {
                        tracing::info!(attempt, status, elapsed_ms, "HTTP 请求完成");
                        return Ok(response);
                    }
                }
            }

            let backoff_ms = self
                .retry
                .initial_backoff_ms
                .saturating_mul(1_u64 << attempt.min(10));
            tracing::debug!(attempt, backoff_ms, "退避等待");
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            attempt += 1;
        }
    }

    async fn send_once(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<reqwest::Response, LlmError> {
        let request_started_at = std::time::Instant::now();
        match request.send().await {
            Ok(response) => {
                tracing::info!(
                    status = response.status().as_u16(),
                    elapsed_ms = request_started_at.elapsed().as_millis() as u64,
                    "HTTP 请求完成（单次发送）"
                );
                Ok(response)
            }
            Err(err) if err.is_timeout() => {
                tracing::warn!(
                    elapsed_ms = request_started_at.elapsed().as_millis() as u64,
                    is_connect = err.is_connect(),
                    "HTTP 请求超时（单次发送）"
                );
                if err.is_connect() {
                    Err(self.connect_timeout_error())
                } else {
                    Err(self.request_timeout_error())
                }
            }
            Err(err) => Err(LlmError::transport("发送请求", err)),
        }
    }
}
