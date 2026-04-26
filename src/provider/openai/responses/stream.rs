//! Responses API 流式状态机。
//!
//! 实现基于 `stream::unfold` 的 SSE 流式处理，将字节流转换为
//! [`ChatChunk`] 序列。与 Chat Completions 流式的主要区别：
//!
//! - 事件以 `type` 字段区分（`response.output_text.delta`、
//!   `response.completed` 等），而非 `choices[0].delta`
//! - `response.completed` / `response.incomplete` 事件携带完整响应快照，
//!   状态机在此处结束并确保 tool calls / finish_reason / usage 被 yield
//! - 需要跟踪 `saw_text_delta` / `saw_thinking_delta` 以避免在完整快照中
//!   重复输出已通过 delta 事件发送过的内容
//!
//! 本模块作为 `responses` 模块的子模块，`impl ResponsesAdapter`
//! 块可跨文件分布，Rust 允许在同一 crate 内的任意文件中为同一 struct 添加 impl。

use std::{collections::VecDeque, pin::Pin};

use futures::{Stream, StreamExt, stream};

use crate::{
    error::LlmError,
    types::{request::ChatRequest, response::ChatChunk},
};

use super::super::{
    RESPONSES_PATH, ChatChunkStream,
    http::{OpenAiRequestBuilder, map_stream_read_error, parse_sse_data, send_request, take_sse_event},
};
use super::ResponsesAdapter;

// ── StreamState —— unfold 状态容器 ────────────────────────────────────────────

/// `stream::unfold` 的携带状态。
///
/// 字段说明：
/// - `source` — 原始 HTTP 字节流
/// - `buffer` — 字节缓冲，存放尚未凑成完整 SSE 帧的字节
/// - `pending` — 已解析但尚未 yield 的 chunk 队列
/// - `started_at` — 请求发起时间，用于日志计时
/// - `first_chunk_logged` — 首个 chunk 延迟是否已记录
/// - `saw_text_delta` — 是否已通过 delta 事件发送过文本内容
///   （若是，完整快照中的文本不再重复 yield）
/// - `saw_thinking_delta` — 是否已通过 delta 事件发送过思考链内容
/// - `done` — 是否已收到结束事件或遇到不可恢复错误
pub(super) struct StreamState {
    pub(super) source:
        Pin<Box<dyn Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>>,
    pub(super) buffer: Vec<u8>,
    pub(super) pending: VecDeque<Result<ChatChunk, LlmError>>,
    pub(super) started_at: std::time::Instant,
    pub(super) first_chunk_logged: bool,
    pub(super) saw_text_delta: bool,
    pub(super) saw_thinking_delta: bool,
    pub(super) done: bool,
}

// ── impl ResponsesAdapter（流式方法）─────────────────────────────────────────

impl ResponsesAdapter {
    /// 建立 Responses API 流式连接并返回 [`ChatChunkStream`]。
    ///
    /// 内部使用 `stream::unfold` 驱动 [`StreamState`] 状态机：
    /// 每次 `poll` 时从 `pending` 队列取出已解析的 chunk，
    /// 队列为空时读取下一批字节，分割 SSE 帧并通过 `parse_stream_event` 解析。
    ///
    /// 当收到 `response.completed` 或 `response.incomplete` 事件时，
    /// 状态机将 `done` 置为 `true` 并不再读取后续字节。
    pub(super) async fn execute_chat_stream(
        &self,
        model: &str,
        req: ChatRequest,
    ) -> Result<ChatChunkStream, LlmError> {
        let request_started_at = std::time::Instant::now();
        let message_count = req.messages.len();
        let tool_count = req.tools.len();
        tracing::info!(
            provider = self.http_context.provider_name(),
            model,
            message_count,
            tool_count,
            stream = true,
            "开始建立 responses_stream"
        );
        let body = self.build_request_body(model, &req, true).await?;
        let response = send_request(self, self.post_json(RESPONSES_PATH).json(&body)).await?;

        let provider_name = self.http_context.provider_name();
        let read_timeout_ms = self.http_context.transport().read_timeout_ms();
        tracing::info!(
            provider = provider_name,
            model,
            elapsed_ms = request_started_at.elapsed().as_millis() as u64,
            "responses_stream 已建立，等待首个 chunk"
        );

        let stream = stream::unfold(
            StreamState {
                source: Box::pin(response.bytes_stream()),
                buffer: Vec::new(),
                pending: VecDeque::new(),
                started_at: request_started_at,
                first_chunk_logged: false,
                saw_text_delta: false,
                saw_thinking_delta: false,
                done: false,
            },
            move |mut state| async move {
                loop {
                    // 优先消费已解析的 chunk
                    if let Some(item) = state.pending.pop_front() {
                        if let Ok(chunk) = &item
                            && !state.first_chunk_logged
                        {
                            state.first_chunk_logged = true;
                            tracing::info!(
                                provider = provider_name,
                                first_chunk_latency_ms =
                                    state.started_at.elapsed().as_millis() as u64,
                                has_text = chunk.text_delta.is_some(),
                                has_thinking = chunk.thinking_delta.is_some(),
                                tool_call_count = chunk.tool_calls.len(),
                                finish_reason = ?chunk.finish_reason,
                                "responses_stream 收到首个 chunk"
                            );
                        }
                        return Some((item, state));
                    }
                    if state.done {
                        tracing::info!(
                            provider = provider_name,
                            total_elapsed_ms = state.started_at.elapsed().as_millis() as u64,
                            saw_first_chunk = state.first_chunk_logged,
                            "responses_stream 结束"
                        );
                        return None;
                    }

                    // 读取下一批字节
                    match state.source.next().await {
                        Some(Ok(bytes)) => {
                            state.buffer.extend_from_slice(&bytes);
                            // 尽可能多地分割出完整 SSE 帧
                            while let Some(event) = take_sse_event(&mut state.buffer) {
                                match parse_sse_data(&event, provider_name) {
                                    Ok(Some(data)) if data == "[DONE]" => {
                                        state.done = true;
                                        break;
                                    }
                                    Ok(Some(data)) => {
                                        match serde_json::from_str::<serde_json::Value>(&data) {
                                            Ok(raw) => {
                                                match ResponsesAdapter::parse_stream_event(
                                                    provider_name,
                                                    &raw,
                                                    &mut state.saw_text_delta,
                                                    &mut state.saw_thinking_delta,
                                                ) {
                                                    Ok(chunks) => {
                                                        // response.completed / incomplete 是终止事件
                                                        if matches!(
                                                            raw.get("type").and_then(|value| value.as_str()),
                                                            Some("response.completed" | "response.incomplete")
                                                        ) {
                                                            state.done = true;
                                                        }
                                                        state.pending.extend(chunks.into_iter().map(Ok));
                                                    }
                                                    Err(err) => {
                                                        state.done = true;
                                                        state.pending.push_back(Err(err));
                                                        break;
                                                    }
                                                }
                                            }
                                            Err(err) => {
                                                state.done = true;
                                                state.pending.push_back(Err(
                                                    LlmError::StreamProtocol {
                                                        provider: provider_name.to_owned(),
                                                        message: format!(
                                                            "stream json 解析失败: {err}"
                                                        ),
                                                    },
                                                ));
                                                break;
                                            }
                                        }
                                    }
                                    Ok(None) => {} // 非 data 帧，跳过
                                    Err(err) => {
                                        state.done = true;
                                        state.pending.push_back(Err(err));
                                        break;
                                    }
                                }
                            }
                        }
                        Some(Err(err)) => {
                            state.done = true;
                            return Some((
                                Err(map_stream_read_error(read_timeout_ms, err)),
                                state,
                            ));
                        }
                        None => return None,
                    }
                }
            },
        );

        Ok(Box::pin(stream))
    }
}
