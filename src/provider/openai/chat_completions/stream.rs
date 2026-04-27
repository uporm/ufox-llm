//! Chat Completions 流式状态机。
//!
//! 实现基于 `stream::unfold` 的 SSE 流式处理，将字节流转换为
//! [`ChatChunk`] 序列。核心逻辑：
//!
//! 1. 从 HTTP 响应字节流中分割出 SSE 帧（[`take_sse_event`]）
//! 2. 提取 `data:` 行内容（[`parse_sse_data`]）
//! 3. 解析 JSON delta，累积 tool call 片段（[`PartialToolCall`]）
//! 4. 在 `finish_reason: tool_calls` 或 `[DONE]` 时最终化 tool calls
//!
//! 本模块作为 `chat_completions` 模块的子模块，`impl ChatCompletionsAdapter`
//! 块可跨文件分布，Rust 允许在同一 crate 内的任意文件中为同一 struct 添加 impl。

use std::{
    collections::{BTreeMap, VecDeque},
    pin::Pin,
};

use futures::{Stream, StreamExt, stream};

use crate::{
    error::LlmError,
    types::{
        content::ToolCall,
        request::ChatRequest,
        response::{ChatChunk, FinishReason, Usage},
    },
};

use super::super::{
    CHAT_COMPLETIONS_PATH, ChatChunkStream,
    http::{
        OpenAiRequestBuilder, map_stream_read_error, parse_finish_reason, parse_sse_data,
        parse_usage, send_request, take_sse_event,
    },
};
use super::ChatCompletionsAdapter;

// ── PartialToolCall —— 累积流式 tool call 片段 ────────────────────────────────

/// 在多个 SSE delta 中累积的不完整 tool call。
///
/// Chat Completions 流式协议将 tool call 分散在若干 delta 中传输：
/// 首帧携带 `id` 与函数名，后续帧追加 `arguments` 片段。
/// `PartialToolCall` 负责合并这些片段，在流结束时通过 [`finalize`] 生成完整 [`ToolCall`]。
///
/// [`finalize`]: PartialToolCall::finalize
#[derive(Default)]
pub(super) struct PartialToolCall {
    pub(super) id: Option<String>,
    pub(super) tool_name: Option<String>,
    /// 已收集的 arguments JSON 片段（待拼接后解析）。
    pub(super) arguments: String,
    /// 是否曾见到过 `arguments` 字段（区分"空字符串"与"字段缺失"）。
    pub(super) arguments_seen: bool,
}

impl PartialToolCall {
    /// 将累积状态最终化为完整 [`ToolCall`]。
    ///
    /// 任一必要字段（`id`、`tool_name`、`arguments`）缺失时返回错误。
    pub(super) fn finalize(self) -> Result<ToolCall, LlmError> {
        let id = self.id.ok_or_else(|| LlmError::ToolProtocol {
            message: "stream tool call 缺少 id".into(),
        })?;
        let tool_name = self.tool_name.ok_or_else(|| LlmError::ToolProtocol {
            message: "stream tool call 缺少 name".into(),
        })?;
        let arguments_raw = if self.arguments_seen {
            self.arguments.as_str()
        } else {
            return Err(LlmError::ToolProtocol {
                message: "stream tool call 缺少 arguments".into(),
            });
        };
        let arguments =
            serde_json::from_str(arguments_raw).map_err(|err| LlmError::ToolProtocol {
                message: format!("stream tool arguments 解析失败: {err}"),
            })?;

        Ok(ToolCall {
            id,
            tool_name,
            arguments,
        })
    }
}

// ── StreamState —— unfold 状态容器 ───────────────────────────────────────────

/// `stream::unfold` 的携带状态。
///
/// 字段说明：
/// - `source` — 原始 HTTP 字节流
/// - `buffer` — 字节缓冲，存放尚未凑成完整 SSE 帧的字节
/// - `pending` — 已解析但尚未 yield 的 chunk 队列
/// - `tool_calls` — 按索引存放的 [`PartialToolCall`]，key 为 delta 中的 `index`
/// - `started_at` — 请求发起时间，用于日志计时
/// - `first_chunk_logged` — 首个 chunk 延迟是否已记录
/// - `done` — 是否已收到 `[DONE]` 或遇到不可恢复错误
pub(super) struct StreamState {
    pub(super) source: Pin<Box<dyn Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send>>,
    pub(super) buffer: Vec<u8>,
    pub(super) pending: VecDeque<Result<ChatChunk, LlmError>>,
    pub(super) tool_calls: BTreeMap<usize, PartialToolCall>,
    pub(super) started_at: std::time::Instant,
    pub(super) first_chunk_logged: bool,
    pub(super) done: bool,
}

impl StreamState {
    /// 取出下一个待产出的 chunk，并在首次成功产出时记录状态。
    pub(super) fn pop_pending(&mut self) -> Option<Result<ChatChunk, LlmError>> {
        let item = self.pending.pop_front()?;
        if item.is_ok() {
            self.first_chunk_logged = true;
        }
        Some(item)
    }

    /// 将流标记为结束。
    pub(super) fn finish(&mut self) {
        self.done = true;
    }

    /// 将最终化后的 tool calls 追加到待产出队列。
    pub(super) fn enqueue_tool_calls(&mut self, tool_calls: Vec<ToolCall>) {
        if tool_calls.is_empty() {
            return;
        }

        self.pending.push_back(Ok(ChatChunk {
            tool_calls,
            ..ChatChunk::default()
        }));
    }
}

// ── impl ChatCompletionsAdapter（流式方法）───────────────────────────────────
impl ChatCompletionsAdapter {
    /// 构造仅包含思考增量的 chunk。
    fn thinking_chunk(thinking: &str) -> ChatChunk {
        ChatChunk {
            thinking_delta: Some(thinking.to_owned()),
            ..ChatChunk::default()
        }
    }

    /// 构造仅包含文本增量的 chunk。
    fn text_chunk(content: &str) -> ChatChunk {
        ChatChunk {
            text_delta: Some(content.to_owned()),
            ..ChatChunk::default()
        }
    }

    /// 构造包含 tool calls 的 chunk，并可附加结束元数据。
    fn tool_calls_chunk(
        tool_calls: Vec<ToolCall>,
        finish_reason: Option<FinishReason>,
        usage: Option<Usage>,
    ) -> ChatChunk {
        ChatChunk {
            tool_calls,
            finish_reason,
            usage,
            ..ChatChunk::default()
        }
    }

    /// 构造仅包含结束元数据的 chunk。
    fn terminal_chunk(finish_reason: Option<FinishReason>, usage: Option<Usage>) -> ChatChunk {
        ChatChunk {
            finish_reason,
            usage,
            ..ChatChunk::default()
        }
    }

    /// 将 `partials` 中所有 [`PartialToolCall`] 最终化并清空。
    pub(super) fn drain_partial_tool_calls(
        partials: &mut BTreeMap<usize, PartialToolCall>,
    ) -> Result<Vec<ToolCall>, LlmError> {
        let mut calls = Vec::with_capacity(partials.len());
        for (_, partial) in std::mem::take(partials) {
            calls.push(partial.finalize()?);
        }
        Ok(calls)
    }

    /// 将单个 `tool_call` delta 合并到对应的 partial。
    fn merge_tool_call_delta(
        item: &serde_json::Value,
        partials: &mut BTreeMap<usize, PartialToolCall>,
    ) -> Result<(), LlmError> {
        let index =
            item.get("index")
                .and_then(|value| value.as_u64())
                .ok_or_else(|| LlmError::ToolProtocol {
                    message: "stream tool call 缺少 index".into(),
                })? as usize;
        let entry = partials.entry(index).or_default();

        if let Some(id) = item.get("id").and_then(|value| value.as_str()) {
            entry.id = Some(id.to_owned());
        }

        let Some(function) = item.get("function").and_then(|value| value.as_object()) else {
            return Ok(());
        };

        if let Some(name) = function.get("name").and_then(|value| value.as_str()) {
            match &mut entry.tool_name {
                Some(existing) => existing.push_str(name),
                None => entry.tool_name = Some(name.to_owned()),
            }
        }

        if let Some(arguments) = function.get("arguments") {
            let arguments = arguments
                .as_str()
                .ok_or_else(|| LlmError::ToolProtocol {
                    message: "stream tool call arguments 不是字符串".into(),
                })?;
            entry.arguments_seen = true;
            entry.arguments.push_str(arguments);
        }

        Ok(())
    }

    /// 将结束原因和 usage 尽量附着到最后一个 chunk，避免额外生成空 chunk。
    fn attach_terminal_metadata(
        chunks: &mut Vec<ChatChunk>,
        finish_reason: Option<FinishReason>,
        usage: Option<Usage>,
    ) {
        if finish_reason.is_none() && usage.is_none() {
            return;
        }

        if let Some(last) = chunks.last_mut()
            && last.finish_reason.is_none()
            && last.usage.is_none()
        {
            last.finish_reason = finish_reason;
            last.usage = usage;
            return;
        }

        chunks.push(Self::terminal_chunk(finish_reason, usage));
    }

    /// 将尚未完成的 partial tool calls 最终化。
    fn finalize_pending_tool_calls(
        partials: &mut BTreeMap<usize, PartialToolCall>,
    ) -> Result<Option<Vec<ToolCall>>, LlmError> {
        let tool_calls = Self::drain_partial_tool_calls(partials)?;
        if tool_calls.is_empty() {
            return Ok(None);
        }

        Ok(Some(tool_calls))
    }

    /// 处理单条 `data:` 负载，并将解析出的 chunk 追加到 `state.pending`。
    fn handle_stream_data(
        provider_name: &str,
        data: &str,
        state: &mut StreamState,
    ) -> Result<(), LlmError> {
        if data == "[DONE]" {
            state.finish();
            if let Some(tool_calls) = Self::finalize_pending_tool_calls(&mut state.tool_calls)? {
                state.enqueue_tool_calls(tool_calls);
            }
            return Ok(());
        }

        let raw = serde_json::from_str::<serde_json::Value>(data).map_err(|err| {
            LlmError::StreamProtocol {
                provider: provider_name.to_owned(),
                message: format!("stream json 解析失败: {err}"),
            }
        })?;

        let chunks = Self::parse_stream_event(provider_name, &raw, &mut state.tool_calls)?;
        state.pending.extend(chunks.into_iter().map(Ok));
        Ok(())
    }

    /// 扫描缓冲区中已完整到达的 SSE 事件。
    ///
    /// 返回 `true` 表示流已经结束或遇到错误，外层应停止继续消费本批字节。
    fn process_buffered_events(provider_name: &str, state: &mut StreamState) -> bool {
        while let Some(event) = take_sse_event(&mut state.buffer) {
            match parse_sse_data(&event, provider_name) {
                Ok(Some(data)) => {
                    if let Err(err) = Self::handle_stream_data(provider_name, &data, state) {
                        state.finish();
                        state.pending.push_back(Err(err));
                        return true;
                    }
                    if state.done {
                        return true;
                    }
                }
                Ok(None) => {}
                Err(err) => {
                    state.finish();
                    state.pending.push_back(Err(err));
                    return true;
                }
            }
        }

        false
    }

    /// 解析单个 SSE JSON 事件，返回零个或多个 [`ChatChunk`]。
    ///
    /// 处理逻辑：
    /// - 提取 `delta.content` 文本片段 → `text_delta` chunk
    /// - 提取 `delta.tool_calls` 更新到 `partials`（尚不 yield）
    /// - 若 `finish_reason == tool_calls`，立即最终化所有 partials → tool_calls chunk
    /// - 其余 finish_reason / usage 附加到最后一个 chunk，或单独生成空 chunk
    pub(super) fn parse_stream_event(
        provider_name: &str,
        raw: &serde_json::Value,
        partials: &mut BTreeMap<usize, PartialToolCall>,
    ) -> Result<Vec<ChatChunk>, LlmError> {
        let choice = raw
            .get("choices")
            .and_then(|value| value.as_array())
            .and_then(|choices| choices.first())
            .ok_or_else(|| LlmError::StreamProtocol {
                provider: provider_name.to_owned(),
                message: "缺少 choices[0]".into(),
            })?;

        let mut chunks = Vec::new();
        let delta = choice.get("delta").and_then(|value| value.as_object());

        // 思考链片段：DeepSeek/Qwen 使用 `reasoning_content`，vLLM >= 0.9 使用 `reasoning`。
        // 优先取 `reasoning_content`，回退到 `reasoning`。
        // 思考链先于正文到达（provider 在思考阶段 content 为 null），先 emit。
        if let Some(thinking) = delta
            .and_then(|value| {
                value
                    .get("reasoning_content")
                    .or_else(|| value.get("reasoning"))
            })
            .and_then(|value| value.as_str())
            .filter(|s| !s.is_empty())
        {
            chunks.push(Self::thinking_chunk(thinking));
        }

        if let Some(content) = delta
            .and_then(|value| value.get("content"))
            .and_then(|value| value.as_str())
            .filter(|content| !content.is_empty())
        {
            chunks.push(Self::text_chunk(content));
        }

        if let Some(items) = delta
            .and_then(|value| value.get("tool_calls"))
            .and_then(|value| value.as_array())
        {
            for item in items {
                Self::merge_tool_call_delta(item, partials)?;
            }
        }

        let finish_reason =
            parse_finish_reason(choice.get("finish_reason").and_then(|value| value.as_str()));
        let usage = parse_usage(raw.get("usage"));

        if matches!(finish_reason, Some(FinishReason::ToolCalls)) {
            chunks.push(Self::tool_calls_chunk(
                Self::drain_partial_tool_calls(partials)?,
                finish_reason,
                usage,
            ));
        } else {
            Self::attach_terminal_metadata(&mut chunks, finish_reason, usage);
        }

        Ok(chunks)
    }

    /// 建立流式连接并返回 [`ChatChunkStream`]。
    ///
    /// 内部使用 `stream::unfold` 驱动 [`StreamState`] 状态机：
    /// 每次 `poll` 时从 `pending` 队列取出已解析的 chunk，
    /// 队列为空时读取下一批字节，分割 SSE 帧并解析。
    pub(super) async fn execute_chat_stream(
        &self,
        model: &str,
        req: ChatRequest,
    ) -> Result<ChatChunkStream, LlmError> {
        let request_started_at = std::time::Instant::now();
        let body = self.to_request_body(model, &req, true).await?;
        let response =
            send_request(self, self.post_json(CHAT_COMPLETIONS_PATH).json(&body)).await?;

        let provider_name = self.http_context.provider_name();
        let read_timeout_ms = self.http_context.transport().read_timeout_ms();
        tracing::debug!(
            provider = provider_name,
            model,
            elapsed_ms = request_started_at.elapsed().as_millis() as u64,
            "chat_stream 已建立，等待首个 chunk"
        );

        let stream = stream::unfold(
            StreamState {
                source: Box::pin(response.bytes_stream()),
                buffer: Vec::new(),
                pending: VecDeque::new(),
                tool_calls: BTreeMap::new(),
                started_at: request_started_at,
                first_chunk_logged: false,
                done: false,
            },
            move |mut state| async move {
                loop {
                    if let Some(item) = state.pop_pending() {
                        return Some((item, state));
                    }
                    if state.done {
                        tracing::debug!(
                            provider = provider_name,
                            total_elapsed_ms = state.started_at.elapsed().as_millis() as u64,
                            saw_first_chunk = state.first_chunk_logged,
                            "chat_stream 结束"
                        );
                        return None;
                    }

                    match state.source.next().await {
                        Some(Ok(bytes)) => {
                            state.buffer.extend_from_slice(&bytes);
                            Self::process_buffered_events(provider_name, &mut state);
                        }
                        Some(Err(err)) => {
                            state.finish();
                            return Some((Err(map_stream_read_error(read_timeout_ms, err)), state));
                        }
                        None => {
                            state.finish();
                            match Self::finalize_pending_tool_calls(&mut state.tool_calls) {
                                Ok(Some(tool_calls)) => {
                                    state.enqueue_tool_calls(tool_calls);
                                }
                                Ok(None) => return None,
                                Err(err) => return Some((Err(err), state)),
                            }
                        }
                    }
                }
            },
        );

        Ok(Box::pin(stream))
    }
}
