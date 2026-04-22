# ufox-llm

面向生产环境的多 Provider `LLM Rust SDK`。

`ufox-llm` 目前提供统一的消息模型、工具调用模型、响应模型，以及面向 `OpenAI`、`Qwen` 和兼容 `OpenAI Chat Completions` 协议服务的客户端实现。调用方可以使用一致的 `Client` API 发送非流式或流式请求，而底层协议转换由各 Provider 适配器负责处理。

## 特性列表

- 统一客户端入口：使用同一套 `Client` / `ClientBuilder` API 接入多个 Provider
- Typestate 构建器：缺少 `api_key` 或 `provider` 时，`build()` 在编译期不可用
- 多 Provider 支持：内置 `OpenAI`、`Qwen`、`Compatible`
- 流式输出支持：基于 `futures::Stream` 暴露统一的 `ChatStream`
- 多模态消息支持：统一建模文本、远程图片 URL 和本地图片文件
- 工具调用支持：统一建模 `Tool`、`ToolCall`、`ToolResult`
- 中文注释与错误信息：面向中文团队项目直接使用
- 零警告要求：当前代码通过 `cargo build`、`cargo clippy`、`cargo test`

## 安装说明

### 添加依赖

```toml
[dependencies]
ufox-llm = "0.1.0"
tokio = { version = "1", features = ["full"] }
```

### TLS Feature

默认启用：

- `openai`
- `qwen`
- `compatible`
- `rustls-tls`

如果你想改用系统 `native-tls`，可以关闭默认特性后手动开启：

```toml
[dependencies]
ufox-llm = { version = "0.1.0", default-features = false, features = ["openai", "qwen", "compatible", "native-tls"] }
tokio = { version = "1", features = ["full"] }
```

## 快速上手

### 基础聊天

```rust
use ufox_llm::{ChatRequest, Client, Message, Provider};

#[tokio::main]
async fn main() -> Result<(), ufox_llm::LlmError> {
    let client = Client::builder()
        .provider(Provider::OpenAI)
        .api_key("sk-xxx")
        .model("gpt-4o")
        .build()?;

    let messages = vec![
        Message::system("你是一位简洁且专业的中文助手。"),
        Message::user("请用一句话介绍 Rust。"),
    ];

    let request = ChatRequest::new(&messages).build();
    let response = client.chat(&request).await?;
    println!("{}", response.content);

    Ok(())
}
```

### 流式输出

```rust
use futures_util::StreamExt;
use ufox_llm::{ChatRequest, Client, Message, Provider};

#[tokio::main]
async fn main() -> Result<(), ufox_llm::LlmError> {
    let client = Client::builder()
        .provider(Provider::OpenAI)
        .api_key("sk-xxx")
        .model("gpt-4o")
        .build()?;

    let messages = vec![Message::user("请分三行介绍 Rust 的优势。")];
    let request = ChatRequest::new(&messages).build();
    let mut stream = client.chat_stream(&request).await?;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;

        if !chunk.delta.is_empty() {
            print!("{}", chunk.delta);
        }
    }

    Ok(())
}
```

### 思考模式

```rust
use futures_util::StreamExt;
use ufox_llm::{ChatRequest, Client, DeltaType, Message, Provider, ReasoningEffort};

#[tokio::main]
async fn main() -> Result<(), ufox_llm::LlmError> {
    let client = Client::builder()
        .provider(Provider::Qwen)
        .api_key("sk-xxx")
        .model("qwen3-max")
        .build()?;

    let messages = vec![Message::user("请分析这道题并给出结论。")];

    let request = ChatRequest::new(&messages)
        .thinking(true)
        .thinking_budget(8_000)
        .reasoning_effort(ReasoningEffort::High)
        .build();
    let response = client.chat(&request).await?;

    if let Some(thinking) = response.thinking_content.as_deref() {
        println!("=== 思考过程 ===\n{thinking}");
    }
    println!("=== 最终回复 ===\n{}", response.content);

    let stream_request = ChatRequest::new(&messages).thinking(true).build();
    let mut stream = client.chat_stream(&stream_request).await?;
    while let Some(chunk) = stream.next().await {
        match chunk?.delta_type() {
            DeltaType::Thinking(text) => print!("{text}"),
            DeltaType::Content(text) => print!("{text}"),
        }
    }

    Ok(())
}
```

说明：

- `Qwen3` 支持 `thinking(true)` 与 `thinking_budget(...)`
- `OpenAI` `o1` / `o3` 系列支持 `reasoning_effort(...)`
- `Compatible` 接入的 `deepseek-reasoner` 会解析 `reasoning_content`
- 对不支持思考模式的模型，SDK 会静默忽略相关参数，并在 `debug` 级别输出提示日志

### 多轮对话

SDK 不管理历史，调用方自己维护 `messages`：

```rust
use ufox_llm::{ChatRequest, Client, Message, Provider};

#[tokio::main]
async fn main() -> Result<(), ufox_llm::LlmError> {
    let client = Client::builder()
        .provider(Provider::OpenAI)
        .api_key("sk-xxx")
        .model("gpt-4o")
        .build()?;

    let mut messages = vec![
        Message::system("你是一位专业的代码审查助手。"),
        Message::user("请简要说明 Rust 所有权系统解决了什么问题。"),
    ];

    let first_request = ChatRequest::new(&messages).build();
    let first = client.chat(&first_request).await?;
    messages.push(Message::assistant(&first.content));

    messages.push(Message::user("请再补充两个它对并发编程的帮助点。"));
    let second_request = ChatRequest::new(&messages).build();
    let second = client.chat(&second_request).await?;
    messages.push(Message::assistant(&second.content));

    Ok(())
}
```

### 多模态消息

当前版本使用 `MessageBuilder` 构造多模态消息：

```rust
use ufox_llm::{ChatRequest, Client, Message, MessageBuilder, Provider};

#[tokio::main]
async fn main() -> Result<(), ufox_llm::LlmError> {
    let client = Client::builder()
        .provider(Provider::Qwen)
        .api_key("sk-xxx")
        .model("qwen-vl-max")
        .build()?;

    let message = MessageBuilder::user()
        .text("请描述这张图片的主要内容。")
        .image_url("https://example.com/photo.jpg")
        .build();

    let messages = vec![
        Message::system("你是一位专业的视觉理解助手。"),
        message,
    ];

    let request = ChatRequest::new(&messages).build();
    let response = client.chat(&request).await?;
    println!("{}", response.content);

    Ok(())
}
```

也可以直接追加本地图片文件：

```rust
use ufox_llm::MessageBuilder;

fn main() {
    let message = MessageBuilder::user()
        .text("请描述这张图片。")
        .image_file("./local.png")
        .build();

    let _ = message;
}
```

### 工具调用

```rust
use serde_json::json;
use ufox_llm::{ChatRequest, Client, JsonType, Message, Provider, Tool, ToolChoice, ToolResult};

#[tokio::main]
async fn main() -> Result<(), ufox_llm::LlmError> {
    let client = Client::builder()
        .provider(Provider::OpenAI)
        .api_key("sk-xxx")
        .model("gpt-4o")
        .build()?;

    let tools = [Tool::function("get_weather")
        .description("获取城市实时天气")
        .param("city", JsonType::String, "城市名称", true)
        .param(
            "unit",
            JsonType::Enum(vec!["celsius".to_string(), "fahrenheit".to_string()]),
            "温度单位",
            false,
        )
        .build()];

    let mut messages = vec![Message::user("请查询杭州天气。")];
    let request = ChatRequest::new(&messages)
        .tools(&tools)
        .tool_choice(ToolChoice::function("get_weather"))
        .parallel_tool_calls(true)
        .build();
    let response = client.chat(&request).await?;

    if let Some(calls) = response.tool_calls.as_ref() {
        let calls = calls.to_vec();
        messages.push(Message::assistant_with_tool_calls(&calls));

        for call in &calls {
            let args = call.arguments_json()?;
            let city = args["city"].as_str().unwrap_or("未知城市");

            let result = ToolResult::json(
                &call.id,
                json!({
                    "city": city,
                    "weather": "晴",
                    "temperature": 26,
                    "unit": "celsius"
                }),
            );

            messages.push(Message::tool_result(&call.id, &result.content));
        }

        let request = ChatRequest::new(&messages).build();
        let final_response = client.chat(&request).await?;
        println!("{}", final_response.content);
    }

    Ok(())
}
```

说明：

- 当前版本已经支持工具声明、工具调用解析、工具结果回填和继续追问模型
- `tool_choice(...)` 支持 `Auto`、`None`、`Required` 以及指定函数名
- `parallel_tool_calls(true)` 可让支持并行工具调用的 Provider 一次返回多个工具调用
- 若 Provider 支持工具调用，推荐按“`assistant_with_tool_calls` -> `tool_result` -> `ChatRequest::new(...)` -> `chat`”的顺序继续对话

## Provider 配置

### 能力矩阵

| 能力 | OpenAI | Qwen | Compatible |
|---|---|---|---|
| 基础聊天 | 支持 | 支持 | 支持 |
| 流式输出 | 支持 | 支持 | 支持 |
| 多模态输入 | 支持 | 支持 | 取决于上游 |
| 工具调用 | 支持 | 支持 | 取决于上游 |
| `thinking(true)` | `o1` / `o3` 等推理模型自动生效，普通模型忽略 | `Qwen3` 支持 | 当前内置识别 `deepseek-reasoner` 及其常见前缀别名 |
| `thinking_budget(...)` | 不支持，自动忽略 | `Qwen3` 支持 | 不支持，自动忽略 |
| `reasoning_effort(...)` | `o1` / `o3` 支持 | 不支持，自动忽略 | 不支持，自动忽略 |
| `tool_choice(...)` | 支持 | 支持 | 取决于上游 |
| `parallel_tool_calls(true)` | 支持 | 支持 | 取决于上游 |
| 默认 `base_url` | 有 | 有 | 无，必须显式设置 |

说明：

- `Compatible` 的实际能力由上游服务决定，SDK 只负责按 `OpenAI-compatible` 协议透传请求和解析响应
- 当前对 `Compatible` 思考模式的内置识别范围为 `deepseek-reasoner` 及其常见前缀别名；若你的网关使用了其它自定义模型名，可能需要额外适配
- 对不支持的能力，SDK 默认采用“静默忽略 + debug 日志提示”的策略，避免破坏跨 Provider 的可移植性

### OpenAI

```rust
use std::time::Duration;
use ufox_llm::{Client, Provider};

fn main() -> Result<(), ufox_llm::LlmError> {
    let client = Client::builder()
        .provider(Provider::OpenAI)
        .api_key("sk-xxx")
        .model("gpt-4o")
        .organization("org-xxx")
        .timeout(Duration::from_secs(30))
        .build()?;

    let _ = client;
    Ok(())
}
```

### Qwen

```rust
use ufox_llm::{Client, Provider};

fn main() -> Result<(), ufox_llm::LlmError> {
    let client = Client::builder()
        .provider(Provider::Qwen)
        .api_key("sk-xxx")
        .model("qwen-max")
        .build()?;

    let _ = client;
    Ok(())
}
```

说明：

- SDK 会在 `Qwen` 流式请求时自动补齐 `X-DashScope-SSE: enable`
- 只有在你确实需要自定义网关或链路透传头时，才需要调用 `.extra_header(...)`

### Compatible

```rust
use ufox_llm::{Client, Provider};

fn main() -> Result<(), ufox_llm::LlmError> {
    let client = Client::builder()
        .provider(Provider::Compatible)
        .base_url("https://api.deepseek.com/v1")
        .api_key("sk-xxx")
        .model("deepseek-chat")
        .build()?;

    let _ = client;
    Ok(())
}
```

说明：

- `Compatible` 没有默认 `base_url`
- 未设置 `.base_url(...)` 时，运行时会返回清晰错误

## 错误处理指南

SDK 使用统一错误类型 [`LlmError`] 暴露调用失败原因。

常见错误分支：

- `LlmError::AuthError`
  - 通常对应 `401`
  - 说明 API Key 无效、缺失或已过期
- `LlmError::RateLimitError { retry_after }`
  - 通常对应 `429`
  - 若 Provider 返回 `Retry-After`，SDK 会尽量解析为 `Duration`
- `LlmError::ApiError { status_code, message, provider }`
  - Provider 返回明确的业务失败
- `LlmError::NetworkError`
  - 网络请求发送失败或连接异常
- `LlmError::ParseError`
  - Provider 返回体无法按预期协议解析
- `LlmError::StreamError`
  - 流式响应解析失败
- `LlmError::UnsupportedFeature`
  - 当前 Provider 或当前 SDK 版本尚未支持某项能力

处理示例：

```rust
use ufox_llm::LlmError;

fn handle_result(some_result: Result<(), LlmError>) {
    match some_result {
        Ok(()) => println!("调用成功"),
        Err(LlmError::AuthError) => eprintln!("鉴权失败，请检查 API Key"),
        Err(LlmError::RateLimitError { retry_after }) => {
            eprintln!("请求频率超限，建议等待：{retry_after:?}");
        }
        Err(LlmError::ApiError {
            status_code,
            message,
            provider,
        }) => {
            eprintln!("调用 {provider} 失败，状态码 {status_code}：{message}");
        }
        Err(other) => eprintln!("其他错误：{other}"),
    }
}
```

## 示例程序

仓库内已提供以下示例：

- `examples/basic_chat.rs`
- `examples/streaming.rs`
- `examples/multi_turn.rs`
- `examples/multimodal.rs`
- `examples/tool_calling.rs`
- `examples/tool_options.rs`

运行示例前，建议先复制示例环境变量文件：

```bash
cp examples/.env.example .env
cargo run --example basic_chat
```

或手动设置环境变量：

```bash 
export UFOX_LLM_PROVIDER=compatible
export UFOX_LLM_BASE_URL=https://api.deepseek.com/v1
export UFOX_LLM_API_KEY=sk-xxx
export UFOX_LLM_MODEL=deepseek-chat
cargo run --example basic_chat
```

## 当前状态

当前版本已经完成：

- 统一类型系统
- `OpenAI` / `Qwen` / `Compatible` Provider 适配器
- 非流式与流式客户端
- 多模态消息建模
- 工具调用建模与解析
- 工具结果回填消息构造器
- 基础示例

后续如需继续增强，比较自然的方向包括：

- 工具结果回填消息辅助构造器
- 更完整的 `README` 使用矩阵与环境变量清单
- 更细粒度的 Provider 能力差异说明
