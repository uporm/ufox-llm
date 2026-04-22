//! 多轮对话示例。
//!
//! 该示例演示如何：
//! 1. 从环境变量读取 Provider、API Key、模型与可选基础地址；
//! 2. 构建一个 [`ufox_llm::Client`]；
//! 3. 由调用方自己维护 `messages` 历史；
//! 4. 连续发起两轮对话，并将上一轮回复追加到历史中。
//!
//! 这也是 SDK 的设计约定之一：历史管理由调用方控制，SDK 只负责按给定消息数组发起请求。
//!
//! 运行前请至少设置：
//! - `UFOX_LLM_PROVIDER`：`openai` / `qwen` / `compatible`
//! - `UFOX_LLM_API_KEY`：对应 Provider 的 API Key
//! - `UFOX_LLM_BASE_URL`：当 `provider=compatible` 时必须设置
//!
//! 可选环境变量：
//! - `UFOX_LLM_MODEL`

use std::env;

use anyhow::{Context, Result, bail};
use tracing_subscriber::EnvFilter;
use ufox_llm::{ChatRequest, Client, Message, Provider};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let _ = dotenvy::dotenv();
    let provider = read_provider()?;
    let api_key = required_env("UFOX_LLM_API_KEY")?;
    let model = env::var("UFOX_LLM_MODEL").unwrap_or_else(|_| default_model(provider).to_string());

    let mut builder = Client::builder()
        .provider(provider)
        .api_key(api_key)
        .model(model);

    if let Some(base_url) = optional_env("UFOX_LLM_BASE_URL") {
        builder = builder.base_url(base_url);
    }

    let client = builder.build().context("构建客户端失败")?;

    let mut messages = vec![
        Message::system("你是一位专业的代码审查助手。"),
        Message::user("请简要说明 Rust 所有权系统解决了什么问题。"),
    ];

    let first_request = ChatRequest::new(&messages).build();
    let first = client
        .chat(&first_request)
        .await
        .context("第一轮对话失败")?;
    println!("第一轮回复：\n{}\n", first.content);
    messages.push(Message::assistant(&first.content));

    messages.push(Message::user("请再补充两个它对并发编程的帮助点。"));

    let second_request = ChatRequest::new(&messages).build();
    let second = client
        .chat(&second_request)
        .await
        .context("第二轮对话失败")?;
    println!("第二轮回复：\n{}\n", second.content);
    messages.push(Message::assistant(&second.content));

    println!("当前消息历史条数：{}", messages.len());

    Ok(())
}

fn init_tracing() {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("ufox_llm=debug"));

    let _ = tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .try_init();
}

fn read_provider() -> Result<Provider> {
    let raw = env::var("UFOX_LLM_PROVIDER").unwrap_or_else(|_| "openai".to_string());

    match raw.trim().to_ascii_lowercase().as_str() {
        "openai" => Ok(Provider::OpenAI),
        "qwen" => Ok(Provider::Qwen),
        "compatible" => Ok(Provider::Compatible),
        other => bail!("不支持的 Provider：{other}，可选值为 openai、qwen、compatible"),
    }
}

fn required_env(key: &str) -> Result<String> {
    env::var(key).with_context(|| format!("缺少必需环境变量：{key}"))
}

fn optional_env(key: &str) -> Option<String> {
    env::var(key).ok().filter(|value| !value.trim().is_empty())
}

fn default_model(provider: Provider) -> &'static str {
    match provider {
        Provider::OpenAI => "gpt-4o",
        Provider::Qwen => "qwen-max",
        Provider::Compatible => "deepseek-chat",
    }
}
