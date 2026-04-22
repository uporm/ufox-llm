//! 工具调用策略示例。
//!
//! 该示例演示如何：
//! 1. 从环境变量读取 Provider、API Key、模型与可选基础地址；
//! 2. 使用 `tool_choice()` 指定工具调用策略；
//! 3. 使用 `parallel_tool_calls(true)` 允许模型一次返回多个工具调用；
//! 4. 打印模型返回的工具调用请求，便于观察不同 Provider 的行为。
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
use ufox_llm::{ChatRequest, Client, JsonType, Message, Provider, Tool, ToolChoice};

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
    let tools = [Tool::function("get_weather")
        .description("获取指定城市的实时天气")
        .param("city", JsonType::String, "城市名称", true)
        .build()];
    let messages = vec![
        Message::system("你是一位会优先使用工具获取准确信息的助手。"),
        Message::user("请分别查询北京和上海的天气。"),
    ];

    let request = ChatRequest::new(&messages)
        .tools(&tools)
        .tool_choice(ToolChoice::Auto)
        .parallel_tool_calls(true)
        .build();
    let response = client
        .chat(&request)
        .await
        .context("发起工具调用请求失败")?;

    if let Some(calls) = response.tool_calls.as_ref() {
        println!("模型返回了 {} 个工具调用：", calls.len());
        for call in calls {
            println!(
                "- id={} name={} arguments={}",
                call.id,
                call.name,
                call.arguments
            );
        }
    } else {
        println!("模型未调用工具，直接回复：\n{}", response.content);
    }

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
