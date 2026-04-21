//! 工具调用示例。
//!
//! 该示例演示如何：
//! 1. 从环境变量读取 Provider、API Key、模型与可选基础地址；
//! 2. 声明一个函数型工具；
//! 3. 构造带工具定义的 `ChatRequest` 并发起请求；
//! 4. 解析模型返回的 `tool_calls`，并在本地分发执行；
//! 5. 将工具结果回填给模型，继续获取最终答复。
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
use serde_json::json;
use tracing_subscriber::EnvFilter;
use ufox_llm::{
    ChatRequest, Client, JsonType, Message, Provider, Tool, ToolCall, ToolChoice, ToolResult,
};

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
        .description("获取城市实时天气")
        .param("city", JsonType::String, "城市名称", true)
        .param(
            "unit",
            JsonType::Enum(vec!["celsius".to_string(), "fahrenheit".to_string()]),
            "温度单位",
            false,
        )
        .build()];
    let mut messages = vec![
        Message::system("你是一位会优先调用工具获取准确信息的助手。"),
        Message::user("请查询杭州当前天气，并告诉我温度。"),
    ];

    let request = ChatRequest::new(&messages)
        .tools(&tools)
        .tool_choice(ToolChoice::Auto)
        .parallel_tool_calls(true)
        .build();
    let response = client.chat(&request).await.context("工具调用请求失败")?;

    if let Some(calls) = response.tool_calls() {
        let calls = calls.to_vec();

        println!("模型请求调用 {} 个工具：", calls.len());
        messages.push(Message::assistant_with_tool_calls(&calls));

        for call in &calls {
            println!("\n工具名：{}", call.name());
            println!("原始参数：{}", call.arguments());

            let result = dispatch_tool(call).context("本地工具分发失败")?;
            println!("工具执行结果：{}", result.content());
            messages.push(Message::tool_result(call.id(), result.content()));
        }

        let request = ChatRequest::new(&messages).build();
        let final_response = client
            .chat(&request)
            .await
            .context("工具结果回填后继续请求失败")?;
        println!("\n最终回复：\n{}", final_response.content());
    } else {
        println!("模型未触发工具调用，直接回复：\n{}", response.content());
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

fn dispatch_tool(call: &ToolCall) -> Result<ToolResult> {
    match call.name() {
        "get_weather" => {
            let arguments = call.arguments_json().context("解析工具参数失败")?;
            let city = arguments
                .get("city")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("未知城市");
            let unit = arguments
                .get("unit")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("celsius");

            // 这里使用固定结果模拟真实工具执行，重点是演示 SDK 的工具调用链路。
            let result = match unit {
                "fahrenheit" => json!({
                    "city": city,
                    "weather": "晴",
                    "temperature": 78,
                    "unit": "fahrenheit"
                }),
                _ => json!({
                    "city": city,
                    "weather": "晴",
                    "temperature": 26,
                    "unit": "celsius"
                }),
            };

            Ok(ToolResult::json(call.id(), result))
        }
        other => bail!("未实现的工具：{other}"),
    }
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
