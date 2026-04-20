//! 多模态聊天示例。
//!
//! 该示例演示如何：
//! 1. 从环境变量读取 Provider、API Key、模型与可选基础地址；
//! 2. 构建一个 [`ufox_llm::Client`]；
//! 3. 使用 [`ufox_llm::MessageBuilder`] 构造包含文本与图片的多模态消息；
//! 4. 调用非流式 `chat()` 并打印模型回复。
//!
//! 运行前请至少设置：
//! - `UFOX_LLM_PROVIDER`：`openai` / `qwen` / `compatible`
//! - `UFOX_LLM_API_KEY`：对应 Provider 的 API Key
//! - `UFOX_LLM_BASE_URL`：当 `provider=compatible` 时必须设置
//!
//! 图片输入二选一即可：
//! - `UFOX_LLM_IMAGE_FILE`：本地图片路径，SDK 会自动读取并转成 `base64 data URL`
//! - `UFOX_LLM_IMAGE_URL`：远程图片 URL
//!
//! 可选环境变量：
//! - `UFOX_LLM_MODEL`

use std::env;

use anyhow::{Context, Result, bail};
use ufox_llm::{Client, Message, MessageBuilder, Provider};

#[tokio::main]
async fn main() -> Result<()> {
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
    let multimodal_message = build_multimodal_message()?;
    let messages = vec![
        Message::system("你是一位专业的视觉理解助手，请使用中文回答。"),
        multimodal_message,
    ];

    let response = client.chat(&messages).await.context("多模态请求失败")?;

    println!("模型回复：\n{}", response.content());

    Ok(())
}

fn build_multimodal_message() -> Result<Message> {
    let mut builder = MessageBuilder::user().text("请描述这张图片的主要内容，并给出一句简短总结。");

    if let Some(path) = optional_env("UFOX_LLM_IMAGE_FILE") {
        builder = builder.image_file(path);
    } else if let Some(url) = optional_env("UFOX_LLM_IMAGE_URL") {
        builder = builder.image_url(url);
    } else {
        bail!(
            "请至少设置一个图片输入：UFOX_LLM_IMAGE_FILE 或 UFOX_LLM_IMAGE_URL"
        );
    }

    Ok(builder.build())
}

fn read_provider() -> Result<Provider> {
    let raw = env::var("UFOX_LLM_PROVIDER").unwrap_or_else(|_| "openai".to_string());

    match raw.trim().to_ascii_lowercase().as_str() {
        "openai" => Ok(Provider::OpenAI),
        "qwen" => Ok(Provider::Qwen),
        "compatible" => Ok(Provider::Compatible),
        other => bail!(
            "不支持的 Provider：{other}，可选值为 openai、qwen、compatible"
        ),
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
        Provider::Qwen => "qwen-vl-max",
        Provider::Compatible => "deepseek-chat",
    }
}
