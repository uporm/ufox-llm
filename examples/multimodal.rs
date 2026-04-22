//! 多模态聊天示例。
//!
//! 该示例演示如何：
//! 1. 从环境变量读取 Provider、API Key、模型与可选基础地址；
//! 2. 构建一个 [`ufox_llm::Client`]；
//! 3. 使用 [`ufox_llm::MessageBuilder`] 构造包含文本与图片的多模态消息；
//! 4. 构造 `ChatRequest`，调用非流式 `chat()` 并打印模型回复。
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
use tracing_subscriber::EnvFilter;
use ufox_llm::{ChatRequest, Client, Message, MessageBuilder, Provider};

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let _ = dotenvy::dotenv();
    let provider = read_provider()?;
    let api_key = required_env("UFOX_LLM_API_KEY")?;
    let model = env::var("UFOX_LLM_MODEL").unwrap_or_else(|_| default_model(provider).to_string());
    let base_url = optional_env("UFOX_LLM_BASE_URL");
    let image_input = resolve_image_input()?;

    print_runtime_overview(provider, &model, base_url.as_deref(), &image_input);

    let mut builder = Client::builder()
        .provider(provider)
        .api_key(api_key)
        .model(model);

    if let Some(base_url) = base_url {
        builder = builder.base_url(base_url);
    }

    let client = builder.build().context("构建客户端失败")?;
    let multimodal_message = build_multimodal_message(&image_input);
    let messages = vec![
        Message::system("你是一位专业的视觉理解助手，请使用中文回答。"),
        multimodal_message,
    ];

    let request = ChatRequest::new(&messages).build();
    let response = client.chat(&request).await.context("多模态请求失败")?;

    println!("模型回复：\n{}", response.content);

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

fn build_multimodal_message(image_input: &ImageInput) -> Message {
    let mut builder = MessageBuilder::user().text("请描述这张图片的主要内容，并给出一句简短总结。");

    match image_input {
        ImageInput::File(path) => {
            builder = builder.image_file(path);
        }
        ImageInput::Url(url) => {
            builder = builder.image_url(url);
        }
    }

    builder.build()
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

#[derive(Debug)]
enum ImageInput {
    File(String),
    Url(String),
}

fn resolve_image_input() -> Result<ImageInput> {
    let file = optional_env("UFOX_LLM_IMAGE_FILE");
    let url = optional_env("UFOX_LLM_IMAGE_URL");

    match (file, url) {
        (Some(path), Some(backup_url)) => {
            println!(
                "提示：检测到同时设置了 UFOX_LLM_IMAGE_FILE 与 UFOX_LLM_IMAGE_URL，优先使用本地文件。URL 将被忽略：{}",
                backup_url
            );
            Ok(ImageInput::File(path))
        }
        (Some(path), None) => Ok(ImageInput::File(path)),
        (None, Some(url)) => Ok(ImageInput::Url(url)),
        (None, None) => bail!("请至少设置一个图片输入：UFOX_LLM_IMAGE_FILE 或 UFOX_LLM_IMAGE_URL"),
    }
}

fn print_runtime_overview(
    provider: Provider,
    model: &str,
    base_url: Option<&str>,
    image_input: &ImageInput,
) {
    println!("====== 多模态运行信息 ======");
    println!("Provider: {}", provider_name(provider));
    println!("Model: {}", model);
    println!("Base URL: {}", base_url.unwrap_or("<provider 默认地址>"));
    match image_input {
        ImageInput::File(path) => println!("Image Input: file ({path})"),
        ImageInput::Url(url) => println!("Image Input: url ({url})"),
    }

    if provider == Provider::Compatible && !looks_like_vision_model(model) {
        println!(
            "警告：当前使用 compatible + `{model}`。该模型名称看起来不像视觉模型，可能会忽略图片输入。"
        );
        println!("建议：切换到明确支持视觉的模型（名称通常包含 vl / vision / omni / gpt-4o 等）。");
    }
}

fn provider_name(provider: Provider) -> &'static str {
    match provider {
        Provider::OpenAI => "openai",
        Provider::Qwen => "qwen",
        Provider::Compatible => "compatible",
    }
}

fn looks_like_vision_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    ["vl", "vision", "omni", "gpt-4o", "gpt-4.1"]
        .iter()
        .any(|key| model.contains(key))
}

fn default_model(provider: Provider) -> &'static str {
    match provider {
        Provider::OpenAI => "gpt-4o",
        Provider::Qwen => "qwen-vl-max",
        Provider::Compatible => "deepseek-chat",
    }
}
