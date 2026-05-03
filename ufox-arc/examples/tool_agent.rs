/// 演示 Phase 3 工具系统：自定义工具 + 内置文件工具的 Agent 调用流程。
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{Value, json};
use ufox_arc::tools::{Tool, ToolError, ToolSpec};
use ufox_arc::{Agent, ArcError, FileReadTool};
use ufox_llm::{Client, ToolResultPayload};

/// 模拟天气查询工具（不发真实网络请求）。
struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn spec(&self) -> &ToolSpec {
        static META: std::sync::OnceLock<ToolSpec> = std::sync::OnceLock::new();
        META.get_or_init(|| ToolSpec {
            name: "get_weather".to_string(),
            description: "Get current weather for a city.".to_string(),
            parameters_schema: json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }),
            timeout: std::time::Duration::from_secs(5),
        })
    }

    async fn execute(&self, params: Value) -> Result<ToolResultPayload, ToolError> {
        let city = params["city"]
            .as_str()
            .ok_or_else(|| ToolError::InvalidParams {
                tool: "get_weather".to_string(),
                message: "city must be a string".to_string(),
            })?;
        // 模拟返回结果
        Ok(ToolResultPayload::text(format!(
            "{city}: 晴，23°C，东南风 3 级"
        )))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let llm = Client::from_env()?;

    let agent = Agent::builder()
        .llm(llm)
        .system("你是一个助手，可以查询天气和读取文件。请用中文回答。")
        .tool(WeatherTool)
        .tool(FileReadTool::new())
        .build()
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let mut session = agent
        .new_session("demo_user")
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;

    // 触发工具调用
    let result = session
        .chat("北京现在天气怎么样？")
        .await
        .map_err(|e: ArcError| anyhow::anyhow!("{e}"))?;

    println!("=== 回复 ===");
    println!("{}", result.response.text);

    println!("\n=== 执行轨迹 ===");
    for step in &result.trace.steps {
        println!(
            "[{:02}] {:?}  ({:.0}ms)",
            step.index,
            step.kind,
            step.duration.as_secs_f64() * 1000.0
        );
    }

    println!("\n=== Token 用量 ===");
    let u = &result.trace.total_usage;
    println!(
        "prompt={} completion={} total={}",
        u.prompt_tokens, u.completion_tokens, u.total_tokens
    );

    Ok(())
}
