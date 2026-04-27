use std::error::Error;

use futures::StreamExt;
use ufox_llm::{
    ChatRequest, Client, ContentPart, Message, Role, Tool, ToolChoice, ToolResult,
    ToolResultPayload,
};

fn run_local_tool(
    name: &str,
    arguments: &serde_json::Value,
) -> Result<serde_json::Value, Box<dyn Error>> {
    match name {
        "get_weather" => Ok(serde_json::json!({
            "city": arguments.get("city").and_then(|v| v.as_str()).unwrap_or("unknown"),
            "weather": "cloudy",
            "temperature_c": 24
        })),
        _ => Err(format!("unknown tool: {name}").into()),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // 默认走环境变量，避免示例里硬编码密钥，也和其他示例保持一致。
    let client = Client::from_env()?;
    // 也可以显式使用 builder：
    // let client = Client::builder()
    //     .provider(Provider::OpenAI)
    //     .api_key("sk-xxx")
    //     .model("gpt-4o")
    //     .build()?;

    let weather_tool = Tool::function(
        "get_weather",
        "查询指定城市的实时天气",
        serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    );

    let mut messages = vec![Message {
        role: Role::User,
        content: vec![ContentPart::text("帮我查询杭州天气，并给出穿衣建议")],
        name: None,
    }];
    let mut started_thinking = false;
    let mut started_answer = false;

    loop {
        let mut stream = client
            .chat_stream(
                ChatRequest::builder()
                    .messages(messages.clone())
                    .tools(vec![weather_tool.clone()])
                    .tool_choice(ToolChoice::Auto)
                    .build(),
            )
            .await?;

        let mut assistant_text = String::new();
        let mut assistant_tool_calls = Vec::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let is_finished = chunk.is_finished();
            if let Some(thinking) = &chunk.thinking_delta {
                if !started_thinking {
                    println!("=== 思考过程 ===\n");
                    started_thinking = true;
                }
                print!("{thinking}");
            }
            if let Some(text) = &chunk.text_delta {
                if !started_answer {
                    if started_thinking {
                        println!("\n\n=== 最终回答 ===\n");
                    } else {
                        println!("=== 最终回答 ===\n");
                    }
                    started_answer = true;
                }
                print!("{text}");
                assistant_text.push_str(text);
            }
            if !chunk.tool_calls.is_empty() {
                assistant_tool_calls.extend(chunk.tool_calls.iter().cloned());
            }
            if is_finished {
                break;
            }
        }

        let tool_calls = assistant_tool_calls.clone();
        let mut assistant_content = Vec::new();
        if !assistant_text.is_empty() {
            assistant_content.push(ContentPart::text(assistant_text));
        }
        for call in assistant_tool_calls {
            assistant_content.push(ContentPart::ToolCall(call));
        }
        messages.push(Message {
            role: Role::Assistant,
            content: assistant_content,
            name: None,
        });

        if tool_calls.is_empty() {
            if started_thinking || started_answer {
                println!();
            }
            break;
        }

        for call in &tool_calls {
            println!("\n[local-tool] 调用 {}", call.tool_name);
            let result = run_local_tool(&call.tool_name, &call.arguments)?;
            println!("[local-tool] 结果 {}", serde_json::to_string(&result).unwrap());
            messages.push(Message {
                role: Role::Tool,
                content: vec![ContentPart::ToolResult(ToolResult {
                    tool_call_id: call.id.clone(),
                    tool_name: Some(call.tool_name.clone()),
                    payload: ToolResultPayload::json(result),
                    is_error: false,
                })],
                name: None,
            });
        }
    }

    Ok(())
}
