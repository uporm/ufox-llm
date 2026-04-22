//! `Qwen` 请求序列化。
//!
//! 将公共消息和工具定义转换为 Qwen 请求体。

use std::fs;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::{
    Content, ContentPart, ImageFile, ImageSource, JsonType, LlmError, Message, Role, Tool,
    ToolChoice, client::RequestOptions,
};

/// 将公共聊天参数转换为 `Qwen` 私有请求体。
/// # Errors
/// - [`LlmError::UnsupportedFeature`]：当当前公共消息暂时无法映射为 `Qwen` 请求时触发
/// - [`LlmError::StreamError`]：当读取本地图片文件失败或构建 `data URL` 失败时触发
pub fn build_chat_request(
    model: &str,
    messages: &[Message],
    tools: Option<&[Tool]>,
    stream: bool,
    options: &RequestOptions,
) -> Result<Value, LlmError> {
    let messages = messages
        .iter()
        .map(QwenMessage::from_public_message)
        .collect::<Result<Vec<_>, _>>()?;
    let parameters = QwenParameters::new(tools, stream, options);

    let request = QwenChatRequest {
        model,
        input: QwenInput { messages },
        parameters,
    };

    let mut request = serde_json::to_value(request).map_err(LlmError::from)?;
    merge_provider_options(
        request
            .as_object_mut()
            .and_then(|body| body.get_mut("parameters"))
            .and_then(Value::as_object_mut),
        &options.provider_options,
    );
    Ok(request)
}

#[derive(Debug, Serialize)]
struct QwenChatRequest<'a> {
    model: &'a str,
    input: QwenInput,
    parameters: QwenParameters,
}

#[derive(Debug, Serialize)]
struct QwenInput {
    messages: Vec<QwenMessage>,
}

#[derive(Debug, Serialize)]
struct QwenParameters {
    result_format: &'static str,
    incremental_output: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<QwenToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<QwenTool>>,
}

impl QwenParameters {
    fn new(tools: Option<&[Tool]>, stream: bool, options: &RequestOptions) -> Self {
        Self {
            result_format: "message",
            incremental_output: stream,
            temperature: options.temperature,
            top_p: options.top_p,
            max_tokens: options.max_tokens,
            presence_penalty: options.presence_penalty,
            frequency_penalty: options.frequency_penalty,
            enable_thinking: options.thinking.then_some(true),
            thinking_budget: options.thinking_budget,
            tool_choice: options
                .tool_choice
                .as_ref()
                .map(QwenToolChoice::from_public_choice),
            parallel_tool_calls: options.parallel_tool_calls,
            tools: tools
                .filter(|items| !items.is_empty())
                .map(QwenTool::from_public_tools),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum QwenToolChoice {
    Builtin(&'static str),
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        function: QwenNamedToolChoice,
    },
}

impl QwenToolChoice {
    fn from_public_choice(choice: &ToolChoice) -> Self {
        match choice {
            ToolChoice::Auto => Self::Builtin("auto"),
            ToolChoice::None => Self::Builtin("none"),
            ToolChoice::Required => Self::Builtin("required"),
            ToolChoice::Function { name } => Self::Function {
                kind: "function",
                function: QwenNamedToolChoice { name: name.clone() },
            },
        }
    }
}

fn merge_provider_options(
    target: Option<&mut Map<String, Value>>,
    provider_options: &Map<String, Value>,
) {
    let Some(target) = target else {
        return;
    };

    for (key, value) in provider_options {
        target.entry(key.clone()).or_insert_with(|| value.clone());
    }
}

#[derive(Debug, Serialize)]
struct QwenNamedToolChoice {
    name: String,
}

#[derive(Debug, Serialize)]
struct QwenMessage {
    role: &'static str,
    content: Option<QwenMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<QwenOutboundToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl QwenMessage {
    fn from_public_message(message: &Message) -> Result<Self, LlmError> {
        let role = match message.role {
            Role::System | Role::User | Role::Assistant | Role::Tool => message.role.as_str(),
        };

        Ok(Self {
            role,
            content: if message.role == Role::Assistant
                && message.tool_calls.is_some()
                && matches!(&message.content, Content::Text(text) if text.is_empty())
            {
                None
            } else {
                Some(QwenMessageContent::from_public_content(&message.content)?)
            },
            name: message.name.clone(),
            tool_calls: message
                .tool_calls
                .as_deref()
                .map(QwenOutboundToolCall::from_public_tool_calls),
            tool_call_id: message.tool_call_id.clone(),
        })
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum QwenMessageContent {
    Text(String),
    Parts(Vec<QwenContentPart>),
}

impl QwenMessageContent {
    fn from_public_content(content: &Content) -> Result<Self, LlmError> {
        match content {
            Content::Text(text) => Ok(Self::Text(text.clone())),
            Content::Parts(parts) => parts
                .iter()
                .map(QwenContentPart::from_public_part)
                .collect::<Result<Vec<_>, _>>()
                .map(Self::Parts),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum QwenContentPart {
    Text { text: String },
    Image { image: String },
}

impl QwenContentPart {
    fn from_public_part(part: &ContentPart) -> Result<Self, LlmError> {
        match part {
            ContentPart::Text { text } => Ok(Self::Text { text: text.clone() }),
            ContentPart::Image { source } => Ok(Self::Image {
                image: image_source_to_qwen_value(source)?,
            }),
            ContentPart::Audio { .. } => Err(LlmError::UnsupportedFeature {
                provider: "qwen".to_string(),
                feature: "音频输入".to_string(),
            }),
            ContentPart::Video { .. } => Err(LlmError::UnsupportedFeature {
                provider: "qwen".to_string(),
                feature: "视频输入".to_string(),
            }),
        }
    }
}

#[derive(Debug, Serialize)]
struct QwenTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: QwenFunctionTool,
}

#[derive(Debug, Serialize)]
struct QwenOutboundToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: QwenOutboundToolFunction,
}

impl QwenOutboundToolCall {
    fn from_public_tool_calls(tool_calls: &[crate::ToolCall]) -> Vec<Self> {
        tool_calls
            .iter()
            .map(|tool_call| Self {
                id: tool_call.id.clone(),
                kind: "function",
                function: QwenOutboundToolFunction {
                    name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                },
            })
            .collect()
    }
}

#[derive(Debug, Serialize)]
struct QwenOutboundToolFunction {
    name: String,
    arguments: String,
}

impl QwenTool {
    fn from_public_tools(tools: &[Tool]) -> Vec<Self> {
        tools
            .iter()
            .map(|tool| Self {
                kind: "function",
                function: QwenFunctionTool::from_public_tool(tool),
            })
            .collect()
    }
}

#[derive(Debug, Serialize)]
struct QwenFunctionTool {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

impl QwenFunctionTool {
    fn from_public_tool(tool: &Tool) -> Self {
        Self {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: build_parameters_schema(tool),
        }
    }
}

fn build_parameters_schema(tool: &Tool) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();

    for parameter in &tool.parameters {
        properties.insert(
            parameter.name.clone(),
            build_parameter_schema(&parameter.json_type, &parameter.description),
        );

        if parameter.required {
            required.push(parameter.name.clone());
        }
    }

    let mut schema = Map::new();
    schema.insert("type".to_string(), Value::String("object".to_string()));
    schema.insert("properties".to_string(), Value::Object(properties));

    if !required.is_empty() {
        schema.insert("required".to_string(), json!(required));
    }

    Value::Object(schema)
}

fn build_parameter_schema(json_type: &JsonType, description: &str) -> Value {
    let mut schema = match json_type {
        JsonType::String => object_schema("string"),
        JsonType::Number => object_schema("number"),
        JsonType::Integer => object_schema("integer"),
        JsonType::Boolean => object_schema("boolean"),
        JsonType::Object => object_schema("object"),
        JsonType::Array(item_type) => {
            let mut schema = object_schema("array");
            schema.insert("items".to_string(), build_parameter_schema(item_type, ""));
            schema
        }
        JsonType::Enum(values) => {
            let mut schema = object_schema("string");
            schema.insert("enum".to_string(), json!(values));
            schema
        }
    };

    if !description.is_empty() {
        schema.insert(
            "description".to_string(),
            Value::String(description.to_string()),
        );
    }

    Value::Object(schema)
}

fn object_schema(type_name: &str) -> Map<String, Value> {
    let mut schema = Map::new();
    schema.insert("type".to_string(), Value::String(type_name.to_string()));
    schema
}

fn image_source_to_qwen_value(source: &ImageSource) -> Result<String, LlmError> {
    match source {
        ImageSource::Url { url } => Ok(url.clone()),
        ImageSource::File(file) => image_file_to_data_url(file),
    }
}

fn image_file_to_data_url(file: &ImageFile) -> Result<String, LlmError> {
    let bytes = fs::read(&file.path).map_err(|error| {
        LlmError::StreamError(format!(
            "读取本地图片文件失败：路径={}，错误={error}",
            file.path.display()
        ))
    })?;
    let mime_type = file.mime_type.as_deref().unwrap_or("application/octet-stream");
    let encoded = STANDARD.encode(bytes);

    Ok(format!("data:{mime_type};base64,{encoded}"))
}

#[cfg(test)]
mod tests {
    use std::{
        env, fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use serde_json::json;

    use super::build_chat_request;
    use crate::{JsonType, Message, MessageBuilder, Role, Tool, client::RequestOptions};

    #[test]
    fn qwen_dashscope() {
        let request = build_chat_request(
            "qwen-max",
            &[Message::user("你好")],
            None,
            false,
            &RequestOptions::default(),
        )
        .expect("请求体应构建成功");

        assert_eq!(request["model"], "qwen-max");
        assert_eq!(request["input"]["messages"][0]["role"], "user");
        assert_eq!(request["input"]["messages"][0]["content"], "你好");
        assert_eq!(request["parameters"]["result_format"], "message");
        assert_eq!(request["parameters"]["incremental_output"], false);
    }

    #[test]
    fn qwen_content_parts() {
        let message = Message::builder(Role::User)
            .text("描述这张图片")
            .image_url("https://example.com/photo.jpg")
            .build();

        let request = build_chat_request(
            "qwen-vl-max",
            &[message],
            None,
            false,
            &RequestOptions::default(),
        )
        .expect("请求体应构建成功");

        assert_eq!(
            request["input"]["messages"][0]["content"][0]["text"],
            "描述这张图片"
        );
        assert_eq!(
            request["input"]["messages"][0]["content"][1]["image"],
            "https://example.com/photo.jpg"
        );
    }

    #[test]
    fn qwen_tools() {
        let tool = Tool::function("get_weather")
            .description("获取城市实时天气")
            .param("city", JsonType::String, "城市名称", true)
            .param(
                "unit",
                JsonType::Enum(vec!["celsius".to_string(), "fahrenheit".to_string()]),
                "温度单位",
                false,
            )
            .build();

        let request = build_chat_request(
            "qwen-max",
            &[Message::user("杭州天气")],
            Some(&[tool]),
            true,
            &RequestOptions::default(),
        )
        .expect("请求体应构建成功");

        assert_eq!(request["parameters"]["incremental_output"], true);
        assert_eq!(request["parameters"]["tools"][0]["type"], "function");
        assert_eq!(
            request["parameters"]["tools"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            request["parameters"]["tools"][0]["function"]["parameters"],
            json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            })
        );
    }

    #[test]
    fn qwen_data_url() {
        let file_path = temp_png_path();
        fs::write(&file_path, [0x89, b'P', b'N', b'G']).expect("应能写入测试图片");

        let message = MessageBuilder::user().image_file(&file_path).build();
        let request = build_chat_request(
            "qwen-vl-max",
            &[message],
            None,
            false,
            &RequestOptions::default(),
        )
        .expect("请求体应构建成功");
        let data_url = request["input"]["messages"][0]["content"][0]["image"]
            .as_str()
            .expect("应为字符串 URL");

        assert!(data_url.starts_with("data:image/png;base64,"));

        fs::remove_file(file_path).expect("应能清理测试文件");
    }

    #[test]
    fn qwen_tool_calls() {
        let calls = vec![crate::ToolCall::new(
            "call_1",
            "get_weather",
            r#"{"city":"杭州"}"#,
        )];
        let request = build_chat_request(
            "qwen-max",
            &[crate::Message::assistant_with_tool_calls(&calls)],
            None,
            false,
            &RequestOptions::default(),
        )
        .expect("请求体应构建成功");

        assert_eq!(request["input"]["messages"][0]["role"], "assistant");
        assert_eq!(
            request["input"]["messages"][0]["content"],
            serde_json::Value::Null
        );
        assert_eq!(
            request["input"]["messages"][0]["tool_calls"][0]["id"],
            "call_1"
        );
    }

    #[test]
    fn qwen_tool_role() {
        let request = build_chat_request(
            "qwen-max",
            &[crate::Message::tool_result("call_1", r#"{"temp":26}"#)],
            None,
            false,
            &RequestOptions::default(),
        )
        .expect("请求体应构建成功");

        assert_eq!(request["input"]["messages"][0]["role"], "tool");
        assert_eq!(request["input"]["messages"][0]["tool_call_id"], "call_1");
        assert_eq!(request["input"]["messages"][0]["content"], r#"{"temp":26}"#);
    }

    #[test]
    fn thinking_qwen_parameters() {
        let request = build_chat_request(
            "qwen3-max",
            &[Message::user("分析这段代码")],
            None,
            true,
            &RequestOptions {
                thinking: true,
                thinking_budget: Some(8192),
                ..RequestOptions::default()
            },
        )
        .expect("请求体应构建成功");

        assert_eq!(request["parameters"]["enable_thinking"], true);
        assert_eq!(request["parameters"]["thinking_budget"], 8192);
    }

    #[test]
    fn qwen_parameters() {
        let tool = Tool::function("get_weather")
            .param("city", JsonType::String, "城市名称", true)
            .build();
        let request = build_chat_request(
            "qwen-max",
            &[Message::user("杭州天气")],
            Some(&[tool]),
            false,
            &RequestOptions {
                tool_choice: Some(crate::ToolChoice::function("get_weather")),
                parallel_tool_calls: Some(true),
                ..RequestOptions::default()
            },
        )
        .expect("请求体应构建成功");

        assert_eq!(request["parameters"]["parallel_tool_calls"], true);
        assert_eq!(request["parameters"]["tool_choice"]["type"], "function");
        assert_eq!(
            request["parameters"]["tool_choice"]["function"]["name"],
            "get_weather"
        );
    }

    #[test]
    fn qwen_sampling_parameters() {
        let request = build_chat_request(
            "qwen-max",
            &[Message::user("讲个故事")],
            None,
            false,
            &RequestOptions {
                temperature: Some(0.7),
                top_p: Some(0.85),
                max_tokens: Some(768),
                presence_penalty: Some(0.3),
                frequency_penalty: Some(0.15),
                ..RequestOptions::default()
            },
        )
        .expect("请求体应构建成功");

        assert!(
            (request["parameters"]["temperature"]
                .as_f64()
                .expect("temperature 应为数字")
                - 0.7)
                .abs()
                < 1e-6
        );
        assert!(
            (request["parameters"]["top_p"]
                .as_f64()
                .expect("top_p 应为数字")
                - 0.85)
                .abs()
                < 1e-6
        );
        assert_eq!(request["parameters"]["max_tokens"], 768);
        assert!(
            (request["parameters"]["presence_penalty"]
                .as_f64()
                .expect("presence_penalty 应为数字")
                - 0.3)
                .abs()
                < 1e-6
        );
        assert!(
            (request["parameters"]["frequency_penalty"]
                .as_f64()
                .expect("frequency_penalty 应为数字")
                - 0.15)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn qwen_provider_options() {
        let request = build_chat_request(
            "qwen-max",
            &[Message::user("讲个故事")],
            None,
            false,
            &RequestOptions {
                top_p: Some(0.85),
                provider_options: serde_json::Map::from_iter([
                    ("seed".to_string(), json!(7)),
                    ("top_p".to_string(), json!(0.2)),
                    ("repetition_penalty".to_string(), json!(1.1)),
                ]),
                ..RequestOptions::default()
            },
        )
        .expect("请求体应构建成功");

        assert_eq!(request["parameters"]["seed"], 7);
        assert_eq!(request["parameters"]["repetition_penalty"], 1.1);
        assert!(
            (request["parameters"]["top_p"]
                .as_f64()
                .expect("top_p 应为数字")
                - 0.85)
                .abs()
                < 1e-6
        );
    }

    fn temp_png_path() -> std::path::PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("系统时间应大于 UNIX_EPOCH")
            .as_nanos();
        env::temp_dir().join(format!(
            "ufox-llm-qwen-request-{timestamp}-{}.png",
            std::process::id()
        ))
    }
}
