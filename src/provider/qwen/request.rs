//! `Qwen` 请求序列化模块。
//!
//! 该模块负责将 SDK 的公共消息模型、工具定义模型转换为 `DashScope` / `Qwen`
//! 所需的私有请求格式。
//!
//! 设计上采用“消息输入与参数配置分层”的方式：
//! 1. `input.messages` 只承载对话内容本身；
//! 2. `parameters` 承载结果格式、流式开关与工具定义，便于后续扩展更多 `Qwen`
//!    特有参数而不污染公共消息模型；
//! 3. 本地图片文件在此阶段被读取并编码为 `data URL`，从而让上层 `client`
//!    只处理统一的 `JSON` 请求发送逻辑。
//!
//! 该模块依赖 `base64` 编码本地图片、依赖标准库 `fs` 读取图片文件，并依赖
//! `types` 模块中的消息与工具定义作为输入。

use std::fs;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::{
    Content, ContentPart, ImageFile, ImageSource, JsonType, LlmError, Message, Role, Tool,
    ToolChoice, client::RequestOptions,
};

/// 将公共聊天参数转换为 `Qwen` 私有请求体。
///
/// # Arguments
/// * `model` - 目标模型名称
/// * `messages` - 发送给模型的消息序列
/// * `tools` - 可选的工具定义列表
/// * `stream` - 是否启用流式输出
///
/// # Returns
/// 符合 `DashScope` 消息格式的请求体 `JSON`。
///
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

    serde_json::to_value(request).map_err(LlmError::from)
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
            enable_thinking: options.thinking().then_some(true),
            thinking_budget: options.thinking_budget(),
            tool_choice: options.tool_choice().map(QwenToolChoice::from_public_choice),
            parallel_tool_calls: options.parallel_tool_calls(),
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
        let role = match message.role() {
            Role::System | Role::User | Role::Assistant | Role::Tool => message.role().as_str(),
        };

        Ok(Self {
            role,
            content: if message.role() == Role::Assistant
                && message.tool_calls().is_some()
                && matches!(message.content(), Content::Text(text) if text.is_empty())
            {
                None
            } else {
                Some(QwenMessageContent::from_public_content(message.content())?)
            },
            name: message.name().map(ToOwned::to_owned),
            tool_calls: message.tool_calls().map(QwenOutboundToolCall::from_public_tool_calls),
            tool_call_id: message.tool_call_id().map(ToOwned::to_owned),
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
    Text {
        text: String,
    },
    Image {
        image: String,
    },
}

impl QwenContentPart {
    fn from_public_part(part: &ContentPart) -> Result<Self, LlmError> {
        match part {
            ContentPart::Text { text } => Ok(Self::Text { text: text.clone() }),
            ContentPart::Image { source } => Ok(Self::Image {
                image: image_source_to_qwen_value(source)?,
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
                id: tool_call.id().to_string(),
                kind: "function",
                function: QwenOutboundToolFunction {
                    name: tool_call.name().to_string(),
                    arguments: tool_call.arguments().to_string(),
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
        tools.iter()
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
            name: tool.name().to_string(),
            description: tool.description().map(ToOwned::to_owned),
            parameters: build_parameters_schema(tool),
        }
    }
}

fn build_parameters_schema(tool: &Tool) -> Value {
    let mut properties = Map::new();
    let mut required = Vec::new();

    for parameter in tool.parameters() {
        properties.insert(
            parameter.name().to_string(),
            build_parameter_schema(parameter.json_type(), parameter.description()),
        );

        if parameter.required() {
            required.push(parameter.name().to_string());
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
    let bytes = fs::read(file.path()).map_err(|error| {
        LlmError::StreamError(format!(
            "读取本地图片文件失败：路径={}，错误={error}",
            file.path().display()
        ))
    })?;
    let mime_type = file.mime_type().unwrap_or("application/octet-stream");
    let encoded = STANDARD.encode(bytes);

    Ok(format!("data:{mime_type};base64,{encoded}"))
}

#[cfg(test)]
mod tests {
    use std::{
        env,
        fs,
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

        assert_eq!(request["input"]["messages"][0]["content"][0]["text"], "描述这张图片");
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
        assert_eq!(request["parameters"]["tools"][0]["function"]["name"], "get_weather");
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
        assert_eq!(request["input"]["messages"][0]["content"], serde_json::Value::Null);
        assert_eq!(request["input"]["messages"][0]["tool_calls"][0]["id"], "call_1");
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
            &RequestOptions::new()
                .with_thinking(true)
                .with_thinking_budget(8192),
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
            &RequestOptions::new()
                .with_tool_choice(crate::ToolChoice::function("get_weather"))
                .with_parallel_tool_calls(true),
        )
        .expect("请求体应构建成功");

        assert_eq!(request["parameters"]["parallel_tool_calls"], true);
        assert_eq!(request["parameters"]["tool_choice"]["type"], "function");
        assert_eq!(
            request["parameters"]["tool_choice"]["function"]["name"],
            "get_weather"
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
