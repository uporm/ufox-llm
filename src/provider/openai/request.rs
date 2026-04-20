//! `OpenAI` 请求序列化模块。
//!
//! 该模块负责将 SDK 的公共消息模型、工具定义模型转换为 `OpenAI Chat Completions`
//! 所需的私有请求格式。
//!
//! 设计上采用“公共类型先归一化，再落为私有 JSON”的方式：
//! 1. 先把消息、多模态片段和工具定义转换为 `OpenAI` 私有结构体；
//! 2. 最终再统一序列化为 `serde_json::Value`，便于上层适配器直接作为请求体发送；
//! 3. 本地图片文件在此阶段被读取并编码为 `data URL`，这样 `client` 层无需感知
//!    多模态细节，只负责发送标准 `JSON` 请求。
//!
//! 该模块依赖 `base64` 编码本地图片、依赖标准库 `fs` 读取图片文件，并依赖
//! `types` 模块中的消息与工具定义作为输入。

use std::fs;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::{
    Content, ContentPart, ImageFile, ImageSource, JsonType, LlmError, Message,
    client::RequestOptions,
    types::response::ReasoningEffort,
    Role, Tool, ToolChoice,
};

/// 将公共聊天参数转换为 `OpenAI` 私有请求体。
///
/// 该函数是 `OpenAI` 适配器构建请求体时的统一入口。
///
/// # Arguments
/// * `model` - 目标模型名称
/// * `messages` - 发送给模型的消息序列
/// * `tools` - 可选的工具定义列表
/// * `stream` - 是否启用流式输出
///
/// # Returns
/// 符合 `OpenAI` `Chat Completions` 协议的请求体 `JSON`。
///
/// # Errors
/// - [`LlmError::UnsupportedFeature`]：当当前公共消息暂时无法映射为 `OpenAI` 请求时触发
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
        .map(OpenAiMessage::from_public_message)
        .collect::<Result<Vec<_>, _>>()?;
    let tools = tools
        .filter(|items| !items.is_empty())
        .map(OpenAiTool::from_public_tools);

    let request = OpenAiChatRequest {
        model,
        messages,
        tools,
        stream,
        reasoning_effort: options.reasoning_effort(),
        tool_choice: options.tool_choice().map(OpenAiToolChoice::from_public_choice),
        parallel_tool_calls: options.parallel_tool_calls(),
    };

    serde_json::to_value(request).map_err(LlmError::from)
}

#[derive(Debug, Serialize)]
struct OpenAiChatRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAiTool>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<OpenAiToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAiToolChoice {
    Builtin(&'static str),
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        function: OpenAiNamedToolChoice,
    },
}

impl OpenAiToolChoice {
    fn from_public_choice(choice: &ToolChoice) -> Self {
        match choice {
            ToolChoice::Auto => Self::Builtin("auto"),
            ToolChoice::None => Self::Builtin("none"),
            ToolChoice::Required => Self::Builtin("required"),
            ToolChoice::Function { name } => Self::Function {
                kind: "function",
                function: OpenAiNamedToolChoice { name: name.clone() },
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiNamedToolChoice {
    name: String,
}

#[derive(Debug, Serialize)]
struct OpenAiMessage {
    role: &'static str,
    content: Option<OpenAiMessageContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiOutboundToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl OpenAiMessage {
    fn from_public_message(message: &Message) -> Result<Self, LlmError> {
        let role = match message.role() {
            Role::System | Role::User | Role::Assistant | Role::Tool => message.role().as_str(),
        };

        let content = if message.role() == Role::Assistant
            && message.tool_calls().is_some()
            && matches!(message.content(), Content::Text(text) if text.is_empty())
        {
            None
        } else {
            Some(OpenAiMessageContent::from_public_content(message.content())?)
        };

        Ok(Self {
            role,
            content,
            name: message.name().map(ToOwned::to_owned),
            tool_calls: message.tool_calls().map(OpenAiOutboundToolCall::from_public_tool_calls),
            tool_call_id: message.tool_call_id().map(ToOwned::to_owned),
        })
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAiMessageContent {
    Text(String),
    Parts(Vec<OpenAiContentPart>),
}

impl OpenAiMessageContent {
    fn from_public_content(content: &Content) -> Result<Self, LlmError> {
        match content {
            Content::Text(text) => Ok(Self::Text(text.clone())),
            Content::Parts(parts) => parts
                .iter()
                .map(OpenAiContentPart::from_public_part)
                .collect::<Result<Vec<_>, _>>()
                .map(Self::Parts),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAiContentPart {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: OpenAiImageUrl,
    },
}

impl OpenAiContentPart {
    fn from_public_part(part: &ContentPart) -> Result<Self, LlmError> {
        match part {
            ContentPart::Text { text } => Ok(Self::Text { text: text.clone() }),
            ContentPart::Image { source } => Ok(Self::ImageUrl {
                image_url: OpenAiImageUrl::from_image_source(source)?,
            }),
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiImageUrl {
    url: String,
}

impl OpenAiImageUrl {
    fn from_image_source(source: &ImageSource) -> Result<Self, LlmError> {
        match source {
            ImageSource::Url { url } => Ok(Self { url: url.clone() }),
            ImageSource::File(file) => Ok(Self {
                url: image_file_to_data_url(file)?,
            }),
        }
    }
}

#[derive(Debug, Serialize)]
struct OpenAiTool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunctionTool,
}

#[derive(Debug, Serialize)]
struct OpenAiOutboundToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiOutboundToolFunction,
}

impl OpenAiOutboundToolCall {
    fn from_public_tool_calls(tool_calls: &[crate::ToolCall]) -> Vec<Self> {
        tool_calls
            .iter()
            .map(|tool_call| Self {
                id: tool_call.id().to_string(),
                kind: "function",
                function: OpenAiOutboundToolFunction {
                    name: tool_call.name().to_string(),
                    arguments: tool_call.arguments().to_string(),
                },
            })
            .collect()
    }
}

#[derive(Debug, Serialize)]
struct OpenAiOutboundToolFunction {
    name: String,
    arguments: String,
}

impl OpenAiTool {
    fn from_public_tools(tools: &[Tool]) -> Vec<Self> {
        tools.iter()
            .map(|tool| Self {
                kind: "function",
                function: OpenAiFunctionTool::from_public_tool(tool),
            })
            .collect()
    }
}

#[derive(Debug, Serialize)]
struct OpenAiFunctionTool {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

impl OpenAiFunctionTool {
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
    schema.insert(
        "additionalProperties".to_string(),
        Value::Bool(false),
    );

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
    fn openai() {
        let request = build_chat_request(
            "gpt-4o",
            &[Message::user("你好")],
            None,
            false,
            &RequestOptions::default(),
        )
            .expect("请求体应构建成功");

        assert_eq!(request["model"], "gpt-4o");
        assert_eq!(request["stream"], false);
        assert_eq!(request["messages"][0]["role"], "user");
        assert_eq!(request["messages"][0]["content"], "你好");
    }

    #[test]
    fn url_image_url() {
        let message = Message::builder(Role::User)
            .text("描述这张图片")
            .image_url("https://example.com/photo.jpg")
            .build();

        let request = build_chat_request(
            "gpt-4o",
            &[message],
            None,
            false,
            &RequestOptions::default(),
        )
        .expect("请求体应构建成功");

        assert_eq!(request["messages"][0]["content"][0]["type"], "text");
        assert_eq!(request["messages"][0]["content"][1]["type"], "image_url");
        assert_eq!(
            request["messages"][0]["content"][1]["image_url"]["url"],
            "https://example.com/photo.jpg"
        );
    }

    #[test]
    fn openai_function_tool() {
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
            "gpt-4o",
            &[Message::user("杭州天气")],
            Some(&[tool]),
            false,
            &RequestOptions::default(),
        )
        .expect("请求体应构建成功");

        assert_eq!(request["tools"][0]["type"], "function");
        assert_eq!(request["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            request["tools"][0]["function"]["parameters"],
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
                "additionalProperties": false,
                "required": ["city"]
            })
        );
    }

    #[test]
    fn data_url() {
        let file_path = temp_png_path();
        fs::write(&file_path, [0x89, b'P', b'N', b'G']).expect("应能写入测试图片");

        let message = MessageBuilder::user().image_file(&file_path).build();
        let request = build_chat_request(
            "gpt-4o",
            &[message],
            None,
            false,
            &RequestOptions::default(),
        )
        .expect("请求体应构建成功");
        let data_url = request["messages"][0]["content"][0]["image_url"]["url"]
            .as_str()
            .expect("应为字符串 URL");

        assert!(data_url.starts_with("data:image/png;base64,"));

        fs::remove_file(file_path).expect("应能清理测试文件");
    }

    #[test]
    fn openai_tool_calls() {
        let calls = vec![crate::ToolCall::new(
            "call_1",
            "get_weather",
            r#"{"city":"杭州"}"#,
        )];
        let request = build_chat_request(
            "gpt-4o",
            &[crate::Message::assistant_with_tool_calls(&calls)],
            None,
            false,
            &RequestOptions::default(),
        )
        .expect("请求体应构建成功");

        assert_eq!(request["messages"][0]["role"], "assistant");
        assert_eq!(request["messages"][0]["content"], serde_json::Value::Null);
        assert_eq!(request["messages"][0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(
            request["messages"][0]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
    }

    #[test]
    fn openai_tool_role() {
        let request = build_chat_request(
            "gpt-4o",
            &[crate::Message::tool_result("call_1", r#"{"temp":26}"#)],
            None,
            false,
            &RequestOptions::default(),
        )
        .expect("请求体应构建成功");

        assert_eq!(request["messages"][0]["role"], "tool");
        assert_eq!(request["messages"][0]["tool_call_id"], "call_1");
        assert_eq!(request["messages"][0]["content"], r#"{"temp":26}"#);
    }

    #[test]
    fn openai_2() {
        let request = build_chat_request(
            "o3-mini",
            &[Message::user("分析这段代码")],
            None,
            false,
            &RequestOptions::new()
                .with_thinking(true)
                .with_reasoning_effort(crate::types::response::ReasoningEffort::High),
        )
        .expect("请求体应构建成功");

        assert_eq!(request["reasoning_effort"], "high");
    }

    #[test]
    fn openai_3() {
        let tool = Tool::function("get_weather")
            .param("city", JsonType::String, "城市名称", true)
            .build();
        let request = build_chat_request(
            "gpt-4o",
            &[Message::user("杭州天气")],
            Some(&[tool]),
            false,
            &RequestOptions::new()
                .with_tool_choice(crate::ToolChoice::function("get_weather"))
                .with_parallel_tool_calls(true),
        )
        .expect("请求体应构建成功");

        assert_eq!(request["parallel_tool_calls"], true);
        assert_eq!(request["tool_choice"]["type"], "function");
        assert_eq!(request["tool_choice"]["function"]["name"], "get_weather");
    }

    fn temp_png_path() -> std::path::PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("系统时间应大于 UNIX_EPOCH")
            .as_nanos();
        env::temp_dir().join(format!(
            "ufox-llm-openai-request-{timestamp}-{}.png",
            std::process::id()
        ))
    }
}
