//! 工具调用模型。
//!
//! 定义工具声明、参数类型、工具调用请求与工具执行结果。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::LlmError;

/// 工具参数的 `JSON` 类型描述。
///
/// 该枚举用于构建工具函数的参数结构，后续会由不同 `Provider` 适配层映射到对应的
/// 请求格式。对于 `enum` 类型，底层仍按字符串类型处理，并附带枚举候选值。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "items", rename_all = "snake_case")]
pub enum JsonType {
    /// 字符串类型。
    String,
    /// 数值类型。
    Number,
    /// 整数类型。
    Integer,
    /// 布尔类型。
    Boolean,
    /// 任意对象类型。
    Object,
    /// 数组类型。
    Array(Box<JsonType>),
    /// 字符串枚举类型。
    Enum(Vec<String>),
}

impl JsonType {
    pub fn enumeration<I, S>(values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Enum(values.into_iter().map(Into::into).collect())
    }

    pub const fn is_enum(&self) -> bool {
        matches!(self, Self::Enum(_))
    }
}

/// 工具参数定义。
///
/// 该类型描述函数工具中的单个入参，包括名称、类型、说明和是否必填。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolParameter {
    name: String,
    json_type: JsonType,
    description: String,
    required: bool,
}

impl ToolParameter {
    pub fn new(
        name: impl Into<String>,
        json_type: JsonType,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        Self {
            name: name.into(),
            json_type,
            description: description.into(),
            required,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn json_type(&self) -> &JsonType {
        &self.json_type
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub const fn required(&self) -> bool {
        self.required
    }
}

/// 工具种类。
///
/// 当前仅支持函数型工具。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// 函数型工具。
    Function,
}

impl ToolKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct FunctionTool {
    name: String,
    description: Option<String>,
    parameters: Vec<ToolParameter>,
}

/// 工具定义。
///
/// 当前 `SDK` 仅支持函数型工具定义，但该结构体保留了稳定的包装层，便于未来扩展更多
/// 工具种类而不破坏对外 `API`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    kind: ToolKind,
    function: FunctionTool,
}

impl Tool {
    pub fn function(name: impl Into<String>) -> ToolBuilder {
        ToolBuilder::new(name)
    }

    pub fn name(&self) -> &str {
        &self.function.name
    }

    pub fn description(&self) -> Option<&str> {
        self.function.description.as_deref()
    }

    pub fn parameters(&self) -> &[ToolParameter] {
        &self.function.parameters
    }

    pub const fn kind(&self) -> ToolKind {
        self.kind
    }
}

/// 工具调用策略。
///
/// 该枚举统一表达不同 Provider 对“是否调用工具、是否强制调用指定工具”的配置方式。
/// 对于不支持某个策略的 Provider，适配层会按能力范围忽略不兼容字段。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolChoice {
    /// 由模型自动决定是否调用工具。
    Auto,
    /// 禁止模型调用工具。
    None,
    /// 强制模型至少调用一个工具。
    Required,
    /// 强制模型调用指定名称的工具。
    Function { name: String },
}

impl ToolChoice {
    pub fn function(name: impl Into<String>) -> Self {
        Self::Function { name: name.into() }
    }

    pub const fn as_str(&self) -> Option<&'static str> {
        match self {
            Self::Auto => Some("auto"),
            Self::None => Some("none"),
            Self::Required => Some("required"),
            Self::Function { .. } => None,
        }
    }

    pub fn function_name(&self) -> Option<&str> {
        match self {
            Self::Function { name } => Some(name),
            Self::Auto | Self::None | Self::Required => None,
        }
    }
}

/// 工具构建器。
///
/// 该构建器用于以声明式方式构建函数型工具定义。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolBuilder {
    name: String,
    description: Option<String>,
    parameters: Vec<ToolParameter>,
}

impl ToolBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: Vec::new(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn param(
        mut self,
        name: impl Into<String>,
        json_type: JsonType,
        description: impl Into<String>,
        required: bool,
    ) -> Self {
        self.parameters.push(ToolParameter {
            name: name.into(),
            json_type,
            description: description.into(),
            required,
        });
        self
    }

    /// 构建工具定义。

    pub fn build(self) -> Tool {
        Tool {
            kind: ToolKind::Function,
            function: FunctionTool {
                name: self.name,
                description: self.description,
                parameters: self.parameters,
            },
        }
    }
}

/// 模型返回的一次工具调用请求。
///
/// 该结构体保留原始参数字符串，调用方可以选择在分发前手动解析，或者使用
/// [`ToolCall::arguments_json`] 获取解析后的 `JSON` 值。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn arguments(&self) -> &str {
        &self.arguments
    }

    /// 将原始参数解析为 `JSON` 值。
    /// # Errors
    /// - [`LlmError::ParseError`]：当 `arguments` 不是合法 `JSON` 时触发
    pub fn arguments_json(&self) -> Result<Value, LlmError> {
        serde_json::from_str(&self.arguments).map_err(LlmError::from)
    }
}

/// 工具执行结果。
///
/// 该结构体描述本地工具执行后需要回传给模型的一条结果消息。其内容统一保存为字符串，
/// 这样可以兼容纯文本结果和调用方自行序列化后的 `JSON` 结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    tool_call_id: String,
    content: String,
}

impl ToolResult {
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }

    /// 从 `JSON` 值创建工具执行结果。

    pub fn json(tool_call_id: impl Into<String>, content: Value) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.to_string(),
        }
    }

    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{JsonType, Tool, ToolCall, ToolChoice, ToolKind, ToolResult};

    #[test]
    fn tool_test() {
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

        assert_eq!(tool.kind(), ToolKind::Function);
        assert_eq!(tool.parameters()[0].name(), "city");
        assert_eq!(tool.parameters()[1].name(), "unit");
    }

    #[test]
    fn json_2() {
        let call = ToolCall::new("call_1", "get_weather", r#"{"city":"杭州"}"#);
        let value = call.arguments_json().expect("参数应为合法 JSON");

        assert_eq!(value["city"], "杭州");
    }

    #[test]
    fn json_3() {
        let result = ToolResult::json("call_1", json!({ "temp": 26, "unit": "celsius" }));

        assert_eq!(result.tool_call_id(), "call_1");
        assert_eq!(result.content(), r#"{"temp":26,"unit":"celsius"}"#);
    }

    #[test]
    fn tool_test_2() {
        let choice = ToolChoice::function("get_weather");

        assert_eq!(choice.function_name(), Some("get_weather"));
        assert_eq!(ToolChoice::Auto.as_str(), Some("auto"));
        assert_eq!(ToolChoice::None.as_str(), Some("none"));
        assert_eq!(ToolChoice::Required.as_str(), Some("required"));
    }
}
