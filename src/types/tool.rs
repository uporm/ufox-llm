//! 工具调用模型。
//!
//! 定义工具声明、参数类型、工具调用请求与工具执行结果。

use std::collections::HashSet;

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

/// 工具定义。
///
/// 当前 `SDK` 仅支持函数调用，因此工具定义保持扁平结构。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Vec<ToolParameter>,
}

impl Tool {
    pub fn function(name: impl Into<String>) -> ToolBuilder {
        ToolBuilder::new(name)
    }
}

/// 工具参数定义。
///
/// 该类型描述函数工具中的单个入参，包括名称、类型、说明和是否必填。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub json_type: JsonType,
    pub description: String,
    pub required: bool,
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
        self.parameters
            .push(ToolParameter::new(name, json_type, description, required));
        self
    }

    /// 构建工具定义。
    pub fn build(self) -> Tool {
        Tool {
            name: self.name,
            description: self.description,
            parameters: self.parameters,
        }
    }

    /// 构建并校验工具定义。
    ///
    /// # Errors
    /// - [`LlmError::ValidationError`]：当工具名为空、参数名重复，或枚举参数无可选值时触发
    pub fn build_checked(self) -> Result<Tool, LlmError> {
        self.validate()?;
        Ok(self.build())
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self.name.trim().is_empty() {
            return Err(LlmError::ValidationError("工具名称不能为空".to_string()));
        }

        let mut parameter_names = HashSet::new();
        for parameter in &self.parameters {
            if parameter.name.trim().is_empty() {
                return Err(LlmError::ValidationError("参数名称不能为空".to_string()));
            }

            if !parameter_names.insert(parameter.name.as_str()) {
                return Err(LlmError::ValidationError(format!(
                    "参数名称重复：{}",
                    parameter.name
                )));
            }

            if let JsonType::Enum(candidates) = &parameter.json_type {
                if candidates.is_empty() {
                    return Err(LlmError::ValidationError(format!(
                        "参数 {} 的枚举候选值不能为空",
                        parameter.name
                    )));
                }
            }
        }

        Ok(())
    }
}

/// 模型返回的一次工具调用请求。
///
/// 该结构体保留原始参数字符串，调用方可以选择在分发前手动解析，或者使用
/// [`ToolCall::arguments_json`] 获取解析后的 `JSON` 值。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
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
    pub tool_call_id: String,
    pub content: String,
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

    /// 转为 Provider 适配层可直接消费的 `JSON` 结构。
    ///
    /// 普通策略映射为字符串，指定函数策略映射为标准对象结构。
    pub fn to_json_value(&self) -> Value {
        match self {
            Self::Auto => Value::String("auto".to_string()),
            Self::None => Value::String("none".to_string()),
            Self::Required => Value::String("required".to_string()),
            Self::Function { name } => serde_json::json!({
                "type": "function",
                "function": { "name": name },
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::LlmError;

    use super::{JsonType, Tool, ToolCall, ToolChoice, ToolResult};

    #[test]
    fn builds_function_tool_with_parameters() {
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

        assert_eq!(tool.name, "get_weather");
        assert_eq!(tool.parameters[0].name, "city");
        assert_eq!(tool.parameters[1].name, "unit");
    }

    #[test]
    fn parses_tool_call_arguments_json() {
        let call = ToolCall::new("call_1", "get_weather", r#"{"city":"杭州"}"#);
        let value = call.arguments_json().expect("参数应为合法 JSON");

        assert_eq!(value["city"], "杭州");
    }

    #[test]
    fn serializes_tool_result_json_content() {
        let result = ToolResult::json("call_1", json!({ "temp": 26, "unit": "celsius" }));

        assert_eq!(result.tool_call_id, "call_1");
        assert_eq!(result.content, r#"{"temp":26,"unit":"celsius"}"#);
    }

    #[test]
    fn exposes_tool_choice_helpers() {
        let choice = ToolChoice::function("get_weather");

        assert_eq!(choice.function_name(), Some("get_weather"));
        assert_eq!(ToolChoice::Auto.as_str(), Some("auto"));
        assert_eq!(ToolChoice::None.as_str(), Some("none"));
        assert_eq!(ToolChoice::Required.as_str(), Some("required"));
        assert_eq!(
            choice.to_json_value(),
            json!({
                "type": "function",
                "function": { "name": "get_weather" }
            })
        );
    }

    #[test]
    fn returns_parse_error_for_invalid_tool_call_arguments() {
        let call = ToolCall::new("call_1", "get_weather", "{invalid-json");
        let error = call
            .arguments_json()
            .expect_err("非法 JSON 应返回 ParseError");
        assert!(matches!(error, LlmError::ParseError(_)));
    }

    #[test]
    fn build_checked_rejects_empty_tool_name() {
        let error = Tool::function("   ")
            .param("city", JsonType::String, "城市名称", true)
            .build_checked()
            .expect_err("空工具名应校验失败");

        assert!(matches!(error, LlmError::ValidationError(_)));
    }

    #[test]
    fn build_checked_rejects_duplicate_parameter_names() {
        let error = Tool::function("get_weather")
            .param("city", JsonType::String, "城市名称", true)
            .param("city", JsonType::String, "城市别名", false)
            .build_checked()
            .expect_err("重复参数名应校验失败");

        assert!(matches!(error, LlmError::ValidationError(_)));
    }

    #[test]
    fn build_checked_rejects_empty_enum_candidates() {
        let error = Tool::function("get_weather")
            .param("unit", JsonType::Enum(vec![]), "温度单位", false)
            .build_checked()
            .expect_err("空枚举候选应校验失败");

        assert!(matches!(error, LlmError::ValidationError(_)));
    }
}
