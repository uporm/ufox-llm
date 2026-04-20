//! 工具调用模型模块。
//!
//! 该模块负责定义 `LLM` 工具调用能力所需的统一中间层类型，包括工具声明、
//! 参数类型、模型返回的工具调用请求以及本地工具执行结果。
//!
//! 设计上采用“声明式工具定义 + 原始参数字符串保留”的方案：
//! 1. `Tool` 使用与 `Provider` 无关的参数描述模型，便于后续转换为 `OpenAI`、
//!    `Qwen` 或兼容接口各自要求的 `JSON Schema` 结构；
//! 2. `ToolCall` 同时保留原始 `arguments` 字符串，避免模型生成的参数在解析前被
//!    提前规范化，从而保留排障和重试所需的原始上下文；
//! 3. 工具参数集合内部使用有序列表保存，确保后续序列化时可以稳定输出，降低测试、
//!    快照与跨 `Provider` 调试时的噪声。
//!
//! 该模块依赖 `serde` 进行序列化，依赖 `serde_json` 解析工具调用参数。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::LlmError;

/// 工具参数的 `JSON` 类型描述。
///
/// 该枚举用于构建工具函数的参数结构，后续会由不同 `Provider` 适配层映射到对应的
/// 请求格式。对于 `enum` 类型，底层仍按字符串类型处理，并附带枚举候选值。
///
/// # 示例
/// ```rust
/// use ufox_llm::JsonType;
///
/// let ty = JsonType::Enum(vec!["celsius".to_string(), "fahrenheit".to_string()]);
/// assert!(ty.is_enum());
/// ```
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
    /// 创建字符串枚举类型。
    ///
    /// # Arguments
    /// * `values` - 可选枚举值列表
    ///
    /// # Returns
    /// 包含枚举候选值的参数类型定义。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::JsonType;
    ///
    /// let ty = JsonType::enumeration(["high", "medium", "low"]);
    /// assert!(ty.is_enum());
    /// ```
    #[must_use]
    pub fn enumeration<I, S>(values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::Enum(values.into_iter().map(Into::into).collect())
    }

    /// 返回该类型是否为枚举类型。
    ///
    /// # Returns
    /// 如果当前类型为 [`JsonType::Enum`]，则返回 `true`。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::JsonType;
    ///
    /// assert!(JsonType::Enum(vec!["a".to_string()]).is_enum());
    /// ```
    #[must_use]
    pub const fn is_enum(&self) -> bool {
        matches!(self, Self::Enum(_))
    }
}

/// 工具定义。
///
/// 当前 `SDK` 仅支持函数型工具定义，但该结构体保留了稳定的包装层，便于未来扩展更多
/// 工具种类而不破坏对外 `API`。
///
/// # 示例
/// ```rust
/// use ufox_llm::{JsonType, Tool};
///
/// let tool = Tool::function("get_weather")
///     .description("获取城市实时天气")
///     .param("city", JsonType::String, "城市名称", true)
///     .build();
///
/// assert_eq!(tool.name(), "get_weather");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tool {
    kind: ToolKind,
    function: FunctionTool,
}

impl Tool {
    /// 创建函数型工具构建器。
    ///
    /// # Arguments
    /// * `name` - 工具函数名称
    ///
    /// # Returns
    /// 可链式配置描述与参数的工具构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{JsonType, Tool};
    ///
    /// let tool = Tool::function("get_weather")
    ///     .description("获取天气")
    ///     .param("city", JsonType::String, "城市名称", true)
    ///     .build();
    ///
    /// assert_eq!(tool.name(), "get_weather");
    /// ```
    #[must_use]
    pub fn function(name: impl Into<String>) -> ToolBuilder {
        ToolBuilder::new(name)
    }

    /// 返回工具名称。
    ///
    /// # Returns
    /// 当前工具的函数名称。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Tool;
    ///
    /// let tool = Tool::function("search_docs").build();
    /// assert_eq!(tool.name(), "search_docs");
    /// ```
    #[must_use]
    pub fn name(&self) -> &str {
        &self.function.name
    }

    /// 返回工具描述。
    ///
    /// # Returns
    /// 若设置了工具描述，则返回该描述。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::Tool;
    ///
    /// let tool = Tool::function("search_docs").description("搜索文档").build();
    /// assert_eq!(tool.description(), Some("搜索文档"));
    /// ```
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.function.description.as_deref()
    }

    /// 返回工具参数列表。
    ///
    /// # Returns
    /// 参数定义的有序切片。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{JsonType, Tool};
    ///
    /// let tool = Tool::function("get_weather")
    ///     .param("city", JsonType::String, "城市名称", true)
    ///     .build();
    ///
    /// assert_eq!(tool.parameters().len(), 1);
    /// ```
    #[must_use]
    pub fn parameters(&self) -> &[ToolParameter] {
        &self.function.parameters
    }

    /// 返回工具种类。
    ///
    /// # Returns
    /// 当前工具的种类。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{Tool, ToolKind};
    ///
    /// let tool = Tool::function("search_docs").build();
    /// assert_eq!(tool.kind(), ToolKind::Function);
    /// ```
    #[must_use]
    pub const fn kind(&self) -> ToolKind {
        self.kind
    }
}

/// 工具种类。
///
/// 当前仅支持函数型工具。
///
/// # 示例
/// ```rust
/// use ufox_llm::ToolKind;
///
/// assert_eq!(ToolKind::Function.as_str(), "function");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// 函数型工具。
    Function,
}

impl ToolKind {
    /// 返回工具种类的稳定字符串表示。
    ///
    /// # Returns
    /// 与主流工具调用协议兼容的小写名称。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolKind;
    ///
    /// assert_eq!(ToolKind::Function.as_str(), "function");
    /// ```
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function",
        }
    }
}

/// 工具调用策略。
///
/// 该枚举统一表达不同 Provider 对“是否调用工具、是否强制调用指定工具”的配置方式。
/// 对于不支持某个策略的 Provider，适配层会按能力范围忽略不兼容字段。
///
/// # 示例
/// ```rust
/// use ufox_llm::ToolChoice;
///
/// assert_eq!(ToolChoice::function("get_weather").function_name(), Some("get_weather"));
/// ```
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
    /// 创建强制调用指定工具的策略。
    ///
    /// # Arguments
    /// * `name` - 需要强制调用的工具名称
    ///
    /// # Returns
    /// 指向指定工具名称的策略对象。
    #[must_use]
    pub fn function(name: impl Into<String>) -> Self {
        Self::Function { name: name.into() }
    }

    /// 返回稳定的字符串表示。
    ///
    /// # Returns
    /// 若为枚举型策略，则返回其固定字符串；若为指定函数策略，则返回 `None`。
    #[must_use]
    pub const fn as_str(&self) -> Option<&'static str> {
        match self {
            Self::Auto => Some("auto"),
            Self::None => Some("none"),
            Self::Required => Some("required"),
            Self::Function { .. } => None,
        }
    }

    /// 返回指定函数名称。
    ///
    /// # Returns
    /// 若当前策略为 [`ToolChoice::Function`]，则返回对应函数名。
    #[must_use]
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
///
/// # 示例
/// ```rust
/// use ufox_llm::{JsonType, ToolBuilder};
///
/// let tool = ToolBuilder::new("get_weather")
///     .description("获取城市实时天气")
///     .param("city", JsonType::String, "城市名称", true)
///     .build();
///
/// assert_eq!(tool.parameters().len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolBuilder {
    name: String,
    description: Option<String>,
    parameters: Vec<ToolParameter>,
}

impl ToolBuilder {
    /// 创建工具构建器。
    ///
    /// # Arguments
    /// * `name` - 工具函数名称
    ///
    /// # Returns
    /// 空的工具构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolBuilder;
    ///
    /// let builder = ToolBuilder::new("search_docs");
    /// assert_eq!(builder.name(), "search_docs");
    /// ```
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters: Vec::new(),
        }
    }

    /// 返回构建器当前的工具名称。
    ///
    /// # Returns
    /// 当前正在构建的工具名称。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolBuilder;
    ///
    /// let builder = ToolBuilder::new("search_docs");
    /// assert_eq!(builder.name(), "search_docs");
    /// ```
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 设置工具描述。
    ///
    /// # Arguments
    /// * `description` - 工具功能描述
    ///
    /// # Returns
    /// 设置描述后的构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolBuilder;
    ///
    /// let tool = ToolBuilder::new("search_docs")
    ///     .description("搜索内部文档")
    ///     .build();
    ///
    /// assert_eq!(tool.description(), Some("搜索内部文档"));
    /// ```
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// 追加一个工具参数。
    ///
    /// 参数按调用顺序保存。之所以不在内部直接改用映射结构，是为了让后续序列化输出尽量
    /// 稳定，减少不同平台或测试场景下的无意义差异。
    ///
    /// # Arguments
    /// * `name` - 参数名称
    /// * `json_type` - 参数的 `JSON` 类型
    /// * `description` - 参数说明
    /// * `required` - 是否必填
    ///
    /// # Returns
    /// 追加参数后的构建器。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{JsonType, ToolBuilder};
    ///
    /// let tool = ToolBuilder::new("get_weather")
    ///     .param("city", JsonType::String, "城市名称", true)
    ///     .build();
    ///
    /// assert_eq!(tool.parameters()[0].name(), "city");
    /// ```
    #[must_use]
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
    ///
    /// # Returns
    /// 完整的函数型工具定义对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolBuilder;
    ///
    /// let tool = ToolBuilder::new("search_docs").build();
    /// assert_eq!(tool.name(), "search_docs");
    /// ```
    #[must_use]
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

/// 工具参数定义。
///
/// 该类型描述函数工具中的单个入参，包括名称、类型、说明和是否必填。
///
/// # 示例
/// ```rust
/// use ufox_llm::{JsonType, ToolParameter};
///
/// let parameter = ToolParameter::new("city", JsonType::String, "城市名称", true);
/// assert!(parameter.required());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolParameter {
    name: String,
    json_type: JsonType,
    description: String,
    required: bool,
}

impl ToolParameter {
    /// 创建工具参数定义。
    ///
    /// # Arguments
    /// * `name` - 参数名称
    /// * `json_type` - 参数类型
    /// * `description` - 参数说明
    /// * `required` - 是否必填
    ///
    /// # Returns
    /// 工具参数定义对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{JsonType, ToolParameter};
    ///
    /// let parameter = ToolParameter::new("city", JsonType::String, "城市名称", true);
    /// assert_eq!(parameter.name(), "city");
    /// ```
    #[must_use]
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

    /// 返回参数名称。
    ///
    /// # Returns
    /// 当前参数名称。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{JsonType, ToolParameter};
    ///
    /// let parameter = ToolParameter::new("city", JsonType::String, "城市名称", true);
    /// assert_eq!(parameter.name(), "city");
    /// ```
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 返回参数类型。
    ///
    /// # Returns
    /// 当前参数的 `JSON` 类型描述。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{JsonType, ToolParameter};
    ///
    /// let parameter = ToolParameter::new("city", JsonType::String, "城市名称", true);
    /// assert_eq!(parameter.json_type(), &JsonType::String);
    /// ```
    #[must_use]
    pub const fn json_type(&self) -> &JsonType {
        &self.json_type
    }

    /// 返回参数说明。
    ///
    /// # Returns
    /// 当前参数说明文本。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{JsonType, ToolParameter};
    ///
    /// let parameter = ToolParameter::new("city", JsonType::String, "城市名称", true);
    /// assert_eq!(parameter.description(), "城市名称");
    /// ```
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// 返回参数是否必填。
    ///
    /// # Returns
    /// 若参数为必填，则返回 `true`。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::{JsonType, ToolParameter};
    ///
    /// let parameter = ToolParameter::new("city", JsonType::String, "城市名称", true);
    /// assert!(parameter.required());
    /// ```
    #[must_use]
    pub const fn required(&self) -> bool {
        self.required
    }
}

/// 模型返回的一次工具调用请求。
///
/// 该结构体保留原始参数字符串，调用方可以选择在分发前手动解析，或者使用
/// [`ToolCall::arguments_json`] 获取解析后的 `JSON` 值。
///
/// # 示例
/// ```rust
/// use ufox_llm::ToolCall;
///
/// let call = ToolCall::new("call_1", "get_weather", r#"{"city":"杭州"}"#);
/// assert_eq!(call.name(), "get_weather");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCall {
    /// 创建一次工具调用请求。
    ///
    /// # Arguments
    /// * `id` - 工具调用唯一标识
    /// * `name` - 工具名称
    /// * `arguments` - 模型生成的原始参数字符串
    ///
    /// # Returns
    /// 工具调用对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolCall;
    ///
    /// let call = ToolCall::new("call_1", "get_weather", r#"{"city":"杭州"}"#);
    /// assert_eq!(call.id(), "call_1");
    /// ```
    #[must_use]
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

    /// 返回工具调用唯一标识。
    ///
    /// # Returns
    /// 工具调用的 `id`。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolCall;
    ///
    /// let call = ToolCall::new("call_1", "get_weather", "{}");
    /// assert_eq!(call.id(), "call_1");
    /// ```
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// 返回工具名称。
    ///
    /// # Returns
    /// 模型请求调用的工具名称。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolCall;
    ///
    /// let call = ToolCall::new("call_1", "get_weather", "{}");
    /// assert_eq!(call.name(), "get_weather");
    /// ```
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 返回原始参数字符串。
    ///
    /// # Returns
    /// 模型生成的原始 `JSON` 参数文本。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolCall;
    ///
    /// let call = ToolCall::new("call_1", "get_weather", r#"{"city":"杭州"}"#);
    /// assert_eq!(call.arguments(), r#"{"city":"杭州"}"#);
    /// ```
    #[must_use]
    pub fn arguments(&self) -> &str {
        &self.arguments
    }

    /// 将原始参数解析为 `JSON` 值。
    ///
    /// # Returns
    /// 成功时返回解析后的 `JSON` 值。
    ///
    /// # Errors
    /// - [`LlmError::ParseError`]：当 `arguments` 不是合法 `JSON` 时触发
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolCall;
    ///
    /// let call = ToolCall::new("call_1", "get_weather", r#"{"city":"杭州"}"#);
    /// let json = call.arguments_json().expect("参数应为合法 JSON");
    ///
    /// assert_eq!(json["city"], "杭州");
    /// ```
    pub fn arguments_json(&self) -> Result<Value, LlmError> {
        serde_json::from_str(&self.arguments).map_err(LlmError::from)
    }
}

/// 工具执行结果。
///
/// 该结构体描述本地工具执行后需要回传给模型的一条结果消息。其内容统一保存为字符串，
/// 这样可以兼容纯文本结果和调用方自行序列化后的 `JSON` 结果。
///
/// # 示例
/// ```rust
/// use ufox_llm::ToolResult;
///
/// let result = ToolResult::new("call_1", r#"{"temp":26}"#);
/// assert_eq!(result.tool_call_id(), "call_1");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    tool_call_id: String,
    content: String,
}

impl ToolResult {
    /// 创建工具执行结果。
    ///
    /// # Arguments
    /// * `tool_call_id` - 对应的工具调用标识
    /// * `content` - 需要回传给模型的结果内容
    ///
    /// # Returns
    /// 工具执行结果对象。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolResult;
    ///
    /// let result = ToolResult::new("call_1", "晴，26 摄氏度");
    /// assert_eq!(result.content(), "晴，26 摄氏度");
    /// ```
    #[must_use]
    pub fn new(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }

    /// 从 `JSON` 值创建工具执行结果。
    ///
    /// # Arguments
    /// * `tool_call_id` - 对应的工具调用标识
    /// * `content` - 结构化结果的 `JSON` 值
    ///
    /// # Returns
    /// 结果内容被序列化为紧凑 `JSON` 字符串后的工具执行结果对象。
    ///
    /// # 示例
    /// ```rust
    /// use serde_json::json;
    /// use ufox_llm::ToolResult;
    ///
    /// let result = ToolResult::json("call_1", json!({ "temp": 26 }));
    /// assert_eq!(result.content(), r#"{"temp":26}"#);
    /// ```
    #[must_use]
    pub fn json(tool_call_id: impl Into<String>, content: Value) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            content: content.to_string(),
        }
    }

    /// 返回对应的工具调用标识。
    ///
    /// # Returns
    /// 原始工具调用的唯一标识。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolResult;
    ///
    /// let result = ToolResult::new("call_1", "ok");
    /// assert_eq!(result.tool_call_id(), "call_1");
    /// ```
    #[must_use]
    pub fn tool_call_id(&self) -> &str {
        &self.tool_call_id
    }

    /// 返回结果内容。
    ///
    /// # Returns
    /// 需要回传给模型的结果字符串。
    ///
    /// # 示例
    /// ```rust
    /// use ufox_llm::ToolResult;
    ///
    /// let result = ToolResult::new("call_1", "ok");
    /// assert_eq!(result.content(), "ok");
    /// ```
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct FunctionTool {
    name: String,
    description: Option<String>,
    parameters: Vec<ToolParameter>,
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
