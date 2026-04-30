use std::time::Duration;

/// Agent 运行参数。
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// 单次 `chat()` 调用允许的最大推理循环次数。
    pub max_iterations: usize,
    /// 单次 `chat()` 调用的最大超时时间。
    pub timeout: Duration,
    /// 应用于 LLM 请求的采样温度；`None` 则使用模型默认值。
    pub temperature: Option<f32>,
    /// 是否启用 Perceive 步骤（从记忆检索上下文）。
    pub enable_perceive: bool,
    /// 是否启用 Observe 步骤（格式化工具结果）。
    pub enable_observe: bool,
    /// 是否启用 Reflect 步骤（自我评估是否重试）。
    pub enable_reflect: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            timeout: Duration::from_secs(300),
            temperature: None,
            // 默认简单模式：只运行 Think/Act/Completion
            enable_perceive: false,
            enable_observe: false,
            enable_reflect: false,
        }
    }
}
