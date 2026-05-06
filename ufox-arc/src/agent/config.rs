use std::time::Duration;

/// 默认反思提示词：要求 LLM 以 VERDICT/REASON 格式评估当前轨迹。
const DEFAULT_REFLECT_PROMPT: &str = concat!(
    "请回顾上面的对话与工具使用情况。 ",
    "智能体是否已经成功完成了用户目标？ ",
    "请严格按照以下格式回复： ",
    "VERDICT: SUCCESS ",
    "REASON: 简要说明原因 ",
    "如果智能体失败了，或者应该尝试不同的方法，请回复： ",
    "VERDICT: RETRY ",
    "REASON: 说明哪里出了问题，以及下一步应如何调整"
);

/// Reflect 步骤配置。
#[derive(Debug, Clone)]
pub struct ReflectConfig {
    /// 发给 LLM 的反思提示词。
    pub prompt: String,
    /// 最多允许重试的次数，防止无限循环。
    pub max_retries: usize,
}

impl Default for ReflectConfig {
    fn default() -> Self {
        Self {
            prompt: DEFAULT_REFLECT_PROMPT.to_string(),
            max_retries: 2,
        }
    }
}

/// Agent 运行参数。
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// 单次 `chat()` 调用允许的最大推理循环次数。
    pub max_iterations: usize,
    /// 单次 `chat()` 调用的最大超时时间。
    pub timeout: Duration,
    /// 应用于 LLM 请求的采样温度；`None` 则使用模型默认值。
    pub temperature: Option<f32>,
    /// Reflect 步骤配置；`None` 表示禁用。
    pub reflect: Option<ReflectConfig>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            timeout: Duration::from_secs(600),
            temperature: None,
            reflect: Some(ReflectConfig::default()),
        }
    }
}
