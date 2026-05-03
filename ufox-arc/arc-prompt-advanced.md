# ufox-arc 第二版（v2）设计

## 13. v2 定位与升级原则

v2 在 v1 核心能力稳定的基础上，新增两类能力：

1. **技能系统（Skills）**：可复用、可组合的命名能力，将系统提示词 + 工具子集 + 执行配置打包成可调用单元
2. **多 Agent 协作（Multi-Agent）**：多个独立 Agent 通过消息传递协作，完成单 Agent 无法完成的复杂任务

**升级原则：**

- v2 完全向后兼容 v1 API，已有代码无需修改
- 技能和多 Agent 只是 v1 核心能力的"组合"，不引入全新的执行模型
- 保持同样的"不过度设计"原则：第一次就跑通主路径，边界情况后补
- Skills 是轻量的：本质是"带配置的 Agent"，不是全新的运行时
- Multi-Agent 是无共享的：各 Agent 拥有独立 Session，通过消息传递协作

## 14. 技能系统（Skills）

### 14.1 设计思路

Skills 遵循 Anthropic Agent Skills 规范：**技能是自主执行单元，不预设配置，由 LLM 在运行时自主发现所需工具和子技能**。

核心模型：

```text
Skill = name + description

执行时：
  - 框架根据 description 自动生成执行上下文
  - 技能内部 LLM 可访问所有已注册工具
  - 技能内部 LLM 可发现并调用其他技能（调用栈保护）
  - 调用方只需 session.chat()，技能选择和调用完全自主
```

**上下文预算问题：**

技能数量增加后，将全部技能注入每次 LLM 调用会撑爆上下文（每个 skill schema ≈ 100–200 token）。框架采用**按需选择**策略：

```text
注册技能数 ≤ max_skills_per_call（默认 8）→ 全量注入

注册技能数 > max_skills_per_call
  → 按当前输入的语义相关性选 top-K 注入
  → 同时注入 discover_skill 元工具，让 LLM 随时按需发现其余技能
```

`discover_skill` 元工具始终可用，LLM 可随时用它检索未注入上下文的技能。

### 14.2 推荐接口

**Skill 结构（极简）：**

```rust
pub struct Skill {
    pub name: String,
    /// 清晰描述技能的能力边界，LLM 根据这个描述决定何时调用或发现该技能
    pub description: String,
}
```

框架执行技能时，自动构建执行提示：

```text
你正在执行技能「{name}」：{description}。
利用所有可用的工具和技能来完成当前任务。
```

**SkillRegistry：**

```rust
pub struct SkillRegistry {
    skills: HashMap<String, Skill>,
}

impl SkillRegistry {
    pub fn register(&mut self, skill: Skill) -> Result<()>;
    /// 替换已有技能定义（不存在时报错）
    pub fn reload(&mut self, skill: Skill) -> Result<()>;
    pub fn unregister(&mut self, name: &str) -> Result<()>;
    pub fn get(&self, name: &str) -> Option<&Skill>;
    pub fn list_names(&self) -> Vec<String>;
}
```

**Agent 动态管理（线程安全，内部持有 `Arc<RwLock<SkillRegistry>>`）：**

```rust
impl AgentBuilder {
    pub fn skill(mut self, skill: Skill) -> Self;
    /// 单次 LLM 调用最多注入几个技能（默认 8）
    pub fn max_skills_per_call(mut self, n: usize) -> Self;
}

impl Agent {
    /// 运行时注册新技能，立即对后续执行生效。
    pub fn add_skill(&self, skill: Skill) -> Result<()>;
    /// 运行时替换技能描述（热更新）。
    pub fn reload_skill(&self, skill: Skill) -> Result<()>;
    pub fn remove_skill(&self, name: &str) -> Result<()>;
}
```

**ExecutionConfig 新增技能配置：**

```rust
pub struct ExecutionConfig {
    // v1 字段不变
    pub max_iterations: usize,
    pub timeout: Duration,
    pub enable_perceive: bool,
    pub enable_observe: bool,
    pub enable_reflect: bool,
    pub max_concurrent_tools: usize,
    /// 单次 LLM 调用注入的最大技能数；超出时按相关性选 top-K（默认 8）
    pub max_skills_per_call: usize,
    /// 技能嵌套最大深度；超出后停止注入技能，LLM 仍可用工具完成任务（默认 3）
    pub max_skill_depth: usize,
}
```

**Session API 不变：**

```rust
// chat() / chat_stream() 完全不变，技能对调用方零感知
let result = session.chat("任务描述").await?;
```

### 14.3 框架内部机制

**Skill → pseudo-tool 转换：**

```json
{
  "name": "<skill.name>",
  "description": "<skill.description>",
  "parameters": {
    "type": "object",
    "properties": {
      "input": { "type": "string", "description": "发给该技能的任务描述" }
    },
    "required": ["input"]
  }
}
```

**discover_skill 元工具（技能数超预算时自动注入）：**

```json
{
  "name": "discover_skill",
  "description": "搜索当前未加载的技能。当现有技能无法满足需求时调用。",
  "parameters": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "描述你需要的能力" }
    },
    "required": ["query"]
  }
}
```

返回匹配的技能名称和描述列表，LLM 可立即调用其中的技能。

**工具列表构建规则：**

```text
每次 LLM 调用前：

1. 取当前输入作为 query
2. 从 SkillRegistry 中排除调用栈已有的技能（防循环）
3. 若剩余技能数 ≤ max_skills_per_call → 全量注入
   否则 → 按 query 相关性选 top-K，并注入 discover_skill 元工具
4. 若当前调用深度 ≥ max_skill_depth → 不注入任何技能（只保留工具）

工具列表 = agent.tools（全量，技能不限制工具访问）
          + 选出的技能 pseudo-tools
          + discover_skill（当技能超预算时）
```

**相关性打分策略（按优先级选用）：**

| 策略 | 条件 | 说明 |
|------|------|------|
| 嵌入向量相似度 | 配置了 embedding 客户端 | 精准，需额外 embed 调用 |
| 关键词匹配 | 默认回退 | 轻量，无需额外调用 |

**调用栈保护：**

- 同名技能已在调用栈 → 从可选列表静默排除（LLM 看不到，不报错）
- 深度 ≥ `max_skill_depth` → 停止注入任何技能 pseudo-tool，LLM 只用工具继续完成任务

两种情况都不中断执行，降级而非报错。

### 14.4 执行流程示例

```text
session.chat("研究 Rust async trait，审查代码，综合成报告")
  │
  ├─ 技能数(3) ≤ max_skills_per_call(8)，全量注入
  ├─ 工具列表：[web_search, file_write, research↗, code_review↗, tech_report↗]
  │
  ├─ LLM Think → 调用 tech_report↗("研究+审查+报告")   depth=0
  │    ├─ 调用栈：[tech_report]，排除 tech_report 自身
  │    ├─ 剩余技能：[research, code_review]，全量注入
  │    ├─ 工具列表：[web_search, file_write, research↗, code_review↗]
  │    │
  │    ├─ tech_report LLM → 调用 research↗("async trait 现状")   depth=1
  │    │    ├─ 调用栈：[tech_report, research]，排除两者
  │    │    ├─ 剩余技能：[code_review]，注入
  │    │    ├─ research LLM 调用 web_search → 整理结果
  │    │    └─ 结果以 ToolResult 返回 tech_report LLM
  │    │
  │    ├─ tech_report LLM → 调用 code_review↗("```rust...```")   depth=1
  │    │    └─ code_review LLM 分析代码 → ToolResult 返回
  │    │
  │    └─ tech_report LLM 综合 → 调用 file_write 保存报告 → Completion
  │
  └─ 主 LLM 收到 tech_report 结果 → Completion
```

### 14.5 使用示例

**示例 1：自动发现与调用**

```rust
use ufox_arc::{Agent, Skill};
use ufox_llm::Client;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Client::from_env()?;

    let agent = Agent::builder()
        .llm(client)
        .tool(WebSearchTool::new())
        .tool(FileWriteTool::new())
        .skill(Skill {
            name: "research".to_string(),
            description: "深度调研技术话题，搜索并核实信息，输出结构化调研报告".to_string(),
        })
        .skill(Skill {
            name: "code_review".to_string(),
            description: "审查 Rust 代码的内存安全、并发安全、错误处理和性能".to_string(),
        })
        .skill(Skill {
            name: "tech_report".to_string(),
            description: "生成综合技术报告，整合调研结论和代码质量评估，保存到文件".to_string(),
        })
        .build()?;

    let mut session = agent.session("user_123", None).await?;

    // LLM 自主判断：直接用 tech_report，tech_report 再自主调用 research 和 code_review
    let result = session.chat(
        "研究 Rust async trait 最新进展，审查这段代码，综合成报告：\n```rust\n...\n```"
    ).await?;
    println!("{}", result.response.text);

    Ok(())
}
```

**示例 2：大量技能，上下文自动管理**

```rust
let agent = Agent::builder()
    .llm(client)
    .max_skills_per_call(5)  // 超出 5 个时按相关性选择 + discover_skill 兜底
    .skill(Skill { name: "research".to_string(),      description: "调研技术话题...".to_string() })
    .skill(Skill { name: "code_review".to_string(),   description: "审查 Rust 代码...".to_string() })
    .skill(Skill { name: "security_audit".to_string(), description: "安全漏洞审计...".to_string() })
    .skill(Skill { name: "perf_analysis".to_string(), description: "性能瓶颈分析...".to_string() })
    .skill(Skill { name: "doc_writer".to_string(),    description: "编写技术文档...".to_string() })
    .skill(Skill { name: "refactor".to_string(),      description: "代码重构建议...".to_string() })
    .skill(Skill { name: "test_writer".to_string(),   description: "生成测试用例...".to_string() })
    .skill(Skill { name: "deploy_check".to_string(),  description: "部署前置检查...".to_string() })
    // 8 个技能 > max_skills_per_call(5)
    // 框架自动按相关性选 5 个 + 注入 discover_skill
    .build()?;

let mut session = agent.session("user_123", None).await?;
let result = session.chat("对这个 PR 做安全审计和性能分析").await?;
// 框架自动选出 security_audit / perf_analysis 等相关技能注入
println!("{}", result.response.text);
```

**示例 3：运行时动态加载**

```rust
agent.add_skill(Skill {
    name: "k8s_ops".to_string(),
    description: "检查和操作 Kubernetes 集群，排查 Pod 问题、扩缩容、查看日志".to_string(),
})?;

agent.reload_skill(Skill {
    name: "code_review".to_string(),
    description: "审查 Rust 代码，重点关注 unsafe 块、生命周期和异步安全".to_string(),
})?;

agent.remove_skill("deploy_check")?;
```

### 14.6 设计决策

- **Skill 只有 name + description**：行为完全由 LLM 在运行时自主决定，框架不预设工具子集或子技能列表
- **技能对调用方零感知**：`chat()` 不变，技能选择和调用由 LLM 驱动
- **上下文预算是硬约束**：`max_skills_per_call` 限制注入数量，超出时选最相关的而不是全量
- **`discover_skill` 作为安全网**：LLM 永远有办法找到未注入的技能，预算限制不丢失能力
- **循环和超深度都静默降级**：不报错、不中断，调用栈中已有的技能被排除出可选列表，深度超限时停止注入技能
- **技能访问全量工具**：技能内部 LLM 可访问所有注册工具，由 LLM 自行判断使用哪些


## 15. 多 Agent 协作

### 15.1 设计思路

多 Agent 系统的核心挑战是：如何让多个 Agent 协作，同时保持架构简单？

**选定方案：协调器模式（Coordinator Pattern）**

```text
用户请求
    ↓
协调器 Agent（Coordinator）
    ↓ 分析任务，决定分配
    ├→ 成员 Agent A（如：研究员）→ 结果 A
    ├→ 成员 Agent B（如：代码专家）→ 结果 B
    └→ 成员 Agent C（如：审查员）→ 结果 C
    ↓ 汇总所有结果
最终回复
```

**关键决策：**

- 每个成员 Agent 拥有独立的 `Session`，不共享状态
- 协调器与成员之间只通过字符串消息传递，不传递内部结构
- 成员 Agent 的结果以 `ToolResult` 的形式回注到协调器的上下文
- 调用方只感知协调器，对成员 Agent 透明

### 15.2 推荐接口

```rust
pub struct AgentTeamMember {
    pub name: String,
    pub description: String,
    pub agent: Agent,
}

pub struct AgentTeam {
    coordinator: Agent,
    members: HashMap<String, AgentTeamMember>,
}

impl AgentTeam {
    pub fn builder() -> AgentTeamBuilder;

    /// 以协调器视角运行整个任务，返回汇总结果。
    pub async fn run(
        &self,
        user_id: impl Into<UserId>,
        task: impl Into<SessionInput>,
    ) -> Result<ExecutionResult>;

    /// 流式版本，事件中包含每个成员 Agent 的中间步骤。
    pub async fn run_stream(
        &self,
        user_id: impl Into<UserId>,
        task: impl Into<SessionInput>,
    ) -> Result<ExecutionEventStream>;
}

pub struct AgentTeamBuilder {
    coordinator: Option<Agent>,
    members: Vec<AgentTeamMember>,
}

impl AgentTeamBuilder {
    pub fn coordinator(mut self, agent: Agent) -> Self;
    pub fn member(mut self, name: impl Into<String>, description: impl Into<String>, agent: Agent) -> Self;
    pub fn build(self) -> Result<AgentTeam>;
}
```

协调器通过工具调用成员 Agent，框架负责路由：

```rust
/// 框架内部生成的"委派工具"，协调器可以用它来调用成员 Agent。
/// 外部使用者不需要手动创建这个工具，AgentTeam 会自动注入。
struct DelegateTool {
    member_name: String,
    member_description: String,
}
```

### 15.3 使用示例

```rust
use ufox_arc::{Agent, AgentTeam};
use ufox_llm::Client;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let client = Client::from_env()?;

    // 成员：研究员 Agent
    let researcher = Agent::builder()
        .llm(Client::from_env()?)
        .system("你是一位严谨的研究员，负责收集和核实信息。")
        .tool(WebSearchTool::new())
        .build()?;

    // 成员：代码专家 Agent
    let coder = Agent::builder()
        .llm(Client::from_env()?)
        .system("你是一位 Rust 专家，负责编写和优化代码。")
        .tool(FileReadTool::new())
        .tool(FileWriteTool::new())
        .build()?;

    // 成员：审查员 Agent
    let reviewer = Agent::builder()
        .llm(Client::from_env()?)
        .system("你是一位严谨的审查员，负责找出潜在问题。")
        .build()?;

    // 协调器
    let coordinator = Agent::builder()
        .llm(client)
        .system(
            "你是一位项目协调员。接到任务后，先分析需要哪些专家，\
             然后委派给对应的成员，最后汇总结果给用户。",
        )
        .build()?;

    let team = AgentTeam::builder()
        .coordinator(coordinator)
        .member("researcher", "收集和核实信息", researcher)
        .member("coder", "编写和优化 Rust 代码", coder)
        .member("reviewer", "审查潜在问题", reviewer)
        .build()?;

    let result = team.run("user_123", "研究 Rust 的 async trait 最佳实践，写一个示例，并审查代码质量").await?;
    println!("{}", result.response.text);

    Ok(())
}
```

### 15.4 执行流程说明

```text
1. team.run(user_id, task)
   ├─ 协调器创建新 Session（session_id 自动生成）
   ├─ 框架为协调器注入 delegate_to_researcher / delegate_to_coder / delegate_to_reviewer 工具
   
2. 协调器 Think 步骤
   └─ LLM 输出：调用 delegate_to_researcher("研究 async trait")

3. 框架拦截 delegate_to_* 工具调用
   ├─ 为对应成员 Agent 创建独立 Session（user_id 相同，session_id 新生成）
   ├─ 在成员 Session 中执行完整的 Think/Act 循环
   └─ 将成员结果转为 ToolResult 回注到协调器上下文

4. 协调器继续 Think
   └─ 可继续调用其他成员，或直接生成最终答复

5. 协调器 Completion → 返回给调用方
```

### 15.5 设计决策

- **成员 Agent 对协调器透明**：协调器只看到"委派工具"，不知道背后是 Agent 还是普通工具
- **不共享 Session**：成员 Agent 拥有独立上下文，避免状态污染
- **不引入消息总线（MessageBus）**：v2 不需要异步事件驱动的消息总线，调用方式就足够
- **不引入 `AgentRole` 枚举**：协调器和成员的区别只在于是否被注入了委派工具，不需要专门的角色概念
- **不引入 `AgentLoop` 新类型**：AgentTeam 复用同一个执行循环，不需要单独定义
- **成员 Agent 的 Memory 是独立的**：如果需要共享记忆，在构建时注入同一个 `MemoryStore` 实例即可

### 15.6 共享记忆（可选）

如果协调器和成员需要共享用户级记忆，在构建时注入同一个 `MemoryStore`：

```rust
use std::sync::Arc;

let shared_memory = Arc::new(SqliteMemory::open("./team_memory.db").await?);

let researcher = Agent::builder()
    .llm(Client::from_env()?)
    .memory(Arc::clone(&shared_memory))
    .build()?;

let coder = Agent::builder()
    .llm(Client::from_env()?)
    .memory(Arc::clone(&shared_memory))
    .build()?;

let coordinator = Agent::builder()
    .llm(client)
    .memory(shared_memory)  // 协调器也共享同一个 MemoryStore
    .build()?;
```

## 16. 并行工具执行

v2 支持在单次 LLM 响应中并发执行多个工具调用，而不是串行等待。

### 16.1 推荐接口

```rust
pub struct ExecutionConfig {
    // v1 字段保持不变
    pub max_iterations: usize,
    pub timeout: Duration,
    pub enable_perceive: bool,
    pub enable_observe: bool,
    pub enable_reflect: bool,
    
    // v2 新增
    /// 工具并发执行时的最大并发数；1 表示串行（v1 行为）
    pub max_concurrent_tools: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            timeout: Duration::from_secs(300),
            enable_perceive: false,
            enable_observe: false,
            enable_reflect: false,
            max_concurrent_tools: 4,  // v2 默认并发
        }
    }
}
```

执行逻辑变化：

```rust
// v1：串行执行
for tool_call in tool_calls {
    let result = manager.execute(&tool_call, interrupt_handler).await?;
    results.push(result);
}

// v2：并发执行（受 max_concurrent_tools 限制）
let results = futures::stream::iter(tool_calls)
    .map(|tc| manager.execute(&tc, interrupt_handler))
    .buffer_unordered(config.max_concurrent_tools)
    .collect::<Result<Vec<_>>>()
    .await?;
```

### 16.2 HITL 与并发执行

当工具需要人工确认时，并发执行会暂停等待确认：

- 需要确认的工具单独排队，等待用户响应
- 不需要确认的工具继续并发执行
- 全部完成后再进入下一个 Think 步骤

## 17. v2 模块目录扩展

在 v1 目录结构基础上，新增以下模块：

```text
ufox-arc/
├── src/
│   ├── ...（v1 保持不变）
│   ├── skills/
│   │   ├── mod.rs
│   │   └── registry.rs
│   └── team/
│       ├── mod.rs
│       ├── builder.rs
│       └── delegate.rs
└── examples/
    ├── ...（v1 保持不变）
    ├── skill_agent.rs
    └── multi_agent_team.rs
```

## 18. v2 分阶段实施计划

v2 在 v1 阶段 7 完成后继续推进，阶段编号延续：

### 阶段 8：技能系统

**目标**：实现技能自动发现与调用，LLM 根据任务自主选择和嵌套使用技能，支持运行时动态加载，解决大量技能注册时的上下文预算问题。

**本阶段必须实现：**

- `Skill` 数据结构（只有 `name` + `description`）
- `SkillRegistry`：`register`/`reload`/`unregister`/`get`/`list_names`
- `Agent` 内部持有 `Arc<RwLock<SkillRegistry>>`
- `AgentBuilder::skill()` 和 `AgentBuilder::max_skills_per_call()` 注册入口
- `Agent::add_skill()` / `reload_skill()` / `remove_skill()` 动态管理
- Skill → pseudo-tool 的自动转换（`src/skills/pseudo_tool.rs`）
- 上下文预算选择逻辑：技能数 ≤ `max_skills_per_call` 时全量注入，否则按相关性 top-K
- `discover_skill` 元工具：技能超预算时自动注入，支持 LLM 按需发现剩余技能
- 关键词匹配作为默认相关性打分（无需 embedding 调用）
- 调用栈维护：同名技能静默排除（不报错）
- 深度超限（≥ `max_skill_depth`）时停止注入技能，降级为只用工具
- 示例 `examples/skill_agent.rs`（含多技能、嵌套、动态加载）

**完成定义：**

- `session.chat()` 不变，技能对调用方完全透明
- LLM 根据技能 `description` 自主决定是否调用、嵌套调用哪些技能
- 技能执行时可访问所有注册工具（不受限制）
- 技能可自主发现并调用其他技能（调用栈排除自身，防止直接循环）
- 注册技能数超过 `max_skills_per_call` 时，`discover_skill` 元工具正确工作
- 动态注册/热更新/移除后，后续执行立即感知
- 深度超限时静默降级，执行不中断
- 示例可以运行

**建议落点：**

- `src/skills/mod.rs`
- `src/skills/registry.rs`
- `src/skills/pseudo_tool.rs`（Skill → Tool schema 转换 + discover_skill）
- `src/skills/selector.rs`（top-K 相关性选择逻辑）
- `src/agent/mod.rs`（builder + 动态管理）
- `src/session/mod.rs`（执行时技能注入与调用栈）
- `examples/skill_agent.rs`

### 阶段 9：多 Agent 协作

**目标**：支持协调器 + 成员 Agent 的团队协作模式。

**本阶段必须实现：**

- `AgentTeamMember` 结构体
- `AgentTeam` 与 `AgentTeamBuilder`
- 委派工具（`DelegateTool`）的自动生成与注入
- `team.run()` 的完整执行流程
- 成员 Agent 的独立 Session 创建与管理
- 成员执行结果转换为 `ToolResult` 回注协调器
- 示例 `examples/multi_agent_team.rs`

**本阶段不要做：**

- 成员 Agent 之间直接通信（绕过协调器）
- 成员并发执行（第一版串行，避免复杂度）
- 动态增减团队成员

**完成定义：**

- 协调器能正确路由任务到成员 Agent
- 成员执行结果正确流回协调器
- 协调器能汇总多个成员结果并输出最终回复
- 支持共享 `MemoryStore` 实现团队级记忆
- 至少有一个协调器 + 2 个成员的完整示例

**建议落点：**

- `src/team/mod.rs`
- `src/team/builder.rs`
- `src/team/delegate.rs`
- `examples/multi_agent_team.rs`

## 19. v2 实施检查清单

### 阶段 8 检查清单

- [ ] `Skill` 结构体只有 `name` + `description` 两个字段
- [ ] `SkillRegistry` 支持 `register`/`reload`/`unregister`
- [ ] `Agent` 内部持有 `Arc<RwLock<SkillRegistry>>`
- [ ] `AgentBuilder::skill()` 和 `max_skills_per_call()` 可以正常使用
- [ ] `Agent::add_skill()` 运行时添加技能，立即生效
- [ ] `Agent::reload_skill()` 运行时热更新技能描述
- [ ] `Agent::remove_skill()` 运行时移除技能
- [ ] Skill 正确转换为 pseudo-tool schema 并注入 LLM 工具列表
- [ ] 技能数 ≤ `max_skills_per_call` 时全量注入
- [ ] 技能数 > `max_skills_per_call` 时按关键词相关性选 top-K
- [ ] 超预算时 `discover_skill` 元工具被正确注入且可用
- [ ] 技能调用时当前技能从可选列表中静默排除（防直接循环）
- [ ] 深度 ≥ `max_skill_depth` 时停止注入技能，执行不中断
- [ ] 技能内部 LLM 可访问所有注册工具
- [ ] `examples/skill_agent.rs` 可以运行（含嵌套与动态加载场景）

### 阶段 9 检查清单

- [ ] `AgentTeam::builder()` 可以正常构造
- [ ] 框架自动为协调器注入委派工具
- [ ] `team.run()` 能完整执行协调器 → 成员 → 协调器的流程
- [ ] 成员 Agent 拥有独立 Session
- [ ] 成员执行结果正确转换为 `ToolResult`
- [ ] 共享 `MemoryStore` 可以在团队成员间生效
- [ ] `examples/multi_agent_team.rs` 可以运行
- [ ] 成员不存在时返回清晰错误

## 20. v2 常见问题

### Q7: 技能（Skill）和工具（Tool）的区别是什么？

**A:** 工具是 LLM 调用的原子操作（查天气、读文件），执行结束返回数据；技能是一个完整的 Agent 执行单元，有自己的系统提示词、工具子集和执行循环。技能在框架层被转换为 pseudo-tool，让 LLM 可以自主发现并调用，但执行机制远比普通工具复杂（完整 Think/Act 循环、可进一步调用子技能）。

### Q8: 多 Agent 中的成员 Agent 能不能共享会话上下文？

**A:** 不能，也不应该。成员 Agent 拥有独立 Session，避免状态污染。如果需要共享信息，通过两种方式：（1）协调器在委派时把上下文摘要传给成员；（2）共享同一个 `MemoryStore` 实例（见第 15.6 节）。

### Q9: 为什么多 Agent 不用消息总线（MessageBus）？

**A:** 消息总线适合异步事件驱动的场景，而这里的场景是同步请求-响应：协调器发出委派，等待结果，再继续。直接函数调用更简单、更容易调试、更容易推断执行顺序。

### Q10: 技能执行失败了怎么办？

**A:** 技能执行失败返回 `ArcError`，与普通 `chat()` 失败的错误处理方式相同。调用方可以 `?` 传播，或者 `.unwrap_or_else()` 降级处理。技能执行前已进入 Session 的消息历史不会回滚。

### Q11: v2 还有哪些东西没有做？

**A:** 以下能力有意推迟到 v3 或更后期：

- 成员 Agent 并发执行（AgentTeam 内并发）
- 分布式 Agent（跨进程/跨机器）
- Agent 持久化与迁移

---

**文档版本：** v2.0  
**最后更新：** 2026-04-29  
**维护者：** ufox-arc team
