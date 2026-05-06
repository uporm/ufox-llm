/// ufox-arc：基于 ufox-llm 的轻量 AI Agent 运行时。
pub mod agent;
pub mod error;
pub mod interrupt;
pub mod memory;
pub mod run;
pub mod thread;
pub mod tools;

pub use agent::{Agent, AgentBuilder, AgentConfig, ReflectConfig};
pub use error::ArcError;
pub use interrupt::{
    AutoApproveHandler, CliInterruptHandler, InterruptDecision, InterruptHandler, InterruptReason,
};
pub use memory::{
    InMemoryBackend, MemoryClient, MemoryFilter, MemoryId, MemoryProvider, MemoryScope,
    SqliteBackend,
};
pub use run::{
    ExecutionState, ExecutionStep, Memory, RunEvent, RunEventStream, RunId, RunInput, RunResult,
    RunTrace, StepInput, StepKind, StepOutput,
};
pub use thread::{
    AttachmentKind, AttachmentRef, InMemoryThreadStore, SqliteThreadStore, Thread, ThreadId,
    ThreadSnapshot, ThreadStore, UserId,
};
pub use tools::{
    Tool, ToolError, ToolSpec,
    builtin::{FileReadTool, FileWriteTool, ShellTool},
};
