/// ufox-arc：基于 ufox-llm 的轻量 AI Agent 运行时。
pub mod agent;
pub mod error;
pub mod interrupt;
pub mod memory;
pub mod run;
pub mod thread;
pub mod tools;

pub use agent::{
    Agent, AgentBuilder, AgentConfig, ExecutionState, ExecutionStep, Memory, MemoryClient,
    StepInput, StepKind, StepOutput,
};
pub use error::ArcError;
pub use interrupt::{
    AutoApproveHandler, CliInterruptHandler, InterruptDecision, InterruptHandler, InterruptReason,
};
pub use memory::{InMemoryStore, MemoryFilter, MemoryId, MemoryScope, MemoryStore, SqliteMemory};
pub use run::{
    RunEvent, RunEventStream, RunId, RunInput, RunRequest, RunResult, RunStatus, RunStep, RunTrace,
};
pub use thread::{
    InMemoryThreadStore, MediaRef, Modality, SqliteThreadStore, Thread, ThreadId, ThreadSnapshot,
    ThreadStore, UserId,
};
pub use tools::{
    Tool, ToolError, ToolSpec,
    builtin::{FileReadTool, FileWriteTool, ShellTool},
};
