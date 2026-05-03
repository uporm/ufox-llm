/// ufox-arc：基于 ufox-llm 的轻量 AI Agent 运行时。
pub mod agent;
pub mod error;
pub mod interrupt;
pub mod memory;
pub mod session;
pub mod tools;

pub use agent::{
    Agent, AgentBuilder, AgentConfig, ExecutionState, ExecutionStep, Memory, StepInput, StepKind,
    StepOutput,
};
pub use error::ArcError;
pub use interrupt::{
    AutoApproveHandler, CliInterruptHandler, InterruptDecision, InterruptHandler, InterruptReason,
};
pub use memory::{InMemoryStore, MemoryFilter, MemoryId, MemoryScope, MemoryStore, SqliteMemory};
pub use session::{
    ExecutionEvent, ExecutionEventStream, ExecutionResult, ExecutionTrace, InMemorySessionStore,
    MediaRef, Modality, Session, SessionId, SessionInput, SessionStore, SqliteSessionStore, UserId,
};
pub use tools::{
    Tool, ToolError, ToolMetadata, ToolRegistry,
    builtin::{FileReadTool, FileWriteTool, ShellTool},
};
