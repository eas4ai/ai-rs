#![doc = include_str!("README.md")]
pub mod agent;
pub mod aggregators;
mod buffer;
pub mod checks;
pub mod hooks;
pub mod middleware;
pub mod runtime;
pub mod tools;

pub use agent::{Agent, AgentStream, Context, RunContext, RunOptions};
pub use aggregators::*;
pub use checks::{RunProgress, StepCheck, StepDecision, StopReason};
pub use hooks::{HookError, HookRegistry};
pub use middleware::Middleware;
pub use runtime::Runtime;
pub use tools::{
    AgentTool, BoundToolCall, ToolContext, ToolError, ToolResolver, ToolRunner, ToolTask,
    UnavailableToolError, deferred_tool_result, error_tool_result, tool_result, yield_from,
};
