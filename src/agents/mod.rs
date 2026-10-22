// Copyright 2026 Vercel, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
