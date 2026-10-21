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

//! Run-scoped wrappers. The first wrapper in a stack is the outermost.
use super::{
    agent::RunContext,
    hooks::{HookError, HookRegistry},
    runtime::Runtime,
};
use crate::{
    Error, Model, StreamOptions,
    providers::EventStream,
    types::{Message, events::ToolCallResult, messages::Metadata},
};
use async_trait::async_trait;
use futures::future::BoxFuture;
use schemars::Schema;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ModelContext {
    pub model: Model,
    pub messages: Vec<Message>,
    pub options: StreamOptions,
}
pub type GenerateContext = ModelContext;
#[derive(Clone, Debug)]
pub struct ToolContext {
    pub tool_call_id: String,
    pub tool_name: String,
    pub kwargs: Value,
}
#[derive(Clone)]
pub struct HookContext {
    pub label: String,
    pub payload: Schema,
    pub payload_name: String,
    pub metadata: Metadata,
    pub tool_call_id: Option<String>,
    pub registry: HookRegistry,
    pub runtime: Runtime,
}
pub type RunNext = Arc<dyn Fn(RunContext) -> EventStream + Send + Sync>;
pub type ModelNext = Arc<dyn Fn(ModelContext) -> EventStream + Send + Sync>;
pub type GenerateNext =
    Arc<dyn Fn(GenerateContext) -> BoxFuture<'static, Result<Message, Error>> + Send + Sync>;
pub type ToolNext =
    Arc<dyn Fn(ToolContext) -> BoxFuture<'static, Result<ToolCallResult, Error>> + Send + Sync>;
pub type HookNext =
    Arc<dyn Fn(HookContext) -> BoxFuture<'static, Result<Value, HookError>> + Send + Sync>;

#[async_trait]
pub trait Middleware: Send + Sync {
    fn wrap_agent_run(&self, call: RunContext, next: RunNext) -> EventStream {
        next(call)
    }
    fn wrap_model(&self, call: ModelContext, next: ModelNext) -> EventStream {
        next(call)
    }
    async fn wrap_generate(
        &self,
        call: GenerateContext,
        next: GenerateNext,
    ) -> Result<Message, Error> {
        next(call).await
    }
    async fn wrap_tool(&self, call: ToolContext, next: ToolNext) -> Result<ToolCallResult, Error> {
        next(call).await
    }
    async fn wrap_hook(&self, call: HookContext, next: HookNext) -> Result<Value, HookError> {
        next(call).await
    }
}

pub fn tool_chain(stack: &[Arc<dyn Middleware>], mut real: ToolNext) -> ToolNext {
    for middleware in stack.iter().rev() {
        let middleware = middleware.clone();
        let next = real;
        real = Arc::new(move |call| {
            let middleware = middleware.clone();
            let next = next.clone();
            Box::pin(async move { middleware.wrap_tool(call, next).await })
        });
    }
    real
}
pub fn hook_chain(stack: &[Arc<dyn Middleware>], mut real: HookNext) -> HookNext {
    for middleware in stack.iter().rev() {
        let middleware = middleware.clone();
        let next = real;
        real = Arc::new(move |call| {
            let middleware = middleware.clone();
            let next = next.clone();
            Box::pin(async move { middleware.wrap_hook(call, next).await })
        });
    }
    real
}
pub fn model_chain(stack: &[Arc<dyn Middleware>], mut real: ModelNext) -> ModelNext {
    for middleware in stack.iter().rev() {
        let middleware = middleware.clone();
        let next = real;
        real = Arc::new(move |call| middleware.wrap_model(call, next.clone()));
    }
    real
}
pub fn run_chain(stack: &[Arc<dyn Middleware>], mut real: RunNext) -> RunNext {
    for middleware in stack.iter().rev() {
        let middleware = middleware.clone();
        let next = real;
        real = Arc::new(move |call| middleware.wrap_agent_run(call, next.clone()));
    }
    real
}
pub fn generate_chain(stack: &[Arc<dyn Middleware>], mut real: GenerateNext) -> GenerateNext {
    for middleware in stack.iter().rev() {
        let middleware = middleware.clone();
        let next = real;
        real = Arc::new(move |call| {
            let middleware = middleware.clone();
            let next = next.clone();
            Box::pin(async move { middleware.wrap_generate(call, next).await })
        });
    }
    real
}
