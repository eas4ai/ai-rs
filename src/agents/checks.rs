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

//! Checks the default loop runs after each step, and why a run stopped.

use std::{fmt, sync::Arc};

use futures::future::BoxFuture;

use crate::{
    Error,
    types::{Message, MessageRole, usage::Usage},
};

/// What the default loop knows after a step.
#[derive(Clone, Debug)]
pub struct RunProgress {
    /// Steps completed in this run. A response replayed from an earlier run
    /// is not a step.
    pub steps: usize,
    /// The history after the step, including its tool results.
    pub messages: Arc<[Message]>,
    /// The summed usage of this run's model responses, or `None` when none
    /// reported usage.
    pub usage: Option<Usage>,
}

impl RunProgress {
    /// The model response of the step that just finished.
    #[must_use]
    pub fn last_response(&self) -> Option<&Message> {
        self.messages
            .iter()
            .rev()
            .find(|message| message.role == MessageRole::Assistant)
    }
}

/// A step check's answer.
#[derive(Clone, Debug, PartialEq)]
#[expect(
    clippy::large_enum_variant,
    reason = "A check returns one short-lived decision per step; boxing would only complicate steering"
)]
pub enum StepDecision {
    /// Let the run go on, or end it if it is done.
    Continue,
    /// Add this user message to the history and call the model again, even
    /// after a final answer.
    Steer(Message),
    /// End the run with this reason.
    Stop(String),
}

type CheckFn = dyn Fn(RunProgress) -> BoxFuture<'static, Result<StepDecision, Error>> + Send + Sync;

/// An async check the default loop runs after each step. A check that fails
/// ends the run with its error.
#[derive(Clone)]
pub struct StepCheck(Arc<CheckFn>);

impl StepCheck {
    pub fn new<F, Fut>(check: F) -> Self
    where
        F: Fn(RunProgress) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<StepDecision, Error>> + Send + 'static,
    {
        Self(Arc::new(move |progress| Box::pin(check(progress))))
    }

    /// Stop once the run has made `count` steps.
    #[must_use]
    pub fn step_count(count: usize) -> Self {
        Self::new(move |progress: RunProgress| {
            let decision = if progress.steps >= count {
                StepDecision::Stop(format!("step count {count} reached"))
            } else {
                StepDecision::Continue
            };
            async move { Ok(decision) }
        })
    }

    /// Stop after a step whose response called one of these tools.
    pub fn tool_called<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let names: Vec<String> = names.into_iter().map(Into::into).collect();
        Self::new(move |progress: RunProgress| {
            let called = progress.last_response().and_then(|response| {
                response
                    .tool_calls()
                    .into_iter()
                    .find(|call| names.contains(&call.tool_name))
                    .map(|call| call.tool_name.clone())
            });
            let decision = match called {
                Some(name) => StepDecision::Stop(format!("tool {name} was called")),
                None => StepDecision::Continue,
            };
            async move { Ok(decision) }
        })
    }

    pub(crate) fn run(
        &self,
        progress: RunProgress,
    ) -> BoxFuture<'static, Result<StepDecision, Error>> {
        (self.0)(progress)
    }
}

impl fmt::Debug for StepCheck {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("StepCheck(..)")
    }
}

/// Why a finished agent run stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The model's last response is a final answer.
    Done,
    /// A step check stopped the run with this reason.
    Stopped(String),
    /// A hook, such as a tool approval, was deferred.
    Deferred,
}
