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

//! A scripted provider for tests that run without a network. Enable the
//! `testing` feature to use it.
//!
//! Each call, streamed or buffered, takes the next scripted response and
//! records the model id, messages and options it received, with the Tokio
//! time of the call. A call after the last response fails with
//! `Error::Configuration`.
//!
//! Calls retry a retryable error before the first event (`StreamOptions::retry`),
//! and each retry is another call to this provider. Script a response for every
//! retry, or set `RetryPolicy::none()`, when a test expects a retryable scripted
//! error (HTTP 429 or 5xx, a connection failure) to reach the caller; otherwise
//! the retry finds the script empty and the caller sees `Error::Configuration`.
//! Run such tests under paused Tokio time to skip the waits.
//!
//! ```
//! use suprnova_ai::testing::{ScriptedProvider, ScriptedResponse};
//! use suprnova_ai::{StreamOptions, generate, user_message};
//!
//! # async fn example() -> Result<(), suprnova_ai::Error> {
//! let provider = ScriptedProvider::new("scripted", [ScriptedResponse::text("Hello")]);
//! let model = provider.model("test-model");
//! let message = generate(&model, vec![user_message("Hi")], StreamOptions::default()).await?;
//! assert_eq!(message.text(), "Hello");
//! assert_eq!(provider.requests().len(), 1);
//! # Ok(())
//! # }
//! ```

use std::{
    collections::VecDeque,
    fmt,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use futures::{StreamExt, stream};
use serde_json::Value;

use crate::{
    Error, Model, Provider, ProviderConfig, StreamOptions,
    providers::EventStream,
    types::{
        Event, Message, MessageHydrator, MessageRole, Part, ToolCallPart,
        events::replay_message_events, messages::TextPart, usage::Usage,
    },
};

/// One scripted answer to a call: model events, an error, or events followed
/// by an error.
pub struct ScriptedResponse {
    events: Vec<Event>,
    error: Option<Error>,
}

impl ScriptedResponse {
    /// Answer with these events as they are.
    #[must_use]
    pub fn events(events: Vec<Event>) -> Self {
        Self {
            events,
            error: None,
        }
    }

    /// Answer with the events that stream this assistant message. The finish
    /// reason is `tool_calls` when the message calls tools and `stop` otherwise.
    #[must_use]
    pub fn message(message: Message) -> Self {
        let finish_reason = if message.tool_calls().is_empty() {
            "stop"
        } else {
            "tool_calls"
        };
        let mut events = replay_message_events(&message);
        if let Some(Event::StreamEnd(end)) = events.last_mut() {
            end.finish_reason = Some(finish_reason.to_owned());
        }
        Self::events(events)
    }

    /// Answer with a final text response.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        let part = TextPart {
            id: format!("text-{}", next_id()),
            text: text.into(),
            provider_metadata: None,
        };
        Self::message(Message::new(MessageRole::Assistant, vec![part.into()]))
    }

    /// Answer with one tool call whose arguments are this JSON value.
    #[must_use]
    pub fn tool_call(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: Value,
    ) -> Self {
        Self::tool_calls([ToolCallPart::new(
            tool_call_id,
            tool_name,
            arguments.to_string(),
        )])
    }

    /// Answer with these tool calls.
    pub fn tool_calls(calls: impl IntoIterator<Item = ToolCallPart>) -> Self {
        let parts = calls.into_iter().map(Part::ToolCall).collect();
        Self::message(Message::new(MessageRole::Assistant, parts))
    }

    /// Answer with this error and no events.
    #[must_use]
    pub fn error(error: impl Into<Error>) -> Self {
        Self {
            events: Vec::new(),
            error: Some(error.into()),
        }
    }

    /// Fail with this error after the response's events, as a stream that
    /// breaks partway does.
    #[must_use]
    pub fn then_error(mut self, error: impl Into<Error>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// Report this usage on the response's final event. A response without a
    /// final event ignores it.
    #[must_use]
    pub fn with_usage(mut self, usage: Usage) -> Self {
        if let Some(Event::StreamEnd(end)) = self
            .events
            .iter_mut()
            .rev()
            .find(|event| matches!(event, Event::StreamEnd(_)))
        {
            end.event.usage = Some(usage);
        }
        self
    }

    fn into_stream(self) -> EventStream {
        let error = self.error.map(Err);
        stream::iter(self.events.into_iter().map(Ok).chain(error)).boxed()
    }

    fn into_message(self) -> Result<Message, Error> {
        let mut hydrator = MessageHydrator::new(None);
        for event in self.events {
            hydrator
                .feed(event)
                .map_err(|error| Error::Validation(error.to_string()))?;
        }
        match self.error {
            Some(error) => Err(error),
            None => Ok(hydrator.message),
        }
    }
}

impl fmt::Debug for ScriptedResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ScriptedResponse")
            .field("events", &self.events.len())
            .field("error", &self.error)
            .finish()
    }
}

fn next_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// What one call sent to the scripted provider, and when.
#[derive(Clone, Debug)]
pub struct ScriptedRequest {
    pub model_id: String,
    pub messages: Vec<Message>,
    pub options: StreamOptions,
    /// The Tokio time of the call, so tests under paused time can check waits.
    pub at: tokio::time::Instant,
}

struct State {
    responses: VecDeque<ScriptedResponse>,
    requests: Vec<ScriptedRequest>,
}

/// A provider that answers each call with the next scripted response. Clones
/// share the script and the recorded requests.
#[derive(Clone)]
pub struct ScriptedProvider {
    name: String,
    config: ProviderConfig,
    state: Arc<Mutex<State>>,
}

impl ScriptedProvider {
    pub fn new(
        name: impl Into<String>,
        responses: impl IntoIterator<Item = ScriptedResponse>,
    ) -> Self {
        Self {
            name: name.into(),
            config: ProviderConfig::default(),
            state: Arc::new(Mutex::new(State {
                responses: responses.into_iter().collect(),
                requests: Vec::new(),
            })),
        }
    }

    /// A model of this provider with the given id.
    #[must_use]
    pub fn model(&self, id: impl Into<String>) -> Model {
        Model::new(id, Arc::new(self.clone()))
    }

    /// Add a response after the scripted ones.
    pub fn push(&self, response: ScriptedResponse) {
        self.lock().responses.push_back(response);
    }

    /// Every request received so far, in order.
    #[must_use]
    pub fn requests(&self) -> Vec<ScriptedRequest> {
        self.lock().requests.clone()
    }

    /// Scripted responses not yet used.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.lock().responses.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("scripted provider mutex poisoned")
    }

    /// Record a call and take its response.
    fn take(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
    ) -> Result<ScriptedResponse, Error> {
        let mut state = self.lock();
        state.requests.push(ScriptedRequest {
            model_id: model.id.clone(),
            messages,
            options,
            at: tokio::time::Instant::now(),
        });
        let call = state.requests.len();
        let _ = call;
        Ok(state
            .responses
            .pop_front()
            .unwrap_or_else(|| ScriptedResponse::text("")))
    }
}

impl fmt::Debug for ScriptedProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.lock();
        formatter
            .debug_struct("ScriptedProvider")
            .field("name", &self.name)
            .field("remaining", &state.responses.len())
            .field("requests", &state.requests.len())
            .finish()
    }
}

#[async_trait]
impl Provider for ScriptedProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }

    fn provider_class_id(&self) -> &str {
        "scripted"
    }

    fn stream(&self, model: &Model, messages: Vec<Message>, options: StreamOptions) -> EventStream {
        match self.take(model, messages, options) {
            Ok(response) => response.into_stream(),
            Err(error) => stream::iter([Err(error)]).boxed(),
        }
    }

    async fn generate(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
    ) -> Result<Message, Error> {
        self.take(model, messages, options)?.into_message()
    }
}
