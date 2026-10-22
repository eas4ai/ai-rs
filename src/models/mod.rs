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

//! Composable model references and single-response inference APIs.

use std::{
    fmt,
    hash::{Hash, Hasher},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use futures::{Stream, StreamExt, stream as futures_stream};
use schemars::{JsonSchema, Schema};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::{
    errors::{Error, ProviderError, Result},
    providers::{EventStream, Provider, ProviderProtocol, ProviderRegistry},
    types::{
        Event, Message, MessageHydrator, ModelEventMeta,
        events::{ToolEnd, replay_message_events},
        messages::ToolCallPart,
        tools::{Tool, ToolChoice},
        usage::Usage,
    },
};

pub mod call;
pub mod files;
pub mod params;

pub use call::RetryPolicy;
pub use params::*;

/// A lightweight reference to one provider model.
#[derive(Clone)]
pub struct Model {
    pub id: String,
    pub provider: Arc<dyn Provider>,
    pub protocol: Option<Arc<dyn ProviderProtocol>>,
}

impl Model {
    #[must_use]
    pub fn new(id: impl Into<String>, provider: Arc<dyn Provider>) -> Self {
        Self {
            id: id.into(),
            provider,
            protocol: None,
        }
    }

    #[must_use]
    pub fn with_protocol(mut self, protocol: Arc<dyn ProviderProtocol>) -> Self {
        self.protocol = Some(protocol);
        self
    }

    /// Serialize the model and the provider configuration needed to restore it.
    pub fn to_value(&self) -> Result<Value> {
        let mut value = Map::new();
        value.insert("id".to_owned(), Value::String(self.id.clone()));
        value.insert("provider".to_owned(), self.provider.to_value()?);
        if let Some(protocol) = &self.protocol {
            value.insert("protocol".to_owned(), protocol.to_value()?);
        }
        Ok(Value::Object(value))
    }

    /// Restore a model through explicitly registered provider and protocol factories.
    pub fn from_value(value: Value, registry: &ProviderRegistry) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| Error::Configuration("model data must be a JSON object".to_owned()))?;
        let id = object
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| {
                Error::Configuration("model data must include a non-empty string id".to_owned())
            })?
            .to_owned();
        let provider = object
            .get("provider")
            .cloned()
            .ok_or_else(|| Error::Configuration("model data must include provider".to_owned()))
            .and_then(|provider| registry.restore_provider(provider))?;
        let protocol = object
            .get("protocol")
            .filter(|protocol| !protocol.is_null())
            .cloned()
            .map(|protocol| registry.restore_protocol(protocol))
            .transpose()?;
        Ok(Self {
            id,
            provider,
            protocol,
        })
    }
}

impl PartialEq for Model {
    fn eq(&self, other: &Self) -> bool {
        if std::ptr::eq(self, other) {
            return true;
        }
        matches!((self.to_value(), other.to_value()), (Ok(left), Ok(right)) if left == right)
    }
}

impl Eq for Model {}

impl Hash for Model {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.to_value() {
            Ok(value) => {
                0_u8.hash(state);
                value.hash(state);
            }
            Err(_) => {
                // Failed snapshots never compare equal except by reflexivity. Keep
                // their hash deterministic without treating the failure as content.
                1_u8.hash(state);
                self.id.hash(state);
            }
        }
    }
}

impl fmt::Debug for Model {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Model")
            .field("id", &self.id)
            .field("provider", &self.provider.name())
            .field(
                "protocol",
                &self
                    .protocol
                    .as_ref()
                    .map(|protocol| protocol.protocol_class_id()),
            )
            .finish()
    }
}

/// Options for one model call. The call applies `retry`, `timeout` and
/// `idle_timeout`; it passes the rest unchanged to the provider protocol.
#[derive(Clone, Debug, Default)]
pub struct StreamOptions {
    pub tools: Vec<Tool>,
    pub tool_choice: Option<ToolChoice>,
    pub output_schema: Option<Schema>,
    pub output_type_name: Option<String>,
    pub params: Option<InferenceRequestParams>,
    /// Retries for a retryable error before the first event. The default
    /// makes 2 retries, waiting 2 s and then 4 s.
    pub retry: RetryPolicy,
    /// Bound on the whole call, retries and waits included. Unset by default.
    pub timeout: Option<Duration>,
    /// Bound on the silence before each event of an attempt, or on the wait
    /// for a buffered response. Unset by default.
    pub idle_timeout: Option<Duration>,
}

impl StreamOptions {
    /// Constrain the provider response to the schema for `T`.
    #[must_use]
    pub fn with_output<T: JsonSchema>(mut self) -> Self {
        self.output_schema = Some(schemars::schema_for!(T));
        self.output_type_name = Some(T::schema_name().into_owned());
        self
    }

    #[must_use]
    pub fn with_tools<T: Into<Tool>>(mut self, tools: impl IntoIterator<Item = T>) -> Self {
        self.tools = tools.into_iter().map(Into::into).collect();
        self
    }
}

/// Resolve a `provider:model` id. The id splits at its first colon, and an id
/// without a colon is an error.
///
/// Configuration is supplied by the registry and its provider objects. This
/// function deliberately never reads process environment variables.
pub fn get_model(model_id: &str, registry: &ProviderRegistry) -> Result<Model> {
    let (provider_id, provider_model_id) =
        model_id.split_once(':').unwrap_or(("gateway", model_id));
    if provider_id.is_empty() || provider_model_id.is_empty() {
        return Err(Error::Configuration(format!(
            "get_model: malformed model_id: {model_id:?}"
        )));
    }
    Ok(Model::new(
        provider_model_id,
        registry.get_provider(provider_id)?,
    ))
}

/// A provider stream plus the message incrementally reconstructed from its events.
pub struct StreamHandle {
    inner: EventStream,
    hydrator: MessageHydrator,
    provider_name: Option<String>,
    source_error_seen: bool,
    exhausted: bool,
}

impl fmt::Debug for StreamHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamHandle")
            .field("message", &self.hydrator.message)
            .field("ended", &self.hydrator.ended)
            .finish_non_exhaustive()
    }
}

impl StreamHandle {
    fn new(
        inner: EventStream,
        seed_message: Option<Message>,
        provider_name: Option<String>,
    ) -> Self {
        Self {
            inner,
            hydrator: MessageHydrator::new(seed_message),
            provider_name,
            source_error_seen: false,
            exhausted: false,
        }
    }

    /// Re-emit a complete message as model stream events.
    #[must_use]
    pub fn replay_message(message: Message) -> Self {
        let seed = Message::with_id(message.id.clone(), message.role, Vec::new());
        let events = replay_message_events(&message)
            .into_iter()
            .map(Ok::<_, Error>);
        Self::new(futures_stream::iter(events).boxed(), Some(seed), None)
    }

    #[must_use]
    pub fn message(&self) -> &Message {
        &self.hydrator.message
    }

    #[must_use]
    pub fn messages(&self) -> &[Message] {
        &self.hydrator.messages
    }

    #[must_use]
    pub fn usage(&self) -> Option<&Usage> {
        self.message().usage.as_ref()
    }

    #[must_use]
    pub fn text(&self) -> String {
        self.message().text()
    }

    #[must_use]
    pub fn tool_calls(&self) -> Vec<&ToolCallPart> {
        self.message().tool_calls()
    }

    #[must_use]
    pub const fn ended(&self) -> bool {
        self.hydrator.ended
    }

    #[must_use]
    pub fn finish_reason(&self) -> Option<&str> {
        self.hydrator.finish_reason.as_deref()
    }

    #[must_use]
    pub fn response_id(&self) -> Option<&str> {
        self.hydrator.response_id.as_deref()
    }

    #[must_use]
    pub fn response_model(&self) -> Option<&str> {
        self.hydrator.response_model.as_deref()
    }

    /// Deserialize the final assistant text into a structured output type.
    pub fn output<T: DeserializeOwned>(&self) -> Result<T> {
        let text = self
            .message()
            .get_output()
            .map_err(|error| Error::Validation(error.to_string()))?;
        serde_json::from_str(&text).map_err(Error::from)
    }
}

impl Stream for StreamHandle {
    type Item = Result<Event>;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.exhausted {
            return Poll::Ready(None);
        }
        match this.inner.as_mut().poll_next(context) {
            Poll::Ready(Some(Ok(event))) => match this.hydrator.feed(event) {
                Ok(event) => Poll::Ready(Some(Ok(event))),
                Err(error) => {
                    this.source_error_seen = true;
                    Poll::Ready(Some(Err(Error::Validation(error.to_string()))))
                }
            },
            Poll::Ready(Some(Err(error))) => {
                this.source_error_seen = true;
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) if this.hydrator.ended || this.source_error_seen => {
                this.exhausted = true;
                Poll::Ready(None)
            }
            Poll::Ready(None) => {
                this.exhausted = true;
                Poll::Ready(Some(Err(ProviderError::incomplete_response(
                    "provider stream ended without a finish event; the response is incomplete",
                    this.provider_name.clone(),
                )
                .into())))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Start one provider response and aggregate it as it is consumed.
#[must_use]
pub fn stream(model: &Model, messages: Vec<Message>, options: StreamOptions) -> StreamHandle {
    if let Some(message) = messages.last().filter(|message| message.replay) {
        let mut events = Vec::new();
        for tool_call in message.tool_calls() {
            let mut event = ModelEventMeta::with_message(message.clone());
            event.replay = true;
            events.push(Ok(Event::ToolEnd(ToolEnd {
                event,
                tool_call: tool_call.clone(),
                tool_call_id: tool_call.tool_call_id.clone(),
            })));
        }
        let mut handle = StreamHandle::new(
            futures_stream::iter(events).boxed(),
            Some(message.clone()),
            Some(model.provider.name().to_owned()),
        );
        // A persisted replay turn is complete even when it has no tool calls.
        handle.hydrator.ended = true;
        return handle;
    }

    StreamHandle::new(
        call::call_stream(model, messages, options),
        None,
        Some(model.provider.name().to_owned()),
    )
}

/// Return a complete provider message, falling back to draining its stream only
/// when native buffered generation is explicitly unsupported. Retries and
/// timeouts apply as they do to `stream`.
pub async fn generate(
    model: &Model,
    messages: Vec<Message>,
    options: StreamOptions,
) -> Result<Message> {
    call::call_generate(model, messages, options).await
}

/// Python-compatible name for the buffered generation API.
pub async fn experimental_generate(
    model: &Model,
    messages: Vec<Message>,
    options: StreamOptions,
) -> Result<Message> {
    generate(model, messages, options).await
}

/// Ask a provider to verify that the model exists and is reachable.
pub async fn probe(model: &Model) -> Result<()> {
    model.provider.probe(model).await
}

/// Compatibility paths matching the Python module layout.
pub mod core;
