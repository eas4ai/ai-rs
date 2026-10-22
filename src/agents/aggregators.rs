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

//! Streaming-tool reducers and the explicit Rust equivalent of an aggregate marker.

use std::{collections::BTreeMap, sync::Arc};

use serde_json::{Value, json};

use crate::{
    Error,
    types::{
        Event, Message, MessageRole,
        events::{Aggregator, AggregatorFactory, ValueAggregator},
        messages::{MessageBundle, SpecialToolResult, ToolResultValue},
    },
};

pub(crate) type ModelInputConverter =
    Arc<dyn Fn(&ToolResultValue) -> Result<ToolResultValue, Error> + Send + Sync>;
type ResultConverter = Arc<dyn Fn(Value) -> Result<ToolResultValue, Error> + Send + Sync>;
type ItemValidator = Arc<dyn Fn(&Value) -> Result<(), Error> + Send + Sync>;

/// Factory metadata attached explicitly to a streaming tool.
/// Captured factory arguments replace Python's `Annotated` marker arguments.
#[derive(Clone)]
pub struct Aggregate {
    pub(crate) factory: AggregatorFactory,
    pub(crate) model_input: Option<ModelInputConverter>,
    result: ResultConverter,
    validate_item: Option<ItemValidator>,
}

impl Aggregate {
    /// Use an existing core factory. Its reducer supplies the live model input.
    pub fn new(factory: AggregatorFactory) -> Self {
        Self {
            factory,
            model_input: None,
            result: Arc::new(|value| Ok(value.into())),
            validate_item: None,
        }
    }

    /// Supply conversion for snapshots reconstructed from a wire round trip.
    pub fn with_model_input(
        mut self,
        convert: impl Fn(&ToolResultValue) -> Result<ToolResultValue, Error> + Send + Sync + 'static,
    ) -> Self {
        self.model_input = Some(Arc::new(convert));
        self
    }

    /// Keep the last streamed value, or JSON null for an empty stream.
    pub fn last() -> Self {
        Self::new(AggregatorFactory::new(LastAggregator::<Value>::default))
            .with_model_input(|value| Ok(value.clone()))
    }

    /// Concatenate string chunks with the configured delimiter.
    pub fn concat(delimiter: impl Into<String>) -> Self {
        let delimiter = delimiter.into();
        let mut aggregate = Self::new(AggregatorFactory::new(move || {
            ConcatValues::new(delimiter.clone())
        }))
        .with_model_input(|value| Ok(value.clone()));
        aggregate.validate_item = Some(Arc::new(|value| {
            if value.is_string() {
                Ok(())
            } else {
                Err(Error::Validation(
                    "concatenating tools must yield strings".into(),
                ))
            }
        }));
        aggregate
    }

    /// Collect sub-agent events as a rich bundle and expose its last assistant text.
    pub fn messages() -> Self {
        let mut aggregate = Self::new(AggregatorFactory::new(MessageValues::default))
            .with_model_input(|value| {
                let bundle = match value {
                    ToolResultValue::Special(SpecialToolResult::Messages(bundle)) => bundle.clone(),
                    ToolResultValue::Json(value) => serde_json::from_value(value.clone())?,
                    _ => return Err(Error::Validation("expected a message bundle".into())),
                };
                Ok(Value::String(MessageAggregator::to_model_input(bundle)).into())
            });
        aggregate.result =
            Arc::new(|value| Ok(serde_json::from_value::<MessageBundle>(value)?.into()));
        aggregate.validate_item = Some(Arc::new(|value| {
            serde_json::from_value::<Event>(value.clone())?;
            Ok(())
        }));
        aggregate
    }

    pub(crate) fn validate(&self, item: &Value) -> Result<(), Error> {
        match &self.validate_item {
            Some(validate) => validate(item),
            None => Ok(()),
        }
    }

    pub(crate) fn result(&self, snapshot: Value) -> Result<ToolResultValue, Error> {
        (self.result)(snapshot)
    }

    /// Create an independent reducer for a new stream or partial-result consumer.
    pub fn factory(&self) -> &AggregatorFactory {
        &self.factory
    }
}

impl From<AggregatorFactory> for Aggregate {
    fn from(factory: AggregatorFactory) -> Self {
        Self::new(factory)
    }
}

/// String chunks joined with a delimiter. Empty streams produce an empty string.
#[derive(Clone, Debug, Default)]
pub struct ConcatAggregator {
    parts: Vec<String>,
    delimiter: String,
}

impl ConcatAggregator {
    pub fn new(delimiter: impl Into<String>) -> Self {
        Self {
            parts: Vec::new(),
            delimiter: delimiter.into(),
        }
    }
}

impl Aggregator<String> for ConcatAggregator {
    type Snapshot = String;
    type ModelInput = String;
    fn feed(&mut self, item: String) {
        self.parts.push(item);
    }
    fn snapshot(&self) -> String {
        self.parts.join(&self.delimiter)
    }
    fn to_model_input(snapshot: String) -> String {
        snapshot
    }
}

/// Last item observed, with `None` before the first item.
#[derive(Clone, Debug)]
pub struct LastAggregator<T> {
    value: Option<T>,
}

impl<T> Default for LastAggregator<T> {
    fn default() -> Self {
        Self { value: None }
    }
}

impl<T: Clone> Aggregator<T> for LastAggregator<T> {
    type Snapshot = Option<T>;
    type ModelInput = Option<T>;
    fn feed(&mut self, item: T) {
        self.value = Some(item);
    }
    fn snapshot(&self) -> Option<T> {
        self.value.clone()
    }
    fn to_model_input(snapshot: Option<T>) -> Option<T> {
        snapshot
    }
}

impl ValueAggregator for LastAggregator<Value> {
    fn feed(&mut self, value: Value) {
        self.value = Some(value);
    }
    fn snapshot(&self) -> Value {
        self.value.clone().unwrap_or(Value::Null)
    }
    fn get_model_input(&self) -> Value {
        ValueAggregator::snapshot(self)
    }
}

#[derive(Clone, Debug)]
struct MessageSnapshots<T> {
    messages: Vec<T>,
    indices: BTreeMap<String, usize>,
    streaming_index: Option<usize>,
}

impl<T> Default for MessageSnapshots<T> {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            indices: BTreeMap::new(),
            streaming_index: None,
        }
    }
}

impl<T> MessageSnapshots<T> {
    fn update(&mut self, id: String, message: T, model_event: bool) {
        let index = if let Some(index) = self.indices.get(&id) {
            self.messages[*index] = message;
            *index
        } else {
            let index = self.messages.len();
            self.indices.insert(id, index);
            self.messages.push(message);
            index
        };
        if model_event {
            self.streaming_index = Some(index);
        }
    }

    fn retry(&mut self) {
        if let Some(index) = self.streaming_index.take() {
            self.messages.truncate(index);
            self.indices.retain(|_, position| *position < index);
        }
    }
}

/// Dedupe message snapshots by id and discard a retried response and its tools.
#[derive(Clone, Debug, Default)]
pub struct MessageAggregator {
    state: MessageSnapshots<Message>,
}

impl Aggregator<Event> for MessageAggregator {
    type Snapshot = MessageBundle;
    type ModelInput = String;
    fn feed(&mut self, item: Event) {
        if matches!(item, Event::Retry(_)) {
            self.state.retry();
            return;
        }
        if matches!(item, Event::PartialToolCallResult(_) | Event::RunBlocked(_)) {
            return;
        }
        if let Some(message) = item.message() {
            self.state.update(
                message.id.clone(),
                message.clone(),
                item.model_event().is_some(),
            );
        }
    }
    fn snapshot(&self) -> MessageBundle {
        MessageBundle::new(self.state.messages.clone())
    }
    fn to_model_input(snapshot: MessageBundle) -> String {
        snapshot
            .messages
            .iter()
            .rev()
            .filter(|message| message.role == MessageRole::Assistant)
            .map(Message::text)
            .find(|text| !text.is_empty())
            .unwrap_or_default()
    }
}

/// JSON adapter used by the core partial-result hydration contract.
struct ConcatValues {
    delimiter: String,
    values: Vec<Value>,
}
impl ConcatValues {
    fn new(delimiter: String) -> Self {
        Self {
            delimiter,
            values: Vec::new(),
        }
    }
}
impl ValueAggregator for ConcatValues {
    fn feed(&mut self, value: Value) {
        self.values.push(value);
    }
    fn snapshot(&self) -> Value {
        // Preserve invalid items rather than silently dropping them. Execution
        // validates each chunk before calling this infallible core interface.
        match self
            .values
            .iter()
            .map(Value::as_str)
            .collect::<Option<Vec<_>>>()
        {
            Some(parts) => Value::String(parts.join(&self.delimiter)),
            None => Value::Array(self.values.clone()),
        }
    }
    fn get_model_input(&self) -> Value {
        self.snapshot()
    }
}

#[derive(Default)]
struct MessageValues {
    state: MessageSnapshots<Value>,
}
impl ValueAggregator for MessageValues {
    fn feed(&mut self, value: Value) {
        let kind = value.get("kind").and_then(Value::as_str);
        match kind {
            Some("retry") => self.state.retry(),
            Some("partial_tool_call_result" | "run_blocked") => {}
            Some(kind) => {
                if let Some(message) = value.get("message")
                    && let Some(id) = message.get("id").and_then(Value::as_str)
                {
                    self.state.update(
                        id.to_owned(),
                        message.clone(),
                        !matches!(kind, "tool_call_result" | "hook"),
                    );
                }
            }
            None => {}
        }
    }
    fn snapshot(&self) -> Value {
        json!({"type":"messages", "messages":self.state.messages})
    }
    fn get_model_input(&self) -> Value {
        let text = self
            .state
            .messages
            .iter()
            .rev()
            .filter(|message| message.get("role").and_then(Value::as_str) == Some("assistant"))
            .filter_map(|message| message.get("parts").and_then(Value::as_array))
            .map(|parts| {
                parts
                    .iter()
                    .filter(|part| part.get("kind").and_then(Value::as_str) == Some("text"))
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect::<String>()
            })
            .find(|text| !text.is_empty())
            .unwrap_or_default();
        Value::String(text)
    }
}
