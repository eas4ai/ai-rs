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

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream as futures_stream};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;
use suprnova_ai::{
    Error, Model, Provider, ProviderConfig, ProviderProtocol, ProviderRegistry, StreamOptions,
    errors::{ProviderError, ProviderErrorKind},
    generate, get_model,
    models::experimental_generate,
    stream,
    types::{
        Event, Message, MessageRole, ModelEventMeta, Part, TextPart, ToolCallPart,
        events::{StreamEnd, StreamStart, TextDelta, TextEnd, TextStart, ToolEnd},
        tools::{Tool, ToolConfig, ToolKind, ToolSpec},
    },
};

#[derive(Clone)]
enum GenerateBehavior {
    Native(Box<Message>),
    Unsupported,
    Failed,
}

struct ScriptedProtocol {
    id: &'static str,
    events: Mutex<Vec<Event>>,
    generate_behavior: Mutex<GenerateBehavior>,
    stream_calls: AtomicUsize,
    generate_calls: AtomicUsize,
    options: Mutex<Vec<StreamOptions>>,
}

impl ScriptedProtocol {
    fn new(id: &'static str, events: Vec<Event>, generate_behavior: GenerateBehavior) -> Self {
        Self {
            id,
            events: Mutex::new(events),
            generate_behavior: Mutex::new(generate_behavior),
            stream_calls: AtomicUsize::new(0),
            generate_calls: AtomicUsize::new(0),
            options: Mutex::new(Vec::new()),
        }
    }
}

#[async_trait]
impl ProviderProtocol for ScriptedProtocol {
    fn protocol_class_id(&self) -> &str {
        self.id
    }

    fn stream(
        &self,
        _model: &Model,
        _messages: Vec<Message>,
        options: StreamOptions,
        _provider: &dyn Provider,
    ) -> suprnova_ai::providers::EventStream {
        self.stream_calls.fetch_add(1, Ordering::SeqCst);
        self.options.lock().unwrap().push(options);
        futures_stream::iter(self.events.lock().unwrap().clone().into_iter().map(Ok)).boxed()
    }

    async fn generate(
        &self,
        _model: &Model,
        _messages: Vec<Message>,
        options: StreamOptions,
        _provider: &dyn Provider,
    ) -> Result<Message, Error> {
        self.generate_calls.fetch_add(1, Ordering::SeqCst);
        self.options.lock().unwrap().push(options);
        match self.generate_behavior.lock().unwrap().clone() {
            GenerateBehavior::Native(message) => Ok(*message),
            GenerateBehavior::Unsupported => Err(Error::UnsupportedOperation("generate".into())),
            GenerateBehavior::Failed => {
                Err(ProviderError::api("upstream rejected request", Some("test".into())).into())
            }
        }
    }
}

struct TestProvider {
    name: &'static str,
    config: ProviderConfig,
    protocol: Arc<dyn ProviderProtocol>,
}

impl TestProvider {
    fn new(name: &'static str, protocol: Arc<dyn ProviderProtocol>) -> Self {
        Self {
            name,
            config: ProviderConfig::default(),
            protocol,
        }
    }
}

#[async_trait]
impl Provider for TestProvider {
    fn name(&self) -> &str {
        self.name
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }

    fn default_protocol(&self) -> Option<Arc<dyn ProviderProtocol>> {
        Some(self.protocol.clone())
    }
}

struct UnserializableProvider {
    config: ProviderConfig,
}

#[async_trait]
impl Provider for UnserializableProvider {
    fn name(&self) -> &str {
        "unserializable"
    }

    fn config(&self) -> &ProviderConfig {
        &self.config
    }

    fn to_value(&self) -> Result<serde_json::Value, Error> {
        Err(Error::Configuration(
            "provider snapshot is unavailable".into(),
        ))
    }
}

fn user_message(text: &str) -> Message {
    Message::new(MessageRole::User, vec![Part::Text(TextPart::new(text))])
}

fn assistant_message(text: &str) -> Message {
    Message::new(
        MessageRole::Assistant,
        vec![Part::Text(TextPart::new(text))],
    )
}

fn text_events(text: &str) -> Vec<Event> {
    vec![
        Event::StreamStart(StreamStart::default()),
        Event::TextStart(TextStart {
            event: ModelEventMeta::default(),
            block_id: "text".into(),
        }),
        Event::TextDelta(TextDelta {
            event: ModelEventMeta::default(),
            chunk: text.into(),
            block_id: "text".into(),
        }),
        Event::TextEnd(TextEnd {
            event: ModelEventMeta::default(),
            block_id: "text".into(),
        }),
        Event::StreamEnd(StreamEnd::default()),
    ]
}

fn model_with(protocol: Arc<ScriptedProtocol>) -> Model {
    let provider: Arc<dyn Provider> = Arc::new(TestProvider::new("test", protocol));
    Model::new("test-model", provider)
}

#[test]
fn model_ids_split_only_once() {
    let direct_protocol = Arc::new(ScriptedProtocol::new(
        "direct",
        text_events("direct"),
        GenerateBehavior::Unsupported,
    ));
    let direct: Arc<dyn Provider> = Arc::new(TestProvider::new("openai", direct_protocol));
    let mut registry = ProviderRegistry::new();
    registry.register("openai", direct.clone()).unwrap();

    let model = get_model("openai:org:model", &registry).unwrap();
    assert_eq!(model.id, "org:model");
    assert!(Arc::ptr_eq(&model.provider, &direct));

    for malformed in ["", ":model", "provider:"] {
        assert!(matches!(
            get_model(malformed, &registry),
            Err(Error::Configuration(_))
        ));
    }
}

#[test]
fn model_snapshots_round_trip_through_registered_factories() {
    let base_protocol = Arc::new(ScriptedProtocol::new(
        "base",
        Vec::new(),
        GenerateBehavior::Unsupported,
    ));
    let provider: Arc<dyn Provider> = Arc::new(TestProvider::new("test", base_protocol));
    let override_protocol = Arc::new(ScriptedProtocol::new(
        "override",
        Vec::new(),
        GenerateBehavior::Unsupported,
    ));
    let model = Model::new("test-model", provider.clone()).with_protocol(override_protocol);

    let mut registry = ProviderRegistry::new();
    // A restore takes credentials from the provider registered under the snapshot's name.
    registry.register("test", provider).unwrap();
    registry
        .register_provider_factory("test", |_value, _registry| {
            let protocol = Arc::new(ScriptedProtocol::new(
                "base",
                Vec::new(),
                GenerateBehavior::Unsupported,
            ));
            Ok(Arc::new(TestProvider::new("test", protocol)))
        })
        .unwrap();
    registry
        .register_protocol_factory("override", |_value| {
            Ok(Arc::new(ScriptedProtocol::new(
                "override",
                Vec::new(),
                GenerateBehavior::Unsupported,
            )))
        })
        .unwrap();

    let snapshot = model.to_value().unwrap();
    assert_eq!(snapshot["id"], "test-model");
    assert_eq!(snapshot["provider"]["provider_class_id"], "test");
    assert_eq!(snapshot["protocol"]["protocol_class_id"], "override");

    let restored = Model::from_value(snapshot, &registry).unwrap();
    assert_eq!(restored, model);
    let mut original_hash = DefaultHasher::new();
    model.hash(&mut original_hash);
    let mut restored_hash = DefaultHasher::new();
    restored.hash(&mut restored_hash);
    assert_eq!(original_hash.finish(), restored_hash.finish());
}

#[test]
fn model_restoration_rejects_malformed_or_unregistered_snapshots() {
    let registry = ProviderRegistry::new();
    for value in [
        json!(null),
        json!({}),
        json!({"id": ""}),
        json!({"id": "test-model"}),
        json!({"id": "test-model", "provider": {}}),
        json!({
            "id": "test-model",
            "provider": {"provider_class_id": "missing"}
        }),
    ] {
        assert!(matches!(
            Model::from_value(value, &registry),
            Err(Error::Configuration(_))
        ));
    }

    let mut registry = ProviderRegistry::new();
    registry
        .register_provider_factory("test", |_value, _registry| {
            let protocol = Arc::new(ScriptedProtocol::new(
                "base",
                Vec::new(),
                GenerateBehavior::Unsupported,
            ));
            Ok(Arc::new(TestProvider::new("test", protocol)))
        })
        .unwrap();
    assert!(matches!(
        registry.register_provider_factory("test", |_value, _registry| unreachable!()),
        Err(Error::Configuration(_))
    ));
    registry
        .register_protocol_factory("known", |_value| {
            Ok(Arc::new(ScriptedProtocol::new(
                "known",
                Vec::new(),
                GenerateBehavior::Unsupported,
            )))
        })
        .unwrap();
    assert!(matches!(
        registry.register_protocol_factory("known", |_value| unreachable!()),
        Err(Error::Configuration(_))
    ));
    assert!(matches!(
        Model::from_value(
            json!({
                "id": "test-model",
                "provider": {"provider_class_id": "test"},
                "protocol": {}
            }),
            &registry,
        ),
        Err(Error::Configuration(_))
    ));
    assert!(matches!(
        Model::from_value(
            json!({
                "id": "test-model",
                "provider": {"provider_class_id": "test"},
                "protocol": {"protocol_class_id": "missing"}
            }),
            &registry,
        ),
        Err(Error::Configuration(_))
    ));
}

#[test]
fn failed_provider_snapshots_are_not_treated_as_equal_content() {
    let provider: Arc<dyn Provider> = Arc::new(UnserializableProvider {
        config: ProviderConfig::default(),
    });
    let model = Model::new("test-model", provider);
    let cloned = model.clone();

    assert!(model.to_value().is_err());
    assert_eq!(model, model);
    assert_ne!(model, cloned);
}

#[tokio::test]
async fn stream_aggregates_text_and_exposes_the_final_message() {
    let protocol = Arc::new(ScriptedProtocol::new(
        "stream",
        text_events("Hello world"),
        GenerateBehavior::Unsupported,
    ));
    let model = model_with(protocol.clone());
    let mut response = stream(&model, vec![user_message("Hi")], StreamOptions::default());
    let mut deltas = String::new();
    while let Some(event) = response.next().await.transpose().unwrap() {
        if let Event::TextDelta(event) = event {
            deltas.push_str(&event.chunk);
        }
    }

    assert_eq!(deltas, "Hello world");
    assert_eq!(response.text(), "Hello world");
    assert!(response.ended());
    assert_eq!(protocol.stream_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn stream_returns_a_retryable_error_when_finish_is_missing() {
    let mut events = text_events("partial");
    events.pop();
    let model = model_with(Arc::new(ScriptedProtocol::new(
        "incomplete",
        events,
        GenerateBehavior::Unsupported,
    )));
    let mut response = stream(&model, vec![user_message("Hi")], StreamOptions::default());
    let error = response.by_ref().try_collect::<Vec<_>>().await.unwrap_err();

    match error {
        Error::Provider(error) => {
            assert_eq!(error.kind, ProviderErrorKind::IncompleteResponse);
            assert!(error.is_retryable);
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(response.text(), "partial");
}

#[tokio::test]
async fn a_model_protocol_override_takes_precedence() {
    let base = Arc::new(ScriptedProtocol::new(
        "base",
        text_events("base"),
        GenerateBehavior::Unsupported,
    ));
    let override_protocol = Arc::new(ScriptedProtocol::new(
        "override",
        text_events("override"),
        GenerateBehavior::Unsupported,
    ));
    let model = model_with(base.clone()).with_protocol(override_protocol.clone());
    let mut response = stream(&model, vec![user_message("Hi")], StreamOptions::default());
    while response.next().await.transpose().unwrap().is_some() {}

    assert_eq!(response.text(), "override");
    assert_eq!(base.stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(override_protocol.stream_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn generate_uses_native_generation_and_forwards_failures() {
    let sentinel = assistant_message("native");
    let native = Arc::new(ScriptedProtocol::new(
        "native",
        text_events("unused"),
        GenerateBehavior::Native(Box::new(sentinel.clone())),
    ));
    let model = model_with(native.clone());
    assert_eq!(
        generate(&model, vec![user_message("Hi")], StreamOptions::default())
            .await
            .unwrap(),
        sentinel
    );
    assert_eq!(native.generate_calls.load(Ordering::SeqCst), 1);
    assert_eq!(native.stream_calls.load(Ordering::SeqCst), 0);

    let failed = Arc::new(ScriptedProtocol::new(
        "failed",
        text_events("must not be used"),
        GenerateBehavior::Failed,
    ));
    let model = model_with(failed.clone());
    assert!(matches!(
        experimental_generate(&model, vec![], StreamOptions::default()).await,
        Err(Error::Provider(_))
    ));
    assert_eq!(failed.stream_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn generate_falls_back_only_when_native_generation_is_unsupported() {
    let protocol = Arc::new(ScriptedProtocol::new(
        "fallback",
        text_events("streamed reply"),
        GenerateBehavior::Unsupported,
    ));
    let model = model_with(protocol.clone());
    let message = experimental_generate(&model, vec![user_message("Hi")], StreamOptions::default())
        .await
        .unwrap();

    assert_eq!(message.text(), "streamed reply");
    assert_eq!(protocol.generate_calls.load(Ordering::SeqCst), 1);
    assert_eq!(protocol.stream_calls.load(Ordering::SeqCst), 1);
}

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Answer {
    value: String,
}

#[tokio::test]
async fn structured_output_supplies_a_schema_and_deserializes_the_result() {
    let protocol = Arc::new(ScriptedProtocol::new(
        "structured",
        text_events(r#"{"value":"ok"}"#),
        GenerateBehavior::Unsupported,
    ));
    let model = model_with(protocol.clone());
    let mut response = stream(
        &model,
        vec![user_message("Hi")],
        StreamOptions::default().with_output::<Answer>(),
    );
    while response.next().await.transpose().unwrap().is_some() {}

    assert_eq!(
        response.output::<Answer>().unwrap(),
        Answer { value: "ok".into() }
    );
    let options = protocol.options.lock().unwrap();
    assert!(options[0].output_schema.is_some());
    assert_eq!(options[0].output_type_name.as_deref(), Some("Answer"));
}

#[tokio::test]
async fn stream_forwards_function_and_provider_tool_declarations() {
    let protocol = Arc::new(ScriptedProtocol::new(
        "tools",
        text_events("done"),
        GenerateBehavior::Unsupported,
    ));
    let model = model_with(protocol.clone());
    let function = Tool::function(
        "answer",
        ToolSpec {
            description: Some("Return a typed answer".into()),
            params: schemars::schema_for!(Answer),
        },
    );
    let provider_tool = Tool::provider(
        "web_search",
        ToolConfig {
            id: Some("provider.web_search".into()),
            args: serde_json::from_value(json!({
                "search_depth": "advanced",
                "result_limit": 8
            }))
            .unwrap(),
        },
    )
    .unwrap();
    let expected = vec![function, provider_tool];

    let mut response = stream(
        &model,
        vec![user_message("Research this")],
        StreamOptions::default().with_tools(expected.clone()),
    );
    while response.next().await.transpose().unwrap().is_some() {}

    let options = protocol.options.lock().unwrap();
    assert_eq!(options[0].tools, expected);
    assert_eq!(options[0].tools[0].kind, ToolKind::Function);
    assert!(options[0].tools[0].spec.is_some());
    assert_eq!(options[0].tools[1].kind, ToolKind::Provider);
    assert_eq!(
        options[0].tools[1].tool_config.as_ref().unwrap().args["search_depth"],
        "advanced"
    );
}

#[tokio::test]
async fn marked_assistant_turns_replay_tool_ends_without_calling_the_provider() {
    let protocol = Arc::new(ScriptedProtocol::new(
        "replay",
        text_events("unused"),
        GenerateBehavior::Unsupported,
    ));
    let model = model_with(protocol.clone());
    let mut replayed = assistant_message("calling tools");
    replayed.parts.push(Part::ToolCall(ToolCallPart::new(
        "tc-1",
        "weather",
        r#"{"city":"SF"}"#,
    )));
    replayed.replay = true;

    let mut response = stream(
        &model,
        vec![user_message("Hi"), replayed],
        StreamOptions::default(),
    );
    let events = response.by_ref().try_collect::<Vec<_>>().await.unwrap();
    assert_eq!(protocol.stream_calls.load(Ordering::SeqCst), 0);
    assert_eq!(response.text(), "calling tools");
    assert_eq!(response.tool_calls().len(), 1);
    assert!(matches!(events.as_slice(), [Event::ToolEnd(ToolEnd { event, .. })] if event.replay));
}

#[tokio::test]
async fn replay_message_rebuilds_parts_and_preserves_the_message_id() {
    let message = assistant_message("cached");
    let id = message.id.clone();
    let mut response = suprnova_ai::StreamHandle::replay_message(message);
    let events = response.by_ref().try_collect::<Vec<_>>().await.unwrap();

    assert!(!events.is_empty());
    assert_eq!(response.message().id, id);
    assert_eq!(response.text(), "cached");
    assert!(response.ended());
}

#[test]
fn request_param_builders_preserve_other_sampling_values() {
    use suprnova_ai::models::{
        InferenceRequestParams, ProviderValue, SamplerParams, TopPSamplerParams,
    };

    let params = InferenceRequestParams {
        sampling: ProviderValue::Value(SamplerParams {
            top_p: Some(TopPSamplerParams {
                top_p: ProviderValue::Value(Some(0.9)),
            }),
            ..SamplerParams::default()
        }),
        ..InferenceRequestParams::default()
    }
    .with_temperature(0.2);

    let ProviderValue::Value(sampling) = params.sampling else {
        panic!("sampling must be explicit");
    };
    assert_eq!(
        sampling.temperature.unwrap().temperature,
        ProviderValue::Value(0.2)
    );
    assert_eq!(
        sampling.top_p.unwrap().top_p,
        ProviderValue::Value(Some(0.9))
    );
}

#[test]
fn model_hash_is_independent_of_json_object_insertion_order() {
    struct OrderedProtocol(bool);
    #[async_trait]
    impl ProviderProtocol for OrderedProtocol {
        fn protocol_class_id(&self) -> &str {
            "ordered"
        }
        fn to_value(&self) -> Result<serde_json::Value, Error> {
            let mut value = serde_json::Map::new();
            let keys = if self.0 { ["a", "b"] } else { ["b", "a"] };
            for key in keys {
                value.insert(key.into(), json!(key));
            }
            value.insert("protocol_class_id".into(), json!("ordered"));
            Ok(serde_json::Value::Object(value))
        }
    }
    let protocol = Arc::new(ScriptedProtocol::new(
        "base",
        vec![],
        GenerateBehavior::Unsupported,
    ));
    let model = model_with(protocol);
    let left = model.clone().with_protocol(Arc::new(OrderedProtocol(true)));
    let right = model.with_protocol(Arc::new(OrderedProtocol(false)));
    assert_eq!(left, right);
    let mut a = DefaultHasher::new();
    let mut b = DefaultHasher::new();
    left.hash(&mut a);
    right.hash(&mut b);
    assert_eq!(a.finish(), b.finish());
}
