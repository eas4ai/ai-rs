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

use serde_json::{Value, json};
use suprnova_ai::types::{
    Event, FilePart, Message, MessageHydrator, MessageRole, Part, ReasoningPart, TextPart,
    ToolCallPart,
    events::{
        HookEvent, ModelEventMeta, OmitEventMessages, Retry, RunStateTracker, StreamEnd,
        StreamStart, TextDelta, TextEnd, TextStart, ToolCallResult, ToolDelta, ToolEnd, ToolStart,
        replay_message_events,
    },
    messages::{BuiltinToolCallPart, BuiltinToolReturnPart, HookPart, HookStatus, ToolResultPart},
};

fn meta(id: &str) -> ModelEventMeta {
    ModelEventMeta::with_message(Message::with_id(id, MessageRole::Assistant, vec![]))
}

#[test]
fn compact_serialization_keeps_only_message_id_and_canonical_kind() {
    let event = Event::TextDelta(TextDelta {
        event: ModelEventMeta::with_message(Message::with_id(
            "message-1",
            MessageRole::Assistant,
            vec![TextPart::new("hello").into()],
        )),
        chunk: "hello".to_owned(),
        block_id: String::new(),
    });
    let value = serde_json::to_value(OmitEventMessages(&event)).expect("compact event");
    assert_eq!(value["kind"], "text_delta");
    assert_eq!(value["message"], json!({"id": "message-1"}));
    assert!(value.get("type").is_none());

    let restored: Event = serde_json::from_value(value).expect("restore compact event");
    assert_eq!(restored.message().expect("model message").id, "message-1");
    assert!(restored.message().expect("model message").parts.is_empty());

    let default = Event::text_delta("hello");
    let default_value = serde_json::to_value(OmitEventMessages(&default)).expect("compact default");
    assert_eq!(default_value["message"], json!({"id": "<unset>"}));
    serde_json::from_value::<Event>(default_value).expect("restore default compact event");
}

#[test]
fn retry_round_trips_as_an_agent_event() {
    let value = serde_json::to_value(Event::Retry(Retry::default())).expect("serialize retry");
    assert_eq!(value, json!({"kind": "retry"}));
    assert!(matches!(
        serde_json::from_value::<Event>(value).expect("deserialize retry"),
        Event::Retry(_)
    ));
}

#[test]
fn hydrator_reconstructs_compact_text_and_message_metadata() {
    let message = Message::with_id("message-1", MessageRole::Assistant, vec![]);
    let source = [
        Event::StreamStart(StreamStart {
            event: ModelEventMeta::with_message(message.clone()),
        }),
        Event::TextStart(TextStart {
            event: ModelEventMeta::with_message(message.clone()),
            block_id: "text-1".to_owned(),
        }),
        Event::TextDelta(TextDelta {
            event: ModelEventMeta::with_message(message.clone()),
            chunk: "hello".to_owned(),
            block_id: "text-1".to_owned(),
        }),
        Event::TextDelta(TextDelta {
            event: ModelEventMeta::with_message(message.clone()),
            chunk: " world".to_owned(),
            block_id: "text-1".to_owned(),
        }),
        Event::TextEnd(TextEnd {
            event: ModelEventMeta::with_message(message.clone()),
            block_id: "text-1".to_owned(),
        }),
        Event::StreamEnd(StreamEnd {
            event: ModelEventMeta {
                provider_metadata: Some([("provider".to_owned(), json!({"id": "1"}))].into()),
                ..ModelEventMeta::with_message(message)
            },
            finish_reason: Some("stop".to_owned()),
            response_id: Some("response-1".to_owned()),
            response_model: Some("model-1".to_owned()),
        }),
    ];
    let mut hydrator = MessageHydrator::default();
    assert!(hydrator.messages.is_empty());
    assert!(hydrator.messages_by_id.is_empty());
    for event in source {
        let compact = serde_json::to_value(OmitEventMessages(&event)).expect("serialize compact");
        let restored: Event = serde_json::from_value(compact).expect("deserialize compact");
        let hydrated = hydrator.feed(restored).expect("hydrate");
        assert_eq!(hydrated.message().expect("model message").id, "message-1");
    }
    assert_eq!(hydrator.message.text(), "hello world");
    assert_eq!(
        hydrator
            .message
            .provider_metadata
            .as_ref()
            .expect("metadata")["provider"],
        json!({"id": "1"})
    );
    assert!(hydrator.ended);
    assert_eq!(hydrator.finish_reason.as_deref(), Some("stop"));
    assert_eq!(hydrator.response_id.as_deref(), Some("response-1"));
    assert_eq!(hydrator.response_model.as_deref(), Some("model-1"));
}

#[test]
fn replay_events_round_trip_reasoning_text_tools_and_files() {
    let mut tool = ToolCallPart::new("tool-1", "weather", "{\"city\":\"SF\"}");
    tool.id = "tool-1".to_owned();
    let original = Message::with_id(
        "message-1",
        MessageRole::Assistant,
        vec![
            ReasoningPart {
                id: "reasoning-1".to_owned(),
                text: "thinking".to_owned(),
                provider_metadata: Some([("signature".to_owned(), json!("sig"))].into()),
            }
            .into(),
            TextPart {
                id: "text-1".to_owned(),
                text: "hello".to_owned(),
                provider_metadata: Some([("block".to_owned(), json!(1))].into()),
            }
            .into(),
            tool.into(),
            Part::BuiltinToolCall(BuiltinToolCallPart {
                id: "builtin-1".to_owned(),
                tool_call_id: "builtin-1".to_owned(),
                tool_name: "web_search".to_owned(),
                tool_args: "{\"q\":\"rust\"}".to_owned(),
                provider_metadata: Some([("call".to_owned(), json!(true))].into()),
            }),
            Part::BuiltinToolReturn(BuiltinToolReturnPart {
                id: "builtin-return-1".to_owned(),
                tool_call_id: "builtin-1".to_owned(),
                tool_name: "web_search".to_owned(),
                result: json!({"hits": 1}),
                is_error: false,
                provider_metadata: Some([("return".to_owned(), json!(true))].into()),
            }),
            FilePart::from_url("https://example.com/chart.png", None)
                .expect("known extension")
                .into(),
        ],
    );
    let original = Message {
        provider_metadata: Some([("response".to_owned(), json!("r1"))].into()),
        ..original
    };
    let mut hydrator = MessageHydrator::default();
    for event in replay_message_events(&original) {
        hydrator.feed(event).expect("hydrate replay");
    }
    assert_eq!(hydrator.message.id, original.id);
    assert_eq!(hydrator.message.text(), "hello");
    assert_eq!(hydrator.message.reasoning(), "thinking");
    let reasoning = match &hydrator.message.parts[0] {
        Part::Reasoning(part) => part,
        _ => panic!("reasoning part"),
    };
    assert_eq!(
        reasoning.provider_metadata.as_ref().expect("signature")["signature"],
        "sig"
    );
    assert_eq!(
        hydrator.message.tool_calls()[0].tool_args,
        "{\"city\":\"SF\"}"
    );
    assert_eq!(hydrator.message.builtin_tool_calls().len(), 1);
    assert_eq!(hydrator.message.builtin_tool_returns().len(), 1);
    assert_eq!(hydrator.message.files().len(), 1);
    assert_eq!(
        hydrator
            .message
            .provider_metadata
            .as_ref()
            .expect("response")["response"],
        "r1"
    );
}

#[test]
fn hydrator_rejects_mismatched_seed_and_tracks_complete_agent_messages() {
    let seed = Message::with_id("seed", MessageRole::Assistant, vec![]);
    let mut hydrator = MessageHydrator::new(Some(seed));
    let error = hydrator
        .feed(Event::StreamStart(StreamStart {
            event: meta("stream"),
        }))
        .expect_err("mismatched seed");
    assert!(error.to_string().contains("seed"));
    assert!(error.to_string().contains("stream"));

    let matching = Message::with_id("matching", MessageRole::Assistant, vec![]);
    let mut seeded = MessageHydrator::new(Some(matching.clone()));
    assert!(seeded.messages.is_empty());
    seeded
        .feed(Event::StreamStart(StreamStart {
            event: ModelEventMeta::with_message(matching),
        }))
        .expect("matching seed");
    assert_eq!(seeded.messages.len(), 1);

    let mut without_start = MessageHydrator::default();
    without_start
        .feed(Event::TextStart(TextStart {
            event: meta("selected"),
            block_id: "text".to_owned(),
        }))
        .expect("select from first content event");
    without_start
        .feed(Event::TextDelta(TextDelta {
            event: meta("selected"),
            chunk: "hello".to_owned(),
            block_id: "text".to_owned(),
        }))
        .expect("hydrate without stream start");
    assert_eq!(without_start.message.text(), "hello");

    let mut hydrator = MessageHydrator::default();
    let tool_message = Message::with_id("tool-message", MessageRole::Tool, vec![]);
    let event = Event::ToolCallResult(ToolCallResult {
        replay: false,
        message: tool_message.clone(),
        results: vec![],
        exception: None,
    });
    assert_eq!(
        hydrator.feed(event.clone()).expect("feed agent event"),
        event
    );
    assert_eq!(hydrator.message, tool_message);
    assert_eq!(hydrator.messages.len(), 1);

    let result = ToolResultPart::new("tc", "tool", json!(1));
    let value = serde_json::to_value(Event::ToolCallResult(ToolCallResult {
        replay: false,
        message: Message::new(MessageRole::Tool, vec![result.clone().into()]),
        results: vec![result],
        exception: None,
    }))
    .expect("serialize tool result event");
    assert_eq!(value["results"][0]["kind"], "tool_result");
    serde_json::from_value::<Event>(value).expect("deserialize tool result event");
    let compact = serde_json::to_value(OmitEventMessages(&event)).expect("compact agent event");
    assert_eq!(
        compact["message"],
        serde_json::to_value(&tool_message).unwrap()
    );
}

#[test]
fn retry_discards_current_response_and_following_tool_results() {
    let mut empty = MessageHydrator::default();
    empty
        .feed(Event::Retry(Retry::default()))
        .expect("retry before stream");
    assert!(empty.messages.is_empty());

    let mut hydrator = MessageHydrator::default();
    let done = Message::with_id("assistant-1", MessageRole::Assistant, vec![]);
    let retried = Message::with_id("assistant-2", MessageRole::Assistant, vec![]);
    for event in [
        Event::StreamStart(StreamStart {
            event: ModelEventMeta::with_message(done.clone()),
        }),
        Event::StreamEnd(StreamEnd {
            event: ModelEventMeta::with_message(done),
            ..StreamEnd::default()
        }),
        Event::StreamStart(StreamStart {
            event: ModelEventMeta::with_message(retried.clone()),
        }),
        Event::ToolStart(ToolStart {
            event: ModelEventMeta::with_message(retried.clone()),
            tool_call_id: "tc1".to_owned(),
            tool_name: "search".to_owned(),
        }),
        Event::ToolDelta(ToolDelta {
            event: ModelEventMeta::with_message(retried.clone()),
            chunk: "{\"q\":\"partial".to_owned(),
            tool_call_id: "tc1".to_owned(),
        }),
        Event::StreamEnd(StreamEnd {
            event: ModelEventMeta::with_message(retried),
            ..StreamEnd::default()
        }),
        Event::ToolCallResult(ToolCallResult {
            replay: false,
            message: Message::with_id("tool-1", MessageRole::Tool, vec![]),
            results: vec![],
            exception: None,
        }),
    ] {
        hydrator.feed(event).expect("hydrate setup");
    }
    hydrator
        .feed(Event::Retry(Retry::default()))
        .expect("retry");
    assert_eq!(
        hydrator
            .messages
            .iter()
            .map(|message| message.id.as_str())
            .collect::<Vec<_>>(),
        ["assistant-1"]
    );
    assert_eq!(hydrator.message.id, "assistant-2");
    assert!(hydrator.message.parts.is_empty());
    assert!(!hydrator.ended);
}

#[test]
fn tool_end_is_rewritten_with_the_aggregated_call() {
    let mut hydrator = MessageHydrator::default();
    let message = Message::with_id("assistant", MessageRole::Assistant, vec![]);
    hydrator
        .feed(Event::ToolStart(ToolStart {
            event: ModelEventMeta::with_message(message.clone()),
            tool_call_id: "tc".to_owned(),
            tool_name: "search".to_owned(),
        }))
        .expect("start tool");
    hydrator
        .feed(Event::ToolDelta(ToolDelta {
            event: ModelEventMeta::with_message(message.clone()),
            chunk: "{\"q\":1}".to_owned(),
            tool_call_id: "tc".to_owned(),
        }))
        .expect("delta tool");
    let end = hydrator
        .feed(Event::ToolEnd(ToolEnd {
            event: ModelEventMeta::with_message(message),
            tool_call: ToolCallPart::new("tc", "search", ""),
            tool_call_id: "tc".to_owned(),
        }))
        .expect("end tool");
    let Event::ToolEnd(end) = end else {
        panic!("tool end")
    };
    assert_eq!(end.tool_call.tool_args, "{\"q\":1}");
}

#[test]
fn run_state_tracker_emits_blocked_transition_once() {
    let tool_call = ToolCallPart::new("tc", "dangerous", "{}");
    let assistant = Message::with_id("assistant", MessageRole::Assistant, vec![tool_call.into()]);
    let hook = HookPart {
        id: "part-hook".to_owned(),
        hook_id: "hook".to_owned(),
        hook_type: "approval".to_owned(),
        status: HookStatus::Pending,
        metadata: Default::default(),
        resolution: None,
        tool_call_id: Some("tc".to_owned()),
    };
    let mut tracker = RunStateTracker::default();
    assert!(
        tracker
            .feed(&Event::StreamStart(StreamStart {
                event: meta("assistant")
            }))
            .is_none()
    );
    assert!(
        tracker
            .feed(&Event::Hook(HookEvent {
                replay: false,
                message: Message::with_id(
                    "hook-message",
                    MessageRole::Internal,
                    vec![Part::Hook(hook.clone())]
                ),
                hook,
            }))
            .is_none()
    );
    let blocked = tracker.feed(&Event::StreamEnd(StreamEnd {
        event: ModelEventMeta::with_message(assistant),
        ..StreamEnd::default()
    }));
    assert_eq!(blocked.expect("blocked transition").hooks.len(), 1);
    assert!(tracker.blocked());

    let result = ToolResultPart::new("tc", "dangerous", Value::Null);
    assert!(
        tracker
            .feed(&Event::ToolCallResult(ToolCallResult {
                replay: false,
                message: Message::new(MessageRole::Tool, vec![result.clone().into()]),
                results: vec![result],
                exception: None,
            }))
            .is_none()
    );
}

#[test]
fn blocked_hook_snapshots_keep_insertion_order() {
    let mut tracker = RunStateTracker::default();
    for id in ["z-hook", "a-hook"] {
        let hook = HookPart {
            id: id.into(),
            hook_id: id.into(),
            hook_type: "approval".into(),
            status: HookStatus::Pending,
            metadata: Default::default(),
            resolution: None,
            tool_call_id: None,
        };
        tracker.feed(&Event::Hook(HookEvent {
            replay: false,
            message: Message::new(MessageRole::Internal, vec![]),
            hook,
        }));
    }
    let blocked = tracker
        .feed(&Event::StreamEnd(StreamEnd::default()))
        .unwrap();
    assert_eq!(
        blocked
            .hooks
            .iter()
            .map(|hook| hook.hook_id.as_str())
            .collect::<Vec<_>>(),
        vec!["z-hook", "a-hook"]
    );
}

#[test]
fn partial_results_keep_runtime_aggregator_factories_and_stable_keys() {
    use suprnova_ai::types::events::{Aggregator, AggregatorFactory};
    struct Sum(u64);
    impl Aggregator<Value> for Sum {
        type Snapshot = Value;
        type ModelInput = Value;
        fn feed(&mut self, value: Value) {
            self.0 += value.as_u64().unwrap();
        }
        fn snapshot(&self) -> Value {
            json!(self.0)
        }
        fn to_model_input(snapshot: Value) -> Value {
            json!({"sum":snapshot})
        }
    }
    let mut event:Event=serde_json::from_value(json!({"kind":"partial_tool_call_result","tool_call_id":"tc","tool_name":"sum","label":"numbers","value":3})).unwrap();
    let Event::PartialToolCallResult(ref mut result) = event else {
        unreachable!()
    };
    result.aggregator_factory = Some(AggregatorFactory::new(|| Sum(0)));
    assert_eq!(result.key(), (Some("tc"), &json!("numbers")));
    let factory = result.aggregator_factory.as_ref().unwrap().clone();
    let mut reducer = factory.create();
    reducer.feed(json!(3));
    reducer.feed(json!(4));
    assert_eq!(reducer.snapshot(), json!(7));
    assert_eq!(reducer.get_model_input(), json!({"sum":7}));
    assert_eq!(factory.create().snapshot(), json!(0));
    assert_eq!(event, event.clone());
    let wire = serde_json::to_value(event).unwrap();
    assert!(wire.get("aggregator_factory").is_none());
    let restored: Event = serde_json::from_value(wire).unwrap();
    let Event::PartialToolCallResult(restored) = restored else {
        unreachable!()
    };
    assert!(restored.aggregator_factory.is_none());
}

#[test]
fn tool_results_keep_original_runtime_error_and_source_chain() {
    use std::error::Error as _;
    use suprnova_ai::types::events::RuntimeError;
    #[derive(Debug, thiserror::Error)]
    #[error("tool failed")]
    struct ToolError {
        #[source]
        cause: std::io::Error,
    }
    let error = RuntimeError::new(ToolError {
        cause: std::io::Error::other("underlying failure"),
    });
    assert!(error.as_error().downcast_ref::<ToolError>().is_some());
    assert_eq!(
        error.as_error().source().unwrap().to_string(),
        "underlying failure"
    );
    assert_eq!(error.source().unwrap().to_string(), "tool failed");
    let event = Event::ToolCallResult(ToolCallResult {
        replay: false,
        message: Message::new(MessageRole::Tool, vec![]),
        results: vec![],
        exception: Some(error),
    });
    assert_eq!(event, event.clone());
    let wire = serde_json::to_value(event).unwrap();
    assert!(wire.get("exception").is_none());
}
