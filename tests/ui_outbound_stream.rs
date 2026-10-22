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

use ai::ui::ai_sdk::ui_events::*;
use ai::{Error, types::events::Event, ui::ai_sdk::outbound_stream::*};
use futures::{StreamExt, stream};
use serde_json::{Value, json};

fn event(value: Value) -> Event {
    serde_json::from_value(value).unwrap()
}

async fn collect(values: Vec<Value>) -> Vec<Value> {
    to_stream(stream::iter(values.into_iter().map(|v| Ok(event(v)))))
        .map(|v| serde_json::to_value(v.unwrap()).unwrap())
        .collect()
        .await
}

fn kinds(values: &[Value]) -> Vec<&str> {
    values.iter().map(|v| v["type"].as_str().unwrap()).collect()
}

#[test]
fn serialization_matches_python_json_and_headers() {
    let start = UIMessageStreamEvent::Start(UIStartEvent {
        message_id: Some("m1".into()),
        ..Default::default()
    });
    assert_eq!(
        serialize_event(&start).unwrap(),
        r#"{"type": "start", "messageId": "m1"}"#
    );
    let delta = UIMessageStreamEvent::TextDelta(UITextDeltaEvent {
        id: "t1".into(),
        delta: "hé 😀\n".into(),
        ..Default::default()
    });
    assert_eq!(
        format_sse(&delta).unwrap(),
        "data: {\"id\": \"t1\", \"delta\": \"h\\u00e9 \\ud83d\\ude00\\n\", \"type\": \"text-delta\"}\n\n"
    );
    let data = UIMessageStreamEvent::Data(UIDataEvent {
        data_type: "custom".into(),
        data: json!({"k": 1}),
        ..Default::default()
    });
    assert_eq!(
        serialize_event(&data).unwrap(),
        r#"{"data": {"k": 1}, "type": "data-custom"}"#
    );
    assert_eq!(format_done_sse(), "data: [DONE]\n\n");
    assert_eq!(
        UI_MESSAGE_STREAM_HEADERS,
        [
            ("Content-Type", "text/event-stream"),
            ("Cache-Control", "no-cache"),
            ("Connection", "keep-alive"),
            ("x-vercel-ai-ui-message-stream", "v1"),
            ("x-accel-buffering", "no")
        ]
    );
}

#[tokio::test]
async fn event_driven_text_streaming_and_finish_metadata() {
    let out = collect(vec![
        json!({"kind":"text_start","block_id":"txt1"}),
        json!({"kind":"text_delta","block_id":"txt1","chunk":"hi"}),
        json!({"kind":"text_end","block_id":"txt1"}),
    ])
    .await;
    assert_eq!(
        kinds(&out),
        [
            "start",
            "start-step",
            "text-start",
            "text-delta",
            "text-end",
            "finish-step",
            "finish"
        ]
    );
    assert!(out[6].get("messageMetadata").is_none());
    let msg = json!({"id":"assistant-1","role":"assistant","parts":[{"kind":"text","id":"text-1","text":"hello"}]});
    let out = collect(vec![
        json!({"kind":"text_delta","block_id":"text-1","chunk":"hello","message":msg}),
    ])
    .await;
    assert_eq!(out[0]["messageId"], "assistant-1");
    assert_eq!(
        out.last().unwrap()["messageMetadata"]["aiPython"]["sourceMessages"],
        json!([{"id":"assistant-1","role":"assistant","turnId":null,"partIds":["text-1"]}])
    );
}

#[tokio::test]
async fn finish_reasons_and_empty_metadata() {
    for (reason, expected) in [
        (Value::Null, "stop"),
        (json!("stop"), "stop"),
        (json!("content_filter"), "content-filter"),
        (json!("tool_call"), "tool-calls"),
        (json!("length"), "length"),
        (json!("error"), "error"),
        (json!("some_provider_reason"), "other"),
    ] {
        let out = collect(vec![
            json!({"kind":"stream_start","message":{"id":"empty","role":"assistant","parts":[]}}),
            json!({"kind":"stream_end","finish_reason":reason}),
        ])
        .await;
        assert_eq!(out.last().unwrap()["finishReason"], expected);
        assert!(out.last().unwrap().get("messageMetadata").is_none());
    }
}

#[tokio::test]
async fn retry_resets_step_reemits_ids_and_autocloses_blocks() {
    let out = collect(vec![
        json!({"kind":"retry"}),
        json!({"kind":"text_delta","block_id":"t1","chunk":"old"}),
        json!({"kind":"reasoning_delta","block_id":"r1","chunk":"why"}),
        json!({"kind":"retry"}),
        json!({"kind":"stream_start"}),
        json!({"kind":"text_delta","block_id":"t1","chunk":"new"}),
    ])
    .await;
    assert_eq!(
        kinds(&out),
        [
            "start",
            "start-step",
            "text-start",
            "text-delta",
            "reasoning-start",
            "reasoning-delta",
            "reset-step",
            "text-start",
            "text-delta",
            "text-end",
            "finish-step",
            "finish"
        ]
    );
    let out = collect(vec![
        json!({"kind":"stream_start"}),
        json!({"kind":"stream_end"}),
        json!({"kind":"stream_start"}),
    ])
    .await;
    assert_eq!(
        kinds(&out),
        [
            "start",
            "start-step",
            "finish-step",
            "start-step",
            "finish-step",
            "finish"
        ]
    );
}

#[tokio::test]
async fn sse_is_lazy_and_errors_do_not_emit_success_or_done() {
    let mut frames = to_sse(stream::iter([
        Ok(event(
            json!({"kind":"text_delta","block_id":"t","chunk":"hi"}),
        )),
        Err(Error::Validation("broken".into())),
    ]));
    let mut received = Vec::new();
    while let Some(frame) = frames.next().await {
        received.push(frame);
    }
    assert_eq!(received.len(), 5);
    assert!(received.last().unwrap().is_err());
    assert!(received[..4].iter().all(
        |v| !v.as_ref().unwrap().contains("finish") && !v.as_ref().unwrap().contains("[DONE]")
    ));
    assert_eq!(
        to_sse(stream::empty())
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(Result::unwrap)
            .collect::<Vec<_>>(),
        ["data: [DONE]\n\n"]
    );
}

#[tokio::test]
async fn tool_results_emit_input_output_errors_and_deduplicate() {
    let call = json!({"kind":"tool_call","id":"call-1","tool_call_id":"tc1","tool_name":"search","tool_args":"{\"q\":\"x\"}"});
    let result =
        json!({"kind":"tool_result","tool_call_id":"tc1","tool_name":"search","result":{"hits":1}});
    let result_event = json!({"kind":"tool_call_result","message":{"id":"tool-1","role":"tool","parts":[call,result.clone()]},"results":[result]});
    let out = collect(vec![result_event.clone(), result_event]).await;
    assert_eq!(
        kinds(&out),
        [
            "start",
            "start-step",
            "tool-input-start",
            "tool-input-available",
            "tool-output-available",
            "finish-step",
            "finish"
        ]
    );
    assert_eq!(out[3]["input"], json!({"q":"x"}));
    assert_eq!(out[4]["output"], json!({"hits":1}));
    for (value, text) in [
        (json!("failed"), "failed"),
        (json!({"error":"bad"}), "bad"),
        (json!({"message":"oops"}), "oops"),
        (json!({"detail":"why"}), "why"),
        (json!({}), "Tool execution failed"),
    ] {
        let out=collect(vec![json!({"kind":"tool_call_result","message":{"role":"tool","parts":[]},"results":[{"kind":"tool_result","tool_call_id":"err","tool_name":"search","result":value,"result_kind":"error"}]})]).await;
        assert_eq!(out[2]["errorText"], text);
    }
    let out=collect(vec![json!({"kind":"tool_call_result","message":{"role":"tool","parts":[]},"results":[{"kind":"tool_result","tool_call_id":"def","tool_name":"search","result":"internal","is_hook_deferred":true}]})]).await;
    assert!(!kinds(&out).contains(&"tool-output-available"));
}

#[tokio::test]
async fn streamed_tool_end_waits_for_result_and_retry_replays_tools() {
    let out=collect(vec![json!({"kind":"tool_start","tool_call_id":"tc1","tool_name":"search"}),json!({"kind":"tool_start","tool_call_id":"tc1","tool_name":"search"}),json!({"kind":"tool_delta","tool_call_id":"tc1","chunk":"{"}),json!({"kind":"retry"}),json!({"kind":"tool_delta","tool_call_id":"tc1","chunk":"{}"}),json!({"kind":"tool_end","tool_call_id":"tc1","tool_call":{"kind":"tool_call","tool_call_id":"tc1","tool_name":"search","tool_args":"{}"}})]).await;
    assert_eq!(
        kinds(&out),
        [
            "start",
            "start-step",
            "tool-input-start",
            "tool-input-delta",
            "reset-step",
            "tool-input-start",
            "tool-input-delta",
            "finish-step",
            "finish"
        ]
    );
    assert_eq!(out[5]["toolName"], "");
}

#[tokio::test]
async fn builtin_tool_stream_marks_provider_execution_and_dynamic() {
    let out=collect(vec![
        json!({"kind":"builtin_tool_start","tool_call_id":"tc1","tool_name":"web_search","provider_metadata":{"provider":{"start":true}}}),
        json!({"kind":"builtin_tool_delta","tool_call_id":"tc1","chunk":"{\"q\":\"ai\"}"}),
        json!({"kind":"builtin_tool_end","tool_call_id":"tc1","tool_call":{"kind":"builtin_tool_call","tool_call_id":"tc1","tool_name":"web_search","tool_args":"{\"q\":\"ai\"}","provider_metadata":{"provider":{"call":true}}}}),
        json!({"kind":"builtin_tool_result","tool_call_id":"tc1","result":{"kind":"builtin_tool_return","tool_call_id":"tc1","tool_name":"web_search","result":{"hits":1},"provider_metadata":{"provider":{"result":true}}}})
    ]).await;
    for index in [2, 4, 5] {
        assert_eq!(out[index]["providerExecuted"], true);
        assert_eq!(out[index]["dynamic"], true);
    }
    assert_eq!(
        out[2]["providerMetadata"],
        json!({"provider":{"start":true}})
    );
    assert_eq!(out[4]["input"], json!({"q":"ai"}));
    assert_eq!(
        out[4]["providerMetadata"],
        json!({"provider":{"call":true}})
    );
    assert_eq!(out[5]["output"], json!({"hits":1}));
    assert_eq!(
        out[5]["providerMetadata"],
        json!({"provider":{"result":true}})
    );
}

#[tokio::test]
async fn file_event_emits_data_url_and_preserves_provider_metadata() {
    use ai::types::{
        events::{FileEvent, ModelEventMeta},
        messages::FileData,
    };
    let events = vec![Ok(Event::File(FileEvent {
        event: ModelEventMeta {
            provider_metadata: Some(
                serde_json::from_value(json!({"provider":{"file":true}})).unwrap(),
            ),
            ..Default::default()
        },
        block_id: String::new(),
        media_type: "image/png".into(),
        data: FileData::Bytes(b"abc".to_vec()),
        filename: None,
    }))];
    let out = to_stream(stream::iter(events)).collect::<Vec<_>>().await;
    let file = serde_json::to_value(out[2].as_ref().unwrap()).unwrap();
    assert_eq!(
        file,
        json!({"type":"file","url":"data:image/png;base64,YWJj","mediaType":"image/png","providerMetadata":{"provider":{"file":true}}})
    );
}

fn hook_event(status: &str, resolution: Value) -> Value {
    json!({"kind":"hook","message":{"id":"internal-1","role":"internal","turn_id":"turn-1","parts":[]},"hook":{"kind":"hook","id":"hook-1","hook_id":"approve_tc1","hook_type":"ToolApproval","status":status,"tool_call_id":"tc1","metadata":{"isAutomatic":true,"providerExecuted":true,"callProviderMetadata":{"provider":{"approval":true}}},"resolution":resolution}})
}

#[tokio::test]
async fn approval_hooks_dedupe_requests_emit_responses_denials_and_cancellation() {
    let pending = hook_event("pending", Value::Null);
    let out = collect(vec![
        pending.clone(),
        pending,
        json!({"kind":"run_blocked"}),
        hook_event("resolved", json!({"granted":false,"reason":"not allowed"})),
        hook_event("cancelled", Value::Null),
    ])
    .await;
    assert_eq!(
        kinds(&out),
        [
            "start",
            "start-step",
            "tool-approval-request",
            "tool-approval-response",
            "tool-output-denied",
            "tool-output-error",
            "finish-step",
            "finish"
        ]
    );
    assert_eq!(out[0]["messageId"], "turn-1");
    assert_eq!(out[2]["isAutomatic"], true);
    assert_eq!(
        out[3],
        json!({"type":"tool-approval-response","approvalId":"approve_tc1","approved":false,"reason":"not allowed","providerExecuted":true,"providerMetadata":{"provider":{"approval":true}}})
    );
    assert_eq!(out[5]["errorText"], "Hook cancelled");
    assert!(
        collect(vec![json!({"kind":"run_blocked"})])
            .await
            .is_empty()
    );
}

#[derive(Default)]
struct Concat(String);
impl ai::types::events::Aggregator<Value> for Concat {
    type Snapshot = Value;
    type ModelInput = Value;
    fn feed(&mut self, value: Value) {
        self.0.push_str(value.as_str().unwrap());
    }
    fn snapshot(&self) -> Value {
        json!(self.0)
    }
    fn to_model_input(snapshot: Value) -> Value {
        snapshot
    }
}

#[tokio::test]
async fn partial_tool_results_accumulate_and_retry_recreates_aggregators() {
    use ai::types::events::{AggregatorFactory, PartialToolCallResult};
    let factory = AggregatorFactory::new(Concat::default);
    let partial = |text: &str, factory: Option<AggregatorFactory>| {
        Event::PartialToolCallResult(PartialToolCallResult {
            replay: false,
            tool_call_id: Some("tc1".into()),
            tool_name: Some("search".into()),
            label: Value::Null,
            value: json!(text),
            aggregator_factory: factory,
        })
    };
    let events = vec![
        Ok(partial("ignored", None)),
        Ok(partial("hit 1, ", Some(factory.clone()))),
        Ok(partial("hit 2, ", Some(factory.clone()))),
        Ok(partial("hit 3", Some(factory.clone()))),
        Ok(Event::Retry(Default::default())),
        Ok(partial("new", Some(factory))),
    ];
    let out = to_stream(stream::iter(events))
        .map(|v| serde_json::to_value(v.unwrap()).unwrap())
        .collect::<Vec<_>>()
        .await;
    let prelim: Vec<_> = out
        .iter()
        .filter(|v| v["preliminary"] == true)
        .map(|v| v["output"].clone())
        .collect();
    assert_eq!(
        prelim,
        vec![
            json!("hit 1, "),
            json!("hit 1, hit 2, "),
            json!("hit 1, hit 2, hit 3"),
            json!("new")
        ]
    );
    assert_eq!(out.iter().filter(|v| v["type"] == "start").count(), 1);
}

#[tokio::test]
async fn adapter_polls_source_only_when_output_queue_is_drained() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let polls = Arc::new(AtomicUsize::new(0));
    let p = polls.clone();
    let source = stream::iter([Ok(event(
        json!({"kind":"text_delta","block_id":"t","chunk":"hi"}),
    ))])
    .inspect(move |_| {
        p.fetch_add(1, Ordering::SeqCst);
    });
    let mut out = to_stream(source);
    assert_eq!(polls.load(Ordering::SeqCst), 0);
    for _ in 0..4 {
        assert!(out.next().await.unwrap().is_ok());
        assert_eq!(polls.load(Ordering::SeqCst), 1);
    }
}

struct Snapshot {
    value: Value,
    typed_bundle: bool,
}
impl ai::types::events::ValueAggregator for Snapshot {
    fn feed(&mut self, _: Value) {}
    fn snapshot(&self) -> Value {
        self.value.clone()
    }
    fn get_model_input(&self) -> Value {
        self.snapshot()
    }
    fn snapshot_result(&self) -> ai::types::messages::ToolResultValue {
        if self.typed_bundle {
            serde_json::from_value::<ai::types::messages::MessageBundle>(self.value.clone())
                .unwrap()
                .into()
        } else {
            self.value.clone().into()
        }
    }
}

#[tokio::test]
async fn typed_partial_message_bundle_is_one_bubble_plain_json_stays_unchanged() {
    use ai::types::events::{AggregatorFactory, PartialToolCallResult};
    let bundle = json!({"type":"messages","messages":[{"id":"inner","role":"assistant","parts":[{"kind":"text","id":"text","text":"hi from sub-agent"}]}]});
    for typed_bundle in [false, true] {
        let value = bundle.clone();
        let factory = AggregatorFactory::from_boxed(move || {
            Box::new(Snapshot {
                value: value.clone(),
                typed_bundle,
            })
        });
        let event = Event::PartialToolCallResult(PartialToolCallResult {
            replay: false,
            tool_call_id: Some("tc1".into()),
            tool_name: Some("research".into()),
            label: Value::Null,
            value: Value::Null,
            aggregator_factory: Some(factory),
        });
        let out = to_stream(stream::iter([Ok(event)]))
            .map(|v| serde_json::to_value(v.unwrap()).unwrap())
            .collect::<Vec<_>>()
            .await;
        if typed_bundle {
            assert_eq!(out[2]["output"]["role"], "assistant");
            assert_eq!(out[2]["output"]["parts"][0]["type"], "text");
        } else {
            assert_eq!(out[2]["output"], bundle);
        }
    }
}

#[tokio::test]
async fn empty_typed_partial_message_bundle_has_no_output_anchor() {
    use ai::types::events::{AggregatorFactory, PartialToolCallResult};
    let factory = AggregatorFactory::from_boxed(|| {
        Box::new(Snapshot {
            value: json!({"type":"messages","messages":[]}),
            typed_bundle: true,
        })
    });
    let event = Event::PartialToolCallResult(PartialToolCallResult {
        replay: false,
        tool_call_id: Some("tc1".into()),
        tool_name: None,
        label: Value::Null,
        value: Value::Null,
        aggregator_factory: Some(factory),
    });
    let out = to_stream(stream::iter([Ok(event)]))
        .map(|v| serde_json::to_value(v.unwrap()).unwrap())
        .collect::<Vec<_>>()
        .await;
    assert_eq!(
        kinds(&out),
        ["start", "start-step", "finish-step", "finish"]
    );
}

#[tokio::test]
async fn builtin_errors_use_python_text_and_empty_metadata_falls_back() {
    let error = json!({"kind":"builtin_tool_result","tool_call_id":"tc","provider_metadata":{"provider":{"fallback":true}},"result":{"kind":"builtin_tool_return","tool_call_id":"tc","tool_name":"builtin","is_error":true,"result":{"message":"bad","retry":false},"provider_metadata":{}}});
    let out = collect(vec![error.clone(), error]).await;
    assert_eq!(
        kinds(&out),
        [
            "start",
            "start-step",
            "tool-output-error",
            "finish-step",
            "finish"
        ]
    );
    let text = out[2]["errorText"].as_str().unwrap();
    assert!(text.contains("'message': 'bad'"));
    assert!(text.contains("'retry': False"));
    assert_eq!(
        out[2]["providerMetadata"],
        json!({"provider":{"fallback":true}})
    );
    assert_eq!(out[2]["providerExecuted"], true);
    assert_eq!(out[2]["dynamic"], true);
}

#[test]
fn serialization_omits_only_top_level_nulls_and_keeps_false() {
    let event = UIMessageStreamEvent::Data(UIDataEvent {
        data_type: "custom".into(),
        data: json!({"null":null,"items":[null,false]}),
        id: None,
        transient: Some(false),
    });
    let wire: Value = serde_json::from_str(&serialize_event(&event).unwrap()).unwrap();
    assert_eq!(
        wire,
        json!({"type":"data-custom","data":{"null":null,"items":[null,false]},"transient":false})
    );
    let event = UIMessageStreamEvent::ToolInputAvailable(UIToolInputAvailableEvent {
        tool_call_id: "tc".into(),
        tool_name: "null_input".into(),
        input: Value::Null,
        ..Default::default()
    });
    assert_eq!(
        serialize_event(&event).unwrap(),
        r#"{"toolCallId": "tc", "toolName": "null_input", "type": "tool-input-available"}"#
    );
}
