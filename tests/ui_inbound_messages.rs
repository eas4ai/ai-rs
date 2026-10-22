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

use ai::types::messages::{
    ContentOutput, ContentPart, HookStatus, MessageBundle, ResultKind, SpecialToolResult,
    ToolResultValue,
};
use ai::types::{Message, MessageRole, Part, TextPart, ToolCallPart, ToolResultPart};
use ai::ui::ai_sdk::inbound_messages::{ToolResultValidator, normalize_ui_messages};
use ai::ui::ai_sdk::{UIMessage, to_messages};
use ai::ui::ai_sdk::{id_utils, tool_utils};
use serde_json::{Value, json};

fn ui(role: &str, parts: Vec<Value>) -> UIMessage {
    serde_json::from_value(json!({"id":"m1", "role":role, "parts":parts}))
        .expect("valid UI message")
}

fn tool(state: &str) -> Value {
    json!({"type":"tool-search","toolCallId":"tc1","state":state,"input":{"q":"x"},"output":{"hits":3}})
}

#[test]
fn user_text_and_tool_boundaries() {
    let (messages, approvals) = to_messages(
        &[ui("user", vec![json!({"type":"text","text":"hello"})])],
        None,
    )
    .unwrap();
    assert_eq!(messages[0].role, MessageRole::User);
    assert_eq!(messages[0].text(), "hello");
    assert!(approvals.is_empty());
    let (messages, _) = to_messages(
        &[ui(
            "assistant",
            vec![
                json!({"type":"text","text":"before"}),
                tool("output-available"),
                json!({"type":"text","text":"after"}),
            ],
        )],
        None,
    )
    .unwrap();
    assert_eq!(
        messages.iter().map(|m| m.role).collect::<Vec<_>>(),
        [
            MessageRole::Assistant,
            MessageRole::Tool,
            MessageRole::Assistant
        ]
    );
    assert_eq!(messages[1].tool_results()[0].tool_call_id, "tc1");
    assert!(messages.iter().all(|m| m.turn_id.as_deref() == Some("m1")));
}

#[test]
fn approval_tombstones_and_trailing_assistant() {
    for (state, approved, expected) in [
        (
            "approval-requested",
            Value::Null,
            vec![MessageRole::Assistant, MessageRole::Internal],
        ),
        (
            "approval-responded",
            json!(true),
            vec![MessageRole::Assistant],
        ),
    ] {
        let message = ui(
            "assistant",
            vec![
                json!({"type":"tool-delete","toolCallId":"tc1","state":state,"approval":{"id":"approve_tc1","approved":approved}}),
            ],
        );
        let (messages, approvals) = to_messages(&[message], None).unwrap();
        assert_eq!(
            messages.iter().map(|m| m.role).collect::<Vec<_>>(),
            expected
        );
        if state == "approval-requested" {
            assert!(approvals.is_empty());
            let Part::Hook(h) = &messages[1].parts[0] else {
                panic!("hook")
            };
            assert_eq!(h.status, HookStatus::Pending);
            assert_eq!(h.hook_type, "ToolApproval");
        } else {
            assert_eq!(approvals[0].hook_id, "approve_tc1");
        }
    }
    let (messages, approvals)=to_messages(&[ui("user",vec![json!({"type":"text","text":"delete it"})]),ui("assistant",vec![json!({"type":"tool-delete","toolCallId":"tc1","state":"approval-responded","approval":{"id":"approve_tc1","approved":true}})])],None).unwrap();
    assert_eq!(
        messages.iter().map(|m| m.role).collect::<Vec<_>>(),
        [MessageRole::User, MessageRole::Assistant]
    );
    assert_eq!(approvals.len(), 1);
}

#[test]
fn normalizes_stale_state_and_rejects_empty_content() {
    let input = ui("assistant", vec![tool("input-available")]);
    let normalized = normalize_ui_messages(std::slice::from_ref(&input));
    assert_eq!(normalized[0].parts[0]["state"], "output-available");
    assert_eq!(input.parts[0]["state"], "input-available");
    for role in ["user", "system"] {
        assert!(to_messages(&[ui(role, vec![])], None).is_err());
    }
    assert!(
        to_messages(&[ui("assistant", vec![])], None)
            .unwrap()
            .0
            .is_empty()
    );
}

#[test]
fn accepts_ui_only_parts_and_metadata() {
    let input = ui(
        "assistant",
        vec![
            json!({"type":"custom","kind":"openai.compaction"}),
            json!({"type":"data-weather","id":"weather-1","data":{"status":"loading"}}),
            json!({"type":"source-url","sourceId":"src-1","url":"https://example.com"}),
            json!({"type":"reasoning-file","mediaType":"image/png","url":"data:image/png;base64,AAAA"}),
            json!({"type":"text","text":"visible","providerMetadata":{"provider":{"k":"v"}}}),
        ],
    );
    let (messages, approvals) = to_messages(&[input], None).unwrap();
    assert!(approvals.is_empty());
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].text(), "visible");
    let Part::Text(text) = &messages[0].parts[0] else {
        panic!("text")
    };
    assert_eq!(
        text.provider_metadata.as_ref().unwrap()["provider"],
        json!({"k":"v"})
    );
}

#[test]
fn dynamic_provider_executed_tool_stays_in_assistant() {
    let input = ui(
        "assistant",
        vec![
            json!({"type":"dynamic-tool","toolName":"web_search","toolCallId":"tc1","state":"output-available","input":{"query":"ai"},"output":[{"title":"result"}],"providerExecuted":true,"callProviderMetadata":{"provider":{"call":1}},"resultProviderMetadata":{"provider":{"result":1}}}),
        ],
    );
    let (messages, _) = to_messages(&[input], None).unwrap();
    assert_eq!(messages.len(), 1);
    let call = messages[0].builtin_tool_calls()[0];
    let result = messages[0].builtin_tool_returns()[0];
    assert_eq!(call.tool_name, "web_search");
    assert_eq!(call.tool_args, r#"{"query": "ai"}"#);
    assert_eq!(
        call.provider_metadata.as_ref().unwrap()["provider"],
        json!({"call":1})
    );
    assert_eq!(result.result, json!([{"title":"result"}]));
    assert_eq!(
        result.provider_metadata.as_ref().unwrap()["provider"],
        json!({"result":1})
    );
}

#[test]
fn special_tool_results_round_trip_metadata() {
    let results: [ToolResultValue; 3] = [
        MessageBundle::new(vec![Message::new(
            MessageRole::Assistant,
            vec![TextPart::new("Mars has two moons.").into()],
        )])
        .into(),
        MessageBundle::new(vec![
            Message::new(MessageRole::Assistant, vec![TextPart::new("first").into()]),
            Message::new(MessageRole::Assistant, vec![TextPart::new("second").into()]),
        ])
        .into(),
        ContentOutput::new(vec![
            ContentPart::Text(TextPart::new("desc")),
            ContentPart::File(
                ai::types::FilePart::from_bytes(b"img-bytes".to_vec(), Some("image/png"), None)
                    .unwrap(),
            ),
        ])
        .into(),
    ];
    for result in results {
        let messages = vec![
            Message {
                turn_id: Some("turn-1".into()),
                ..Message::with_id(
                    "a1",
                    MessageRole::Assistant,
                    vec![ToolCallPart::new("tc1", "research", "{}").into()],
                )
            },
            Message {
                turn_id: Some("turn-1".into()),
                ..Message::with_id(
                    "t1",
                    MessageRole::Tool,
                    vec![ToolResultPart::new("tc1", "research", result).into()],
                )
            },
        ];
        let ui = ai::ui::ai_sdk::to_ui_messages(&messages).unwrap();
        let (restored, _) = to_messages(&ui, None).unwrap();
        let part = restored
            .iter()
            .find(|m| m.role == MessageRole::Tool)
            .unwrap()
            .tool_results()[0];
        assert_eq!(part.result_kind, ResultKind::Special);
        assert!(!part.has_model_input());
        assert_eq!(restored[0].id, "a1");
        assert_eq!(restored[1].id, "t1");
        match &part.result {
            ToolResultValue::Special(SpecialToolResult::Messages(bundle)) => {
                assert!(!bundle.messages[0].text().is_empty())
            }
            ToolResultValue::Special(SpecialToolResult::Content(content)) => {
                assert_eq!(content.value.len(), 2)
            }
            _ => panic!("special result"),
        }
    }
}

struct Weather;
impl ToolResultValidator for Weather {
    fn validate_result(&self, name: &str, mut value: Value) -> Result<ToolResultValue, ai::Error> {
        if name == "weather" {
            let n = value["temp"]
                .as_str()
                .and_then(|s| s.parse::<i64>().ok())
                .ok_or_else(|| ai::Error::Validation("expected temperature".into()))?;
            value["temp"] = json!(n);
        }
        if name == "research" {
            return Ok(serde_json::from_value::<MessageBundle>(value)?.into());
        }
        Ok(value.into())
    }
}

#[test]
fn validates_tool_return_and_aggregator_output_with_explicit_registry() {
    // Rust uses the explicit registry for return annotations and aggregator outputs.
    let input = ui(
        "assistant",
        vec![json!({
            "type": "tool-weather", "toolCallId": "tc1", "state": "output-available",
            "input": {}, "output": {"temp": "72"}
        })],
    );
    let (messages, _) = to_messages(&[input], Some(&Weather)).unwrap();
    assert_eq!(
        messages[1].tool_results()[0].result.as_json().unwrap()["temp"],
        72
    );

    let input = ui(
        "assistant",
        vec![json!({
            "type": "tool-research", "toolCallId": "tc1", "state": "output-available",
            "output": {"type": "messages", "messages": [
                {"role": "assistant", "parts": [{"kind": "text", "text": "ok"}]}
            ]}
        })],
    );
    let (messages, _) = to_messages(&[input], Some(&Weather)).unwrap();
    let ToolResultValue::Special(SpecialToolResult::Messages(bundle)) =
        &messages[1].tool_results()[0].result
    else {
        panic!("bundle")
    };
    assert_eq!(bundle.messages[0].text(), "ok");

    let malformed = ui(
        "assistant",
        vec![json!({
            "type": "tool-weather", "toolCallId": "tc1", "state": "output-available",
            "output": {"temp": "invalid"}
        })],
    );
    assert!(matches!(
        to_messages(&[malformed], Some(&Weather)),
        Err(ai::Error::Validation(_))
    ));
}

#[test]
fn passthrough_keeps_wire_shape_and_does_not_infer_special_results() {
    for output in [
        json!({"pong":true}),
        json!({"type":"messages","messages":[]}),
    ] {
        let mut part = tool("output-available");
        part["output"] = output.clone();
        let (messages, _) = to_messages(&[ui("assistant", vec![part])], None).unwrap();
        assert_eq!(
            messages[1].tool_results()[0].result.as_json(),
            Some(&output)
        );
    }
}

#[test]
fn defaults_aliases_unknown_parts_and_extra_fields() {
    let message: UIMessage=serde_json::from_value(json!({"role":"assistant","extraMessage":true,"parts":[{"type":"unknown","ignore":true},{"type":"text","text":"ok","provider_metadata":{"p":1},"extraPart":true},{"type":"dynamic-tool","tool_name":"x","tool_call_id":"tc1","state":"input-available","provider_executed":true}]})).unwrap();
    assert!(message.id.starts_with("msg_"));
    assert_eq!(message.parts.len(), 2);
    let wire = serde_json::to_value(message).unwrap();
    assert_eq!(wire["extraMessage"], true);
    assert_eq!(wire["parts"][0]["providerMetadata"], json!({"p":1}));
    assert_eq!(wire["parts"][0]["extraPart"], true);
    assert!(wire["parts"][0]["state"].is_null());
    assert_eq!(wire["parts"][1]["toolName"], "x");
    for raw in [
        json!({"role":"tool","parts":[]}),
        json!({"role":"assistant","parts":[{"type":"text"}]}),
        json!({"role":"assistant","parts":[{"type":"tool-x","toolCallId":"x","state":"bad"}]}),
    ] {
        assert!(serde_json::from_value::<UIMessage>(raw).is_err());
    }
}

#[test]
fn deferred_approvals_patch_only_missing_trailing_results() {
    let input = ui(
        "assistant",
        vec![
            json!({"type":"tool-ok","toolCallId":"tc1","state":"output-available","output":{"ok":true}}),
            json!({"type":"tool-wait","toolCallId":"tc2","state":"approval-responded","approval":{"id":"gate","approved":true}}),
        ],
    );
    let (messages, _) = to_messages(&[input], None).unwrap();
    let results = messages[1].tool_results();
    assert_eq!(results.len(), 2);
    assert!(results[1].is_hook_deferred);
    assert_eq!(
        results[1].result.as_json(),
        Some(&json!("Deferred on hook 'gate'"))
    );
}

#[test]
fn tool_output_errors_denials_and_legacy_invocation() {
    for (part, expected) in [
        (
            json!({"type":"tool-x","toolCallId":"tc1","state":"output-error","input":null,"rawInput":{"raw":1},"errorText":"broken","output":{"value":5}}),
            json!({"error":"broken","value":5}),
        ),
        (
            json!({"type":"tool-x","toolCallId":"tc1","state":"output-denied","approval":{"id":"gate","reason":"no"}}),
            json!({"type":"error-text","value":"no"}),
        ),
    ] {
        let (messages, _) = to_messages(&[ui("assistant", vec![part])], None).unwrap();
        let result = messages[1].tool_results()[0];
        assert_eq!(result.result_kind, ResultKind::Error);
        assert_eq!(result.result.as_json(), Some(&expected));
    }
    let (messages,_)=to_messages(&[ui("assistant",vec![json!({"type":"tool-invocation","toolInvocationId":"tc1","toolName":"old","args":{"x":1},"state":"output-available","result":[1,2]})])],None).unwrap();
    assert_eq!(messages[0].tool_calls()[0].tool_name, "old");
    assert_eq!(
        messages[1].tool_results()[0].result.as_json(),
        Some(&json!({"value":[1,2]}))
    );
}

#[test]
fn id_metadata_ignores_malformed_entries_and_restores_part_ids() {
    let messages = vec![Message::with_id(
        "a",
        MessageRole::Assistant,
        vec![TextPart::new("x").into()],
    )];
    let metadata = id_utils::metadata_for(&messages);
    let sources = id_utils::source_messages_from(&metadata);
    let restored = id_utils::restore_source_ids(
        vec![Message::new(
            MessageRole::Assistant,
            vec![TextPart::new("x").into()],
        )],
        &sources,
    );
    assert_eq!(restored, messages);
    assert!(id_utils::source_messages_from(&json!({"aiPython":{"sourceMessages":[null,{"id":5,"role":"assistant"},{"id":"x","role":"invalid"}]}})).is_empty());
    assert!(
        id_utils::tool_result_kinds_from(&json!({"aiPython":{"toolResultKinds":true}})).is_empty()
    );
}

#[test]
fn tool_utilities_match_python_json_format() {
    assert_eq!(
        tool_utils::normalize_tool_input(r#"{"x":1}"#),
        json!({"x":1})
    );
    assert_eq!(tool_utils::normalize_tool_input("bad"), json!("bad"));
    assert_eq!(tool_utils::normalize_tool_args(&Value::Null), "{}");
    assert_eq!(tool_utils::normalize_tool_args(&json!("bad")), "bad");
    assert_eq!(
        tool_utils::normalize_tool_args(&json!({"x":["é",true,null]})),
        r#"{"x": ["\u00e9", true, null]}"#
    );
}

#[test]
fn preserves_arbitrary_extra_fields_and_validates_declared_fields() {
    let message:UIMessage=serde_json::from_value(json!({"role":"assistant","parts":[{"type":"text","text":"ok","provider_executed":{"custom":true},"providerExecuted":[],"title":42},{"type":"step-start","providerMetadata":5,"transient":"extra"},{"type":"data-x","data":null,"provider_metadata":"retained"}]})).unwrap();
    assert_eq!(
        message.parts[0]["provider_executed"],
        json!({"custom":true})
    );
    assert_eq!(message.parts[0]["providerExecuted"], json!([]));
    assert_eq!(message.parts[0]["title"], 42);
    assert_eq!(message.parts[1]["providerMetadata"], 5);
    assert_eq!(message.parts[2]["provider_metadata"], "retained");
    for extra in [json!({"reason":5}), json!({"isAutomatic":{}})] {
        let mut approval = json!({"id":"gate"});
        approval
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        assert!(serde_json::from_value::<UIMessage>(json!({"role":"assistant","parts":[{"type":"tool-x","toolCallId":"tc1","state":"approval-requested","approval":approval}]})).is_err());
    }
    assert!(
        serde_json::from_value::<UIMessage>(
            json!({"role":"assistant","parts":[{"type":"data-x","id":2,"data":null}]})
        )
        .is_err()
    );
    let message:UIMessage=serde_json::from_value(json!({"role":"assistant","parts":[{"type":"tool-x","toolCallId":"tc1","state":"approval-responded","approval":{"id":"gate","approved":"true","isAutomatic":0},"providerExecuted":"off"}]})).unwrap();
    assert_eq!(message.parts[0]["approval"]["approved"], true);
    assert_eq!(message.parts[0]["approval"]["isAutomatic"], false);
    assert_eq!(message.parts[0]["providerExecuted"], false);
}

#[test]
fn tool_arguments_match_python_finite_float_spelling() {
    for (input, expected) in [
        ("1e-8", "1e-08"),
        ("1e-5", "1e-05"),
        ("1e16", "1e+16"),
        ("1e20", "1e+20"),
        ("-0.0", "-0.0"),
        ("0.0", "0.0"),
        ("1.0", "1.0"),
        ("0.0001", "0.0001"),
        ("0.00009999999999999999", "9.999999999999999e-05"),
        ("1.2345678901234567", "1.2345678901234567"),
        ("1.7976931348623157e308", "1.7976931348623157e+308"),
        ("5e-324", "5e-324"),
        ("2.2250738585072014e-308", "2.2250738585072014e-308"),
        ("9007199254740992.0", "9007199254740992.0"),
        ("9999999999999998.0", "9999999999999998.0"),
        ("1000000000000000100.0", "1.0000000000000001e+18"),
        ("-1.234e-5", "-1.234e-05"),
        ("1.2e-7", "1.2e-07"),
        ("1e15", "1000000000000000.0"),
        ("1.234567890123456e16", "1.234567890123456e+16"),
    ] {
        let value = Value::from(input.parse::<f64>().unwrap());
        assert_eq!(
            tool_utils::normalize_tool_args(&value),
            expected,
            "input {input}"
        );
    }
}
