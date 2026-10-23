use std::collections::{BTreeMap, BTreeSet};

use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures::{TryStreamExt, stream};
use serde_json::{Value, json};
use suprnova_ai::{
    types::{Event, Message, messages::FileData},
    ui::ai_sdk::{
        UIMessage, outbound_stream::serialize_event, to_messages, to_stream, to_ui_messages,
        ui_events::*,
    },
};

fn fixtures() -> Value {
    serde_json::from_str(include_str!("fixtures/ui-python-0.8.0.json")).unwrap()
}

fn strings(value: &Value, result: &mut BTreeSet<String>) {
    match value {
        Value::String(value) => {
            result.insert(value.clone());
        }
        Value::Array(values) => values.iter().for_each(|v| strings(v, result)),
        Value::Object(values) => values.values().for_each(|v| strings(v, result)),
        _ => {}
    }
}

// IDs created during inbound parsing are intentionally random in both ports.
// Preserve every input ID, and compare the identity relationships of new IDs.
fn canonical_ids(value: &mut Value, input: &Value) {
    fn visit(value: &mut Value, protected: &BTreeSet<String>, ids: &mut BTreeMap<String, String>) {
        match value {
            Value::String(value) if !protected.contains(value) => {
                let generated = ["part_", "msg_"].iter().any(|prefix| {
                    value.strip_prefix(prefix).is_some_and(|suffix| {
                        suffix.len() == 12 && suffix.bytes().all(|b| b.is_ascii_hexdigit())
                    })
                });
                if generated {
                    let next = format!("generated-{}", ids.len());
                    *value = ids.entry(value.clone()).or_insert(next).clone();
                }
            }
            Value::Array(values) => values.iter_mut().for_each(|v| visit(v, protected, ids)),
            Value::Object(values) => {
                // Sort keys even when downstream applications enable preserve_order.
                let keys: BTreeSet<_> = values.keys().cloned().collect();
                for key in keys {
                    visit(values.get_mut(&key).unwrap(), protected, ids);
                }
            }
            _ => {}
        }
    }
    let mut protected = BTreeSet::new();
    strings(input, &mut protected);
    visit(value, &protected, &mut BTreeMap::new());
}

#[test]
fn inbound_matches_captured_python_requests() {
    for (index, case) in fixtures()["inbound"].as_array().unwrap().iter().enumerate() {
        let input: Vec<UIMessage> = serde_json::from_value(case["input"].clone()).unwrap();
        let result = to_messages(&input, None);
        if case["error"] == true {
            assert!(result.is_err(), "inbound case {index}");
            continue;
        }
        let (messages, approvals) = result.unwrap_or_else(|e| panic!("inbound case {index}: {e}"));
        let mut actual = json!({"messages":messages,"approvals":approvals});
        let mut expected = json!({"messages":case["messages"],"approvals":case["approvals"]});
        canonical_ids(&mut actual, &case["input"]);
        canonical_ids(&mut expected, &case["input"]);
        assert_eq!(actual, expected, "inbound case {index}");
    }
}

#[test]
fn outbound_matches_captured_python_histories() {
    for (index, case) in fixtures()["outbound"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let input: Vec<Message> = serde_json::from_value(case["input"].clone()).unwrap();
        let actual = serde_json::to_value(to_ui_messages(&input).unwrap()).unwrap();
        assert_eq!(actual, case["messages"], "outbound case {index}");
    }
}

#[test]
fn serialization_matches_exact_python_json() {
    for case in fixtures()["serialization"].as_array().unwrap() {
        let value = &case["event"];
        let text = |key: &str| value[key].as_str().unwrap().to_owned();
        let event = match value["type"].as_str().unwrap() {
            "start" => UIMessageStreamEvent::Start(UIStartEvent {
                message_id: value["messageId"].as_str().map(str::to_owned),
                ..Default::default()
            }),
            "text-delta" => UIMessageStreamEvent::TextDelta(UITextDeltaEvent {
                id: text("id"),
                delta: text("delta"),
                ..Default::default()
            }),
            "text-start" => UIMessageStreamEvent::TextStart(UITextStartEvent {
                id: text("id"),
                ..Default::default()
            }),
            "text-end" => UIMessageStreamEvent::TextEnd(UITextEndEvent {
                id: text("id"),
                ..Default::default()
            }),
            "tool-approval-response" => {
                UIMessageStreamEvent::ToolApprovalResponse(UIToolApprovalResponseEvent {
                    approval_id: text("approvalId"),
                    approved: value["approved"].as_bool().unwrap(),
                    reason: value["reason"].as_str().map(str::to_owned),
                    provider_executed: value["providerExecuted"].as_bool(),
                    provider_metadata: Some(
                        serde_json::from_value(value["providerMetadata"].clone()).unwrap(),
                    ),
                })
            }
            "start-step" => UIMessageStreamEvent::StartStep(UIStartStepEvent {}),
            "finish-step" => UIMessageStreamEvent::FinishStep(UIFinishStepEvent {}),
            "reset-step" => UIMessageStreamEvent::ResetStep(UIResetStepEvent {}),
            "finish" => UIMessageStreamEvent::Finish(UIFinishEvent {
                finish_reason: Some(FinishReason::Stop),
                ..Default::default()
            }),
            kind if kind.starts_with("data-") => UIMessageStreamEvent::Data(UIDataEvent {
                data_type: kind.strip_prefix("data-").unwrap().to_owned(),
                data: value["data"].clone(),
                ..Default::default()
            }),
            kind => panic!("unhandled fixture event {kind}"),
        };
        assert_eq!(
            serialize_event(&event).unwrap(),
            case["json"].as_str().unwrap()
        );
    }
}

#[test]
fn tool_arguments_match_python_spacing_escaping_and_object_order() {
    for case in fixtures()["tool_arguments"].as_array().unwrap() {
        assert_eq!(
            suprnova_ai::ui::ai_sdk::tool_utils::normalize_tool_args(&case["input"]),
            case["args"].as_str().unwrap()
        );
    }
}

#[tokio::test]
async fn stream_matches_captured_python_events() {
    for (index, case) in fixtures()["stream"].as_array().unwrap().iter().enumerate() {
        let mut input: Vec<Event> = serde_json::from_value(case["input"].clone()).unwrap();
        for (event, spec) in input.iter_mut().zip(case["input"].as_array().unwrap()) {
            if let (Event::File(file), Some(bytes)) = (event, spec["runtime_file_bytes"].as_str()) {
                file.data = FileData::Bytes(STANDARD.decode(bytes).unwrap());
            }
        }
        let result = to_stream(stream::iter(input.into_iter().map(Ok)))
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual = serde_json::to_value(result).unwrap();
        assert_eq!(actual, case["events"], "stream case {index}");
    }
}
