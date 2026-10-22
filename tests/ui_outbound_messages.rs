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

use std::collections::BTreeMap;

use ai::{
    types::{FilePart, Message, MessageRole, Part, TextPart, messages::MessageBundle},
    ui::ai_sdk::{
        inbound_messages::to_messages,
        outbound_messages::{
            bundle_to_wire_output, dedupe_tool_parts, merge_approval_signals, merge_tool_results,
            to_ui_messages, to_ui_parts,
        },
    },
};
use serde_json::{Value, json};

fn message(value: Value) -> Message {
    serde_json::from_value(value).expect("valid core message fixture")
}

fn part(value: Value) -> Part {
    serde_json::from_value(value).expect("valid core part fixture")
}

fn call() -> Part {
    part(
        json!({"kind":"tool_call","id":"call-0","tool_call_id":"tc1","tool_name":"search","tool_args":"{\"q\":\"x\"}"}),
    )
}

fn hook(status: &str, granted: Value) -> Part {
    part(
        json!({"kind":"hook","id":"hook-part","hook_id":"approve_tc1","hook_type":"ToolApproval","status":status,"tool_call_id":"tc1","resolution":{"granted":granted,"reason":null}}),
    )
}

fn parallel_tool_turn(turn: &str, prefix: &str) -> Vec<Message> {
    vec![
        message(
            json!({"id":format!("{prefix}:assistant:0"),"turn_id":turn,"role":"assistant","parts":[
                {"kind":"text","id":format!("{prefix}:text:0"),"text":"I will run two tools."},
                {"kind":"tool_call","id":format!("{prefix}:call:bash"),"tool_call_id":"tc-bash","tool_name":"bash","tool_args":"{\"command\":\"date\"}"},
                {"kind":"tool_call","id":format!("{prefix}:call:web"),"tool_call_id":"tc-web","tool_name":"web_fetch","tool_args":"{\"url\":\"https://httpbin.org/get\"}"}
            ]}),
        ),
        message(
            json!({"id":format!("{prefix}:tool:0"),"turn_id":turn,"role":"tool","parts":[
                {"kind":"tool_result","id":format!("{prefix}:result:bash"),"tool_call_id":"tc-bash","tool_name":"bash","result":"Tue May 19 2026"},
                {"kind":"tool_result","id":format!("{prefix}:result:web"),"tool_call_id":"tc-web","tool_name":"web_fetch","result":{"status":200}}
            ]}),
        ),
        message(
            json!({"id":format!("{prefix}:assistant:1"),"turn_id":turn,"role":"assistant","parts":[
                {"kind":"text","id":format!("{prefix}:text:1"),"text":"Both tools finished."}
            ]}),
        ),
    ]
}

fn tool_counts(messages: &[Message]) -> BTreeMap<(String, String), usize> {
    let mut counts = BTreeMap::new();
    for message in messages {
        for part in &message.parts {
            let key = match part {
                Part::ToolCall(p) => ("tool_call", &p.tool_call_id),
                Part::ToolResult(p) => ("tool_result", &p.tool_call_id),
                _ => continue,
            };
            *counts.entry((key.0.into(), key.1.clone())).or_default() += 1;
        }
    }
    counts
}

#[test]
fn to_ui_parts_text_and_reasoning() {
    let parts = to_ui_parts(&[
        part(json!({"kind":"reasoning","text":"thinking"})),
        TextPart::new("hi").into(),
        TextPart::new("").into(),
        part(json!({"kind":"reasoning","text":""})),
    ])
    .expect("outbound parts");
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], "reasoning");
    assert_eq!(parts[0]["text"], "thinking");
    assert_eq!(parts[1]["type"], "text");
    assert_eq!(parts[1]["text"], "hi");
}

#[test]
fn to_ui_parts_tool_call_parses_json_args() {
    let parts = to_ui_parts(&[call()]).expect("tool part");
    assert_eq!(parts[0]["type"], "tool-search");
    assert_eq!(parts[0]["input"], json!({"q":"x"}));
    assert_eq!(parts[0]["state"], "input-available");
}

#[test]
fn merge_tool_results_updates_state_and_output() {
    let mut parts = to_ui_parts(&[call()]).expect("tool part");
    merge_tool_results(&mut parts, &[part(json!({"kind":"tool_result","tool_call_id":"tc1","tool_name":"search","result":{"hits":3}}))]).expect("merge result");
    assert_eq!(parts[0]["state"], "output-available");
    assert_eq!(parts[0]["output"], json!({"hits":3}));
}

#[test]
fn merge_approval_signals_deferred_then_resolved() {
    let mut parts = to_ui_parts(&[call()]).expect("tool part");
    merge_approval_signals(&mut parts, &[hook("pending", Value::Null)]);
    assert_eq!(parts[0]["state"], "approval-requested");
    assert_eq!(parts[0]["approval"]["id"], "approve_tc1");
    merge_approval_signals(&mut parts, &[hook("resolved", json!(true))]);
    assert_eq!(parts[0]["state"], "approval-responded");
    assert_eq!(parts[0]["approval"]["approved"], true);
}

#[test]
fn approval_resolution_state_uses_python_truthiness_before_boolean_coercion() {
    for (granted, state, approved, output) in [
        (
            json!(1),
            "approval-responded",
            json!(true),
            json!({"hits":3}),
        ),
        (
            json!("true"),
            "approval-responded",
            json!(true),
            json!({"hits":3}),
        ),
        (
            json!("false"),
            "approval-responded",
            json!(false),
            json!({"hits":3}),
        ),
        (json!(0), "output-denied", json!(false), Value::Null),
        (json!(false), "output-denied", json!(false), Value::Null),
    ] {
        let messages = [
            Message::with_id("a1", MessageRole::Assistant, vec![call()]),
            Message::new(
                MessageRole::Tool,
                vec![part(
                    json!({"kind":"tool_result","tool_call_id":"tc1","tool_name":"search","result":{"hits":3}}),
                )],
            ),
            Message::new(
                MessageRole::Internal,
                vec![hook("resolved", granted.clone())],
            ),
        ];
        let ui = to_ui_messages(&messages).expect("accepted approval resolution");
        let tool = &ui[0].parts[0];
        assert_eq!(tool["state"], state, "resolution granted={granted}");
        assert_eq!(
            tool["approval"]["approved"], approved,
            "resolution granted={granted}"
        );
        assert_eq!(tool["output"], output, "resolution granted={granted}");
    }
}

#[test]
fn to_ui_messages_user_and_assistant() {
    let ui = to_ui_messages(&[
        Message::with_id("u1", MessageRole::User, vec![TextPart::new("hi").into()]),
        Message::with_id(
            "a1",
            MessageRole::Assistant,
            vec![TextPart::new("hello back").into()],
        ),
    ])
    .expect("UI messages");
    assert_eq!(ui.len(), 2);
    assert_eq!(ui[0].role, "user");
    assert_eq!(ui[1].role, "assistant");
    assert_eq!(ui[1].id, "a1");
}

#[test]
fn to_ui_messages_merges_assistant_tool_internal() {
    let ui = to_ui_messages(&[
        Message::with_id("a1", MessageRole::Assistant, vec![TextPart::new("calling").into(), call()]),
        message(json!({"role":"tool","parts":[{"kind":"tool_result","tool_call_id":"tc1","tool_name":"search","result":{"hits":2}}]})),
        Message::new(MessageRole::Assistant, vec![TextPart::new("done").into()]),
    ]).expect("UI messages");
    assert_eq!(ui.len(), 1);
    assert_eq!(ui[0].id, "a1");
    assert_eq!(ui[0].parts[0]["text"], "calling");
    assert_eq!(ui[0].parts[1]["state"], "output-available");
    assert_eq!(ui[0].parts[1]["output"], json!({"hits":2}));
    assert_eq!(ui[0].parts[2]["text"], "done");
}

#[test]
fn to_ui_messages_records_source_messages_in_metadata() {
    let source = parallel_tool_turn("turn-1", "turn-1");
    let ui = to_ui_messages(&source).expect("UI messages");
    assert_eq!(ui[0].id, "turn-1");
    let expected: Vec<Value> = source.iter().map(|m| json!({"id":m.id,"role":m.role,"turnId":m.turn_id,"partIds":m.parts.iter().map(Part::id).collect::<Vec<_>>()})).collect();
    assert_eq!(
        ui[0].metadata,
        json!({"aiPython":{"sourceMessages":expected}})
    );
}

#[test]
fn to_ui_messages_internal_role_merges_approval() {
    let ui = to_ui_messages(&[
        Message::with_id("a1", MessageRole::Assistant, vec![call()]),
        Message::new(MessageRole::Internal, vec![hook("pending", Value::Null)]),
    ])
    .expect("UI messages");
    assert_eq!(ui[0].parts[0]["state"], "approval-requested");
    assert_eq!(ui[0].parts[0]["approval"]["id"], "approve_tc1");
}

#[test]
fn to_ui_messages_user_message_uses_own_id() {
    let mut user = Message::with_id("u1", MessageRole::User, vec![TextPart::new("a").into()]);
    user.turn_id = Some("different-turn".into());
    assert_eq!(to_ui_messages(&[user]).expect("UI user")[0].id, "u1");
}

#[test]
fn to_ui_messages_uses_first_assistant_id_as_bubble_id() {
    let ui = to_ui_messages(&[
        Message::with_id(
            "a1",
            MessageRole::Assistant,
            vec![TextPart::new("first").into()],
        ),
        Message::with_id(
            "a2",
            MessageRole::Assistant,
            vec![TextPart::new("second").into()],
        ),
    ])
    .expect("UI messages");
    assert_eq!(ui.len(), 1);
    assert_eq!(ui[0].id, "a1");
}

#[test]
fn to_ui_messages_preserves_provider_metadata_and_files() {
    let mut file =
        FilePart::from_bytes(b"abc".to_vec(), Some("image/png"), Some("image.png".into()))
            .expect("file");
    file.provider_metadata = Some([("provider".into(), json!({"file":true}))].into());
    let ui = to_ui_messages(&[Message::with_id("a1", MessageRole::Assistant, vec![
        part(json!({"kind":"text","text":"hello","provider_metadata":{"provider":{"text":true}}})),
        file.into(),
    ])]).expect("UI files");
    assert_eq!(
        ui[0].parts[0]["providerMetadata"],
        json!({"provider":{"text":true}})
    );
    assert_eq!(ui[0].parts[1]["url"], "data:image/png;base64,YWJj");
    assert_eq!(ui[0].parts[1]["filename"], "image.png");
    assert_eq!(
        ui[0].parts[1]["providerMetadata"],
        json!({"provider":{"file":true}})
    );
}

#[test]
fn to_ui_messages_maps_builtin_tools_to_dynamic_parts() {
    let ui = to_ui_messages(&[message(json!({"id":"a1","role":"assistant","parts":[
        {"kind":"builtin_tool_call","tool_call_id":"tc1","tool_name":"web_search","tool_args":"{\"q\":\"ai\"}","provider_metadata":{"provider":{"call":true}}},
        {"kind":"builtin_tool_return","tool_call_id":"tc1","tool_name":"web_search","result":{"hits":1},"provider_metadata":{"provider":{"result":true}}}
    ]}))]).expect("dynamic tools");
    assert_eq!(ui[0].parts.len(), 1);
    let p = &ui[0].parts[0];
    assert_eq!(p["type"], "dynamic-tool");
    assert_eq!(p["providerExecuted"], true);
    assert_eq!(p["state"], "output-available");
    assert_eq!(p["input"], json!({"q":"ai"}));
    assert_eq!(p["output"], json!({"hits":1}));
    assert_eq!(p["callProviderMetadata"], json!({"provider":{"call":true}}));
    assert_eq!(
        p["resultProviderMetadata"],
        json!({"provider":{"result":true}})
    );
}

#[test]
fn collapsed_assistant_turn_roundtrips_internal_ids() {
    let original = vec![
        message(
            json!({"id":"assistant-alpha","turn_id":"turn-arbitrary","role":"assistant","parts":[
                {"kind":"text","id":"text-alpha","text":"calling first"},
                {"kind":"tool_call","id":"call-alpha","tool_call_id":"tc-first","tool_name":"search","tool_args":"{\"q\":\"first\"}"}
            ]}),
        ),
        message(
            json!({"id":"tool-beta","turn_id":"turn-arbitrary","role":"tool","parts":[
                {"kind":"tool_result","id":"result-beta","tool_call_id":"tc-first","tool_name":"search","result":{"hits":1}}
            ]}),
        ),
        message(
            json!({"id":"assistant-gamma","turn_id":"turn-arbitrary","role":"assistant","parts":[
                {"kind":"text","id":"text-gamma","text":"calling second"},
                {"kind":"tool_call","id":"call-gamma","tool_call_id":"tc-second","tool_name":"lookup","tool_args":"{\"id\":2}"}
            ]}),
        ),
        message(
            json!({"id":"tool-delta","turn_id":"turn-arbitrary","role":"tool","parts":[
                {"kind":"tool_result","id":"result-delta","tool_call_id":"tc-second","tool_name":"lookup","result":{"value":2}}
            ]}),
        ),
        message(
            json!({"id":"assistant-epsilon","turn_id":"turn-arbitrary","role":"assistant","parts":[
                {"kind":"text","id":"text-epsilon","text":"all done"}
            ]}),
        ),
    ];
    let ui = to_ui_messages(&original).expect("UI messages");
    let (back, approvals) = to_messages(&ui, None).expect("roundtrip");
    assert!(approvals.is_empty());
    assert_eq!(ui[0].id, "turn-arbitrary");
    assert_eq!(
        back.iter()
            .map(|m| (
                &m.id,
                m.role,
                &m.turn_id,
                m.parts.iter().map(Part::id).collect::<Vec<_>>()
            ))
            .collect::<Vec<_>>(),
        original
            .iter()
            .map(|m| (
                &m.id,
                m.role,
                &m.turn_id,
                m.parts.iter().map(Part::id).collect::<Vec<_>>()
            ))
            .collect::<Vec<_>>()
    );
}

#[test]
fn common_id_upsert_persistence_is_idempotent_after_reload() {
    let mut store = vec![Message::with_id(
        "user-1",
        MessageRole::User,
        vec![TextPart::new("run two tools").into()],
    )];
    store.extend(parallel_tool_turn("turn-1", "turn-1"));
    let (mut second, _) =
        to_messages(&to_ui_messages(&store).expect("reload UI"), None).expect("request history");
    second.push(Message::with_id(
        "user-2",
        MessageRole::User,
        vec![TextPart::new("do nothing").into()],
    ));
    second.push(Message::with_id(
        "turn-2:assistant:0",
        MessageRole::Assistant,
        vec![TextPart::new("standing by").into()],
    ));
    for message in second {
        if let Some(existing) = store.iter_mut().find(|m| m.id == message.id) {
            *existing = message;
        } else {
            store.push(message);
        }
    }
    assert_eq!(
        tool_counts(&store),
        BTreeMap::from([
            (("tool_call".into(), "tc-bash".into()), 1),
            (("tool_result".into(), "tc-bash".into()), 1),
            (("tool_call".into(), "tc-web".into()), 1),
            (("tool_result".into(), "tc-web".into()), 1),
        ])
    );
}

#[test]
fn duplicate_tool_copies_do_not_reach_model_integrity() {
    let mut history = parallel_tool_turn("turn-1", "server");
    history.extend(parallel_tool_turn("turn-1", "client"));
    let (back, _) = to_messages(&to_ui_messages(&history).expect("reloaded UI"), None)
        .expect("request history");
    assert_eq!(
        tool_counts(&back).values().copied().collect::<Vec<_>>(),
        vec![1, 1, 1, 1]
    );
    let calls = back
        .iter()
        .flat_map(Message::tool_calls)
        .map(|p| &p.tool_call_id)
        .collect::<Vec<_>>();
    let results = back
        .iter()
        .flat_map(Message::tool_results)
        .map(|p| &p.tool_call_id)
        .collect::<Vec<_>>();
    assert_eq!(calls, results);
}

#[test]
fn deferred_approval_hook_roundtrips_tool_call_id() {
    for hook_id in ["approve_tc-1", "my_custom_gate"] {
        let history = vec![
            message(
                json!({"id":"assistant-1","turn_id":"turn-1","role":"assistant","parts":[{"kind":"tool_call","id":"call-1","tool_call_id":"tc-1","tool_name":"bash","tool_args":"{\"command\":\"ls\"}"}]}),
            ),
            message(
                json!({"id":"hook-1","turn_id":"turn-1","role":"internal","parts":[{"kind":"hook","id":"part-1","hook_id":hook_id,"hook_type":"ToolApproval","status":"pending","tool_call_id":"tc-1","metadata":{"tool":"bash","kwargs":{"command":"ls"}}}]}),
            ),
        ];
        let ui = to_ui_messages(&history).expect("parked turn");
        assert_eq!(ui[0].parts[0]["state"], "approval-requested");
        let (back, approvals) = to_messages(&ui, None).expect("roundtrip hook");
        assert!(approvals.is_empty());
        let hooks = back
            .iter()
            .flat_map(|m| &m.parts)
            .filter_map(|p| match p {
                Part::Hook(h) => Some(h),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(hooks.len(), 1);
        assert_eq!(hooks[0].hook_id, hook_id);
        assert_eq!(hooks[0].tool_call_id.as_deref(), Some("tc-1"));
    }
}

#[test]
fn tool_merge_keeps_first_position_and_highest_state_and_non_null_fields() {
    let parts = dedupe_tool_parts(vec![
        json!({"type":"tool-search","toolCallId":"tc1","state":"input-streaming","input":null}),
        json!({"type":"text","text":"between"}),
        json!({"type":"tool-search","toolCallId":"tc1","state":"output-available","input":{"q":"x"},"output":1,"title":"Search"}),
        json!({"type":"tool-search","toolCallId":"tc1","state":"approval-requested","input":{"q":"other"},"approval":{"id":"gate"},"output":null}),
    ]);
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["state"], "output-available");
    assert_eq!(parts[0]["input"], json!({"q":"x"}));
    assert_eq!(parts[0]["output"], 1);
    assert_eq!(parts[0]["title"], "Search");
    assert_eq!(parts[0]["approval"]["id"], "gate");
    assert_eq!(parts[1]["text"], "between");
}

#[test]
fn denied_output_ignores_result_and_deferred_result_is_skipped() {
    let mut parts = to_ui_parts(&[call()]).expect("tool part");
    merge_approval_signals(&mut parts, &[hook("resolved", json!(false))]);
    merge_tool_results(&mut parts, &[part(json!({"kind":"tool_result","tool_call_id":"tc1","tool_name":"search","result":"hidden"}))]).expect("denied merge");
    assert_eq!(parts[0]["state"], "output-denied");
    assert!(parts[0]["output"].is_null());
    let mut parts = to_ui_parts(&[call()]).expect("tool part");
    merge_tool_results(&mut parts, &[part(json!({"kind":"tool_result","tool_call_id":"tc1","tool_name":"search","is_hook_deferred":true,"result":"parked"}))]).expect("deferred merge");
    assert_eq!(parts[0]["state"], "input-available");
    merge_approval_signals(&mut parts, &[hook("cancelled", Value::Null)]);
    assert_eq!(parts[0]["errorText"], "Hook cancelled");
}

#[test]
fn tool_result_error_uses_json_text_and_preserves_metadata() {
    let mut parts = to_ui_parts(&[call()]).expect("tool part");
    merge_tool_results(&mut parts, &[part(json!({"kind":"tool_result","tool_call_id":"tc1","tool_name":"search","result_kind":"error","result":{"failed":true,"detail":"é"},"provider_metadata":{"p":{"error":true}}}))]).expect("error merge");
    assert_eq!(parts[0]["state"], "output-error");
    assert_eq!(
        parts[0]["errorText"],
        "{\"failed\": true, \"detail\": \"\\u00e9\"}"
    );
    assert_eq!(
        parts[0]["resultProviderMetadata"],
        json!({"p":{"error":true}})
    );
}

#[test]
fn builtin_error_result_formats_float_numbers_like_python_repr() {
    let parts = to_ui_parts(&[part(json!({
        "kind":"builtin_tool_return","tool_call_id":"builtin-error","tool_name":"compute", "is_error":true,
        "result":{"small":0.00001,"threshold":0.0001,"large":1e16,"whole":1.0}
    }))]).expect("builtin error");
    assert_eq!(
        parts[0]["errorText"],
        "{'small': 1e-05, 'threshold': 0.0001, 'large': 1e+16, 'whole': 1.0}"
    );
}

#[test]
fn content_output_and_empty_bundle_keep_special_result_metadata() {
    for (result, kind, expected) in [
        (
            json!({"type":"content","value":[{"kind":"text","id":"content-text","text":"image caption","provider_metadata":null}]}),
            "content",
            json!([{"kind":"text","id":"content-text","text":"image caption","provider_metadata":null}]),
        ),
        (
            json!({"type":"messages","messages":[]}),
            "messages",
            json!([]),
        ),
    ] {
        let ui = to_ui_messages(&[
            Message::with_id("a1", MessageRole::Assistant, vec![call()]),
            message(json!({"role":"tool","parts":[{"kind":"tool_result","tool_call_id":"tc1","tool_name":"search","result_kind":"special","result":result}]})),
        ]).expect("special result");
        assert_eq!(ui[0].parts[0]["output"], expected);
        assert_eq!(ui[0].metadata["aiPython"]["toolResultKinds"]["tc1"], kind);
        let (back, _) = to_messages(&ui, None).expect("special result roundtrip");
        assert!(matches!(
            back.iter()
                .flat_map(Message::tool_results)
                .next()
                .expect("result")
                .result,
            ai::types::messages::ToolResultValue::Special(_)
        ));
    }
}

#[test]
fn bundle_wire_output_uses_single_list_or_null_and_python_dump_field_names() {
    assert!(
        bundle_to_wire_output(&MessageBundle::new(vec![]))
            .expect("empty")
            .is_null()
    );
    let message = Message::with_id("a1", MessageRole::Assistant, vec![call()]);
    let single = bundle_to_wire_output(&MessageBundle::new(vec![message.clone()])).expect("single");
    assert_eq!(single["id"], "a1");
    assert_eq!(single["parts"][0]["tool_call_id"], "tc1");
    assert!(single["parts"][0].get("toolCallId").is_none());
    assert!(
        single["parts"][0]
            .get("provider_executed")
            .is_some_and(Value::is_null)
    );
    let multiple = bundle_to_wire_output(&MessageBundle::new(vec![
        message,
        Message::with_id("u1", MessageRole::User, vec![]),
    ]))
    .expect("multiple");
    assert_eq!(multiple.as_array().expect("array").len(), 2);
}

#[test]
fn grouping_respects_turn_boundaries_and_skips_orphan_messages() {
    let messages = vec![
        Message::new(MessageRole::Tool, vec![]),
        Message::new(MessageRole::Internal, vec![]),
        message(json!({"id":"a1","role":"assistant","turn_id":"turn-1","parts":[]})),
        message(json!({"id":"a2","role":"assistant","turn_id":"turn-2","parts":[]})),
        Message::with_id(
            "s1",
            MessageRole::System,
            vec![TextPart::new("rules").into()],
        ),
    ];
    let ui = to_ui_messages(&messages).expect("turn groups");
    assert_eq!(
        ui.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
        vec!["turn-1", "turn-2", "s1"]
    );
    assert_eq!(ui[2].role, "system");

    let empty_turn = message(json!({"id":"a-empty","turn_id":"","role":"assistant","parts":[]}));
    assert_eq!(
        to_ui_messages(&[empty_turn]).expect("empty turn terminates")[0].id,
        "a-empty"
    );
}
