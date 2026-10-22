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

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use suprnova_ai::types::{
    ContentOutput, FilePart, Message, MessageRole, Part, TextPart, ToolCallPart, ToolResultPart,
    Usage,
    messages::{
        BuiltinToolCallPart, BuiltinToolReturnPart, ContentPart, FileData, MessageBundle,
        ModelInputKind, ResultKind, ToolResultValue, with_id_generator,
    },
};

#[test]
fn usage_add_merges_optional_fields_and_drops_raw() {
    let total = Usage {
        input_tokens: 100,
        output_tokens: 50,
        cache_read_tokens: Some(20),
        ..Usage::default()
    } + Usage {
        input_tokens: 200,
        output_tokens: 80,
        reasoning_tokens: Some(10),
        raw: Some(json!({"provider": true})),
        ..Usage::default()
    };
    assert_eq!(total.input_tokens, 300);
    assert_eq!(total.output_tokens, 130);
    assert_eq!(total.total_tokens(), 430);
    assert_eq!(total.reasoning_tokens, Some(10));
    assert_eq!(total.cache_read_tokens, Some(20));
    assert_eq!(total.cache_write_tokens, None);
    assert_eq!(total.raw, None);
}

#[test]
fn part_union_round_trips_with_canonical_kind_tag() {
    let message = Message {
        provider_metadata: Some([("provider".to_owned(), json!({"nested": 1}))].into()),
        ..Message::with_id(
            "m1",
            MessageRole::User,
            vec![
                TextPart {
                    provider_metadata: Some([("block".to_owned(), json!("text"))].into()),
                    ..TextPart::new("look at this")
                }
                .into(),
                FilePart::from_url("https://example.com/cat.jpg", None)
                    .expect("known extension")
                    .into(),
            ],
        )
    };
    let value = serde_json::to_value(&message).expect("serialize message");
    assert_eq!(value["parts"][0]["kind"], "text");
    assert_eq!(value["parts"][1]["kind"], "file");
    assert!(value["parts"][0].get("type").is_none());
    assert_eq!(
        serde_json::from_value::<Message>(value).expect("deserialize"),
        message
    );
}

#[test]
fn file_constructors_infer_override_and_reject_like_python() {
    assert_eq!(
        FilePart::from_url("data:audio/wav;base64,AAAA", None)
            .expect("data URL")
            .media_type,
        "audio/wav"
    );
    assert_eq!(
        FilePart::from_url("https://example.com/blob", Some("image/webp"))
            .expect("explicit type")
            .media_type,
        "image/webp"
    );
    assert!(FilePart::from_url("https://example.com/blob", None).is_err());

    assert_eq!(
        FilePart::from_bytes(b"RIFF\0\0\0\0WAVE".to_vec(), None, None)
            .expect("detect WAV")
            .media_type,
        "audio/wav"
    );
    assert_eq!(
        FilePart::from_bytes(b"\0\0\0\x18ftypisom".to_vec(), None, None)
            .expect("detect MP4")
            .media_type,
        "video/mp4"
    );
    assert!(FilePart::from_bytes(vec![0, 1, 2, 3], None, None).is_err());
}

#[test]
fn byte_file_serializes_as_standard_base64_and_restores_as_text() {
    let raw = b"\xff\xd8\xff\xe0\0\x10JFIF".repeat(3);
    let file = FilePart::from_bytes(raw.clone(), Some("image/jpeg"), None).expect("explicit type");
    let value = serde_json::to_value(&file).expect("serialize file");
    let encoded = value["data"].as_str().expect("string data").to_owned();
    assert!(!encoded.contains(['-', '_']));
    assert_eq!(STANDARD.decode(&encoded).expect("valid base64"), raw);

    let restored: FilePart = serde_json::from_value(value).expect("deserialize file");
    assert_eq!(restored.data, FileData::Text(encoded));
}

#[test]
fn special_tool_results_and_model_inputs_round_trip_independently() {
    let content = ContentOutput::new(vec![ContentPart::File(
        FilePart::from_bytes(b"image".to_vec(), Some("image/png"), None).expect("explicit type"),
    )]);
    assert_eq!(
        serde_json::to_value(&content).expect("serialize content output")["type"],
        "content"
    );
    let mut part = ToolResultPart::new("tc1", "draw", json!({"image_id": "img1"}));
    part.set_model_input(content.clone());

    let value = serde_json::to_value(&part).expect("serialize result");
    assert_eq!(value["result_kind"], "json");
    assert_eq!(value["model_input_kind"], "special");
    let restored: ToolResultPart = serde_json::from_value(value).expect("deserialize result");
    assert!(matches!(restored.result, ToolResultValue::Json(_)));
    assert!(matches!(
        restored.get_model_input(),
        ToolResultValue::Special(_)
    ));

    let bundle = MessageBundle::new(vec![Message::new(MessageRole::Assistant, vec![])]);
    let bundled = ToolResultPart::new("tc2", "delegate", bundle);
    let bundled: ToolResultPart =
        serde_json::from_value(serde_json::to_value(bundled).expect("serialize message bundle"))
            .expect("deserialize message bundle");
    assert!(matches!(bundled.result, ToolResultValue::Special(_)));

    let mut special_result = ToolResultPart::new("tc3", "draw", content.clone());
    special_result.set_model_input(json!({"type": "content", "value": "plain data"}));
    let restored: ToolResultPart = serde_json::from_value(
        serde_json::to_value(special_result).expect("serialize independent kinds"),
    )
    .expect("deserialize independent kinds");
    assert!(matches!(restored.result, ToolResultValue::Special(_)));
    assert_eq!(restored.model_input_kind, ModelInputKind::Json);
    assert!(matches!(
        restored.get_model_input(),
        ToolResultValue::Json(_)
    ));
}

#[test]
fn model_input_unset_and_explicit_null_have_distinct_wire_forms() {
    let mut part = ToolResultPart::new("tc", "tool", json!({"ok": true}));
    let unset = serde_json::to_value(&part).expect("serialize unset");
    assert!(unset.get("model_input").is_none());
    assert!(unset.get("model_input_kind").is_none());
    assert_eq!(part.get_model_input().as_json(), Some(&json!({"ok": true})));

    part.set_model_input(Value::Null);
    let explicit_null = serde_json::to_value(&part).expect("serialize null");
    assert_eq!(explicit_null.get("model_input"), Some(&Value::Null));
    assert!(explicit_null.get("model_input_kind").is_none());
    let restored: ToolResultPart = serde_json::from_value(explicit_null).expect("deserialize null");
    assert!(restored.has_model_input());
    assert_eq!(restored.model_input_kind, ModelInputKind::Json);
    assert_eq!(restored.get_model_input().as_json(), Some(&Value::Null));
    assert_ne!(restored.get_model_input(), &restored.result);
}

#[test]
fn gc_regression_wire_omissions_are_preserved_by_rust_ownership() {
    let mut call = ToolCallPart::new("tc", "tool", "{}");
    let call_value = serde_json::to_value(&call).expect("serialize call");
    assert!(call_value.get("cached_result").is_none());
    call.cached_result = Some(ToolResultPart::new("tc", "tool", json!(1)));
    let cached = serde_json::to_value(Part::ToolCall(call)).expect("serialize cached result");
    assert_eq!(cached["cached_result"]["kind"], "tool_result");

    let message = Message::new(MessageRole::User, vec![]);
    let message_value = serde_json::to_value(&message).expect("serialize message");
    assert!(message_value.get("replay").is_none());
    let replay_value = serde_json::to_value(Message {
        replay: true,
        ..message
    })
    .expect("serialize replay");
    assert_eq!(replay_value["replay"], true);
}

#[test]
fn output_requires_a_final_assistant_message() {
    let message = Message::new(MessageRole::Assistant, vec![TextPart::new("hello").into()]);
    assert_eq!(message.get_output().expect("final output"), "hello");

    let pending = Message::new(
        MessageRole::Assistant,
        vec![ToolCallPart::new("tc", "tool", "{}").into()],
    );
    assert!(pending.get_output().is_err());
}

#[test]
fn builtin_tool_accessors_filter_each_part_kind() {
    let call = BuiltinToolCallPart {
        id: "call-part".to_owned(),
        tool_call_id: "call".to_owned(),
        tool_name: "web_search".to_owned(),
        tool_args: "{}".to_owned(),
        provider_metadata: None,
    };
    let returned = BuiltinToolReturnPart {
        id: "return-part".to_owned(),
        tool_call_id: "call".to_owned(),
        tool_name: "web_search".to_owned(),
        result: json!({"hits": 1}),
        is_error: false,
        provider_metadata: None,
    };
    let message = Message::new(
        MessageRole::Assistant,
        vec![
            Part::BuiltinToolCall(call),
            TextPart::new("done").into(),
            Part::BuiltinToolReturn(returned),
        ],
    );
    assert_eq!(message.builtin_tool_calls()[0].tool_call_id, "call");
    assert_eq!(message.builtin_tool_returns()[0].tool_call_id, "call");
}

#[tokio::test]
async fn scoped_id_generator_is_deterministic_across_awaits() {
    async fn sequence() -> (String, String) {
        let message = Message::new(MessageRole::User, vec![]).id;
        tokio::task::yield_now().await;
        let part = TextPart::new("hi").id;
        (message, part)
    }

    let first = with_id_generator(
        {
            let mut n = 0;
            move || {
                n += 1;
                n
            }
        },
        sequence(),
    )
    .await;
    let second = with_id_generator(
        {
            let mut n = 0;
            move || {
                n += 1;
                n
            }
        },
        sequence(),
    )
    .await;
    assert_eq!(first, second);
    assert!(first.0.starts_with("msg_"));
    assert!(first.1.starts_with("part_"));
}

#[tokio::test]
async fn spawned_tasks_opt_in_to_their_own_id_scope() {
    let (outer, spawned) = with_id_generator(|| 7, async {
        let outer = Message::new(MessageRole::User, vec![]).id;
        let spawned = tokio::spawn(with_id_generator(|| 7, async {
            Message::new(MessageRole::User, vec![]).id
        }))
        .await
        .expect("spawned ID task");
        (outer, spawned)
    })
    .await;
    assert_eq!(outer, spawned);
    assert_eq!(outer, "msg_000000000007");
}

#[test]
fn error_result_kind_is_preserved() {
    let part = ToolResultPart::error("tc", "tool", json!("failed"));
    assert_eq!(part.result_kind, ResultKind::Error);
    let restored: ToolResultPart =
        serde_json::from_str(&serde_json::to_string(&part).expect("serialize error result"))
            .expect("deserialize error result");
    assert!(restored.is_error());

    for value in [
        json!("hello"),
        Value::Null,
        json!([1, 2, 3]),
        json!({"key": "value"}),
    ] {
        let part = ToolResultPart::new("tc", "tool", value.clone());
        let restored: ToolResultPart =
            serde_json::from_value(serde_json::to_value(part).expect("serialize plain result"))
                .expect("deserialize plain result");
        assert_eq!(restored.result.as_json(), Some(&value));
    }
}
