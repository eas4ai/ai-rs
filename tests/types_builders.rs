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

use serde_json::json;
use suprnova_ai::types::{
    Message, MessageRole, Part, TextPart,
    builders::{
        PartLike, ToolMessageError, assistant_message, content_output, file_part, message,
        system_message, text_part, thinking, tool_message, tool_result_message, tool_result_part,
        user_message,
    },
};

#[test]
fn message_builders_coerce_strings_and_keep_parts() {
    let file = file_part("https://example.com/img.png", None, None).expect("known extension");
    let built = user_message(vec![
        PartLike::from("Describe this:"),
        PartLike::from(file),
        PartLike::from("Thanks"),
    ]);
    assert_eq!(built.role, MessageRole::User);
    assert!(matches!(built.parts[0], Part::Text(_)));
    assert!(matches!(built.parts[1], Part::File(_)));
    assert!(matches!(built.parts[2], Part::Text(_)));

    assert_eq!(system_message("hi").role, MessageRole::System);
    assert_eq!(assistant_message("hi").role, MessageRole::Assistant);
    assert_eq!(message(MessageRole::User, "hi").role, MessageRole::User);
}

#[test]
fn text_and_content_helpers_preserve_metadata_and_mixed_content() {
    let text = text_part("hello", Some([("k".to_owned(), json!("v"))].into()));
    assert_eq!(text.provider_metadata.as_ref().expect("metadata")["k"], "v");

    let file = file_part("https://example.com/img.png", None, None).expect("known extension");
    let output = content_output([
        suprnova_ai::types::builders::ContentLike::from("Here:"),
        suprnova_ai::types::builders::ContentLike::from(file),
    ]);
    assert_eq!(output.value.len(), 2);

    let reasoning = thinking("hmm", Some([("signature".to_owned(), json!("sig"))].into()));
    assert_eq!(reasoning.text, "hmm");
    assert_eq!(
        reasoning.provider_metadata.expect("metadata")["signature"],
        "sig"
    );
}

#[test]
fn byte_file_helper_detects_png_and_rejects_unknown_data() {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend([0; 32]);
    let file = file_part(png, None, Some("cat.png".to_owned())).expect("detect PNG");
    assert_eq!(file.media_type, "image/png");
    assert_eq!(file.filename.as_deref(), Some("cat.png"));
    assert!(file_part(&[0, 0, 0], None, None).is_err());
}

#[test]
fn tool_message_merges_results_and_rejects_invalid_messages() {
    let first = Message::new(
        MessageRole::Tool,
        vec![tool_result_part("tc-1", json!(1), "a", false).into()],
    );
    let second = Message::new(
        MessageRole::Tool,
        vec![tool_result_part("tc-2", json!(2), "b", false).into()],
    );
    let merged = tool_message([
        suprnova_ai::types::builders::ToolMessageItem::from(first),
        suprnova_ai::types::builders::ToolMessageItem::from(second),
    ])
    .expect("merge messages");
    let ids: Vec<_> = merged
        .tool_results()
        .into_iter()
        .map(|part| part.tool_call_id.as_str())
        .collect();
    assert_eq!(ids, ["tc-1", "tc-2"]);

    let wrong_role = user_message("hello");
    assert!(matches!(
        tool_message([suprnova_ai::types::builders::ToolMessageItem::from(
            wrong_role
        )]),
        Err(ToolMessageError::WrongRole(MessageRole::User))
    ));
    let invalid = Message::new(MessageRole::Tool, vec![TextPart::new("bad").into()]);
    assert_eq!(
        tool_message([suprnova_ai::types::builders::ToolMessageItem::from(invalid)]),
        Err(ToolMessageError::NonResultPart)
    );
    assert_eq!(
        tool_message(Vec::<suprnova_ai::types::builders::ToolMessageItem>::new()),
        Err(ToolMessageError::Empty)
    );

    let direct = tool_result_message("tc-3", json!({"ok": true}), "c", false);
    assert_eq!(direct.role, MessageRole::Tool);
    assert_eq!(direct.tool_results()[0].tool_call_id, "tc-3");
}

#[test]
fn builders_accept_tool_calls_hooks_events_and_message_lists() {
    use suprnova_ai::types::{HookPart, HookStatus, ToolCallPart, events::ToolCallResult};
    let call = ToolCallPart::new("tc", "tool", "{}");
    assert!(matches!(
        assistant_message(call.clone()).parts[0],
        Part::ToolCall(_)
    ));
    assert!(matches!(
        assistant_message(vec![call]).parts[0],
        Part::ToolCall(_)
    ));
    let hook = HookPart {
        id: "part".into(),
        hook_id: "hook".into(),
        hook_type: "approval".into(),
        status: HookStatus::Pending,
        metadata: Default::default(),
        resolution: None,
        tool_call_id: None,
    };
    assert!(matches!(user_message(hook.clone()).parts[0], Part::Hook(_)));
    assert!(matches!(user_message(vec![hook]).parts[0], Part::Hook(_)));
    let result = tool_result_part("tc", json!(42), "tool", false);
    let message = tool_result_message("tc", json!(42), "tool", false);
    let event = ToolCallResult {
        replay: false,
        message: message.clone(),
        results: vec![result],
        exception: None,
    };
    assert_eq!(tool_message([event]).unwrap().parts, message.parts);
    assert_eq!(
        tool_message([vec![message.clone(), message.clone()]])
            .unwrap()
            .parts
            .len(),
        2
    );
    assert!(
        tool_message([Vec::<Message>::new()])
            .unwrap()
            .parts
            .is_empty()
    );
    assert!(tool_message([vec![user_message("wrong role")]]).is_err());
}
