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

//! Adapter metadata preserves IDs while UI bubbles collapse internal messages.

use crate::types::messages::{Message, MessageRole, Part, SpecialToolResult, ToolResultValue};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const ADAPTER_METADATA_KEY: &str = "aiPython";
pub const SOURCE_MESSAGES_KEY: &str = "sourceMessages";
pub const TOOL_RESULT_KINDS_KEY: &str = "toolResultKinds";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceMessage {
    pub id: String,
    pub role: MessageRole,
    pub turn_id: Option<String>,
    pub part_ids: Vec<String>,
}

#[must_use]
pub fn metadata_for(messages: &[Message]) -> Value {
    let sources:Vec<_>=messages.iter().map(|message|json!({"id":message.id,"role":message.role,"turnId":message.turn_id,"partIds":message.parts.iter().map(Part::id).collect::<Vec<_>>()})).collect();
    let mut adapter = json!({SOURCE_MESSAGES_KEY:sources});
    let mut kinds = BTreeMap::new();
    for message in messages {
        for part in &message.parts {
            if let Part::ToolResult(part) = part
                && let ToolResultValue::Special(result) = &part.result
            {
                kinds.insert(
                    part.tool_call_id.clone(),
                    match result {
                        SpecialToolResult::Content(_) => "content",
                        SpecialToolResult::Messages(_) => "messages",
                    },
                );
            }
        }
    }
    if !kinds.is_empty() {
        adapter[TOOL_RESULT_KINDS_KEY] = json!(kinds);
    }
    json!({ADAPTER_METADATA_KEY:adapter})
}

#[must_use]
pub fn source_messages_from(metadata: &Value) -> Vec<SourceMessage> {
    let Some(raw) = metadata
        .get(ADAPTER_METADATA_KEY)
        .and_then(|v| v.get(SOURCE_MESSAGES_KEY))
        .and_then(Value::as_array)
    else {
        return vec![];
    };
    raw.iter()
        .filter_map(|source| {
            let id = source.get("id")?.as_str()?.to_owned();
            let role = match source.get("role")?.as_str()? {
                "user" => MessageRole::User,
                "assistant" => MessageRole::Assistant,
                "system" => MessageRole::System,
                "tool" => MessageRole::Tool,
                "internal" => MessageRole::Internal,
                _ => return None,
            };
            let turn_id = source
                .get("turnId")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let part_ids = source
                .get("partIds")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
            Some(SourceMessage {
                id,
                role,
                turn_id,
                part_ids,
            })
        })
        .collect()
}

#[must_use]
pub fn tool_result_kinds_from(metadata: &Value) -> BTreeMap<String, String> {
    metadata
        .get(ADAPTER_METADATA_KEY)
        .and_then(|v| v.get(TOOL_RESULT_KINDS_KEY))
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .filter_map(|(id, kind)| kind.as_str().map(|kind| (id.clone(), kind.to_owned())))
        .collect()
}

#[must_use]
pub fn restore_source_ids(messages: Vec<Message>, sources: &[SourceMessage]) -> Vec<Message> {
    let mut source_index = 0;
    messages
        .into_iter()
        .map(|mut message| {
            if let Some(offset) = sources[source_index..]
                .iter()
                .position(|source| source.role == message.role)
            {
                let source = &sources[source_index + offset];
                source_index += offset + 1;
                message.id.clone_from(&source.id);
                message.turn_id.clone_from(&source.turn_id);
                if source.part_ids.len() == message.parts.len() {
                    for (part, id) in message.parts.iter_mut().zip(&source.part_ids) {
                        match part {
                            Part::Text(p) => p.id.clone_from(id),
                            Part::Reasoning(p) => p.id.clone_from(id),
                            Part::File(p) => p.id.clone_from(id),
                            Part::ToolCall(p) => p.id.clone_from(id),
                            Part::ToolResult(p) => p.id.clone_from(id),
                            Part::BuiltinToolCall(p) => p.id.clone_from(id),
                            Part::BuiltinToolReturn(p) => p.id.clone_from(id),
                            Part::Hook(p) => p.id.clone_from(id),
                        }
                    }
                }
            }
            message
        })
        .collect()
}
