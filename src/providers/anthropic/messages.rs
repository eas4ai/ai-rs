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

use crate::{
    Error,
    types::{media, messages::*},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::collections::HashSet;

pub(super) fn metadata(values: Value) -> Metadata {
    [("anthropic".into(), values)].into()
}
fn anthropic_metadata(metadata: &Option<Metadata>, key: &str) -> Option<Value> {
    metadata
        .as_ref()?
        .get("anthropic")?
        .as_object()?
        .get(key)
        .cloned()
}

pub fn file_to_anthropic(part: &FilePart) -> Result<Value, Error> {
    let mt = part.media_type.as_str();
    let block_type = if mt.starts_with("image/") {
        "image"
    } else if matches!(mt, "application/pdf" | "text/plain") {
        "document"
    } else {
        return Err(Error::Validation(format!(
            "Unsupported media type for Anthropic: {mt}"
        )));
    };
    if let FileData::Text(data) = &part.data
        && media::is_url(data)
    {
        return Ok(json!({"type":block_type,"source":{"type":"url","url":data}}));
    }
    let source = if mt == "text/plain" {
        let bytes = match &part.data {
            FileData::Bytes(data) => data.clone(),
            FileData::Text(data) => STANDARD.decode(data).map_err(|_| {
                Error::Validation("Anthropic text document must contain base64 UTF-8 data".into())
            })?,
        };
        let text = String::from_utf8(bytes).map_err(|_| {
            Error::Validation("Anthropic text document must contain UTF-8 data".into())
        })?;
        json!({"type":"text","media_type":mt,"data":text})
    } else {
        let data = match &part.data {
            FileData::Text(data) => media::data_to_base64(data),
            FileData::Bytes(data) => media::data_to_base64(data.as_slice()),
        };
        json!({"type":"base64","media_type":if mt=="image/*" {"image/jpeg"}else{mt},"data":data})
    };
    Ok(json!({"type":block_type,"source":source}))
}
pub fn tool_result_to_anthropic(value: &ToolResultValue) -> Result<Value, Error> {
    match value {
        ToolResultValue::Special(SpecialToolResult::Content(content)) => content
            .value
            .iter()
            .map(|item| match item {
                ContentPart::Text(p) => Ok(json!({"type":"text","text":p.text})),
                ContentPart::File(p) => file_to_anthropic(p),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        ToolResultValue::Json(Value::Null) => Ok(json!("")),
        ToolResultValue::Json(Value::String(s)) => Ok(json!(s)),
        other => Ok(json!(serde_json::to_string(other)?)),
    }
}

fn check_tool_ids(messages: &[Message]) -> Result<(), Error> {
    let mut calls = HashSet::new();
    let mut results = HashSet::new();
    let mut pending = HashSet::new();
    for msg in messages {
        if matches!(msg.role, MessageRole::User | MessageRole::Assistant) {
            pending.clear();
        }
        for part in &msg.parts {
            match (msg.role, part) {
                (MessageRole::Assistant, Part::ToolCall(call)) => {
                    if !calls.insert(call.tool_call_id.clone()) {
                        return Err(Error::Validation(
                            "message history has duplicate-tool-call".into(),
                        ));
                    }
                    pending.insert(call.tool_call_id.clone());
                }
                (MessageRole::Tool, Part::ToolResult(result)) => {
                    if !results.insert(result.tool_call_id.clone()) {
                        return Err(Error::Validation(
                            "message history has duplicate-tool-result".into(),
                        ));
                    }
                    if !pending.remove(&result.tool_call_id) {
                        return Err(Error::Validation(
                            "message history has orphaned-tool-result".into(),
                        ));
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}
fn flush_pending(pending: &mut Vec<ToolCallPart>, result: &mut Vec<Message>) {
    if !pending.is_empty() {
        result.push(Message::new(
            MessageRole::Tool,
            pending
                .drain(..)
                .map(|c| {
                    Part::ToolResult(ToolResultPart::error(
                        c.tool_call_id,
                        c.tool_name,
                        json!("Tool result not available"),
                    ))
                })
                .collect(),
        ));
    }
}
fn repair(messages: &[Message]) -> Result<Vec<Message>, Error> {
    let messages: Vec<_> = messages
        .iter()
        .filter(|m| m.role != MessageRole::Internal)
        .cloned()
        .filter_map(|mut m| {
            m.parts.retain(|p| !matches!(p, Part::Hook(_)));
            for p in &mut m.parts {
                if let Part::ToolCall(c) = p
                    && serde_json::from_str::<Value>(&c.tool_args).is_err()
                {
                    c.tool_args = "{}".into();
                }
            }
            (!m.parts.is_empty()).then_some(m)
        })
        .collect();
    check_tool_ids(&messages)?;
    let answered: HashSet<_> = messages
        .iter()
        .filter(|m| m.role == MessageRole::Tool)
        .flat_map(|m| &m.parts)
        .filter_map(|p| match p {
            Part::ToolResult(r) => Some(r.tool_call_id.clone()),
            _ => None,
        })
        .collect();
    let mut pending = Vec::new();
    let mut result = Vec::new();
    for msg in messages {
        if matches!(msg.role, MessageRole::User | MessageRole::Assistant) {
            flush_pending(&mut pending, &mut result);
        }
        if msg.role == MessageRole::Assistant {
            for part in &msg.parts {
                if let Part::ToolCall(c) = part
                    && !answered.contains(&c.tool_call_id)
                {
                    pending.push(c.clone());
                }
            }
        }
        result.push(msg);
    }
    flush_pending(&mut pending, &mut result);
    Ok(result)
}
fn assistant_content(parts: &[Part]) -> Result<Vec<Value>, Error> {
    let mut content = Vec::new();
    for part in parts {
        match part {
            Part::Reasoning(p)=>if let Some(signature)=anthropic_metadata(&p.provider_metadata,"signature") && signature.as_str().is_some_and(|s|!s.is_empty()) {content.push(json!({"type":"thinking","thinking":p.text,"signature":signature}));},
            Part::Text(p)=>content.push(json!({"type":"text","text":p.text})),
            Part::ToolCall(p)=>content.push(json!({"type":"tool_use","id":p.tool_call_id,"name":p.tool_name,"input":serde_json::from_str::<Value>(&p.tool_args)?})),
            Part::BuiltinToolCall(p)=>content.push(json!({"type":"server_tool_use","id":p.tool_call_id,"name":p.tool_name,"input":if p.tool_args.is_empty() {json!({})}else{serde_json::from_str::<Value>(&p.tool_args)?}})),
            Part::BuiltinToolReturn(p)=>content.push(json!({"type":anthropic_metadata(&p.provider_metadata,"resultType").filter(|v|v.as_str().is_some_and(|s|!s.is_empty())).unwrap_or_else(||json!(format!("{}_tool_result",p.tool_name))),"tool_use_id":p.tool_call_id,"content":p.result})),
            _=>{}
        }
    }
    Ok(content)
}
fn content_list(value: Value) -> Vec<Value> {
    match value {
        Value::Array(v) => v,
        other => vec![json!({"type":"text","text":other})],
    }
}
/// Convert messages after dropping internal parts and repairing orphaned calls.
pub fn messages_to_anthropic(messages: &[Message]) -> Result<(Option<String>, Vec<Value>), Error> {
    let mut systems = Vec::new();
    let mut wire: Vec<Value> = Vec::new();
    for msg in repair(messages)? {
        let content = match msg.role {
            MessageRole::System => {
                systems.push(msg.text());
                continue;
            }
            MessageRole::Assistant => {
                let c = assistant_content(&msg.parts)?;
                if c.is_empty() {
                    continue;
                }
                Value::Array(c)
            }
            MessageRole::Tool => {
                let mut c = Vec::new();
                for p in &msg.parts {
                    if let Part::ToolResult(r) = p {
                        let mut v = json!({"type":"tool_result","tool_use_id":r.tool_call_id,"content":tool_result_to_anthropic(r.get_model_input())?});
                        if r.is_error() {
                            v["is_error"] = json!(true);
                        }
                        c.push(v);
                    }
                }
                if c.is_empty() {
                    continue;
                }
                Value::Array(c)
            }
            MessageRole::User => {
                if msg.parts.iter().any(|p| matches!(p, Part::File(_))) {
                    Value::Array(
                        msg.parts
                            .iter()
                            .filter_map(|p| match p {
                                Part::Text(t) => Some(Ok(json!({"type":"text","text":t.text}))),
                                Part::File(f) => Some(file_to_anthropic(f)),
                                _ => None,
                            })
                            .collect::<Result<_, Error>>()?,
                    )
                } else {
                    json!(msg.text())
                }
            }
            MessageRole::Internal => continue,
        };
        let role = if msg.role == MessageRole::Assistant {
            "assistant"
        } else {
            "user"
        };
        if let Some(prev) = wire.last_mut()
            && prev["role"] == role
        {
            let mut merged = content_list(prev["content"].take());
            merged.extend(content_list(content));
            prev["content"] = json!(merged);
        } else {
            wire.push(json!({"role":role,"content":content}));
        }
    }
    let system = systems.join("\n\n");
    Ok(((!system.is_empty()).then_some(system), wire))
}
