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

use super::params::invalid;
use crate::{
    Error,
    types::{
        Message, MessageRole, Part,
        media::{self, MediaDataRef},
        messages::*,
    },
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::collections::BTreeSet;

fn media_ref(data: &FileData) -> MediaDataRef<'_> {
    match data {
        FileData::Text(s) => MediaDataRef::Text(s),
        FileData::Bytes(b) => MediaDataRef::Bytes(b),
    }
}
fn file(part: &FilePart, responses: bool) -> Result<Value, Error> {
    let mt = if part.media_type == "image/*" {
        "image/jpeg"
    } else {
        &part.media_type
    };
    let data = media_ref(&part.data);
    if mt.starts_with("image/") {
        let url = media::data_to_data_url(data, mt);
        return Ok(if responses {
            json!({"type":"input_image","image_url":url})
        } else {
            json!({"type":"image_url","image_url":{"url":url}})
        });
    }
    let url = part
        .data
        .as_text()
        .filter(|s| media::is_downloadable_url(s));
    if mt == "application/pdf" {
        if let Some(url) = url {
            if responses {
                return Ok(json!({"type":"input_file","file_url":url}));
            }
            return Err(invalid(
                "chat PDF URLs require a download transport; supply file bytes or configure protocol downloads",
            ));
        }
        let filename = part.filename.as_deref().unwrap_or("document.pdf");
        let data = media::data_to_data_url(data, mt);
        return Ok(if responses {
            json!({"type":"input_file","filename":filename,"file_data":data})
        } else {
            json!({"type":"file","file":{"filename":filename,"file_data":data}})
        });
    }
    if mt.starts_with("audio/") && !responses {
        if url.is_some() {
            return Err(invalid(
                "chat audio URLs require a download transport; supply file bytes or configure protocol downloads",
            ));
        }
        return Ok(
            json!({"type":"input_audio","input_audio":{"format":mt.split_once('/').map_or(mt,|(_,v)|v),"data":media::data_to_base64(data)}}),
        );
    }
    if mt.starts_with("text/") {
        let text = match &part.data {
            FileData::Bytes(b) => String::from_utf8(b.clone())
                .map_err(|_| invalid("OpenAI text input is not UTF-8"))?,
            FileData::Text(s) if media::is_url(s) => s.clone(),
            FileData::Text(s) => String::from_utf8(
                STANDARD
                    .decode(s)
                    .map_err(|_| invalid("OpenAI text input is not base64"))?,
            )
            .map_err(|_| invalid("OpenAI text input is not UTF-8"))?,
        };
        return Ok(json!({"type":if responses {"input_text"}else {"text"},"text":text}));
    }
    Err(invalid(format!("Unsupported media type for OpenAI: {mt}")))
}
fn tool_result(value: &ToolResultValue, responses: bool) -> Result<Value, Error> {
    match value {
        ToolResultValue::Json(Value::Null) => Ok(json!("")),
        ToolResultValue::Json(Value::String(s)) => Ok(json!(s)),
        ToolResultValue::Special(SpecialToolResult::Content(content)) => {
            let mut parts = vec![];
            for p in &content.value {
                parts.push(match p {
                    ContentPart::Text(t) => {
                        json!({"type":if responses {"input_text"}else {"text"},"text":t.text})
                    }
                    ContentPart::File(f) => {
                        let url = media::data_to_data_url(media_ref(&f.data), &f.media_type);
                        if f.media_type.starts_with("image/") {
                            if responses {
                                json!({"type":"input_image","image_url":url})
                            } else {
                                json!({"type":"image_url","image_url":{"url":url}})
                            }
                        } else if responses {
                            let mut value = json!({"type":"input_file","file_data":url});
                            if let Some(name) = &f.filename {
                                value["filename"] = json!(name);
                            }
                            value
                        } else {
                            json!({"type":"text","text":format!("[file: {}]",f.media_type)})
                        }
                    }
                });
            }
            Ok(parts.into())
        }
        _ => Ok(json!(serde_json::to_string(value)?)),
    }
}
fn repair(messages: Vec<Message>) -> Result<Vec<Message>, Error> {
    let mut cleaned = vec![];
    for mut msg in messages {
        if msg.role == MessageRole::Internal {
            continue;
        }
        msg.parts.retain(|p| !matches!(p, Part::Hook(_)));
        if msg.parts.is_empty() {
            continue;
        }
        for p in &mut msg.parts {
            if let Part::ToolCall(c) = p
                && serde_json::from_str::<Value>(&c.tool_args).is_err()
            {
                c.tool_args = "{}".into();
            }
        }
        cleaned.push(msg);
    }
    let mut calls = BTreeSet::new();
    let mut results = BTreeSet::new();
    let mut pending = BTreeSet::new();
    for msg in &cleaned {
        if matches!(msg.role, MessageRole::User | MessageRole::Assistant) {
            pending.clear();
        }
        for p in &msg.parts {
            match p {
                Part::ToolCall(c) if msg.role == MessageRole::Assistant => {
                    if !calls.insert(c.tool_call_id.clone()) {
                        return Err(invalid("duplicate tool-call id in history"));
                    }
                    pending.insert(c.tool_call_id.clone());
                }
                Part::ToolResult(r) if msg.role == MessageRole::Tool => {
                    if !results.insert(r.tool_call_id.clone()) {
                        return Err(invalid("duplicate tool-result id in history"));
                    }
                    if !pending.remove(&r.tool_call_id) {
                        return Err(invalid("orphaned tool result in history"));
                    }
                }
                _ => {}
            }
        }
    }
    let mut output = vec![];
    let mut unanswered: Vec<ToolCallPart> = vec![];
    for msg in cleaned {
        if matches!(msg.role, MessageRole::User | MessageRole::Assistant) {
            flush_pending(&mut output, &mut unanswered);
        }
        if msg.role == MessageRole::Assistant {
            for c in msg.tool_calls() {
                if !results.contains(&c.tool_call_id) {
                    unanswered.push(c.clone());
                }
            }
        }
        output.push(msg);
    }
    flush_pending(&mut output, &mut unanswered);
    Ok(output)
}
fn flush_pending(output: &mut Vec<Message>, pending: &mut Vec<ToolCallPart>) {
    if !pending.is_empty() {
        output.push(Message::new(
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
fn metadata(part: &Part) -> Option<&Value> {
    (match part {
        Part::Text(p) => p.provider_metadata.as_ref(),
        Part::Reasoning(p) => p.provider_metadata.as_ref(),
        Part::ToolCall(p) => p.provider_metadata.as_ref(),
        Part::BuiltinToolCall(p) => p.provider_metadata.as_ref(),
        Part::BuiltinToolReturn(p) => p.provider_metadata.as_ref(),
        Part::File(p) => p.provider_metadata.as_ref(),
        Part::ToolResult(p) => p.provider_metadata.as_ref(),
        _ => None,
    })
    .and_then(|m| m.get("openai"))
}
fn flush_content(out: &mut Vec<Value>, content: &mut Vec<Value>) {
    if !content.is_empty() {
        out.push(json!({"role":"assistant","content":std::mem::take(content)}));
    }
}
pub(super) fn convert(
    messages: Vec<Message>,
    responses: bool,
    references: bool,
) -> Result<Vec<Value>, Error> {
    let mut out = vec![];
    for msg in repair(messages)? {
        match msg.role {
            MessageRole::System => {
                let text = msg.text();
                if !responses || !text.is_empty() {
                    out.push(json!({"role":"system","content":text}));
                }
            }
            MessageRole::User => {
                if !responses && !msg.parts.iter().any(|p| matches!(p, Part::File(_))) {
                    out.push(json!({"role":"user","content":msg.text()}));
                    continue;
                }
                let mut content = vec![];
                for p in msg.parts {
                    match p {
                        Part::Text(t) => content.push(
                            json!({"type":if responses {"input_text"}else {"text"},"text":t.text}),
                        ),
                        Part::File(f) => content.push(file(&f, responses)?),
                        _ => {}
                    }
                }
                out.push(json!({"role":"user","content":content}));
            }
            MessageRole::Assistant if !responses => {
                let mut entry = json!({"role":"assistant"});
                let text = msg.text();
                let reasoning = msg.reasoning();
                if !text.is_empty() {
                    entry["content"] = json!(text);
                }
                if !reasoning.is_empty() {
                    entry["reasoning"] = json!(reasoning);
                }
                let mut calls = vec![];
                for p in &msg.parts {
                    match p {
                    Part::ToolCall(c)=>calls.push(json!({"id":c.tool_call_id,"type":"function","function":{"name":c.tool_name,"arguments":c.tool_args}})),
                    Part::BuiltinToolCall(_)|Part::BuiltinToolReturn(_)=>return Err(Error::UnsupportedOperation("OpenAI chat completions does not support BuiltinTool history".into())),_=>{}
                }
                }
                if !calls.is_empty() {
                    entry["tool_calls"] = calls.into();
                }
                out.push(entry);
            }
            MessageRole::Assistant => {
                let mut content = vec![];
                for p in &msg.parts {
                    let md = metadata(p);
                    let id = md
                        .and_then(|m| m.get("item_id").or_else(|| m.get("itemId")))
                        .and_then(Value::as_str);
                    if references && let Some(id) = id {
                        flush_content(&mut out, &mut content);
                        out.push(json!({"type":"item_reference","id":id}));
                        continue;
                    }
                    if let Some(raw) = md.and_then(|m| m.get("raw_item")).filter(|r| r.is_object())
                    {
                        flush_content(&mut out, &mut content);
                        out.push(raw.clone());
                        continue;
                    }
                    match p {
                        Part::Text(t) => content.push(json!({"type":"output_text","text":t.text})),
                        Part::Reasoning(r) => {
                            flush_content(&mut out, &mut content);
                            if let Some(encrypted) = md.and_then(|m| {
                                m.get("reasoning_encrypted_content")
                                    .or_else(|| m.get("reasoningEncryptedContent"))
                            }) {
                                out.push(json!({"type":"reasoning","summary":[{"type":"summary_text","text":r.text}],"encrypted_content":encrypted}));
                            }
                        }
                        Part::ToolCall(c) => {
                            flush_content(&mut out, &mut content);
                            out.push(json!({"type":"function_call","call_id":c.tool_call_id,"name":c.tool_name,"arguments":c.tool_args}));
                        }
                        Part::BuiltinToolCall(_) | Part::BuiltinToolReturn(_) => {
                            flush_content(&mut out, &mut content)
                        }
                        _ => {}
                    }
                }
                flush_content(&mut out, &mut content);
            }
            MessageRole::Tool => {
                for p in &msg.parts {
                    if let Part::ToolResult(r) = p {
                        let value = tool_result(r.get_model_input(), responses)?;
                        out.push(if responses {json!({"type":"function_call_output","call_id":r.tool_call_id,"output":value})}else {json!({"role":"tool","tool_call_id":r.tool_call_id,"content":value})});
                    }
                }
            }
            MessageRole::Internal => {}
        }
    }
    Ok(out)
}

pub(super) async fn download_files(
    mut messages: Vec<Message>,
    transport: Option<&dyn crate::models::files::DownloadTransport>,
) -> Result<Vec<Message>, Error> {
    let Some(transport) = transport else {
        return Ok(messages);
    };
    for msg in &mut messages {
        if msg.role != MessageRole::User {
            continue;
        }
        for part in &mut msg.parts {
            let Part::File(file) = part else {
                continue;
            };
            if !(file.media_type.starts_with("audio/") || file.media_type == "application/pdf") {
                continue;
            }
            let Some(url) = file
                .data
                .as_text()
                .filter(|s| media::is_downloadable_url(s))
            else {
                continue;
            };
            let (bytes, _) = crate::models::files::download(
                transport,
                url,
                crate::models::files::DEFAULT_MAX_BYTES,
            )
            .await
            .map_err(|e| {
                crate::errors::ProviderError::new(
                    crate::errors::ProviderErrorKind::Response,
                    "OpenAI input download failed",
                )
                .with_source(e)
            })?;
            file.data = FileData::Bytes(bytes.to_vec());
        }
    }
    Ok(messages)
}
