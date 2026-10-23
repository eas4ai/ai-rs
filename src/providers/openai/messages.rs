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

use super::{
    dialect,
    params::invalid,
    profile::{Dialect, ResponsesDialect},
    protocol::Wire,
};
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
fn dialect_of(wire: Wire) -> Dialect {
    match wire {
        Wire::Chat(profile) => profile.dialect,
        Wire::Responses(_) => Dialect::OpenAI,
    }
}
fn file(part: &FilePart, wire: Wire) -> Result<Value, Error> {
    let responses = matches!(wire, Wire::Responses(_));
    let mt = if part.media_type == "image/*" {
        "image/jpeg"
    } else {
        &part.media_type
    };
    match wire.responses_dialect() {
        Some(ResponsesDialect::Xai) => return xai_file(part, mt),
        Some(ResponsesDialect::HuggingFace) => return huggingface_file(part, mt),
        _ => {}
    }
    match dialect_of(wire) {
        Dialect::OpenAI | Dialect::VertexXai => {}
        Dialect::DeepSeek => return deepseek_file(part, mt),
        Dialect::Moonshot => return moonshot_file(part, mt),
        Dialect::Mistral => return mistral_file(part, mt),
        // The TypeScript provider downloads every file URL for MaaS.
        Dialect::VertexMaas if fetch_url(part).is_some() => {
            return Err(invalid(
                "Google Vertex AI MaaS does not fetch file URLs; supply the file bytes or build the provider with a download transport",
            ));
        }
        Dialect::VertexMaas => {}
    }
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
        let text = text_file(part)?;
        return Ok(json!({"type":if responses {"input_text"}else {"text"},"text":text}));
    }
    Err(invalid(format!("Unsupported media type for OpenAI: {mt}")))
}
fn text_file(part: &FilePart) -> Result<String, Error> {
    Ok(match &part.data {
        FileData::Bytes(b) => {
            String::from_utf8(b.clone()).map_err(|_| invalid("OpenAI text input is not UTF-8"))?
        }
        FileData::Text(s) if media::is_url(s) => s.clone(),
        FileData::Text(s) => String::from_utf8(
            STANDARD
                .decode(s)
                .map_err(|_| invalid("OpenAI text input is not base64"))?,
        )
        .map_err(|_| invalid("OpenAI text input is not UTF-8"))?,
    })
}
/// The media URL of a file, which a host that cannot fetch URLs cannot take.
fn fetch_url(part: &FilePart) -> Option<&str> {
    part.data
        .as_text()
        .filter(|s| media::is_downloadable_url(s))
}
/// DeepSeek takes GIF, JPEG, PNG and WebP images only
/// (`convert-to-deepseek-chat-messages.ts`).
fn deepseek_file(part: &FilePart, mt: &str) -> Result<Value, Error> {
    if ![
        "image/gif",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    ]
    .contains(&mt)
    {
        return Err(invalid(format!(
            "DeepSeek takes only GIF, JPEG, PNG and WebP images, not {mt}"
        )));
    }
    let mt = if mt == "image/jpg" { "image/jpeg" } else { mt };
    let url = media::data_to_data_url(media_ref(&part.data), mt);
    Ok(json!({"type":"image_url","image_url":{"url":url}}))
}
/// Moonshot AI takes images, videos and text, inline: it does not fetch
/// URLs, so a provider downloads them first when it has a download transport
/// (`convert-to-moonshotai-chat-messages.ts`).
fn moonshot_file(part: &FilePart, mt: &str) -> Result<Value, Error> {
    const IMAGES: [&str; 7] = [
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/bmp",
        "image/heic",
        "image/heif",
    ];
    const VIDEOS: [&str; 9] = [
        "video/mp4",
        "video/mpeg",
        "video/mov",
        "video/avi",
        "video/x-flv",
        "video/mpg",
        "video/webm",
        "video/wmv",
        "video/3gpp",
    ];
    if fetch_url(part).is_some() {
        return Err(invalid(
            "Moonshot AI does not fetch media URLs; supply the file bytes or build the provider with a download transport",
        ));
    }
    if mt.starts_with("text/") {
        return Ok(json!({"type":"text","text":text_file(part)?}));
    }
    let kind = if IMAGES.contains(&mt) {
        "image_url"
    } else if VIDEOS.contains(&mt) {
        "video_url"
    } else {
        return Err(invalid(format!("Moonshot AI does not take {mt} files")));
    };
    let url = media::data_to_data_url(media_ref(&part.data), mt);
    Ok(json!({"type":kind,kind:{"url":url}}))
}
/// xAI takes images, and other files only as URLs
/// (`convert-to-xai-responses-input.ts`).
fn xai_file(part: &FilePart, mt: &str) -> Result<Value, Error> {
    if mt.starts_with("image/") {
        let url = media::data_to_data_url(media_ref(&part.data), mt);
        return Ok(json!({"type":"input_image","image_url":url}));
    }
    match fetch_url(part) {
        Some(url) => Ok(json!({"type":"input_file","file_url":url})),
        None => Err(invalid(format!(
            "xAI takes {mt} files only as URLs; supply the file as an http or https URL"
        ))),
    }
}
/// Hugging Face takes images only
/// (`convert-to-huggingface-responses-messages.ts`).
fn huggingface_file(part: &FilePart, mt: &str) -> Result<Value, Error> {
    if !mt.starts_with("image/") {
        return Err(invalid(format!("Hugging Face does not take {mt} files")));
    }
    let url = media::data_to_data_url(media_ref(&part.data), mt);
    Ok(json!({"type":"input_image","image_url":url}))
}
/// Mistral takes images and PDFs (`convert-to-mistral-chat-messages.ts`).
fn mistral_file(part: &FilePart, mt: &str) -> Result<Value, Error> {
    let url = media::data_to_data_url(media_ref(&part.data), mt);
    if mt.starts_with("image/") {
        return Ok(json!({"type":"image_url","image_url":url}));
    }
    if mt == "application/pdf" {
        if fetch_url(part).is_some_and(|url| !url.starts_with("https://")) {
            return Err(invalid(
                "Mistral fetches PDF URLs over https only; supply the file bytes",
            ));
        }
        return Ok(json!({"type":"document_url","document_url":url}));
    }
    Err(invalid(format!("Mistral does not take {mt} files")))
}
fn tool_result(value: &ToolResultValue, wire: Wire) -> Result<Value, Error> {
    let responses = matches!(wire, Wire::Responses(_));
    // Only OpenAI's Responses API takes files other than images in a result.
    let result_files = wire.responses_dialect() == Some(ResponsesDialect::OpenAI);
    match value {
        ToolResultValue::Json(Value::Null) => Ok(json!("")),
        ToolResultValue::Json(Value::String(s)) => Ok(json!(s)),
        // Moonshot AI and Mistral take only text; DeepSeek takes parts only
        // with an image.
        ToolResultValue::Special(SpecialToolResult::Content(content))
            if match dialect_of(wire) {
                Dialect::OpenAI | Dialect::VertexMaas | Dialect::VertexXai => false,
                Dialect::DeepSeek => !content.value.iter().any(
                    |p| matches!(p, ContentPart::File(f) if f.media_type.starts_with("image/")),
                ),
                Dialect::Moonshot | Dialect::Mistral => true,
            } =>
        {
            Ok(json!(serde_json::to_string(&content.value)?))
        }
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
                        } else if result_files {
                            let mut value = json!({"type":"input_file","file_data":url});
                            if let Some(name) = &f.filename {
                                value["filename"] = json!(name);
                            }
                            value
                        } else {
                            json!({"type":if responses {"input_text"} else {"text"},"text":format!("[file: {}]",f.media_type)})
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
/// A part's metadata under the host's key, such as `openai` or `xai`.
fn metadata<'a>(part: &'a Part, key: &str) -> Option<&'a Value> {
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
    .and_then(|m| m.get(key))
}
/// A part's item id and encrypted reasoning, from its metadata.
fn item_id(md: Option<&Value>) -> Option<&str> {
    md.and_then(|m| m.get("item_id").or_else(|| m.get("itemId")))
        .and_then(Value::as_str)
}
fn encrypted_reasoning(md: Option<&Value>) -> Option<&Value> {
    md.and_then(|m| {
        m.get("reasoning_encrypted_content")
            .or_else(|| m.get("reasoningEncryptedContent"))
    })
    .filter(|value| !value.is_null())
}
/// An xAI assistant turn: an item for each text part, tool call and
/// reasoning part that has an item id or encrypted content, with the xAI
/// item ids and no references to stored items
/// (`convert-to-xai-responses-input.ts`).
fn xai_assistant(msg: &Message, key: &str, out: &mut Vec<Value>) {
    for part in &msg.parts {
        let md = metadata(part, key);
        let id = item_id(md);
        match part {
            Part::Text(t) => {
                let mut item = json!({"role":"assistant","content":t.text});
                if let Some(id) = id {
                    item["id"] = json!(id);
                }
                out.push(item);
            }
            Part::ToolCall(c) => out.push(json!({
                "type":"function_call",
                "id":id.unwrap_or(&c.tool_call_id),
                "call_id":c.tool_call_id,
                "name":c.tool_name,
                "arguments":c.tool_args,
                "status":"completed",
            })),
            Part::Reasoning(r) => {
                let encrypted = encrypted_reasoning(md);
                if id.is_none() && encrypted.is_none() {
                    continue;
                }
                let summary = if r.text.is_empty() {
                    json!([])
                } else {
                    json!([{"type":"summary_text","text":r.text}])
                };
                let mut item = json!({"type":"reasoning","id":id.unwrap_or(""),"summary":summary,"status":"completed"});
                if let Some(encrypted) = encrypted {
                    item["encrypted_content"] = encrypted.clone();
                }
                out.push(item);
            }
            _ => {}
        }
    }
}
/// A Hugging Face assistant turn: each text and reasoning part as an
/// assistant message of output text, and tool calls as `function_call`
/// items (`convert-to-huggingface-responses-messages.ts`, which drops the
/// tool calls; HST-024 keeps them).
fn huggingface_assistant(msg: &Message, out: &mut Vec<Value>) {
    for part in &msg.parts {
        match part {
            Part::Text(TextPart { text, .. }) | Part::Reasoning(ReasoningPart { text, .. }) => {
                out.push(json!({"role":"assistant","content":[{"type":"output_text","text":text}]}));
            }
            Part::ToolCall(c) => out.push(json!({"type":"function_call","call_id":c.tool_call_id,"name":c.tool_name,"arguments":c.tool_args})),
            _ => {}
        }
    }
}
/// A tool call the application runs, as OpenAI's Responses API takes it in
/// history: its `call_id`, name and arguments, with the namespace its item
/// recorded (`convert-to-openai-responses-input.ts`).
fn function_call(call: &ToolCallPart, md: Option<&Value>) -> Value {
    let mut item = json!({"type":"function_call","call_id":call.tool_call_id,"name":call.tool_name,"arguments":call.tool_args});
    let namespace = md
        .and_then(|m| {
            m.get("namespace")
                .or_else(|| m.get("raw_item").and_then(|raw| raw.get("namespace")))
        })
        .filter(|namespace| namespace.is_string());
    if let Some(namespace) = namespace {
        item["namespace"] = namespace.clone();
    }
    item
}
fn flush_content(out: &mut Vec<Value>, content: &mut Vec<Value>) {
    if !content.is_empty() {
        out.push(json!({"role":"assistant","content":std::mem::take(content)}));
    }
}
pub(super) fn convert(
    messages: Vec<Message>,
    wire: Wire,
    model: &str,
    references: bool,
) -> Result<Vec<Value>, Error> {
    let responses = matches!(wire, Wire::Responses(_));
    let reasoning_field = match wire {
        Wire::Chat(profile) => profile.reasoning_field,
        Wire::Responses(_) => "reasoning",
    };
    let dialect = dialect_of(wire);
    let messages = repair(messages)?;
    let last = messages.len().saturating_sub(1);
    let last_user = messages.iter().rposition(|m| m.role == MessageRole::User);
    let mut out = vec![];
    for (index, msg) in messages.into_iter().enumerate() {
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
                        Part::File(f) => content.push(file(&f, wire)?),
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
                match dialect {
                    // V4 needs the field on every assistant turn; earlier
                    // models get no reasoning from before the last user turn.
                    Dialect::DeepSeek if !dialect::deepseek_legacy(model) => {
                        entry[reasoning_field] = json!(reasoning);
                    }
                    Dialect::DeepSeek if last_user.is_some_and(|user| index < user) => {}
                    // Mistral sends reasoning as thinking chunks of the content,
                    // in the turn's order, and always sends `content`.
                    Dialect::Mistral => {
                        entry["content"] = if reasoning.is_empty() {
                            json!(text)
                        } else {
                            mistral_content(&msg)
                        };
                        if index == last {
                            entry["prefix"] = json!(true);
                        }
                    }
                    _ if !reasoning.is_empty() => {
                        entry[reasoning_field] = json!(reasoning);
                    }
                    _ => {}
                }
                let mut calls = vec![];
                for p in &msg.parts {
                    match p {
                    Part::ToolCall(c)=>calls.push(json!({"id":c.tool_call_id,"type":"function","function":{"name":c.tool_name,"arguments":c.tool_args}})),
                    Part::BuiltinToolCall(_)|Part::BuiltinToolReturn(_)=>return Err(Error::UnsupportedOperation(format!("{} chat completions does not support BuiltinTool history", wire.api().label))),_=>{}
                }
                }
                if !calls.is_empty() {
                    if dialect == Dialect::Moonshot && text.is_empty() {
                        entry["content"] = Value::Null;
                    }
                    entry["tool_calls"] = calls.into();
                }
                out.push(entry);
            }
            MessageRole::Assistant if wire.responses_dialect() == Some(ResponsesDialect::Xai) => {
                if let Wire::Responses(profile) = wire {
                    xai_assistant(&msg, profile.metadata_key, &mut out);
                }
            }
            MessageRole::Assistant
                if wire.responses_dialect() == Some(ResponsesDialect::HuggingFace) =>
            {
                huggingface_assistant(&msg, &mut out);
            }
            MessageRole::Assistant => {
                let mut content = vec![];
                for p in &msg.parts {
                    let md = metadata(p, "openai");
                    // A call the application runs pairs with its output only
                    // by `call_id`, so it is never a reference or its stored item.
                    if let Part::ToolCall(c) = p {
                        flush_content(&mut out, &mut content);
                        out.push(function_call(c, md));
                        continue;
                    }
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
                        let value = tool_result(r.get_model_input(), wire)?;
                        let mut entry = if responses {
                            json!({"type":"function_call_output","call_id":r.tool_call_id,"output":value})
                        } else {
                            json!({"role":"tool","tool_call_id":r.tool_call_id,"content":value})
                        };
                        if dialect == Dialect::Mistral {
                            entry["name"] = json!(r.tool_name);
                        }
                        out.push(entry);
                    }
                }
            }
            MessageRole::Internal => {}
        }
    }
    Ok(out)
}

/// A Mistral assistant turn's text and reasoning as content chunks, in order.
fn mistral_content(msg: &Message) -> Value {
    let mut content = vec![];
    for part in &msg.parts {
        match part {
            Part::Text(t) if !t.text.is_empty() => {
                content.push(json!({"type":"text","text":t.text}));
            }
            Part::Reasoning(r) if !r.text.is_empty() => content.push(
                json!({"type":"thinking","thinking":[{"type":"text","text":r.text}],"closed":true}),
            ),
            _ => {}
        }
    }
    content.into()
}
/// Whether the host needs a file of this media type inline, so that a media
/// URL is downloaded first when the protocol has a download transport.
fn downloads(wire: Wire, media_type: &str) -> bool {
    match dialect_of(wire) {
        Dialect::OpenAI => media_type.starts_with("audio/") || media_type == "application/pdf",
        Dialect::Moonshot => ["image/", "video/", "text/"]
            .iter()
            .any(|kind| media_type.starts_with(kind)),
        Dialect::DeepSeek | Dialect::Mistral => false,
        Dialect::VertexMaas => true,
        Dialect::VertexXai => !media_type.starts_with("image/"),
    }
}
/// The URL of a user file that the protocol downloads before the request.
fn download_url(file: &FilePart, wire: Wire) -> Option<&str> {
    file.data
        .as_text()
        .filter(|s| downloads(wire, &file.media_type) && media::is_downloadable_url(s))
}
/// The messages with each file the protocol would download given inline as
/// empty bytes, to check the request before any download; `None` when there
/// is no such file.
pub(super) fn without_downloads(messages: &[Message], wire: Wire) -> Option<Vec<Message>> {
    let mut found = false;
    let mut messages = messages.to_vec();
    for msg in messages.iter_mut().filter(|m| m.role == MessageRole::User) {
        for part in &mut msg.parts {
            if let Part::File(file) = part
                && download_url(file, wire).is_some()
            {
                file.data = FileData::Bytes(Vec::new());
                found = true;
            }
        }
    }
    found.then_some(messages)
}
pub(super) async fn download_files(
    mut messages: Vec<Message>,
    wire: Wire,
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
            let Some(url) = download_url(file, wire) else {
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
                    format!("{} input download failed", wire.api().label),
                )
                .with_source(e)
            })?;
            file.data = FileData::Bytes(bytes.to_vec());
        }
    }
    Ok(messages)
}
