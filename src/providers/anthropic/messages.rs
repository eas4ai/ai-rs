//! The history of a Messages request, as `convert-to-anthropic-prompt.ts`
//! sends it (ANT-009 to ANT-012, ANT-015).
use super::{
    capabilities::Notes,
    options::{FileOptions, SystemOptions, ToolChange, part_options},
    params::add_betas,
    tools::ToolNames,
};
use crate::{
    Error,
    types::{media, messages::*},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};
use std::collections::{BTreeSet, HashSet};

pub(super) fn metadata(values: Value) -> Metadata {
    [("anthropic".into(), values)].into()
}
/// The `anthropic` entry of a part's or message's metadata.
fn anthropic(metadata: Option<&Metadata>) -> Option<&Value> {
    metadata?.get("anthropic").filter(|v| v.is_object())
}
fn part_metadata(part: &Part) -> Option<&Value> {
    anthropic(match part {
        Part::Text(p) => p.provider_metadata.as_ref(),
        Part::Reasoning(p) => p.provider_metadata.as_ref(),
        Part::ToolCall(p) => p.provider_metadata.as_ref(),
        Part::BuiltinToolCall(p) => p.provider_metadata.as_ref(),
        Part::BuiltinToolReturn(p) => p.provider_metadata.as_ref(),
        Part::File(p) => p.provider_metadata.as_ref(),
        Part::ToolResult(p) => p.provider_metadata.as_ref(),
        _ => None,
    })
}
fn string<'a>(value: Option<&'a Value>, key: &str) -> Option<&'a str> {
    value?.get(key)?.as_str()
}

/// The `cacheControl`, else `cache_control`, of an `anthropic` metadata
/// entry.
pub(crate) fn cache_control(metadata: Option<&Value>) -> Option<Value> {
    let metadata = metadata?;
    metadata
        .get("cacheControl")
        .or_else(|| metadata.get("cache_control"))
        .filter(|v| !v.is_null())
        .cloned()
}

/// A document the call sent with citations enabled, for the sources of its
/// citations (ANT-013).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Document {
    pub title: String,
    pub filename: Option<String>,
    pub media_type: String,
}

/// The documents of the user's files that enable citations, in order.
pub(crate) fn citation_documents(messages: &[Message]) -> Vec<Document> {
    messages
        .iter()
        .filter(|m| m.role == MessageRole::User)
        .flat_map(|m| &m.parts)
        .filter_map(|part| match part {
            Part::File(file)
                if matches!(file.media_type.as_str(), "application/pdf" | "text/plain") =>
            {
                let options: FileOptions =
                    part_options(anthropic(file.provider_metadata.as_ref()), "file part").ok()?;
                options
                    .citations
                    .is_some_and(|c| c.enabled)
                    .then(|| Document {
                        title: file
                            .filename
                            .clone()
                            .unwrap_or_else(|| "Untitled Document".into()),
                        filename: file.filename.clone(),
                        media_type: file.media_type.clone(),
                    })
            }
            _ => None,
        })
        .collect()
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
/// The history without internal messages and hooks, with each call that has
/// no result closed by an error result.
fn repair(messages: &[Message]) -> Result<Vec<Message>, Error> {
    let messages: Vec<_> = messages
        .iter()
        .filter(|m| m.role != MessageRole::Internal)
        .cloned()
        .filter_map(|mut m| {
            m.parts.retain(|p| !matches!(p, Part::Hook(_)));
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

/// A tool call's input: its arguments when they are a JSON object, else
/// `{"rawInvalidInput": <value>}` (`toAnthropicToolInput`).
fn tool_input(args: &str) -> Value {
    match serde_json::from_str::<Value>(args) {
        Ok(Value::Object(input)) => Value::Object(input),
        Ok(other) => json!({"rawInvalidInput": other}),
        Err(_) => json!({"rawInvalidInput": args}),
    }
}

/// A file's data as base64.
fn base64(data: &FileData) -> String {
    match data {
        FileData::Text(text) => media::data_to_base64(text.as_str()),
        FileData::Bytes(bytes) => media::data_to_base64(bytes.as_slice()),
    }
}
/// The media type a wildcard's top-level type detects in the data.
fn detect(wildcard: &str, data: media::MediaDataRef<'_>) -> Option<&'static str> {
    match wildcard {
        "image/*" => media::detect_image_media_type(data),
        "audio/*" => media::detect_audio_media_type(data),
        "video/*" => media::detect_video_media_type(data),
        _ => None,
    }
}
/// A user file's media type, a wildcard resolved from its bytes.
fn full_media_type(file: &FilePart) -> Result<String, Error> {
    let mt = file.media_type.as_str();
    if !mt.ends_with("/*") {
        return Ok(mt.to_owned());
    }
    let detected = match &file.data {
        FileData::Text(text) if media::is_downloadable_url(text) => None,
        FileData::Text(text) => detect(mt, media::MediaDataRef::from(text.as_str())),
        FileData::Bytes(bytes) => detect(mt, media::MediaDataRef::from(bytes.as_slice())),
    };
    detected.map(str::to_owned).ok_or_else(|| {
        Error::Validation(format!(
            "a file of media type {mt:?} must name its subtype, which cannot be detected"
        ))
    })
}

/// The UTF-8 text of a text file's data.
fn text_of(data: &FileData) -> Result<String, Error> {
    let bytes = match data {
        FileData::Bytes(data) => data.clone(),
        FileData::Text(data) => STANDARD
            .decode(media::data_to_base64(data.as_str()))
            .map_err(|_| {
                Error::Validation("Anthropic text document must contain base64 UTF-8 data".into())
            })?,
    };
    String::from_utf8(bytes)
        .map_err(|_| Error::Validation("Anthropic text document must contain UTF-8 data".into()))
}

fn with_cache(mut block: Value, cache: Option<Value>) -> Value {
    if let Some(cache) = cache {
        block["cache_control"] = cache;
    }
    block
}

/// A user file as an `image`, `document` or `container_upload` block.
fn user_file(
    file: &FilePart,
    cache: Option<Value>,
    betas: &mut Vec<String>,
) -> Result<Value, Error> {
    let mt = file.media_type.as_str();
    let options: FileOptions =
        part_options(anthropic(file.provider_metadata.as_ref()), "file part")?;
    let text = file.data.as_text();
    if let Some(id) = text.filter(|t| t.starts_with("file_")) {
        add_betas(betas, ["files-api-2025-04-14"]);
        if options.container_upload == Some(true) {
            return Ok(json!({"type": "container_upload", "file_id": id}));
        }
        let kind = if mt.starts_with("image/") {
            "image"
        } else {
            "document"
        };
        return Ok(with_cache(
            json!({"type": kind, "source": {"type": "file", "file_id": id}}),
            cache,
        ));
    }
    let url = text.filter(|t| media::is_downloadable_url(t));
    let document = |source: Value| {
        let mut block = json!({"type": "document", "source": source});
        if let Some(title) = options.title.clone().or_else(|| file.filename.clone()) {
            block["title"] = json!(title);
        }
        if let Some(context) = options.context.as_ref().filter(|c| !c.is_empty()) {
            block["context"] = json!(context);
        }
        if options.citations.as_ref().is_some_and(|c| c.enabled) {
            block["citations"] = json!({"enabled": true});
        }
        block
    };
    let block = if mt.starts_with("image/") {
        let source = match url {
            Some(url) => json!({"type": "url", "url": url}),
            None => json!({
                "type": "base64",
                "media_type": full_media_type(file)?,
                "data": base64(&file.data),
            }),
        };
        json!({"type": "image", "source": source})
    } else if mt == "application/pdf" {
        add_betas(betas, ["pdfs-2024-09-25"]);
        document(match url {
            Some(url) => json!({"type": "url", "url": url}),
            None => json!({
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64(&file.data),
            }),
        })
    } else if mt == "text/plain" {
        if url.is_some() {
            return Err(Error::Validation(
                "an Anthropic text file given as a URL needs the provider's download transport"
                    .into(),
            ));
        }
        document(json!({
            "type": "text",
            "media_type": "text/plain",
            "data": text_of(&file.data)?,
        }))
    } else {
        return Err(Error::Validation(format!(
            "Unsupported media type for Anthropic: {mt}"
        )));
    };
    Ok(with_cache(block, cache))
}

/// A file of a tool result's content, or `None` with a log when Anthropic
/// takes no such file there.
fn result_file(file: &FilePart, notes: &mut Notes, betas: &mut Vec<String>) -> Option<Value> {
    let image = file.media_type.starts_with("image/");
    let kind = if image { "image" } else { "document" };
    let text = file.data.as_text();
    // A file id or URL names its file by its media type alone; only an
    // image or a PDF goes there.
    let named = text.filter(|t| t.starts_with("file_") || media::is_downloadable_url(t));
    if let Some(named) = named
        && (image || file.media_type == "application/pdf")
    {
        if named.starts_with("file_") {
            add_betas(betas, ["files-api-2025-04-14"]);
            return Some(json!({"type": kind, "source": {"type": "file", "file_id": named}}));
        }
        return Some(json!({"type": kind, "source": {"type": "url", "url": named}}));
    }
    let media_type = named
        .is_none()
        .then(|| full_media_type(file).ok())
        .flatten();
    if image && let Some(media_type) = media_type {
        return Some(json!({"type": "image", "source": {
            "type": "base64", "media_type": media_type, "data": base64(&file.data)}}));
    }
    if media_type.as_deref() == Some("application/pdf") {
        add_betas(betas, ["pdfs-2024-09-25"]);
        return Some(json!({"type": "document", "source": {
            "type": "base64", "media_type": "application/pdf", "data": base64(&file.data)}}));
    }
    notes.note(
        "tool result content",
        format!(
            "a file of media type {} cannot go in a tool result; it is left out",
            file.media_type
        ),
    );
    None
}

/// The content of a tool result: its text, the JSON text of its value, or
/// its parts.
fn result_content(
    value: &ToolResultValue,
    notes: &mut Notes,
    betas: &mut Vec<String>,
) -> Result<Value, Error> {
    match value {
        ToolResultValue::Special(SpecialToolResult::Content(content)) => Ok(Value::Array(
            content
                .value
                .iter()
                .filter_map(|item| match item {
                    ContentPart::Text(p) => Some(json!({"type": "text", "text": p.text})),
                    ContentPart::File(p) => result_file(p, notes, betas),
                })
                .collect(),
        )),
        ToolResultValue::Json(Value::Null) => Ok(json!("")),
        ToolResultValue::Json(Value::String(s)) => Ok(json!(s)),
        other => Ok(json!(serde_json::to_string(other)?)),
    }
}

/// The caller of a call or result in the wire form, from its `anthropic`
/// metadata's `caller` (`getAnthropicCaller`).
fn caller(metadata: Option<&Value>) -> Option<Value> {
    let caller = metadata?.get("caller")?;
    match caller.get("type")?.as_str()? {
        kind @ ("code_execution_20250825" | "code_execution_20260120") => {
            let tool_id = caller.get("toolId")?.as_str().filter(|id| !id.is_empty())?;
            Some(json!({"type": kind, "tool_id": tool_id}))
        }
        "direct" => Some(json!({"type": "direct"})),
        _ => None,
    }
}

/// An error code of a built-in error result, `unavailable` when it names
/// none.
fn error_code(result: &Value) -> Value {
    let value = match result {
        Value::String(text) => serde_json::from_str(text).unwrap_or(Value::Null),
        other => other.clone(),
    };
    value
        .get("errorCode")
        .filter(|code| code.is_string())
        .cloned()
        .unwrap_or_else(|| json!("unavailable"))
}

/// What an assistant turn's conversion needs from the whole call.
pub(crate) struct Context<'a> {
    pub send_reasoning: bool,
    pub names: &'a ToolNames,
}

/// A built-in call of the history as a `server_tool_use` or `mcp_tool_use`.
fn builtin_call(
    call: &BuiltinToolCallPart,
    context: &Context<'_>,
    cache: Option<Value>,
    mcp: &mut BTreeSet<String>,
    notes: &mut Notes,
) -> Option<Value> {
    let md = anthropic(call.provider_metadata.as_ref());
    let input = if call.tool_args.is_empty() {
        json!({})
    } else {
        serde_json::from_str(&call.tool_args).unwrap_or_else(|_| json!({}))
    };
    let mut block = if string(md, "type") == Some("mcp-tool-use") {
        mcp.insert(call.tool_call_id.clone());
        let Some(server) = string(md, "serverName") else {
            notes.note(
                "mcp tool use",
                "an MCP call needs a string server name; it is left out",
            );
            return None;
        };
        json!({"type": "mcp_tool_use", "id": call.tool_call_id, "name": call.tool_name,
               "input": input, "server_name": server})
    } else {
        let name = context.names.provider(&call.tool_name);
        let typed = input.get("type").and_then(Value::as_str);
        let (name, input) = match (name, typed) {
            (
                "code_execution",
                Some(sub @ ("bash_code_execution" | "text_editor_code_execution")),
            ) => {
                let mut rest = input.clone();
                rest.as_object_mut().map(|o| o.remove("type"));
                (sub.to_owned(), rest)
            }
            ("code_execution", Some("programmatic-tool-call")) => {
                let mut rest = input.clone();
                rest.as_object_mut().map(|o| o.remove("type"));
                ("code_execution".to_owned(), rest)
            }
            (
                "code_execution"
                | "web_fetch"
                | "web_search"
                | "tool_search_tool_regex"
                | "tool_search_tool_bm25",
                _,
            ) => (name.to_owned(), input),
            ("advisor", _) => ("advisor".to_owned(), json!({})),
            _ => {
                notes.note("provider tool call", format!(
                        "a call of the provider tool {:?} cannot go back to Anthropic; it is left out",
                        call.tool_name
                    ),
                );
                return None;
            }
        };
        let mut block = json!({"type": "server_tool_use", "id": call.tool_call_id, "name": name, "input": input});
        if let Some(caller) = caller(md) {
            block["caller"] = caller;
        }
        block
    };
    if let Some(cache) = cache {
        block["cache_control"] = cache;
    }
    Some(block)
}

/// A built-in result of the history in Anthropic's shapes.
fn builtin_result(
    result: &BuiltinToolReturnPart,
    context: &Context<'_>,
    cache: Option<Value>,
    mcp: &BTreeSet<String>,
    notes: &mut Notes,
) -> Option<Value> {
    let md = anthropic(result.provider_metadata.as_ref());
    let id = &result.tool_call_id;
    let value = &result.result;
    // A result kept before the crate kept TypeScript's shapes goes back as
    // it came.
    if let Some(kind) = string(md, "resultType").filter(|k| !k.is_empty()) {
        return Some(with_cache(
            json!({"type": kind, "tool_use_id": id, "content": value}),
            cache,
        ));
    }
    let name = context.names.provider(&result.tool_name);
    let mut block = if mcp.contains(id) || string(md, "type") == Some("mcp-tool-use") {
        json!({"type": "mcp_tool_result", "tool_use_id": id, "is_error": result.is_error,
               "content": value})
    } else {
        match name {
            "code_execution" => {
                let Some(block) = code_execution_result(result) else {
                    notes.note(
                        "provider tool result",
                        format!(
                            "the result of code execution call {id} is not a valid code execution result; it is left out"
                        ),
                    );
                    return None;
                };
                block
            }
            "web_fetch" if result.is_error => json!({"type": "web_fetch_tool_result",
                "tool_use_id": id, "content": {"type": "web_fetch_tool_result_error",
                "error_code": error_code(value)}}),
            "web_fetch" => {
                // A field the result does not hold stays out.
                let pick = |value: &Value, keys: &[(&str, &str)]| {
                    let mut out = Map::new();
                    for (from, to) in keys {
                        if let Some(v) = value.get(*from) {
                            out.insert((*to).into(), v.clone());
                        }
                    }
                    out
                };
                let content = &value["content"];
                let mut document = pick(content, &[("title", "title"), ("citations", "citations")]);
                document.insert("type".into(), json!("document"));
                document.insert(
                    "source".into(),
                    Value::Object(pick(
                        &content["source"],
                        &[
                            ("type", "type"),
                            ("mediaType", "media_type"),
                            ("data", "data"),
                        ],
                    )),
                );
                let mut fetched = pick(value, &[("url", "url"), ("retrievedAt", "retrieved_at")]);
                fetched.insert("type".into(), json!("web_fetch_result"));
                fetched.insert("content".into(), Value::Object(document));
                json!({"type": "web_fetch_tool_result", "tool_use_id": id, "content": fetched})
            }
            "web_search" if result.is_error => json!({"type": "web_search_tool_result",
                "tool_use_id": id, "content": {"type": "web_search_tool_result_error",
                "error_code": error_code(value)}}),
            "web_search" => json!({"type": "web_search_tool_result", "tool_use_id": id,
                "content": value.as_array().into_iter().flatten().map(|r| {
                    let mut out = Map::new();
                    for (from, to) in [("url", "url"), ("title", "title"), ("pageAge", "page_age"),
                        ("encryptedContent", "encrypted_content"), ("type", "type")] {
                        if let Some(v) = r.get(from) {
                            out.insert(to.into(), v.clone());
                        }
                    }
                    Value::Object(out)
                }).collect::<Vec<_>>()}),
            "tool_search_tool_regex" | "tool_search_tool_bm25"
                if result.is_error || value["type"] == "tool_search_tool_result_error" =>
            {
                json!({"type": "tool_search_tool_result", "tool_use_id": id, "content": {
                    "type": "tool_search_tool_result_error", "error_code": error_code(value)}})
            }
            "tool_search_tool_regex" | "tool_search_tool_bm25" => {
                json!({"type": "tool_search_tool_result", "tool_use_id": id, "content": {
                    "type": "tool_search_tool_search_result",
                    "tool_references": value.as_array().into_iter().flatten().map(|r| json!({
                        "type": "tool_reference", "tool_name": r["toolName"]})).collect::<Vec<_>>()}})
            }
            "advisor" => {
                let mut content = match value["type"].as_str() {
                    Some("advisor_result") => {
                        json!({"type": "advisor_result", "text": value["text"]})
                    }
                    Some("advisor_redacted_result") => json!({"type": "advisor_redacted_result",
                        "encrypted_content": value["encryptedContent"]}),
                    _ => {
                        json!({"type": "advisor_tool_result_error", "error_code": error_code(value)})
                    }
                };
                if content["type"] != "advisor_tool_result_error"
                    && let Some(reason) = value.get("stopReason").filter(|r| !r.is_null())
                {
                    content["stop_reason"] = reason.clone();
                }
                json!({"type": "advisor_tool_result", "tool_use_id": id, "content": content})
            }
            _ => {
                notes.note("provider tool result", format!(
                        "a result of the provider tool {:?} cannot go back to Anthropic; it is left out",
                        result.tool_name
                    ),
                );
                return None;
            }
        }
    };
    if matches!(name, "web_fetch" | "web_search")
        && let Some(caller) = caller(md)
    {
        block["caller"] = caller;
    }
    Some(with_cache(block, cache))
}

/// A code execution result, by the type of its value.
fn code_execution_result(result: &BuiltinToolReturnPart) -> Option<Value> {
    let id = &result.tool_call_id;
    let value = &result.result;
    if result.is_error {
        let parsed = match value {
            Value::String(text) => serde_json::from_str(text).unwrap_or(Value::Null),
            other => other.clone(),
        };
        let code = parsed
            .get("errorCode")
            .cloned()
            .unwrap_or_else(|| json!("unknown"));
        return Some(if parsed["type"] == "code_execution_tool_result_error" {
            json!({"type": "code_execution_tool_result", "tool_use_id": id, "content": {
                    "type": "code_execution_tool_result_error", "error_code": code}})
        } else {
            json!({"type": "bash_code_execution_tool_result", "tool_use_id": id, "content": {
                    "type": "bash_code_execution_tool_result_error", "error_code": code}})
        });
    }
    let content = |value: &Value| value.get("content").cloned().unwrap_or_else(|| json!([]));
    Some(match value["type"].as_str()? {
        "code_execution_result" => json!({"type": "code_execution_tool_result", "tool_use_id": id,
            "content": {"type": "code_execution_result", "stdout": value["stdout"],
                "stderr": value["stderr"], "return_code": value["return_code"], "content": content(value)}}),
        "encrypted_code_execution_result" => json!({"type": "code_execution_tool_result",
            "tool_use_id": id, "content": {"type": "encrypted_code_execution_result",
                "encrypted_stdout": value["encrypted_stdout"], "stderr": value["stderr"],
                "return_code": value["return_code"], "content": content(value)}}),
        "bash_code_execution_result" => json!({"type": "bash_code_execution_tool_result",
            "tool_use_id": id, "content": {"type": "bash_code_execution_result",
                "stdout": value["stdout"], "stderr": value["stderr"],
                "return_code": value["return_code"], "content": value["content"]}}),
        "bash_code_execution_tool_result_error" => {
            json!({"type": "bash_code_execution_tool_result",
            "tool_use_id": id, "content": value})
        }
        _ => json!({"type": "text_editor_code_execution_tool_result", "tool_use_id": id,
            "content": value}),
    })
}

/// The blocks of an assistant run, with each run of blocks between thinking
/// and fallback blocks ending in its `tool_use` blocks
/// (`moveToolUseBlocksToEnd`).
fn tool_uses_last(content: Vec<Value>) -> Vec<Value> {
    let mut result = Vec::new();
    let mut segment: Vec<Value> = Vec::new();
    let flush = |segment: &mut Vec<Value>, result: &mut Vec<Value>| {
        let (uses, others): (Vec<_>, Vec<_>) =
            segment.drain(..).partition(|b| b["type"] == "tool_use");
        result.extend(others);
        result.extend(uses);
    };
    for block in content {
        if matches!(
            block["type"].as_str(),
            Some("thinking" | "redacted_thinking" | "fallback")
        ) {
            flush(&mut segment, &mut result);
            result.push(block);
        } else {
            segment.push(block);
        }
    }
    flush(&mut segment, &mut result);
    result
}

/// A converted history: the top-level system blocks, the messages and the
/// betas they need.
pub(crate) struct Converted {
    pub system: Option<Vec<Value>>,
    pub messages: Vec<Value>,
    pub betas: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    System,
    User,
    Assistant,
}
fn block_of(role: MessageRole) -> Block {
    match role {
        MessageRole::System => Block::System,
        MessageRole::Assistant => Block::Assistant,
        _ => Block::User,
    }
}

/// Convert a history, after dropping internal parts and closing calls
/// without a result.
pub(crate) fn convert(
    messages: &[Message],
    context: &Context<'_>,
    notes: &mut Notes,
) -> Result<Converted, Error> {
    let messages = repair(messages)?;
    let mut blocks: Vec<(Block, Vec<&Message>)> = Vec::new();
    for msg in &messages {
        let block = block_of(msg.role);
        match blocks.last_mut() {
            Some((last, run)) if *last == block => run.push(msg),
            _ => blocks.push((block, vec![msg])),
        }
    }
    let mut betas = Vec::new();
    let mut system: Option<Vec<Value>> = None;
    let mut out: Vec<Value> = Vec::new();
    let mut last_user = None;
    let count = blocks.len();
    for (index, (block, run)) in blocks.into_iter().enumerate() {
        let last_block = index + 1 == count;
        match block {
            Block::System => system_block(
                index,
                &run,
                context,
                notes,
                &mut system,
                &mut out,
                &mut betas,
            )?,
            Block::User => {
                let mut content = Vec::new();
                for msg in run {
                    let message_cache = cache_control(anthropic(msg.provider_metadata.as_ref()));
                    let parts: Vec<&Part> = msg
                        .parts
                        .iter()
                        .filter(|p| {
                            matches!(p, Part::Text(_) | Part::File(_) | Part::ToolResult(_))
                        })
                        .collect();
                    if msg.role == MessageRole::User && !parts.is_empty() {
                        last_user = Some(out.len());
                    }
                    for (j, part) in parts.iter().enumerate() {
                        let last = j + 1 == parts.len();
                        let own = cache_control(part_metadata(part));
                        match part {
                            Part::Text(t) => {
                                let cache = notes.cache(
                                    own.or_else(|| last.then(|| message_cache.clone()).flatten()),
                                    "user message part",
                                    true,
                                );
                                content.push(with_cache(
                                    json!({"type": "text", "text": t.text}),
                                    cache,
                                ));
                            }
                            Part::File(f) => {
                                let cache = notes.cache(
                                    own.or_else(|| last.then(|| message_cache.clone()).flatten()),
                                    "user message part",
                                    true,
                                );
                                content.push(user_file(f, cache, &mut betas)?);
                            }
                            Part::ToolResult(r) => {
                                let content_cache = match r.get_model_input() {
                                    ToolResultValue::Special(SpecialToolResult::Content(c)) => c
                                        .value
                                        .iter()
                                        .find_map(|item| {
                                            let md = match item {
                                                ContentPart::Text(p) => {
                                                    p.provider_metadata.as_ref()
                                                }
                                                ContentPart::File(p) => {
                                                    p.provider_metadata.as_ref()
                                                }
                                            };
                                            anthropic(md).map(|md| cache_control(Some(md)))
                                        })
                                        .flatten(),
                                    _ => None,
                                };
                                let cache = notes.cache(
                                    own.or(content_cache)
                                        .or_else(|| last.then(|| message_cache.clone()).flatten()),
                                    "tool result part",
                                    true,
                                );
                                let mut result =
                                    json!({"type": "tool_result", "tool_use_id": r.tool_call_id});
                                let toolset = context
                                    .names
                                    .toolset(&r.tool_name)
                                    .map(str::to_owned)
                                    .or_else(|| {
                                        string(part_metadata(part), "toolsetName")
                                            .map(str::to_owned)
                                    });
                                if let Some(toolset) = toolset {
                                    result["toolset_name"] = json!(toolset);
                                }
                                result["content"] =
                                    result_content(r.get_model_input(), notes, &mut betas)?;
                                if r.is_error() {
                                    result["is_error"] = json!(true);
                                }
                                content.push(with_cache(result, cache));
                            }
                            _ => {}
                        }
                    }
                }
                out.push(json!({"role": "user", "content": content}));
            }
            Block::Assistant => {
                let content = assistant_block(&run, last_block, context, notes, &mut betas)?;
                if !content.is_empty() {
                    out.push(json!({"role": "assistant", "content": tool_uses_last(content)}));
                }
            }
        }
    }
    prune_callers(&mut out, last_user, notes);
    Ok(Converted {
        system,
        messages: place_system(out, notes)?,
        betas,
    })
}

/// Whether Anthropic takes a system message with content after this
/// message: a user message, or an assistant message that ends in a server
/// tool result.
fn takes_system_after(message: &Value) -> bool {
    match message["role"].as_str() {
        Some("user") => true,
        Some("assistant") => message["content"]
            .as_array()
            .and_then(|content| content.last())
            .and_then(|block| block["type"].as_str())
            .is_some_and(|kind| kind.ends_with("_tool_result") && kind != "tool_result"),
        _ => false,
    }
}

/// Put each run of later system messages where Anthropic takes it (ANT-011).
/// A run with content stays when it follows a user message, or an assistant
/// message that ends in a server tool result, and comes before an assistant
/// message or ends the history. Elsewhere it moves to just after the last
/// user message before the next assistant message or the end, and the user
/// messages it separated go as one. A run of effort-only messages renders
/// nothing, and Anthropic takes it anywhere.
fn place_system(messages: Vec<Value>, notes: &mut Notes) -> Result<Vec<Value>, Error> {
    let role = |m: &Value| m["role"].as_str().map(str::to_owned);
    let roles: Vec<Option<String>> = messages.iter().map(role).collect();
    // The role of the first message from `i` on that is not a system message.
    let next_role = |i: usize| {
        roles[i..]
            .iter()
            .find(|r| r.as_deref() != Some("system"))
            .cloned()
            .flatten()
    };
    let no_user = || {
        Error::Validation(
            "a system message with content needs a user message after it: Anthropic takes one \
             only after a user message or a server tool result, and before the model's reply"
                .into(),
        )
    };
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    let mut pending: Vec<Value> = Vec::new();
    let mut messages = messages.into_iter().enumerate().peekable();
    while let Some((i, mut message)) = messages.next() {
        match roles[i].as_deref() {
            Some("system") => {
                let mut run = vec![message];
                while let Some((_, next)) =
                    messages.next_if(|(j, _)| roles[*j].as_deref() == Some("system"))
                {
                    run.push(next);
                }
                let end = i + run.len();
                let content = run
                    .iter()
                    .any(|m| m["content"].as_array().is_some_and(|c| !c.is_empty()));
                let after = out
                    .iter()
                    .rev()
                    .find(|m| m["role"] != "system")
                    .is_some_and(takes_system_after);
                let before = matches!(
                    roles.get(end).cloned().flatten().as_deref(),
                    None | Some("assistant")
                );
                if !content || (pending.is_empty() && after && before) {
                    out.extend(run);
                } else {
                    notes.note(
                        "system message",
                        "Anthropic does not take a system message with content here; it moves to \
                         just after the last user message before the model's reply",
                    );
                    pending.extend(run);
                }
            }
            Some("user") => {
                match out.last_mut() {
                    Some(last) if last["role"] == "user" => {
                        if let (Some(merged), Value::Array(content)) =
                            (last["content"].as_array_mut(), message["content"].take())
                        {
                            merged.extend(content);
                        }
                    }
                    _ => out.push(message),
                }
                if matches!(next_role(i + 1).as_deref(), None | Some("assistant")) {
                    out.append(&mut pending);
                }
            }
            _ => {
                if !pending.is_empty() {
                    return Err(no_user());
                }
                out.push(message);
            }
        }
    }
    if !pending.is_empty() {
        return Err(no_user());
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn system_block(
    index: usize,
    run: &[&Message],
    context: &Context<'_>,
    notes: &mut Notes,
    system: &mut Option<Vec<Value>>,
    out: &mut Vec<Value>,
    betas: &mut Vec<String>,
) -> Result<(), Error> {
    struct Converted {
        content: Vec<Value>,
        options: SystemOptions,
        tool_changes: usize,
    }
    let mut converted = Vec::new();
    for msg in run {
        let message_md = anthropic(msg.provider_metadata.as_ref());
        let first_part_md = msg.parts.iter().find_map(part_metadata);
        let options: SystemOptions = part_options(message_md, "system message")?;
        let text = msg.text();
        let changes = options.tool_changes.clone().unwrap_or_default();
        let mut content = Vec::new();
        if !text.is_empty()
            || (changes.is_empty() && options.clear_at.is_none() && options.effort.is_none())
        {
            let cache = notes.cache(
                cache_control(first_part_md).or_else(|| cache_control(message_md)),
                "system message",
                true,
            );
            content.push(with_cache(json!({"type": "text", "text": text}), cache));
        }
        for change in &changes {
            let (kind, name) = match change {
                ToolChange::ToolAddition { tool_name } => ("tool_addition", tool_name),
                ToolChange::ToolRemoval { tool_name } => ("tool_removal", tool_name),
            };
            content.push(json!({"type": kind, "tool": {"type": "tool_reference",
                "name": context.names.provider(name)}}));
        }
        converted.push(Converted {
            content,
            options,
            tool_changes: changes.len(),
        });
    }
    let tool_changes: usize = converted.iter().map(|c| c.tool_changes).sum();
    let inline = converted
        .iter()
        .any(|c| c.options.clear_at.is_some() || c.options.effort.is_some());
    if index == 0 || (system.is_none() && tool_changes == 0 && !inline) {
        if tool_changes > 0 {
            notes.note(
                "toolChanges",
                "tool changes on the first system message are not taken; they are left out",
            );
        }
        for c in &converted {
            if c.content.is_empty() && c.options.clear_at.is_none() {
                if let Some(effort) = c.options.effort {
                    out.push(json!({"role": "system", "content": [], "output_config": {"effort": effort.wire()}}));
                    add_betas(betas, ["mid-conversation-output-config-2026-07-01"]);
                }
            } else if c.options.clear_at.is_some() || c.options.effort.is_some() {
                notes.note(
                    "clearAt and effort",
                    "clearAt and effort on the first system text are not taken; they are left out",
                );
            }
        }
        *system = Some(
            converted
                .iter()
                .flat_map(|c| c.content.iter().filter(|b| b["type"] == "text").cloned())
                .collect(),
        );
    } else {
        add_betas(betas, ["mid-conversation-system-2026-04-07"]);
        for c in converted {
            let mut message = json!({"role": "system", "content": c.content});
            if c.options.clear_at.is_some() {
                message["clear_at"] = json!("next_user_message");
                add_betas(betas, ["mid-conversation-system-clear-at-2026-08-21"]);
            }
            if let Some(effort) = c.options.effort {
                message["output_config"] = json!({"effort": effort.wire()});
                add_betas(betas, ["mid-conversation-output-config-2026-07-01"]);
            }
            if c.tool_changes > 0 {
                add_betas(betas, ["mid-conversation-tool-changes-2026-07-01"]);
            }
            out.push(message);
        }
    }
    Ok(())
}

fn assistant_block(
    run: &[&Message],
    last_block: bool,
    context: &Context<'_>,
    notes: &mut Notes,
    betas: &mut Vec<String>,
) -> Result<Vec<Value>, Error> {
    let mut content = Vec::new();
    let mut mcp = BTreeSet::new();
    for (j, msg) in run.iter().enumerate() {
        let last_message = j + 1 == run.len();
        let message_cache = cache_control(anthropic(msg.provider_metadata.as_ref()));
        for (k, part) in msg.parts.iter().enumerate() {
            let last_part = k + 1 == msg.parts.len();
            let md = part_metadata(part);
            let own = cache_control(md);
            let fallback_cache = || last_part.then(|| message_cache.clone()).flatten();
            match part {
                Part::Text(t) => {
                    match string(md, "type") {
                        Some("compaction") => {
                            if t.text.is_empty() {
                                continue;
                            }
                            let cache = notes.cache(
                                own.or_else(fallback_cache),
                                "assistant message part",
                                true,
                            );
                            let mut block = json!({"type": "compaction", "content": t.text});
                            if let Some(signature) = string(md, "signature") {
                                block["signature"] = json!(signature);
                                add_betas(betas, ["compact-2026-09-04"]);
                            }
                            content.push(with_cache(block, cache));
                        }
                        Some("fallback") => {
                            let from = md
                                .and_then(|m| m.get("from"))
                                .filter(|f| f["model"].is_string());
                            let to = md
                                .and_then(|m| m.get("to"))
                                .filter(|t| t["model"].is_string());
                            let (Some(from), Some(to)) = (from, to) else {
                                notes.note("fallback", "fallback metadata needs from.model and to.model; it is left out",
                            );
                                continue;
                            };
                            add_betas(betas, ["server-side-fallback-2026-06-01"]);
                            content.push(json!({"type": "fallback", "from": from, "to": to}));
                        }
                        Some("container_upload") => {}
                        _ => {
                            let cache = notes.cache(
                                own.or_else(fallback_cache),
                                "assistant message part",
                                true,
                            );
                            let text = if last_block && last_message && last_part {
                                t.text.trim().to_owned()
                            } else {
                                t.text.clone()
                            };
                            let mut block = json!({"type": "text", "text": text});
                            if let Some(citations) =
                                md.and_then(|m| m.get("citations")).filter(|c| !c.is_null())
                            {
                                block["citations"] = citations.clone();
                            }
                            content.push(with_cache(block, cache));
                        }
                    }
                }
                Part::Reasoning(r) => {
                    if !context.send_reasoning {
                        notes.note(
                            "sendReasoning",
                            "sending reasoning is turned off; the reasoning is left out",
                        );
                        continue;
                    }
                    if let Some(signature) = string(md, "signature").filter(|s| !s.is_empty()) {
                        notes.cache(own, "thinking block", false);
                        content.push(
                            json!({"type": "thinking", "thinking": r.text, "signature": signature}),
                        );
                    } else if let Some(data) = string(md, "redactedData") {
                        notes.cache(own, "redacted thinking block", false);
                        content.push(json!({"type": "redacted_thinking", "data": data}));
                    } else {
                        notes.note("reasoning", "reasoning without a signature or redacted data cannot go back; it is left out",
                        );
                    }
                }
                Part::ToolCall(c) => {
                    let cache =
                        notes.cache(own.or_else(fallback_cache), "assistant message part", true);
                    let input = tool_input(&c.tool_args);
                    let toolset = context
                        .names
                        .toolset(&c.tool_name)
                        .map(str::to_owned)
                        .or_else(|| string(md, "toolsetName").map(str::to_owned));
                    let mut block = match toolset {
                        Some(toolset) => {
                            let action = input.get("action").and_then(Value::as_str);
                            if action.is_none() {
                                notes.note(
                                    "toolset call",
                                    format!("the toolset call of {:?} has no action", c.tool_name),
                                );
                            }
                            json!({"type": "tool_use", "id": c.tool_call_id,
                                "name": action.unwrap_or(&toolset), "toolset_name": toolset, "input": input})
                        }
                        None => {
                            json!({"type": "tool_use", "id": c.tool_call_id,
                                "name": context.names.provider(&c.tool_name), "input": input})
                        }
                    };
                    if let Some(caller) = caller(md) {
                        block["caller"] = caller;
                    }
                    content.push(with_cache(block, cache));
                }
                Part::BuiltinToolCall(c) => {
                    let cache =
                        notes.cache(own.or_else(fallback_cache), "assistant message part", true);
                    if let Some(block) = builtin_call(c, context, cache, &mut mcp, notes) {
                        content.push(block);
                    }
                }
                Part::BuiltinToolReturn(r) => {
                    let cache =
                        notes.cache(own.or_else(fallback_cache), "assistant message part", true);
                    if let Some(block) = builtin_result(r, context, cache, &mcp, notes) {
                        content.push(block);
                    }
                }
                _ => {}
            }
        }
    }
    Ok(content)
}

/// Leave out, before the last user message, the caller of a call whose code
/// execution call the history no longer holds.
fn prune_callers(messages: &mut [Value], last_user: Option<usize>, notes: &mut Notes) {
    let code_execution: BTreeSet<String> = messages
        .iter()
        .filter(|m| m["role"] == "assistant")
        .flat_map(|m| m["content"].as_array().into_iter().flatten())
        .filter(|b| b["type"] == "server_tool_use" && b["name"] == "code_execution")
        .filter_map(|b| b["id"].as_str().map(str::to_owned))
        .collect();
    let mut warned = BTreeSet::new();
    let end = last_user.unwrap_or(0);
    for message in messages.iter_mut().take(end) {
        if message["role"] != "assistant" {
            continue;
        }
        for block in message["content"].as_array_mut().into_iter().flatten() {
            let Some(caller) = block.get("caller") else {
                continue;
            };
            let kind = caller["type"].as_str().unwrap_or_default();
            let tool = caller["tool_id"].as_str().unwrap_or_default();
            if kind == "direct" || code_execution.contains(tool) {
                continue;
            }
            let tool = tool.to_owned();
            let id = block
                .get("id")
                .or_else(|| block.get("tool_use_id"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            block.as_object_mut().map(|o| o.remove("caller"));
            if warned.insert(id.clone()) {
                notes.note("caller", format!(
                        "the caller of tool call {id} is left out: its code execution call {tool} is missing from the history"
                    ),
                );
            }
        }
    }
}

/// A user file as Anthropic takes it, with no part options.
pub fn file_to_anthropic(part: &FilePart) -> Result<Value, Error> {
    user_file(part, None, &mut Vec::new())
}
/// A tool result's content as Anthropic takes it.
pub fn tool_result_to_anthropic(value: &ToolResultValue) -> Result<Value, Error> {
    result_content(value, &mut Notes::new(""), &mut Vec::new())
}
/// Convert messages after dropping internal parts and repairing orphaned
/// calls: the top-level system text and the messages.
pub fn messages_to_anthropic(messages: &[Message]) -> Result<(Option<String>, Vec<Value>), Error> {
    let names = ToolNames::default();
    let context = Context {
        send_reasoning: true,
        names: &names,
    };
    let converted = convert(messages, &context, &mut Notes::new(""))?;
    let system = converted.system.map(|blocks| {
        blocks
            .iter()
            .filter_map(|b| b["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    });
    Ok((system.filter(|s| !s.is_empty()), converted.messages))
}
