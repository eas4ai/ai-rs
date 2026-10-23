//! Persisted-message conversion for AI SDK UI messages.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::{
    Error,
    types::{
        Message, MessageRole, Part,
        media::data_to_data_url,
        messages::{
            BuiltinToolReturnPart, FileData, HookStatus, MessageBundle, SpecialToolResult,
            ToolResultPart, ToolResultValue,
        },
    },
};

use super::{
    approvals::tool_call_id_for,
    id_utils,
    tool_utils::{normalize_tool_input, python_json_dumps, python_truthy},
    ui_messages::UIMessage,
};

const MERGEABLE_TOOL_FIELDS: &[&str] = &[
    "rawInput",
    "output",
    "errorText",
    "approval",
    "providerExecuted",
    "callProviderMetadata",
    "resultProviderMetadata",
    "toolMetadata",
    "preliminary",
    "title",
];

fn state_rank(part: &Value) -> u8 {
    match part["state"].as_str() {
        Some("input-available") => 1,
        Some("approval-requested") => 2,
        Some("approval-responded") => 3,
        Some("output-denied") => 4,
        Some("output-error") => 5,
        Some("output-available") => 6,
        _ => 0,
    }
}

fn tool_call_id(part: &Value) -> Option<&str> {
    let kind = part.get("type")?.as_str()?;
    (kind == "dynamic-tool" || kind.starts_with("tool-"))
        .then(|| part.get("toolCallId").and_then(Value::as_str))
        .flatten()
}

fn tool_index(parts: &[Value]) -> HashMap<String, usize> {
    parts
        .iter()
        .enumerate()
        .filter_map(|(i, part)| tool_call_id(part).map(|id| (id.to_owned(), i)))
        .collect()
}

fn merge_tool_part(existing: &mut Value, candidate: Value) {
    if state_rank(&candidate) >= state_rank(existing) {
        existing["state"] = candidate["state"].clone();
        if candidate["state"] == "output-denied" {
            existing["output"] = Value::Null;
        }
    }
    if existing["input"].is_null() && !candidate["input"].is_null() {
        existing["input"] = candidate["input"].clone();
    }
    for field in MERGEABLE_TOOL_FIELDS {
        if let Some(value) = candidate.get(*field).filter(|value| !value.is_null()) {
            existing[*field] = value.clone();
        }
    }
}

/// Collapse duplicate tool parts while retaining their first display position.
#[must_use]
pub fn dedupe_tool_parts(parts: Vec<Value>) -> Vec<Value> {
    let mut result = Vec::with_capacity(parts.len());
    let mut index = HashMap::<String, usize>::new();
    for part in parts {
        let Some(id) = tool_call_id(&part) else {
            result.push(part);
            continue;
        };
        if let Some(&position) = index.get(id) {
            merge_tool_part(&mut result[position], part);
        } else {
            index.insert(id.to_owned(), result.len());
            result.push(part);
        }
    }
    result
}

/// Serialize a sub-agent transcript as one message, a list, or null if empty.
pub fn bundle_to_wire_output(bundle: &MessageBundle) -> Result<Value, Error> {
    let messages = to_ui_messages(&bundle.messages)?;
    let mut dumped = messages
        .into_iter()
        .map(|message| {
            let mut value = serde_json::to_value(message)?;
            if let Some(parts) = value.get_mut("parts").and_then(Value::as_array_mut) {
                for part in parts {
                    python_part_dump(part);
                }
            }
            Ok(value)
        })
        .collect::<Result<Vec<Value>, Error>>()?;
    Ok(match dumped.len() {
        0 => Value::Null,
        1 => dumped.pop().unwrap_or(Value::Null),
        _ => Value::Array(dumped),
    })
}

// Python model_dump uses field names rather than wire aliases by default.
fn python_part_dump(part: &mut Value) {
    let Some(object) = part.as_object_mut() else {
        return;
    };
    for (wire, python) in [
        ("toolCallId", "tool_call_id"),
        ("toolName", "tool_name"),
        ("rawInput", "raw_input"),
        ("errorText", "error_text"),
        ("providerExecuted", "provider_executed"),
        ("callProviderMetadata", "call_provider_metadata"),
        ("resultProviderMetadata", "result_provider_metadata"),
        ("toolMetadata", "tool_metadata"),
        ("mediaType", "media_type"),
        ("providerReference", "provider_reference"),
        ("providerMetadata", "provider_metadata"),
    ] {
        if let Some(value) = object.remove(wire) {
            object.insert(python.to_owned(), value);
        }
    }
    if let Some(approval) = object.get_mut("approval").and_then(Value::as_object_mut)
        && let Some(automatic) = approval.remove("isAutomatic")
    {
        approval.insert("is_automatic".to_owned(), automatic);
    }
}

fn output_view(part: &ToolResultPart) -> Result<Value, Error> {
    Ok(match &part.result {
        ToolResultValue::Special(SpecialToolResult::Content(output)) => {
            json!({"state":"output-available", "output":output.value})
        }
        ToolResultValue::Special(SpecialToolResult::Messages(bundle)) => {
            let output = bundle_to_wire_output(bundle)?;
            json!({"state":"output-available", "output":if output.is_null() { json!([]) } else { output }})
        }
        ToolResultValue::Json(value) if part.is_error() => {
            json!({"state":"output-error", "errorText":match value.as_str() {
                Some(text) => text.to_owned(),
                None => python_json_dumps(value)?,
            }})
        }
        ToolResultValue::Json(value) => json!({"state":"output-available", "output":value}),
    })
}

/// Merge result records into the corresponding UI tool parts.
pub fn merge_tool_results(ui_parts: &mut [Value], tool_parts: &[Part]) -> Result<(), Error> {
    let index = tool_index(ui_parts);
    for part in tool_parts {
        let (id, updates) = match part {
            Part::ToolResult(part) if part.is_hook_deferred => continue,
            Part::ToolResult(part) => {
                let mut updates = output_view(part)?;
                updates["resultProviderMetadata"] = json!(part.provider_metadata);
                (&part.tool_call_id, updates)
            }
            Part::BuiltinToolReturn(part) => {
                let mut updates = json!({
                    "state":if part.is_error {"output-error"} else {"output-available"},
                    "providerExecuted":true, "resultProviderMetadata":part.provider_metadata,
                });
                if part.is_error {
                    updates["errorText"] = json!(python_str(&part.result));
                } else {
                    updates["output"] = part.result.clone();
                }
                (&part.tool_call_id, updates)
            }
            _ => continue,
        };
        let Some(&position) = index.get(id) else {
            continue;
        };
        let existing = &mut ui_parts[position];
        if existing["state"] == "output-denied" {
            continue;
        }
        if let (Some(existing), Some(updates)) = (existing.as_object_mut(), updates.as_object()) {
            existing.extend(updates.clone());
        }
    }
    Ok(())
}

/// Merge approval hooks into the corresponding UI tool parts.
pub fn merge_approval_signals(ui_parts: &mut [Value], internal_parts: &[Part]) {
    let index = tool_index(ui_parts);
    for part in internal_parts {
        let Part::Hook(part) = part else {
            continue;
        };
        let Some(position) = tool_call_id_for(part).and_then(|id| index.get(id)).copied() else {
            continue;
        };
        let existing = &mut ui_parts[position];
        if let Some(value) = part
            .metadata
            .get("providerExecuted")
            .and_then(Value::as_bool)
        {
            existing["providerExecuted"] = json!(value);
        }
        let automatic = part.metadata.get("isAutomatic").and_then(Value::as_bool);
        match part.status {
            HookStatus::Pending => {
                existing["state"] = json!("approval-requested");
                existing["approval"] = json!({"id":part.hook_id, "approved":null, "reason":null, "isAutomatic":automatic});
            }
            HookStatus::Resolved => {
                let resolution = part.resolution.as_ref().and_then(Value::as_object);
                let granted = resolution
                    .and_then(|r| r.get("granted"))
                    .cloned()
                    .unwrap_or(Value::Null);
                let reason = resolution
                    .and_then(|r| r.get("reason"))
                    .cloned()
                    .unwrap_or(Value::Null);
                let approved = python_truthy(&granted);
                existing["approval"] = json!({"id":part.hook_id, "approved":granted, "reason":reason, "isAutomatic":automatic});
                existing["state"] = json!(if approved {
                    "approval-responded"
                } else {
                    "output-denied"
                });
                if !approved {
                    existing["output"] = Value::Null;
                }
            }
            HookStatus::Cancelled => {
                existing["state"] = json!("output-error");
                existing["errorText"] = json!("Hook cancelled");
            }
        }
    }
}

fn tool_part(kind: String, id: &str, state: &str, input: Value) -> Value {
    json!({"type":kind,"toolCallId":id,"state":state,"input":input,"output":null,
        "rawInput":null,"errorText":null,"approval":null,"providerExecuted":null,
        "callProviderMetadata":null,"resultProviderMetadata":null,"toolMetadata":null,
        "preliminary":null,"title":null})
}

fn builtin_return_part(part: &BuiltinToolReturnPart) -> Value {
    let mut value = tool_part(
        "dynamic-tool".to_owned(),
        &part.tool_call_id,
        if part.is_error {
            "output-error"
        } else {
            "output-available"
        },
        Value::Null,
    );
    value["toolName"] = json!(part.tool_name);
    value["providerExecuted"] = json!(true);
    value["resultProviderMetadata"] = json!(part.provider_metadata);
    if part.is_error {
        value["errorText"] = json!(python_str(&part.result));
    } else {
        value["output"] = part.result.clone();
    }
    value
}

/// Convert core parts to UI message parts, omitting empty text and hidden parts.
pub fn to_ui_parts(parts: &[Part]) -> Result<Vec<Value>, Error> {
    let mut result = Vec::new();
    for part in parts {
        let output = match part {
            Part::Text(part) if !part.text.is_empty() => {
                json!({"type":"text","text":part.text,"state":null,"providerMetadata":part.provider_metadata})
            }
            Part::Reasoning(part) if !part.text.is_empty() => {
                json!({"type":"reasoning","text":part.text,"state":null,"providerMetadata":part.provider_metadata})
            }
            Part::ToolCall(part) => {
                let mut value = tool_part(
                    format!("tool-{}", part.tool_name),
                    &part.tool_call_id,
                    "input-available",
                    normalize_tool_input(&part.tool_args),
                );
                value["callProviderMetadata"] = json!(part.provider_metadata);
                value
            }
            Part::BuiltinToolCall(part) => {
                let mut value = tool_part(
                    "dynamic-tool".to_owned(),
                    &part.tool_call_id,
                    "input-available",
                    normalize_tool_input(&part.tool_args),
                );
                value["toolName"] = json!(part.tool_name);
                value["providerExecuted"] = json!(true);
                value["callProviderMetadata"] = json!(part.provider_metadata);
                value
            }
            Part::BuiltinToolReturn(part) => builtin_return_part(part),
            Part::File(part) => {
                let url = match &part.data {
                    FileData::Text(text) => data_to_data_url(text.as_str(), &part.media_type),
                    FileData::Bytes(bytes) => data_to_data_url(bytes.as_slice(), &part.media_type),
                };
                json!({"type":"file","mediaType":part.media_type,"url":url,"filename":part.filename,"providerReference":null,"providerMetadata":part.provider_metadata})
            }
            _ => continue,
        };
        result.push(output);
    }
    Ok(result)
}

/// Group persisted core messages into UI message bubbles.
pub fn to_ui_messages(messages: &[Message]) -> Result<Vec<UIMessage>, Error> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        let message = &messages[i];
        match message.role {
            MessageRole::User | MessageRole::System => {
                result.push(serde_json::from_value(json!({
                    "id":message.id,"role":message.role,"metadata":id_utils::metadata_for(std::slice::from_ref(message)),
                    "parts":to_ui_parts(&message.parts)?,
                }))?);
                i += 1;
            }
            MessageRole::Assistant => {
                let (bubble, consumed) = assistant_bubble(&messages[i..])?;
                result.push(bubble);
                i += consumed;
            }
            _ => i += 1,
        }
    }
    Ok(result)
}

fn assistant_bubble(messages: &[Message]) -> Result<(UIMessage, usize), Error> {
    let first = &messages[0];
    let bubble_id = first
        .turn_id
        .as_deref()
        .filter(|id| !id.is_empty())
        .unwrap_or(&first.id);
    let mut parts = Vec::new();
    let mut consumed = 0;
    for current in messages {
        if consumed > 0 && current.turn_id.as_deref().is_some_and(|id| id != bubble_id) {
            break;
        }
        match current.role {
            MessageRole::Assistant => {
                parts.extend(to_ui_parts(&current.parts)?);
                parts = dedupe_tool_parts(parts);
                // A provider's approval request, such as OpenAI's MCP one,
                // is a hook in the assistant message that raised it.
                merge_approval_signals(&mut parts, &current.parts);
            }
            MessageRole::Tool => merge_tool_results(&mut parts, &current.parts)?,
            MessageRole::Internal => merge_approval_signals(&mut parts, &current.parts),
            _ => break,
        }
        consumed += 1;
    }
    let bubble = serde_json::from_value(json!({
        "id":bubble_id,"role":"assistant","metadata":id_utils::metadata_for(&messages[..consumed]),
        "parts":dedupe_tool_parts(parts),
    }))?;
    Ok((bubble, consumed))
}

pub(crate) fn python_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => "None".into(),
        Value::Bool(value) => if *value { "True" } else { "False" }.into(),
        Value::Number(_) => python_json_dumps(value).expect("JSON number writing to memory"),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!("{}: {}", python_repr(&json!(key)), python_repr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn python_repr(value: &Value) -> String {
    if let Value::String(text) = value {
        let quote = if text.contains('\'') && !text.contains('"') {
            '"'
        } else {
            '\''
        };
        let mut result = String::new();
        result.push(quote);
        for c in text.chars() {
            match c {
                '\\' => result.push_str("\\\\"),
                '\n' => result.push_str("\\n"),
                '\r' => result.push_str("\\r"),
                '\t' => result.push_str("\\t"),
                c if c == quote => {
                    result.push('\\');
                    result.push(c);
                }
                c if c.is_control() => {
                    use std::fmt::Write as _;
                    let _ = write!(result, "\\x{:02x}", u32::from(c));
                }
                c => result.push(c),
            }
        }
        result.push(quote);
        result
    } else {
        python_str(value)
    }
}
