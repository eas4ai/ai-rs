//! AI SDK v6 UI messages. Known part payloads retain provider and application fields.

use crate::types::messages::generate_id;
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Browser message with supported parts in their canonical camelCase wire shape.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct UIMessage {
    pub id: String,
    pub role: String,
    pub metadata: Value,
    pub parts: Vec<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl<'de> Deserialize<'de> for UIMessage {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            #[serde(default = "message_id")]
            id: String,
            role: String,
            #[serde(default)]
            metadata: Value,
            #[serde(default)]
            parts: Vec<Value>,
            #[serde(flatten)]
            extra: BTreeMap<String, Value>,
        }
        let wire = Wire::deserialize(deserializer)?;
        if !matches!(wire.role.as_str(), "user" | "assistant" | "system") {
            return Err(de::Error::custom(
                "UI role must be user, assistant, or system",
            ));
        }
        let parts = wire
            .parts
            .into_iter()
            .filter_map(|part| match parse_ui_part(part) {
                Ok(None) => None,
                result => Some(result),
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(de::Error::custom)?
            .into_iter()
            .flatten()
            .collect();
        Ok(Self {
            id: wire.id,
            role: wire.role,
            metadata: wire.metadata,
            parts,
            extra: wire.extra,
        })
    }
}

fn message_id() -> String {
    generate_id("msg")
}

fn alias(object: &mut Map<String, Value>, snake: &str, camel: &str) {
    if !object.contains_key(camel)
        && let Some(value) = object.remove(snake)
    {
        object.insert(camel.to_owned(), value);
    }
}

fn required_string(object: &Map<String, Value>, field: &str) -> Result<(), String> {
    if object.get(field).is_some_and(Value::is_string) {
        Ok(())
    } else {
        Err(format!("UI part requires string {field}"))
    }
}

fn optional_fields(object: &mut Map<String, Value>, fields: &[&str]) {
    for field in fields {
        object.entry((*field).to_owned()).or_insert(Value::Null);
    }
}

/// Parse a supported part, accepting Python field names and browser aliases.
/// Unsupported types are omitted, as in the Python adapter.
pub fn parse_ui_part(part: Value) -> Result<Option<Value>, String> {
    let mut object = part
        .as_object()
        .cloned()
        .ok_or("UI parts must be objects")?;
    let Some(kind) = object
        .get("type")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return Ok(None);
    };
    let supported = matches!(
        kind.as_str(),
        "text"
            | "reasoning"
            | "custom"
            | "tool-invocation"
            | "step-start"
            | "file"
            | "reasoning-file"
            | "source-url"
            | "source-document"
            | "dynamic-tool"
    ) || kind.starts_with("tool-")
        || kind.starts_with("data-");
    if !supported {
        return Ok(None);
    }
    let (required, optional): (&[&str], &[&str]) = match kind.as_str() {
        "text" | "reasoning" => (&["text"], &["state", "providerMetadata"]),
        "custom" => (&["kind"], &["providerMetadata"]),
        "tool-invocation" => (
            &["toolInvocationId", "toolName"],
            &["args", "state", "result", "providerExecuted"],
        ),
        "file" => (
            &["mediaType", "url"],
            &["filename", "providerReference", "providerMetadata"],
        ),
        "reasoning-file" => (&["mediaType", "url"], &["providerMetadata"]),
        "source-url" => (&["sourceId", "url"], &["title", "providerMetadata"]),
        "source-document" => (
            &["sourceId", "mediaType", "title"],
            &["filename", "providerMetadata"],
        ),
        "step-start" => (&[], &[]),
        t if t == "dynamic-tool" || t.starts_with("tool-") => {
            let required: &[&str] = if t == "dynamic-tool" {
                &["toolName", "toolCallId", "state"]
            } else {
                &["toolCallId", "state"]
            };
            (
                required,
                &[
                    "input",
                    "output",
                    "rawInput",
                    "errorText",
                    "approval",
                    "providerExecuted",
                    "callProviderMetadata",
                    "resultProviderMetadata",
                    "toolMetadata",
                    "preliminary",
                    "title",
                ],
            )
        }
        _ => (&[], &["id", "transient"]),
    };
    for (snake, camel) in [
        ("provider_metadata", "providerMetadata"),
        ("tool_invocation_id", "toolInvocationId"),
        ("tool_name", "toolName"),
        ("tool_call_id", "toolCallId"),
        ("provider_executed", "providerExecuted"),
        ("raw_input", "rawInput"),
        ("error_text", "errorText"),
        ("call_provider_metadata", "callProviderMetadata"),
        ("result_provider_metadata", "resultProviderMetadata"),
        ("tool_metadata", "toolMetadata"),
        ("media_type", "mediaType"),
        ("provider_reference", "providerReference"),
        ("source_id", "sourceId"),
    ] {
        if required.contains(&camel) || optional.contains(&camel) {
            alias(&mut object, snake, camel);
        }
    }
    for field in required {
        required_string(&object, field)?;
    }
    if kind == "tool-invocation" {
        object
            .entry("args")
            .or_insert_with(|| Value::Object(Map::new()));
        if !object["args"].is_object() {
            return Err("UI invocation args must be an object".into());
        }
        object
            .entry("state")
            .or_insert_with(|| Value::String("input-available".into()));
    }
    optional_fields(&mut object, optional);
    if matches!(kind.as_str(), "text" | "reasoning")
        && !object["state"].is_null()
        && !matches!(object["state"].as_str(), Some("streaming" | "done"))
    {
        return Err("invalid UI text state".into());
    }
    if kind == "dynamic-tool" || kind.starts_with("tool-") {
        validate_tool_state(&object)?;
        if kind != "tool-invocation" {
            if kind != "dynamic-tool"
                && !object["input"].is_null()
                && !object["input"].is_string()
                && !object["input"].is_object()
            {
                return Err("UI tool input must be an object, string, or null".into());
            }
            if !object["approval"].is_null() {
                let approval = object
                    .get_mut("approval")
                    .and_then(Value::as_object_mut)
                    .ok_or("UI approval must be an object")?;
                required_string(approval, "id")?;
                alias(approval, "is_automatic", "isAutomatic");
                optional_fields(approval, &["approved", "reason", "isAutomatic"]);
                optional_bool(approval, "approved")?;
                optional_bool(approval, "isAutomatic")?;
                optional_string(approval, "reason")?;
            }
        }
    }
    if kind.starts_with("data-") && !object.contains_key("data") {
        return Err("UI data part requires data".into());
    }
    for field in optional.iter().copied() {
        match field {
            "providerMetadata"
            | "callProviderMetadata"
            | "resultProviderMetadata"
            | "toolMetadata"
            | "providerReference"
                if !object[field].is_null() && !object[field].is_object() =>
            {
                return Err(format!("UI {field} must be an object"));
            }
            "providerExecuted" | "preliminary" | "transient" => optional_bool(&mut object, field)?,
            "errorText" | "filename" | "title" | "id" => optional_string(&object, field)?,
            _ => {}
        }
    }
    Ok(Some(Value::Object(object)))
}

fn validate_tool_state(object: &Map<String, Value>) -> Result<(), String> {
    if matches!(
        object.get("state").and_then(Value::as_str),
        Some(
            "input-streaming"
                | "input-available"
                | "approval-requested"
                | "approval-responded"
                | "output-available"
                | "output-error"
                | "output-denied"
        )
    ) {
        Ok(())
    } else {
        Err("invalid UI tool state".into())
    }
}

fn optional_string(object: &Map<String, Value>, field: &str) -> Result<(), String> {
    if object
        .get(field)
        .is_none_or(|v| v.is_null() || v.is_string())
    {
        Ok(())
    } else {
        Err(format!("UI {field} must be a string"))
    }
}

fn optional_bool(object: &mut Map<String, Value>, field: &str) -> Result<(), String> {
    let Some(value) = object.get_mut(field) else {
        return Ok(());
    };
    if value.is_null() || value.is_boolean() {
        return Ok(());
    }
    let parsed = match value {
        Value::String(text) => match text.to_ascii_lowercase().as_str() {
            "1" | "true" | "t" | "yes" | "y" | "on" => Some(true),
            "0" | "false" | "f" | "no" | "n" | "off" => Some(false),
            _ => None,
        },
        Value::Number(number) => match number.as_f64() {
            Some(0.0) => Some(false),
            Some(1.0) => Some(true),
            _ => None,
        },
        _ => None,
    };
    if let Some(parsed) = parsed {
        *value = Value::Bool(parsed);
        Ok(())
    } else {
        Err(format!("UI {field} must be boolean"))
    }
}
