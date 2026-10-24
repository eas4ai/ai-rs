//! The output schema Anthropic's constrained decoder takes, as
//! `sanitize-json-schema.ts` writes it (ANT-005): the keywords it keeps,
//! `oneOf` as `anyOf`, closed objects, the string formats it lists, and the
//! other constraints as description text.
use crate::Error;
use serde_json::{Map, Value, json};

/// The string formats Anthropic takes.
const FORMATS: &[&str] = &[
    "date-time",
    "time",
    "date",
    "duration",
    "email",
    "hostname",
    "uri",
    "ipv4",
    "ipv6",
    "uuid",
];

/// The constraints written into the description, in this order.
const CONSTRAINTS: &[&str] = &[
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "not",
];

/// The deepest schema nesting the crate converts.
const MAX_DEPTH: usize = 64;

/// Sanitize an output schema for `output_config.format`.
pub(crate) fn sanitize(schema: &Value) -> Result<Value, Error> {
    definition(schema, 0)
}

fn definition(value: &Value, depth: usize) -> Result<Value, Error> {
    if depth > MAX_DEPTH {
        return Err(Error::Validation(format!(
            "output schema is nested more than {MAX_DEPTH} levels"
        )));
    }
    match value {
        Value::Object(schema) => object(schema, depth),
        other => Ok(other.clone()),
    }
}

fn object(schema: &Map<String, Value>, depth: usize) -> Result<Value, Error> {
    if let Some(reference) = schema.get("$ref").filter(|r| !r.is_null()) {
        return Ok(json!({"$ref": reference}));
    }
    let mut out = Map::new();
    let present = |key: &str| schema.get(key).filter(|v| !v.is_null());
    for key in ["$schema", "$id", "title", "description"] {
        if let Some(value) = present(key) {
            out.insert(key.into(), value.clone());
        }
    }
    // `default` and `const` are kept even when `null`.
    for key in ["default", "const"] {
        if let Some(value) = schema.get(key) {
            out.insert(key.into(), value.clone());
        }
    }
    for key in ["enum", "type"] {
        if let Some(value) = present(key) {
            out.insert(key.into(), value.clone());
        }
    }
    let list = |items: &Value| -> Result<Value, Error> {
        Ok(Value::Array(
            items
                .as_array()
                .into_iter()
                .flatten()
                .map(|item| definition(item, depth + 1))
                .collect::<Result<_, _>>()?,
        ))
    };
    if let Some(any) = present("anyOf") {
        out.insert("anyOf".into(), list(any)?);
    } else if let Some(one) = present("oneOf") {
        out.insert("anyOf".into(), list(one)?);
    }
    if let Some(all) = present("allOf") {
        out.insert("allOf".into(), list(all)?);
    }
    let map = |defs: &Value| -> Result<Value, Error> {
        Ok(Value::Object(
            defs.as_object()
                .into_iter()
                .flatten()
                .map(|(name, d)| Ok((name.clone(), definition(d, depth + 1)?)))
                .collect::<Result<_, Error>>()?,
        ))
    };
    for key in ["definitions", "$defs"] {
        if let Some(defs) = present(key) {
            out.insert(key.into(), map(defs)?);
        }
    }
    if schema.get("type") == Some(&json!("object")) || present("properties").is_some() {
        if let Some(properties) = present("properties") {
            out.insert("properties".into(), map(properties)?);
        }
        out.insert("additionalProperties".into(), json!(false));
        if let Some(required) = present("required") {
            out.insert("required".into(), required.clone());
        }
    }
    if let Some(items) = present("items") {
        out.insert(
            "items".into(),
            if items.is_array() {
                list(items)?
            } else {
                definition(items, depth + 1)?
            },
        );
    }
    let format = schema.get("format").and_then(Value::as_str);
    if let Some(format) = format.filter(|f| FORMATS.contains(f)) {
        out.insert("format".into(), json!(format));
    }
    if let Some(text) = constraints(schema, format) {
        let description = match out.get("description").and_then(Value::as_str) {
            Some(description) => format!("{description}\n{text}"),
            None => text,
        };
        out.insert("description".into(), json!(description));
    }
    Ok(Value::Object(out))
}

/// The constraints Anthropic does not take, as `minimum: 1; pattern: x.`.
fn constraints(schema: &Map<String, Value>, format: Option<&str>) -> Option<String> {
    let mut parts: Vec<String> = CONSTRAINTS
        .iter()
        .filter_map(|key| {
            let value = schema.get(*key)?;
            if value.is_null() || *value == Value::Bool(false) {
                return None;
            }
            Some(format!("{}: {}", name(key), text(value)))
        })
        .collect();
    if let Some(format) = format.filter(|f| !FORMATS.contains(f)) {
        parts.push(format!("format: {format}"));
    }
    (!parts.is_empty()).then(|| format!("{}.", parts.join("; ")))
}

/// A constraint's name with each capital letter as a space and the lower
/// case letter: `minLength` as `min length`.
fn name(key: &str) -> String {
    let mut out = String::new();
    for c in key.chars() {
        if c.is_ascii_uppercase() {
            out.push(' ');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::sanitize;
    use serde_json::json;

    #[test]
    fn schemas_are_sanitized_as_typescript_does() {
        let schema = json!({
            "type": "object",
            "properties": {
                "n": {"type": "integer", "minimum": 1, "maximum": 9, "description": "Count"},
                "s": {"type": "string", "pattern": "^a", "format": "email"},
                "r": {"type": "string", "format": "regex"},
                "u": {"oneOf": [{"type": "string"}, {"type": "null"}]},
                "a": {"type": "array", "items": {"type": "object", "properties": {}}, "uniqueItems": true}
            },
            "required": ["n"],
            "additionalProperties": true
        });
        assert_eq!(
            sanitize(&schema).unwrap(),
            json!({
                "type": "object",
                "properties": {
                    "n": {"description": "Count\nminimum: 1; maximum: 9.", "type": "integer"},
                    "s": {"type": "string", "format": "email", "description": "pattern: ^a."},
                    "r": {"type": "string", "description": "format: regex."},
                    "u": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "a": {"type": "array", "items": {"type": "object", "properties": {}, "additionalProperties": false},
                          "description": "unique items: true."}
                },
                "additionalProperties": false,
                "required": ["n"]
            })
        );
    }
}
