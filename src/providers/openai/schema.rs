//! JSON schemas as OpenAI takes them (`normalize-openai-json-schema.ts`),
//! OAI-027.

use serde_json::{Map, Value};

use super::{capabilities::dropped, params::invalid};
use crate::Error;

/// A schema without what OpenAI refuses: a string `propertyNames` and a
/// `pattern` with a regex lookaround are removed and logged, another
/// `propertyNames` fails, and a one-item `allOf` of a `$ref` becomes the
/// `$ref`, expanded at the root.
pub(super) fn normalize(schema: &Value, model: &str) -> Result<Value, Error> {
    let mut removed = Removed::default();
    let normalized = node(schema, true, &mut removed)?;
    if removed.property_names {
        dropped(
            "propertyNames",
            model,
            "OpenAI does not take JSON Schema propertyNames, so it will not enforce property-name constraints",
        );
    }
    if removed.lookaround {
        dropped(
            "pattern",
            model,
            "OpenAI does not take regex lookaround in JSON Schema patterns, so it will not enforce that pattern",
        );
    }
    Ok(normalized)
}

#[derive(Default)]
struct Removed {
    property_names: bool,
    lookaround: bool,
}

fn definition(value: &Value, removed: &mut Removed) -> Result<Value, Error> {
    match value {
        Value::Object(_) => node(value, false, removed),
        other => Ok(other.clone()),
    }
}

fn record(value: &Value, removed: &mut Removed) -> Result<Value, Error> {
    let Some(map) = value.as_object() else {
        return Ok(value.clone());
    };
    let mut out = Map::new();
    for (key, schema) in map {
        out.insert(key.clone(), definition(schema, removed)?);
    }
    Ok(Value::Object(out))
}

fn node(schema: &Value, root: bool, removed: &mut Removed) -> Result<Value, Error> {
    let Some(object) = schema.as_object() else {
        return Ok(schema.clone());
    };
    let mut out = object.clone();
    if let Some(names) = out.remove("propertyNames").filter(|v| !v.is_null()) {
        if names.get("type").and_then(Value::as_str) != Some("string") {
            return Err(invalid(
                "OpenAI does not take a JSON Schema propertyNames that is not a string schema",
            ));
        }
        removed.property_names = true;
    }
    if out
        .get("pattern")
        .and_then(Value::as_str)
        .is_some_and(lookaround)
    {
        out.remove("pattern");
        removed.lookaround = true;
    }
    for key in ["properties", "patternProperties", "definitions", "$defs"] {
        if let Some(value) = out.get(key) {
            let value = record(value, removed)?;
            out.insert(key.into(), value);
        }
    }
    for key in [
        "additionalProperties",
        "additionalItems",
        "contains",
        "not",
        "if",
        "then",
        "else",
    ] {
        if let Some(value) = out.get(key) {
            let value = definition(value, removed)?;
            out.insert(key.into(), value);
        }
    }
    if let Some(items) = out.get("items") {
        let items = match items {
            Value::Array(list) => Value::Array(
                list.iter()
                    .map(|item| definition(item, removed))
                    .collect::<Result<_, _>>()?,
            ),
            other => definition(other, removed)?,
        };
        out.insert("items".into(), items);
    }
    for key in ["allOf", "anyOf", "oneOf"] {
        if let Some(Value::Array(list)) = out.get(key) {
            let list = list
                .iter()
                .map(|item| definition(item, removed))
                .collect::<Result<Vec<_>, _>>()?;
            out.insert(key.into(), Value::Array(list));
        }
    }
    if let Some(Value::Object(dependencies)) = out.get("dependencies") {
        let mut normalized = Map::new();
        for (key, dependency) in dependencies {
            let value = match dependency {
                Value::Array(_) => dependency.clone(),
                other => definition(other, removed)?,
            };
            normalized.insert(key.clone(), value);
        }
        out.insert("dependencies".into(), Value::Object(normalized));
    }
    let Some(reference) = singleton_reference(&out) else {
        return Ok(Value::Object(out));
    };
    let mut without = out.clone();
    without.remove("allOf");
    if !root {
        without.insert("$ref".into(), Value::String(reference));
        return Ok(Value::Object(without));
    }
    let Some(referenced) = local_reference(&reference, &out) else {
        return Ok(Value::Object(out));
    };
    let mut expanded = referenced;
    expanded.extend(without);
    Ok(Value::Object(expanded))
}

/// The `$ref` of an `allOf` that holds only `{"$ref"}`.
fn singleton_reference(schema: &Map<String, Value>) -> Option<String> {
    let [only] = schema.get("allOf")?.as_array()?.as_slice() else {
        return None;
    };
    let only = only.as_object()?;
    (only.len() == 1)
        .then(|| only.get("$ref")?.as_str().map(str::to_owned))
        .flatten()
}

/// The schema a local `#/definitions/<name>` or `#/$defs/<name>` names.
fn local_reference(reference: &str, schema: &Map<String, Value>) -> Option<Map<String, Value>> {
    let (keyword, name) = reference
        .strip_prefix("#/definitions/")
        .map(|name| ("definitions", name))
        .or_else(|| {
            reference
                .strip_prefix("#/$defs/")
                .map(|name| ("$defs", name))
        })?;
    let name = name.replace("~1", "/").replace("~0", "~");
    schema.get(keyword)?.get(&name)?.as_object().cloned()
}

/// Whether a pattern holds a regex lookaround, `(?=`, `(?!`, `(?<=` or
/// `(?<!`, outside a character class (`containsRegexLookaround`).
fn lookaround(pattern: &str) -> bool {
    let chars: Vec<char> = pattern.chars().collect();
    let mut escaped = false;
    let mut class = false;
    for (index, &c) in chars.iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' => escaped = true,
            '[' => class = true,
            ']' => class = false,
            '(' if !class && chars.get(index + 1) == Some(&'?') => match chars.get(index + 2) {
                Some('=' | '!') => return true,
                Some('<') if matches!(chars.get(index + 3), Some('=' | '!')) => return true,
                _ => {}
            },
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::lookaround;

    #[test]
    fn lookarounds_are_found_outside_character_classes() {
        assert!(lookaround("^(?=a)"));
        assert!(lookaround("(?<!b)c"));
        assert!(!lookaround("[(?=]"));
        assert!(!lookaround("\\(?=x"));
        assert!(!lookaround("(?:a)"));
    }
}
