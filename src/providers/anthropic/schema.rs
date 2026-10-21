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

use crate::Error;
use serde_json::{Map, Value, json};
// The Python SDK lowers Pydantic schemas to the provider's supported subset.
// Schemars also emits type arrays, which carry the same union as anyOf.
pub(super) fn transform(schema: &Value, depth: usize) -> Result<Value, Error> {
    if depth > 128 {
        return Err(Error::Validation(
            "Anthropic output schema exceeds 128 levels".into(),
        ));
    }
    let mut source = schema
        .as_object()
        .cloned()
        .ok_or_else(|| Error::Validation("Anthropic output schema must be an object".into()))?;
    let mut out = Map::new();
    if let Some(defs) = source.remove("$defs") {
        out.insert("$defs".into(), map_schemas(&defs, depth + 1)?);
    }
    if let Some(reference) = source.remove("$ref") {
        out.insert("$ref".into(), reference);
        return Ok(Value::Object(out));
    }
    let ty = source.remove("type");
    let alternatives = source.remove("anyOf").or_else(|| source.remove("oneOf"));
    if let Some(values) = alternatives {
        out.insert("anyOf".into(), transform_list(&values, depth + 1)?);
    } else if let Some(values) = source.remove("allOf") {
        out.insert("allOf".into(), transform_list(&values, depth + 1)?);
    } else if let Some(Value::Array(types)) = &ty {
        let variants = types
            .iter()
            .map(|ty| {
                let mut branch = source.clone();
                branch.insert("type".into(), ty.clone());
                transform(&Value::Object(branch), depth + 1)
            })
            .collect::<Result<Vec<_>, _>>()?;
        out.insert("anyOf".into(), json!(variants));
        return Ok(Value::Object(out));
    } else if let Some(ty) = &ty {
        out.insert("type".into(), ty.clone());
    } else {
        return Err(Error::Validation(
            "Anthropic output schema needs type, anyOf, oneOf, allOf, or $ref".into(),
        ));
    }
    for key in ["enum", "description", "title"] {
        if let Some(v) = source.remove(key) {
            out.insert(key.into(), v);
        }
    }
    match ty.as_ref().and_then(Value::as_str) {
        Some("object") => {
            let properties = source.remove("properties").unwrap_or(json!({}));
            out.insert("properties".into(), map_schemas(&properties, depth + 1)?);
            source.remove("additionalProperties");
            out.insert("additionalProperties".into(), json!(false));
            if let Some(required) = source.remove("required") {
                out.insert("required".into(), required);
            }
        }
        Some("array") => {
            if let Some(items) = source.remove("items") {
                out.insert("items".into(), transform(&items, depth + 1)?);
            }
            if let Some(min) = source.remove("minItems") {
                if min == 0 || min == 1 {
                    out.insert("minItems".into(), min);
                } else {
                    source.insert("minItems".into(), min);
                }
            }
        }
        Some("string") => {
            if let Some(format) = source.remove("format") {
                if matches!(
                    format.as_str(),
                    Some(
                        "date-time"
                            | "time"
                            | "date"
                            | "duration"
                            | "email"
                            | "hostname"
                            | "uri"
                            | "ipv4"
                            | "ipv6"
                            | "uuid"
                    )
                ) {
                    out.insert("format".into(), format);
                } else {
                    source.insert("format".into(), format);
                }
            }
        }
        Some("boolean" | "integer" | "number" | "null") | None => {}
        _ => {
            return Err(Error::Validation(
                "unsupported Anthropic output schema type".into(),
            ));
        }
    }
    // Keep unsupported constraints as guidance, matching the Python SDK.
    if !source.is_empty() {
        let extra = source
            .iter()
            .map(|(k, v)| format!("{k}: {}", description_value(v)))
            .collect::<Vec<_>>()
            .join(", ");
        let prior = out
            .get("description")
            .and_then(Value::as_str)
            .map(|s| format!("{s}\n\n"))
            .unwrap_or_default();
        out.insert("description".into(), json!(format!("{prior}{{{extra}}}")));
    }
    Ok(Value::Object(out))
}
fn description_value(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}
fn map_schemas(value: &Value, depth: usize) -> Result<Value, Error> {
    let values = value.as_object().ok_or_else(|| {
        Error::Validation("Anthropic schema definitions and properties must be objects".into())
    })?;
    values
        .iter()
        .map(|(k, v)| Ok((k.clone(), transform(v, depth)?)))
        .collect::<Result<Map<_, _>, _>>()
        .map(Value::Object)
}
fn transform_list(value: &Value, depth: usize) -> Result<Value, Error> {
    let values = value
        .as_array()
        .ok_or_else(|| Error::Validation("Anthropic schema alternatives must be arrays".into()))?;
    values
        .iter()
        .map(|v| transform(v, depth))
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}
