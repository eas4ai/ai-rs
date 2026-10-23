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

//! Function call arguments that Vertex AI streams as `partialArgs`, built
//! into JSON text deltas that join to the JSON of the final arguments
//! object, keys in arrival order (`google-json-accumulator.ts`).

use serde_json::{Map, Number, Value};

#[derive(Clone, Debug, PartialEq)]
enum Segment {
    Key(String),
    Index(usize),
}

#[derive(Debug)]
struct Open {
    segment: Option<Segment>,
    is_array: bool,
    children: usize,
}

/// Accumulates one function call's `partialArgs`.
#[derive(Debug, Default)]
pub(crate) struct Accumulator {
    args: Map<String, Value>,
    text: String,
    stack: Vec<Open>,
    string_open: bool,
}

impl Accumulator {
    /// Take a chunk's `partialArgs` and return the text to append.
    pub(crate) fn feed(&mut self, partial_args: &[Value]) -> String {
        let mut delta = String::new();
        for arg in partial_args {
            let Some(path) = arg.get("jsonPath").and_then(Value::as_str) else {
                continue;
            };
            let raw = path.strip_prefix("$.").unwrap_or(path);
            if raw.is_empty() || raw == "$" {
                continue;
            }
            let segments = parse_path(raw);
            if segments.is_empty() {
                continue;
            }
            let string = arg.get("stringValue").and_then(Value::as_str);
            if let Some(string) = string
                && let Some(Value::String(existing)) = get(&self.args, &segments)
            {
                let joined = format!("{existing}{string}");
                if !set(&mut self.args, &segments, Value::String(joined)) {
                    continue;
                }
                let escaped = Value::String(string.to_owned()).to_string();
                delta.push_str(&escaped[1..escaped.len() - 1]);
                continue;
            }
            let Some((value, json)) = resolve(arg) else {
                continue;
            };
            if !set(&mut self.args, &segments, value) {
                continue;
            }
            let will_continue = arg.get("willContinue") == Some(&Value::Bool(true));
            delta.push_str(&self.navigate(&segments, string.is_some() && will_continue, &json));
        }
        self.text.push_str(&delta);
        delta
    }

    /// The final arguments as JSON text and the text still to append.
    pub(crate) fn finish(&self) -> (String, String) {
        let full = Value::Object(self.args.clone()).to_string();
        let closing = full.get(self.text.len()..).unwrap_or_default().to_owned();
        (full, closing)
    }

    fn navigate(&mut self, segments: &[Segment], open_string: bool, json: &str) -> String {
        let mut out = String::new();
        if self.string_open {
            out.push('"');
            self.string_open = false;
        }
        if self.stack.is_empty() {
            self.stack.push(Open {
                segment: None,
                is_array: false,
                children: 0,
            });
            out.push('{');
        }
        let (leaf, containers) = segments.split_last().expect("segments are not empty");
        // Keep the open containers shared with the new path; close the rest.
        let shared = self
            .stack
            .iter()
            .skip(1)
            .zip(containers)
            .take_while(|(open, segment)| open.segment.as_ref() == Some(*segment))
            .count();
        while self.stack.len() > shared + 1 {
            let open = self.stack.pop().expect("stack is deeper than the root");
            out.push(if open.is_array { ']' } else { '}' });
        }
        for index in (self.stack.len() - 1)..containers.len() {
            let segment = &containers[index];
            let parent = self.stack.last_mut().expect("the root is open");
            if parent.children > 0 {
                out.push(',');
            }
            parent.children += 1;
            if let Segment::Key(key) = segment {
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
            }
            let child = containers.get(index + 1).unwrap_or(leaf);
            let is_array = matches!(child, Segment::Index(_));
            out.push(if is_array { '[' } else { '{' });
            self.stack.push(Open {
                segment: Some(segment.clone()),
                is_array,
                children: 0,
            });
        }
        let container = self.stack.last_mut().expect("the root is open");
        if container.children > 0 {
            out.push(',');
        }
        container.children += 1;
        if let Segment::Key(key) = leaf {
            out.push_str(&Value::String(key.clone()).to_string());
            out.push(':');
        }
        if open_string {
            out.push_str(&json[..json.len() - 1]);
            self.string_open = true;
        } else {
            out.push_str(json);
        }
        out
    }
}

/// `recipe.ingredients[0].name` as `["recipe", "ingredients", 0, "name"]`.
fn parse_path(raw: &str) -> Vec<Segment> {
    let mut segments = Vec::new();
    for part in raw.split('.') {
        match part.find('[') {
            None => segments.push(Segment::Key(part.to_owned())),
            Some(bracket) => {
                if bracket > 0 {
                    segments.push(Segment::Key(part[..bracket].to_owned()));
                }
                let mut rest = &part[bracket..];
                while let Some(open) = rest.find('[') {
                    let Some(close) = rest[open..].find(']') else {
                        break;
                    };
                    if let Ok(index) = rest[open + 1..open + close].parse() {
                        segments.push(Segment::Index(index));
                    }
                    rest = &rest[open + close + 1..];
                }
            }
        }
    }
    segments
}

/// The first value a partial arg holds, with its JSON text.
fn resolve(arg: &Value) -> Option<(Value, String)> {
    let value = arg
        .get("stringValue")
        .filter(|v| v.is_string())
        .or_else(|| arg.get("numberValue").filter(|v| v.is_number()))
        .or_else(|| arg.get("boolValue").filter(|v| v.is_boolean()))
        .cloned();
    match value {
        Some(Value::Number(number)) => {
            // Integral numbers print without a fraction, as in JSON.stringify.
            let number = number
                .as_f64()
                .filter(|f| f.fract() == 0.0 && f.abs() < 9_007_199_254_740_992.0)
                .map_or(number, |f| Number::from(f as i64));
            let value = Value::Number(number);
            let json = value.to_string();
            Some((value, json))
        }
        Some(value) => {
            let json = value.to_string();
            Some((value, json))
        }
        None if arg.get("nullValue").is_some() => Some((Value::Null, "null".into())),
        None => None,
    }
}

fn get<'a>(root: &'a Map<String, Value>, segments: &[Segment]) -> Option<&'a Value> {
    let (first, rest) = segments.split_first()?;
    let Segment::Key(key) = first else {
        return None;
    };
    let mut current = root.get(key)?;
    for segment in rest {
        current = match (segment, current) {
            (Segment::Key(key), Value::Object(map)) => map.get(key)?,
            (Segment::Index(index), Value::Array(items)) => items.get(*index)?,
            _ => return None,
        };
    }
    Some(current)
}

/// Write `value` at the path. Returns false, writing nothing, for a path the
/// arguments cannot take: one that does not start with a key, or an array
/// index past the end of its array, which would leave holes. Streamed
/// arguments fill arrays in order, so this also bounds what a response can
/// make the accumulator allocate.
fn set(root: &mut Map<String, Value>, segments: &[Segment], value: Value) -> bool {
    let Some((Segment::Key(first), rest)) = segments.split_first() else {
        return false;
    };
    if !fits(root.get(first), rest) {
        return false;
    }
    if rest.is_empty() {
        root.insert(first.clone(), value);
        return true;
    }
    let slot = root.entry(first.clone()).or_insert(Value::Null);
    set_in(slot, rest, value);
    true
}

/// Whether each array index of the path is at most its array's length.
fn fits(slot: Option<&Value>, segments: &[Segment]) -> bool {
    let Some((segment, rest)) = segments.split_first() else {
        return true;
    };
    match segment {
        Segment::Key(key) => {
            let next = slot.and_then(Value::as_object).and_then(|map| map.get(key));
            fits(next, rest)
        }
        Segment::Index(index) => {
            let items = slot.and_then(Value::as_array);
            let len = items.map_or(0, Vec::len);
            *index <= len && fits(items.and_then(|items| items.get(*index)), rest)
        }
    }
}

fn set_in(slot: &mut Value, segments: &[Segment], value: Value) {
    let (segment, rest) = segments.split_first().expect("segments are not empty");
    match segment {
        Segment::Key(key) => {
            if !slot.is_object() {
                *slot = Value::Object(Map::new());
            }
            let map = slot.as_object_mut().expect("slot is an object");
            if rest.is_empty() {
                map.insert(key.clone(), value);
            } else {
                set_in(map.entry(key.clone()).or_insert(Value::Null), rest, value);
            }
        }
        Segment::Index(index) => {
            if !slot.is_array() {
                *slot = Value::Array(Vec::new());
            }
            let items = slot.as_array_mut().expect("slot is an array");
            if items.len() == *index {
                items.push(Value::Null);
            }
            if rest.is_empty() {
                items[*index] = value;
            } else {
                set_in(&mut items[*index], rest, value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deltas_join_to_the_final_arguments() {
        let mut accumulator = Accumulator::default();
        let mut text = String::new();
        text += &accumulator
            .feed(&[json!({"jsonPath": "$.location", "stringValue": "Bos", "willContinue": true})]);
        text += &accumulator.feed(&[json!({"jsonPath": "$.location", "stringValue": "ton"})]);
        text += &accumulator.feed(&[json!({"jsonPath": "$.units", "stringValue": "metric"})]);
        text += &accumulator
            .feed(&[json!({"jsonPath": "$.recipe.ingredients[0].name", "stringValue": "Noodles"})]);
        text += &accumulator.feed(&[json!({"jsonPath": "$.count", "numberValue": 2})]);
        let (full, closing) = accumulator.finish();
        text += &closing;
        assert_eq!(text, full);
        assert_eq!(
            serde_json::from_str::<Value>(&full).unwrap(),
            json!({"location": "Boston", "units": "metric", "recipe": {"ingredients": [{"name": "Noodles"}]}, "count": 2})
        );
    }

    #[test]
    fn an_index_past_the_end_of_its_array_is_skipped() {
        let mut accumulator = Accumulator::default();
        let delta =
            accumulator.feed(&[json!({"jsonPath": "$.items[999999999999]", "stringValue": "x"})]);
        assert_eq!(delta, "");
        accumulator.feed(&[json!({"jsonPath": "$.items[0]", "stringValue": "a"})]);
        accumulator.feed(&[json!({"jsonPath": "$.items[1]", "stringValue": "b"})]);
        assert_eq!(accumulator.finish().0, r#"{"items":["a","b"]}"#);
    }
}
