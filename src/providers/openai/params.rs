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

//! OpenAI-specific parameters. Use the stable `openai` provider-parameter key.
use crate::{
    Error,
    models::params::*,
    types::tools::{ToolChoice, ToolChoiceMode},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenAIParams {
    pub store: Option<bool>,
}
impl OpenAIParams {
    pub const KEY: &str = "openai";
    pub fn apply(self, params: InferenceRequestParams) -> InferenceRequestParams {
        params.with_provider_param(Self::KEY, json!({"store":self.store}))
    }
}
pub(super) fn invalid(message: impl Into<String>) -> Error {
    Error::Validation(message.into())
}
fn put<T: Serialize>(
    out: &mut Map<String, Value>,
    key: &str,
    value: &ProviderValue<T>,
) -> Result<(), Error> {
    if let ProviderValue::Value(v) = value {
        out.insert(key.into(), serde_json::to_value(v)?);
    }
    Ok(())
}
fn reject<T>(value: &ProviderValue<Option<T>>, name: &str) -> Result<(), Error> {
    if matches!(value, ProviderValue::Value(Some(_))) {
        return Err(invalid(format!("OpenAI does not support {name}")));
    }
    Ok(())
}
pub(super) fn tool_choice(choice: &ToolChoice, responses: bool) -> Result<Value, Error> {
    let named = |name: String| {
        if responses {
            json!({"type":"function","name":name})
        } else {
            json!({"type":"function","function":{"name":name}})
        }
    };
    Ok(match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(t) => named(t.to_string()),
        ToolChoice::Selection(s) => {
            if s.mode == ToolChoiceMode::None {
                return Err(invalid(
                    "OpenAI allowed tool selection does not support none",
                ));
            }
            let tools: Vec<_> = s.tools.iter().map(|t| named(t.to_string())).collect();
            if responses {
                json!({"type":"allowed_tools","mode":s.mode,"tools":tools})
            } else {
                json!({"type":"allowed_tools","allowed_tools":{"mode":s.mode,"tools":tools}})
            }
        }
    })
}
pub(super) fn build(
    p: &InferenceRequestParams,
    responses: bool,
) -> Result<Map<String, Value>, Error> {
    let mut out = Map::new();
    if let ProviderValue::Value(s) = &p.sampling {
        if let Some(t) = &s.temperature {
            put(&mut out, "temperature", &t.temperature)?;
        }
        if let Some(t) = &s.top_p {
            put(&mut out, "top_p", &t.top_p)?;
        }
        if let Some(t) = &s.top_k {
            reject(&t.top_k, "top_k")?;
        }
        if let Some(t) = &s.min_p {
            reject(&t.min_p, "min_p")?;
        }
        if let Some(t) = &s.seed
            && let ProviderValue::Value(Some(Seed::Value(seed))) = &t.seed
            && *seed != -1
        {
            if responses {
                return Err(invalid("OpenAI responses does not support seed"));
            }
            out.insert("seed".into(), json!(seed));
        }
        if let Some(r) = &s.repetition_penalty {
            put(&mut out, "frequency_penalty", &r.frequency_penalty)?;
            put(&mut out, "presence_penalty", &r.presence_penalty)?;
            reject(&r.repetition_penalty, "repetition_penalty")?;
            reject(&r.consideration_window, "consideration_window")?;
        }
    }
    let mut reasoning = Map::new();
    if let ProviderValue::Value(r) = &p.reasoning
        && let ProviderValue::Value(effort) = &r.effort
    {
        let effort = json!(effort.as_deref().unwrap_or("none"));
        if responses {
            reasoning.insert("effort".into(), effort);
        } else {
            out.insert("reasoning_effort".into(), effort);
        }
    }
    if let Some(o) = &p.output {
        if responses {
            put(&mut reasoning, "summary", &o.reasoning_summary)?;
        } else {
            reject(
                &o.reasoning_summary,
                "reasoning summary in chat completions",
            )?;
        }
        if let Some(max) = o.max_tokens {
            out.insert(
                if responses {
                    "max_output_tokens"
                } else {
                    "max_completion_tokens"
                }
                .into(),
                json!(max),
            );
        }
        if let Some(include) = &o.include {
            if !responses {
                return Err(invalid(
                    "OpenAI chat completions does not support output include",
                ));
            }
            out.insert("include".into(), json!(include));
        }
        if responses {
            let mut text = Map::new();
            put(&mut text, "verbosity", &o.text_verbosity)?;
            if !text.is_empty() {
                out.insert("text".into(), text.into());
            }
        } else {
            reject(&o.text_verbosity, "text verbosity in chat completions")?;
        }
    }
    if !reasoning.is_empty() {
        out.insert("reasoning".into(), reasoning.into());
    }
    if let Some(t) = &p.tool_calling {
        if responses {
            put(&mut out, "max_tool_calls", &t.max_tool_calls)?;
        } else {
            reject(&t.max_tool_calls, "max_tool_calls in chat completions")?;
        }
        put(&mut out, "parallel_tool_calls", &t.parallel_tool_calls)?;
        out.insert(
            "tool_choice".into(),
            tool_choice(&t.tool_choice, responses)?,
        );
    }
    if let Some(s) = &p.provider_service {
        put(&mut out, "service_tier", &s.service_tier)?;
    }
    if let Some(s) = &p.safety_identifier {
        out.insert("safety_identifier".into(), json!(s));
    }
    if let Some(m) = &p.metadata {
        out.insert("metadata".into(), json!(m));
    }
    if let Some(provider) = p
        .provider_params
        .as_ref()
        .and_then(|p| p.get(OpenAIParams::KEY))
    {
        let options: OpenAIParams = serde_json::from_value(provider.clone())
            .map_err(|_| invalid("openai provider_params must be an OpenAIParams object"))?;
        if let Some(store) = options.store {
            out.insert("store".into(), json!(store));
        }
    }
    if let Some(threshold) = p.context_management.as_ref().and_then(|m| m.compaction) {
        if !responses {
            return Err(invalid(
                "OpenAI chat completions does not support context management",
            ));
        }
        out.insert(
            "context_management".into(),
            json!([{"type":"compaction","compact_threshold":threshold.0}]),
        );
    }
    if let Some(c) = &p.cache {
        if let Some(key) = &c.key {
            out.insert("prompt_cache_key".into(), json!(key));
        }
        put(&mut out, "prompt_cache_retention", &c.retention)?;
    }
    Ok(out)
}
