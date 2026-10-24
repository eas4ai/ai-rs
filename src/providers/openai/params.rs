//! OpenAI-specific parameters. Use the stable `openai` provider-parameter key.
use super::protocol::Wire;
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
    /// Merge these options into the call's `openai` provider parameters,
    /// keeping the options already there. With `store` `None` the
    /// parameters stay as they are.
    pub fn apply(self, mut params: InferenceRequestParams) -> InferenceRequestParams {
        if let Some(store) = self.store {
            let options = params
                .provider_params
                .get_or_insert_with(Default::default)
                .entry(Self::KEY.to_owned())
                .or_insert_with(|| json!({}));
            // Another value is not OpenAI options, and the call refuses it.
            if let Value::Object(options) = options {
                options.insert("store".into(), json!(store));
            }
        }
        params
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
fn reject<T>(label: &str, value: &ProviderValue<Option<T>>, name: &str) -> Result<(), Error> {
    if matches!(value, ProviderValue::Value(Some(_))) {
        return Err(invalid(format!("{label} does not support {name}")));
    }
    Ok(())
}
/// The value of one OpenAI option in the call's provider parameters.
fn option<'a>(p: &'a InferenceRequestParams, wire: Wire, name: &str) -> Option<&'a Value> {
    let key = match wire {
        Wire::Chat(profile) => profile.openai_params.then_some(OpenAIParams::KEY),
        Wire::Responses(profile) => profile.params_key,
    };
    key.and_then(|key| p.provider_params.as_ref()?.get(key))?
        .get(name)
}
/// Whether the call's provider parameters set `store` false: the history and
/// the stream reader follow it (OAI-002).
pub(super) fn unstored(p: &InferenceRequestParams, wire: Wire) -> bool {
    option(p, wire, "store").and_then(Value::as_bool) == Some(false)
}
/// Whether the call asks for logprobs, `true` or a count: the stream reader
/// keeps the text deltas' logprobs only then (OAI-013).
pub(super) fn asks_logprobs(p: &InferenceRequestParams, wire: Wire) -> bool {
    option(p, wire, "logprobs").is_some_and(|v| v.as_bool() == Some(true) || v.is_number())
}
pub(super) fn tool_choice(choice: &ToolChoice, wire: Wire) -> Result<Value, Error> {
    let responses = matches!(wire, Wire::Responses(_));
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
                return Err(invalid(format!(
                    "{} allowed tool selection does not support none",
                    wire.api().label
                )));
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
pub(super) fn build(p: &InferenceRequestParams, wire: Wire) -> Result<Map<String, Value>, Error> {
    let responses = matches!(wire, Wire::Responses(_));
    let chat = match wire {
        Wire::Chat(profile) => Some(profile),
        Wire::Responses(_) => None,
    };
    let label = wire.api().label;
    let rejects_penalties_and_seed = chat.is_some_and(|c| c.rejects_penalties_and_seed);
    // xAI's Responses API takes top-k, min-p and seeds.
    let sampling_extras = wire.responses_dialect() == Some(super::profile::ResponsesDialect::Xai);
    let mut out = Map::new();
    if let ProviderValue::Value(s) = &p.sampling {
        if let Some(t) = &s.temperature {
            put(&mut out, "temperature", &t.temperature)?;
        }
        if let Some(t) = &s.top_p {
            put(&mut out, "top_p", &t.top_p)?;
        }
        if let Some(t) = &s.top_k {
            if sampling_extras {
                put(&mut out, "top_k", &t.top_k)?;
            } else {
                reject(label, &t.top_k, "top_k")?;
            }
        }
        if let Some(t) = &s.min_p {
            if sampling_extras {
                put(&mut out, "min_p", &t.min_p)?;
            } else {
                reject(label, &t.min_p, "min_p")?;
            }
        }
        if let Some(t) = &s.seed
            && let ProviderValue::Value(Some(Seed::Value(seed))) = &t.seed
            && *seed != -1
        {
            if responses && !sampling_extras {
                return Err(invalid(format!("{label} responses does not support seed")));
            }
            if rejects_penalties_and_seed {
                return Err(invalid(format!("{label} does not support seed")));
            }
            out.insert("seed".into(), json!(seed));
        }
        if let Some(r) = &s.repetition_penalty {
            if rejects_penalties_and_seed {
                reject(label, &r.frequency_penalty, "frequency_penalty")?;
                reject(label, &r.presence_penalty, "presence_penalty")?;
            } else {
                put(&mut out, "frequency_penalty", &r.frequency_penalty)?;
                put(&mut out, "presence_penalty", &r.presence_penalty)?;
            }
            reject(label, &r.repetition_penalty, "repetition_penalty")?;
            reject(label, &r.consideration_window, "consideration_window")?;
        }
    }
    let mut reasoning = Map::new();
    if let ProviderValue::Value(r) = &p.reasoning
        && let ProviderValue::Value(effort) = &r.effort
    {
        let effort = effort.as_deref().unwrap_or("none");
        match chat {
            None => {
                reasoning.insert("effort".into(), json!(effort));
            }
            Some(profile) => {
                out.insert("reasoning_effort".into(), json!(profile.effort(effort)));
            }
        }
    }
    if let Some(o) = &p.output {
        if responses {
            put(&mut reasoning, "summary", &o.reasoning_summary)?;
        } else {
            reject(
                label,
                &o.reasoning_summary,
                "reasoning summary in chat completions",
            )?;
        }
        if let Some(max) = o.max_tokens {
            out.insert(
                match chat {
                    None => "max_output_tokens",
                    Some(profile) => profile.max_tokens_field,
                }
                .into(),
                json!(max),
            );
        }
        if let Some(include) = &o.include {
            if !responses {
                return Err(invalid(format!(
                    "{label} chat completions does not support output include"
                )));
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
            reject(
                label,
                &o.text_verbosity,
                "text verbosity in chat completions",
            )?;
        }
    }
    if !reasoning.is_empty() {
        out.insert("reasoning".into(), reasoning.into());
    }
    if let Some(t) = &p.tool_calling {
        if responses {
            put(&mut out, "max_tool_calls", &t.max_tool_calls)?;
        } else {
            reject(
                label,
                &t.max_tool_calls,
                "max_tool_calls in chat completions",
            )?;
        }
        put(&mut out, "parallel_tool_calls", &t.parallel_tool_calls)?;
        out.insert("tool_choice".into(), tool_choice(&t.tool_choice, wire)?);
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
    // The provider parameters that hold `store`: `openai` for OpenAI, and a
    // host's own key, such as `xai`, for a host that reads them.
    let params_key = match wire {
        Wire::Chat(profile) => profile.openai_params.then_some(OpenAIParams::KEY),
        Wire::Responses(profile) => profile.params_key,
    };
    if let Some(key) = params_key
        && let Some(provider) = p.provider_params.as_ref().and_then(|p| p.get(key))
    {
        let options: OpenAIParams = serde_json::from_value(provider.clone()).map_err(|_| {
            invalid(format!(
                "{key} provider_params must be an OpenAIParams object"
            ))
        })?;
        if let Some(store) = options.store {
            out.insert("store".into(), json!(store));
        }
    }
    if let Some(threshold) = p.context_management.as_ref().and_then(|m| m.compaction) {
        if !responses {
            return Err(invalid(format!(
                "{label} chat completions does not support context management"
            )));
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
