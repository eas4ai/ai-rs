//! The core parameters of a Messages call, read and checked before the
//! request is built (ANT-003). A setting Anthropic never takes fails with
//! `Error::Validation`.
use crate::{
    Error,
    models::params::*,
    types::tools::{ToolChoice, ToolChoiceMode},
};
use serde_json::Value;
use std::collections::BTreeMap;

/// The core reasoning setting of a call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) enum Reasoning {
    /// Left to the provider.
    #[default]
    Default,
    /// Turned off (`none`).
    Disabled,
    /// A reasoning level: `minimal`, `low`, `medium`, `high`, `xhigh` or
    /// `max`.
    Level(String),
}

/// What a call's core parameters ask for.
#[derive(Default)]
pub(super) struct Core {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub max_tokens: Option<u64>,
    pub reasoning: Reasoning,
    /// The reasoning summary, `Some(None)` for none.
    pub summary: Option<Option<String>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub service_tier: Option<String>,
    pub safety_identifier: Option<String>,
    /// A cache setting, with its retention.
    pub cache: Option<Option<String>>,
    pub compaction: Option<u64>,
    pub headers: BTreeMap<String, String>,
    pub query: BTreeMap<String, Value>,
    pub extra_body: BTreeMap<String, Value>,
}

fn unsupported(label: &str, name: &str) -> Error {
    Error::Validation(format!("{label} does not support {name}"))
}
fn reject_value<T>(label: &str, value: &ProviderValue<Option<T>>, name: &str) -> Result<(), Error> {
    if matches!(value, ProviderValue::Value(Some(_))) {
        Err(unsupported(label, name))
    } else {
        Ok(())
    }
}

/// Add betas the request needs, each once, keeping their order.
pub(super) fn add_betas<'a>(betas: &mut Vec<String>, more: impl IntoIterator<Item = &'a str>) {
    for beta in more {
        if !betas.iter().any(|known| known == beta) {
            betas.push(beta.to_owned());
        }
    }
}

/// The reasoning levels a core reasoning effort may name.
const LEVELS: &[&str] = &["minimal", "low", "medium", "high", "xhigh", "max"];

/// Read and check a call's core parameters.
pub(super) fn read(params: Option<&InferenceRequestParams>, label: &str) -> Result<Core, Error> {
    let mut core = Core::default();
    let Some(p) = params else {
        return Ok(core);
    };
    if let ProviderValue::Value(s) = &p.sampling {
        if let Some(t) = &s.temperature
            && let ProviderValue::Value(v) = t.temperature
        {
            if !v.is_finite() {
                return Err(Error::Validation("temperature must be finite".into()));
            }
            core.temperature = Some(v);
        }
        if let Some(t) = &s.top_p
            && let ProviderValue::Value(v) = t.top_p
        {
            if v.is_some_and(|v| !v.is_finite()) {
                return Err(Error::Validation("top_p must be finite".into()));
            }
            core.top_p = v;
        }
        if let Some(t) = &s.top_k
            && let ProviderValue::Value(v) = t.top_k
        {
            core.top_k =
                Some(v.ok_or_else(|| Error::Validation(format!("{label} top_k cannot be None")))?);
        }
        if let Some(t) = &s.seed
            && let ProviderValue::Value(Some(Seed::Value(seed))) = t.seed
            && seed != -1
        {
            return Err(unsupported(label, "seed"));
        }
        if let Some(t) = &s.min_p {
            reject_value(label, &t.min_p, "min_p")?;
        }
        if let Some(r) = &s.repetition_penalty {
            reject_value(label, &r.repetition_penalty, "repetition_penalty")?;
            reject_value(label, &r.frequency_penalty, "frequency_penalty")?;
            reject_value(label, &r.presence_penalty, "presence_penalty")?;
            reject_value(label, &r.consideration_window, "consideration_window")?;
        }
    }
    if let ProviderValue::Value(r) = &p.reasoning
        && let ProviderValue::Value(effort) = &r.effort
    {
        core.reasoning = match effort.as_deref() {
            None | Some("none") => Reasoning::Disabled,
            Some(level) if LEVELS.contains(&level) => Reasoning::Level(level.to_owned()),
            Some(other) => {
                return Err(Error::Validation(format!(
                    "{label} takes the reasoning efforts none, {}, not {other:?}",
                    LEVELS.join(", ")
                )));
            }
        };
    }
    if let Some(output) = &p.output {
        if let ProviderValue::Value(summary) = &output.reasoning_summary {
            core.summary = Some(summary.clone());
        }
        core.max_tokens = output.max_tokens;
        if output.include.is_some() {
            return Err(unsupported(label, "output include"));
        }
        reject_value(label, &output.text_verbosity, "text verbosity")?;
    }
    if let Some(tc) = &p.tool_calling {
        reject_value(label, &tc.max_tool_calls, "max_tool_calls")?;
        if let ToolChoice::Selection(s) = &tc.tool_choice
            && !(s.tools.len() == 1
                && matches!(s.mode, ToolChoiceMode::Auto | ToolChoiceMode::Required))
        {
            return Err(Error::Validation(
                "Anthropic does not support allowed tool subsets".into(),
            ));
        }
        if let ProviderValue::Value(parallel) = tc.parallel_tool_calls {
            if tc.tool_choice == ToolChoice::None {
                return Err(Error::Validation(format!(
                    "{label} cannot set parallel tool calls with tool none"
                )));
            }
            core.parallel_tool_calls = Some(parallel);
        }
        core.tool_choice = Some(tc.tool_choice.clone());
    }
    if let Some(s) = &p.provider_service
        && let ProviderValue::Value(v) = &s.service_tier
    {
        core.service_tier = Some(v.clone());
    }
    if p.metadata.as_ref().is_some_and(|m| !m.is_empty()) {
        return Err(Error::Validation(format!(
            "{label} takes no request metadata other than a user id; set the safety identifier"
        )));
    }
    core.safety_identifier = p.safety_identifier.clone();
    if let Some(cache) = &p.cache {
        if cache.key.is_some() {
            return Err(unsupported(label, "cache keys"));
        }
        if matches!(cache.mode, ProviderValue::Value(_)) {
            return Err(unsupported(label, "a cache mode"));
        }
        core.cache = Some(match &cache.retention {
            ProviderValue::Value(ttl) => Some(ttl.clone()),
            ProviderValue::Default => None,
        });
    }
    if let Some(headers) = &p.extra_headers {
        for (k, v) in headers {
            if let ExtraHeaderValue::Value(v) = v {
                core.headers.insert(k.clone(), v.clone());
            }
        }
    }
    if let Some(context) = &p.context_management {
        core.compaction = context.compaction.map(|t| t.0);
    }
    if let Some(query) = &p.extra_query {
        core.query = query.clone();
    }
    if let Some(extra) = &p.extra_body {
        core.extra_body = extra.clone();
    }
    Ok(core)
}
