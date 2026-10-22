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

use crate::{
    Error,
    models::params::*,
    types::tools::{ToolChoice, ToolChoiceMode},
};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Default)]
pub(super) struct RequestParams {
    pub body: Map<String, Value>,
    pub headers: BTreeMap<String, String>,
    pub query: BTreeMap<String, Value>,
    pub extra_body: BTreeMap<String, Value>,
}
fn unsupported(name: &str) -> Error {
    Error::Validation(format!("Anthropic does not support {name}"))
}
fn reject_value<T>(value: &ProviderValue<Option<T>>, name: &str) -> Result<(), Error> {
    if matches!(value, ProviderValue::Value(Some(_))) {
        Err(unsupported(name))
    } else {
        Ok(())
    }
}
pub(super) fn tool_choice(choice: &ToolChoice) -> Result<Value, Error> {
    Ok(match choice {
        ToolChoice::Auto => json!({"type":"auto"}),
        ToolChoice::None => json!({"type":"none"}),
        ToolChoice::Required => json!({"type":"any"}),
        ToolChoice::Tool(t) => json!({"type":"tool","name":t.0}),
        ToolChoice::Selection(s)
            if s.tools.len() == 1
                && matches!(s.mode, ToolChoiceMode::Auto | ToolChoiceMode::Required) =>
        {
            json!({"type":"tool","name":s.tools.first().map(|t|t.0.as_str()).unwrap_or("")})
        }
        _ => {
            return Err(Error::Validation(
                "Anthropic does not support allowed tool subsets".into(),
            ));
        }
    })
}
fn sampling(
    body: &mut Map<String, Value>,
    sampling: &ProviderValue<SamplerParams>,
) -> Result<(), Error> {
    let ProviderValue::Value(s) = sampling else {
        return Ok(());
    };
    if let Some(p) = &s.temperature
        && let ProviderValue::Value(v) = p.temperature
    {
        if !v.is_finite() {
            return Err(Error::Validation("temperature must be finite".into()));
        }
        body.insert("temperature".into(), json!(v));
    }
    if let Some(p) = &s.top_p
        && let ProviderValue::Value(v) = p.top_p
    {
        if v.is_some_and(|v| !v.is_finite()) {
            return Err(Error::Validation("top_p must be finite".into()));
        }
        body.insert("top_p".into(), json!(v));
    }
    if let Some(p) = &s.top_k
        && let ProviderValue::Value(v) = p.top_k
    {
        let v = v.ok_or_else(|| Error::Validation("Anthropic top_k cannot be None".into()))?;
        body.insert("top_k".into(), json!(v));
    }
    if let Some(p) = &s.seed
        && let ProviderValue::Value(Some(Seed::Value(seed))) = p.seed
        && seed != -1
    {
        return Err(unsupported("seed"));
    }
    if let Some(p) = &s.min_p {
        reject_value(&p.min_p, "min_p")?;
    }
    if let Some(p) = &s.repetition_penalty {
        reject_value(&p.repetition_penalty, "repetition_penalty")?;
        reject_value(&p.frequency_penalty, "frequency_penalty")?;
        reject_value(&p.presence_penalty, "presence_penalty")?;
        reject_value(&p.consideration_window, "consideration_window")?;
    }
    Ok(())
}
pub(super) fn add_betas(headers: &mut BTreeMap<String, String>, betas: &BTreeSet<String>) {
    if !betas.is_empty()
        && !headers
            .keys()
            .any(|k| k.eq_ignore_ascii_case("anthropic-beta"))
    {
        headers.insert(
            "anthropic-beta".into(),
            betas.iter().cloned().collect::<Vec<_>>().join(","),
        );
    }
}
pub(super) fn translate(params: Option<&InferenceRequestParams>) -> Result<RequestParams, Error> {
    let mut out = RequestParams::default();
    let Some(p) = params else {
        return Ok(out);
    };
    let b = &mut out.body;
    sampling(b, &p.sampling)?;
    if let ProviderValue::Value(reasoning) = &p.reasoning
        && let ProviderValue::Value(effort) = &reasoning.effort
    {
        match effort {
            None => {
                b.insert("thinking".into(), json!({"type":"disabled"}));
            }
            Some(v) => {
                b.insert("output_config".into(), json!({"effort":v}));
            }
        }
    }
    if let Some(output) = &p.output {
        if let ProviderValue::Value(summary) = &output.reasoning_summary
            && b.get("thinking").and_then(|v| v.get("type")) != Some(&json!("disabled"))
        {
            b.insert(
                "thinking".into(),
                json!({"type":"adaptive","display":summary.as_deref().unwrap_or("omitted")}),
            );
        }
        if let Some(max) = output.max_tokens {
            b.insert("max_tokens".into(), json!(max));
        }
        if output.include.is_some() {
            return Err(unsupported("output include"));
        }
        reject_value(&output.text_verbosity, "text verbosity")?;
    }
    if let Some(tc) = &p.tool_calling {
        reject_value(&tc.max_tool_calls, "max_tool_calls")?;
        let mut choice = tool_choice(&tc.tool_choice)?;
        if let ProviderValue::Value(parallel) = tc.parallel_tool_calls {
            if choice["type"] == "none" {
                return Err(Error::Validation(
                    "Anthropic cannot set parallel tool calls with tool none".into(),
                ));
            }
            choice["disable_parallel_tool_use"] = json!(!parallel);
        }
        b.insert("tool_choice".into(), choice);
    }
    if let Some(s) = &p.provider_service
        && let ProviderValue::Value(v) = &s.service_tier
    {
        b.insert("service_tier".into(), json!(v));
    }
    let mut metadata = p.metadata.clone().unwrap_or_default();
    if let Some(id) = &p.safety_identifier {
        metadata.insert("user_id".into(), id.clone());
    }
    if !metadata.is_empty() {
        b.insert("metadata".into(), json!(metadata));
    }
    if let Some(cache) = &p.cache {
        if cache.key.is_some() {
            return Err(unsupported("cache keys"));
        }
        let mut control = json!({"type":"ephemeral"});
        if let ProviderValue::Value(ttl) = &cache.retention {
            control["ttl"] = json!(ttl);
        }
        b.insert("cache_control".into(), control);
    }
    if let Some(headers) = &p.extra_headers {
        for (k, v) in headers {
            if let ExtraHeaderValue::Value(v) = v {
                out.headers.insert(k.clone(), v.clone());
            }
        }
    }
    if let Some(context) = &p.context_management
        && let Some(threshold) = context.compaction
    {
        out.extra_body.insert("context_management".into(),json!({"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":threshold.0}}]}));
        add_betas(
            &mut out.headers,
            &[
                "compact-2026-01-12".into(),
                "context-management-2025-06-27".into(),
            ]
            .into(),
        );
    }
    if let Some(query) = &p.extra_query {
        out.query = query.clone();
    }
    if let Some(extra) = &p.extra_body {
        out.extra_body.extend(extra.clone());
    }
    Ok(out)
}
