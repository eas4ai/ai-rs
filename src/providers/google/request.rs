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

//! The generate content request body (`prepareRequest` and `getArgs` in
//! `google-language-model.ts`).

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};

use super::{
    api::Flavor,
    capabilities,
    messages::{self, Target},
    options::{LanguageOptions, ThinkingConfig},
    tools,
};
use crate::{
    Error, StreamOptions,
    models::params::*,
    types::{
        Message,
        tools::{Tool, ToolChoice, ToolKind, ToolRef, ToolSpec},
    },
};

/// A prepared request: its body and headers, and the name of the function
/// tool that carries an output schema, whose call becomes the response text.
#[derive(Clone, Debug)]
pub(crate) struct Prepared {
    pub body: Map<String, Value>,
    pub headers: BTreeMap<String, String>,
    pub json_tool: Option<String>,
}

fn reject(what: &str) -> Error {
    Error::Validation(format!("Google does not support {what}"))
}

fn set<T: Into<Value>>(map: &mut Map<String, Value>, key: &str, value: Option<T>) {
    if let Some(value) = value {
        map.insert(key.into(), value.into());
    }
}

/// The settings of the crate's call parameters that Gemini takes.
struct Settings {
    max_tokens: Option<u64>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
    seed: Option<i64>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    service_tier: Option<String>,
    /// `None` leaves reasoning to the model; `Some(None)` disables it.
    effort: Option<Option<String>>,
    summary: bool,
    tool_choice: Option<ToolChoice>,
}

/// Read the call parameters, rejecting the ones Gemini never takes
/// (GGL-002).
fn settings(
    params: &InferenceRequestParams,
    options: &StreamOptions,
    model_id: &str,
    flavor: Flavor,
) -> Result<Settings, Error> {
    let finite = |value: f64, name: &str| {
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::Validation(format!("{name} must be finite")))
        }
    };
    let mut out = Settings {
        max_tokens: params.output.as_ref().and_then(|output| output.max_tokens),
        temperature: None,
        top_p: None,
        top_k: None,
        seed: None,
        frequency_penalty: None,
        presence_penalty: None,
        service_tier: None,
        effort: None,
        summary: false,
        tool_choice: options.tool_choice.clone(),
    };
    if let ProviderValue::Value(sampling) = &params.sampling {
        if let Some(t) = &sampling.temperature
            && let ProviderValue::Value(v) = t.temperature
        {
            out.temperature = Some(finite(v, "temperature")?);
        }
        if let Some(t) = &sampling.top_p
            && let ProviderValue::Value(Some(v)) = t.top_p
        {
            out.top_p = Some(finite(v, "top_p")?);
        }
        if let Some(t) = &sampling.top_k
            && let ProviderValue::Value(Some(v)) = t.top_k
        {
            out.top_k = Some(v);
        }
        if let Some(seed) = &sampling.seed
            && let ProviderValue::Value(Some(Seed::Value(v))) = seed.seed
        {
            out.seed = Some(v);
        }
        if let Some(min_p) = &sampling.min_p
            && matches!(min_p.min_p, ProviderValue::Value(Some(_)))
        {
            return Err(reject("min-p"));
        }
        if let Some(penalties) = &sampling.repetition_penalty {
            if matches!(penalties.repetition_penalty, ProviderValue::Value(Some(_))) {
                return Err(reject("a repetition penalty"));
            }
            if matches!(
                penalties.consideration_window,
                ProviderValue::Value(Some(_))
            ) {
                return Err(reject("a repetition consideration window"));
            }
            if let ProviderValue::Value(Some(v)) = penalties.frequency_penalty {
                out.frequency_penalty = Some(finite(v, "frequency_penalty")?);
            }
            if let ProviderValue::Value(Some(v)) = penalties.presence_penalty {
                out.presence_penalty = Some(finite(v, "presence_penalty")?);
            }
            if flavor == Flavor::Gemini
                && capabilities::is_gemini_25(model_id)
                && (out.frequency_penalty.is_some() || out.presence_penalty.is_some())
            {
                return Err(Error::Validation(format!(
                    "the Gemini API does not take frequency or presence penalties for {model_id}"
                )));
            }
        }
    }
    if let ProviderValue::Value(reasoning) = &params.reasoning
        && let ProviderValue::Value(effort) = &reasoning.effort
    {
        out.effort = Some(effort.clone());
    }
    if let Some(output) = &params.output {
        if output.include.is_some() {
            return Err(reject("an output include list"));
        }
        if matches!(output.text_verbosity, ProviderValue::Value(Some(_))) {
            return Err(reject("text verbosity"));
        }
        out.summary = matches!(output.reasoning_summary, ProviderValue::Value(Some(_)));
    }
    if let Some(calling) = &params.tool_calling {
        if matches!(calling.max_tool_calls, ProviderValue::Value(Some(_))) {
            return Err(reject("a maximum tool call count"));
        }
        if matches!(calling.parallel_tool_calls, ProviderValue::Value(_)) {
            return Err(reject("parallel tool calls"));
        }
        if out.tool_choice.is_none() {
            out.tool_choice = Some(calling.tool_choice.clone());
        }
    }
    if let Some(ToolChoice::Selection(_)) = &out.tool_choice {
        return Err(reject("tool selections"));
    }
    if params.safety_identifier.is_some() {
        return Err(reject("a safety identifier"));
    }
    if params.metadata.is_some() {
        return Err(reject("request metadata"));
    }
    if let Some(cache) = &params.cache {
        if cache.key.is_some() {
            return Err(reject("a cache key"));
        }
        if matches!(cache.retention, ProviderValue::Value(_)) {
            return Err(reject("a cache retention"));
        }
        if matches!(cache.mode, ProviderValue::Value(_)) {
            return Err(reject("a cache mode; use the cachedContent option"));
        }
    }
    if params.context_management.is_some() {
        return Err(reject("context management"));
    }
    if let Some(service) = &params.provider_service
        && let ProviderValue::Value(tier) = &service.service_tier
    {
        if flavor.is_vertex() {
            return Err(Error::Validation(
                "Vertex AI does not take a service tier; use the sharedRequestType and requestType options".into(),
            ));
        }
        out.service_tier = Some(tier.clone());
    }
    Ok(out)
}

/// The thinking config a reasoning effort maps to (GGL-003).
fn derived_thinking(model_id: &str, effort: &Option<String>) -> Result<ThinkingConfig, Error> {
    let mut config = ThinkingConfig::default();
    if capabilities::uses_thinking_level(model_id) {
        let minimum = capabilities::minimum_thinking_level(model_id);
        let level = match effort.as_deref() {
            None | Some("minimal") => minimum,
            Some("low") => "low",
            Some("medium") => "high",
            Some("high" | "xhigh" | "max") => "high",
            Some(other) => {
                return Err(Error::Validation(format!(
                    "unknown reasoning effort {other:?}"
                )));
            }
        };
        config.thinking_level = Some(level.into());
    } else {
        let budget = match effort.as_deref() {
            None => 0,
            Some(effort) => {
                let share = match effort {
                    "minimal" => 0.02,
                    "low" => 0.1,
                    "medium" => 0.3,
                    "high" => 0.6,
                    "xhigh" => 0.9,
                    "max" => 0.95,
                    other => {
                        return Err(Error::Validation(format!(
                            "unknown reasoning effort {other:?}"
                        )));
                    }
                };
                let budget = (65536.0_f64 * share).round() as u64;
                budget.min(capabilities::max_thinking_budget(model_id))
            }
        };
        config.thinking_budget = Some(budget.into());
    }
    Ok(config)
}

fn thinking(
    model_id: &str,
    settings: &Settings,
    option: Option<&ThinkingConfig>,
) -> Result<Option<Value>, Error> {
    let option_sets_amount =
        option.is_some_and(|o| o.thinking_budget.is_some() || o.thinking_level.is_some());
    let mut derived = match &settings.effort {
        Some(effort) if !option_sets_amount => Some(derived_thinking(model_id, effort)?),
        _ => None,
    };
    if settings.summary {
        derived
            .get_or_insert_with(ThinkingConfig::default)
            .include_thoughts = Some(true);
    }
    let merged = match (derived, option) {
        (None, None) => return Ok(None),
        (Some(derived), None) => derived,
        (derived, Some(option)) => {
            let mut merged = derived.unwrap_or_default();
            if option.thinking_budget.is_some() {
                merged.thinking_budget.clone_from(&option.thinking_budget);
            }
            if option.thinking_level.is_some() {
                merged.thinking_level.clone_from(&option.thinking_level);
            }
            if option.include_thoughts.is_some() {
                merged.include_thoughts = option.include_thoughts;
            }
            merged
        }
    };
    Ok(Some(serde_json::to_value(merged)?))
}

/// Replace every `const` with a one-value `enum`, in the places Google reads
/// (`sanitize-response-json-schema.ts`).
pub(crate) fn sanitize_schema(schema: &Value) -> Value {
    let Value::Object(object) = schema else {
        return schema.clone();
    };
    let mut out = Map::new();
    for (key, value) in object {
        match key.as_str() {
            "const" => {
                out.insert("const".into(), value.clone());
            }
            "properties" | "$defs" => {
                out.insert(key.clone(), sanitize_map(value));
            }
            "items" => {
                out.insert(
                    key.clone(),
                    match value {
                        Value::Array(items) => {
                            Value::Array(items.iter().map(sanitize_schema).collect())
                        }
                        other => sanitize_schema(other),
                    },
                );
            }
            "additionalProperties" => {
                out.insert(key.clone(), sanitize_schema(value));
            }
            "anyOf" | "oneOf" => {
                out.insert(
                    key.clone(),
                    match value {
                        Value::Array(items) => {
                            Value::Array(items.iter().map(sanitize_schema).collect())
                        }
                        other => other.clone(),
                    },
                );
            }
            _ => {
                out.insert(key.clone(), value.clone());
            }
        }
    }
    Value::Object(out)
}

fn sanitize_map(value: &Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(name, schema)| (name.clone(), sanitize_schema(schema)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Build the generate content body for a model call. `streaming` is set for
/// `streamGenerateContent`.
pub(crate) fn build(
    target: Target<'_>,
    history: &[Message],
    options: &StreamOptions,
    streaming: bool,
) -> Result<Prepared, Error> {
    let model_id = target.model_id;
    let flavor = target.flavor;
    let params = options.params.clone().unwrap_or_default();
    let mut settings = settings(&params, options, model_id, flavor)?;
    let google: LanguageOptions = super::options::parse(flavor, params.provider_params.as_ref())?;
    google.validate(flavor)?;

    // Gemini before 3 rejects a JSON response format together with function
    // tools, and Gemini 3 takes the two only with automatic tool choice: the
    // schema goes out as a function tool whose call is the response text.
    let mut tools_list = options.tools.clone();
    let mut output_schema = options.output_schema.clone();
    let mut json_tool = None;
    let has_functions = tools_list.iter().any(|t| t.kind == ToolKind::Function);
    if let Some(schema) = &output_schema
        && has_functions
        && capabilities::is_gemini_path(model_id)
        && (!capabilities::uses_gemini_3_features(model_id)
            || matches!(
                settings.tool_choice,
                Some(ToolChoice::Required | ToolChoice::Tool(_))
            ))
    {
        let mut name = "json".to_owned();
        let mut suffix = 1;
        while tools_list.iter().any(|t| t.name == name) {
            name = format!("json_{suffix}");
            suffix += 1;
        }
        tools_list.push(Tool::function(
            name.clone(),
            ToolSpec {
                description: Some("Respond with a JSON object.".into()),
                params: schema.clone(),
            },
        ));
        settings.tool_choice = Some(match &settings.tool_choice {
            Some(ToolChoice::Tool(tool)) => ToolChoice::Tool(tool.clone()),
            Some(ToolChoice::None) => ToolChoice::Tool(ToolRef::new(name.clone())),
            _ => ToolChoice::Required,
        });
        output_schema = None;
        json_tool = Some(name);
    }

    let mut generation = Map::new();
    set(&mut generation, "maxOutputTokens", settings.max_tokens);
    set(&mut generation, "temperature", settings.temperature);
    set(&mut generation, "topP", settings.top_p);
    set(
        &mut generation,
        "frequencyPenalty",
        settings.frequency_penalty,
    );
    set(
        &mut generation,
        "presencePenalty",
        settings.presence_penalty,
    );
    set(&mut generation, "seed", settings.seed);
    if let Some(schema) = &output_schema {
        generation.insert("responseMimeType".into(), json!("application/json"));
        if google.structured_outputs != Some(false) {
            generation.insert(
                "responseJsonSchema".into(),
                sanitize_schema(&serde_json::to_value(schema)?),
            );
        }
    }
    if google.audio_timestamp == Some(true) {
        generation.insert("audioTimestamp".into(), json!(true));
    }
    set(
        &mut generation,
        "responseModalities",
        google.response_modalities.clone().map(|m| json!(m)),
    );
    set(
        &mut generation,
        "thinkingConfig",
        thinking(model_id, &settings, google.thinking_config.as_ref())?,
    );
    set(
        &mut generation,
        "mediaResolution",
        google.media_resolution.clone(),
    );
    if let Some(config) = &google.image_config {
        generation.insert("imageConfig".into(), serde_json::to_value(config)?);
    }

    let (system, contents) = messages::convert(history, target)?;
    let (wire_tools, mut tool_config) =
        tools::prepare(&tools_list, settings.tool_choice.as_ref(), model_id, flavor)?;
    if streaming && flavor.is_vertex() && google.stream_function_call_arguments == Some(true) {
        let config = tool_config.get_or_insert_with(|| json!({}));
        let calling = config
            .as_object_mut()
            .expect("tool config is an object")
            .entry("functionCallingConfig")
            .or_insert_with(|| json!({}));
        calling["streamFunctionCallArguments"] = json!(true);
    }
    if let Some(retrieval) = &google.retrieval_config {
        tool_config.get_or_insert_with(|| json!({}))["retrievalConfig"] =
            serde_json::to_value(retrieval)?;
    }

    let mut body = Map::new();
    body.insert("generationConfig".into(), Value::Object(generation));
    body.insert("contents".into(), json!(contents));
    set(&mut body, "systemInstruction", system);
    set(&mut body, "tools", wire_tools);
    set(&mut body, "toolConfig", tool_config);
    set(&mut body, "cachedContent", google.cached_content.clone());
    set(&mut body, "labels", google.labels.clone().map(|l| json!(l)));
    if !flavor.is_vertex() {
        set(
            &mut body,
            "serviceTier",
            google.service_tier.clone().or(settings.service_tier),
        );
    }
    set(
        &mut body,
        "continuationToken",
        google.continuation_token.clone(),
    );
    if let Some(extra) = &params.extra_body {
        body.extend(extra.clone());
    }

    let mut headers = BTreeMap::new();
    if flavor.is_vertex() {
        if let Some(value) = &google.shared_request_type {
            headers.insert("X-Vertex-AI-LLM-Shared-Request-Type".into(), value.clone());
        }
        if let Some(value) = &google.request_type {
            headers.insert("X-Vertex-AI-LLM-Request-Type".into(), value.clone());
        }
    }
    if let Some(extra) = &params.extra_headers {
        for (name, value) in extra {
            if let ExtraHeaderValue::Value(value) = value {
                headers.insert(name.clone(), value.clone());
            }
        }
    }
    Ok(Prepared {
        body,
        headers,
        json_tool,
    })
}
