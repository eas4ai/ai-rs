//! OpenAI's own Responses requests: the call's `openai` options, sent as
//! TypeScript sends them, and the settings left out for the model
//! (`prepareRequest` in `responses/openai-responses-language-model.ts`).

use serde_json::{Map, Value, json};

use super::{
    capabilities::{Capabilities, dropped},
    options::ResponsesOptions,
    params::invalid,
};
use crate::{
    Error,
    models::params::{InferenceRequestParams, ProviderValue},
    types::tools::{Tool, ToolKind},
};

/// One Responses call to OpenAI: its options and what its model takes.
pub(super) struct Call {
    pub options: ResponsesOptions,
    caps: Capabilities,
    /// The model is a reasoning model, or `forceReasoning` makes it one.
    pub reasoning: bool,
}

impl Call {
    /// Parse the call's options and fail on the settings OpenAI never takes.
    pub(super) fn new(model: &str, p: &InferenceRequestParams) -> Result<Self, Error> {
        let options = ResponsesOptions::parse(p)?;
        if let ProviderValue::Value(sampling) = &p.sampling
            && let Some(penalties) = &sampling.repetition_penalty
        {
            for (name, value) in [
                ("frequency_penalty", &penalties.frequency_penalty),
                ("presence_penalty", &penalties.presence_penalty),
            ] {
                if matches!(value, ProviderValue::Value(Some(_))) {
                    return Err(invalid(format!(
                        "OpenAI responses does not support {name}"
                    )));
                }
            }
        }
        let caps = Capabilities::of(model);
        let reasoning = options.force_reasoning.unwrap_or(caps.reasoning);
        Ok(Self {
            options,
            caps,
            reasoning,
        })
    }

    /// The role system messages go out with, or `None` to leave them out.
    pub(super) fn system_role(&self) -> Option<&'static str> {
        match self.options.system_message_mode.as_deref() {
            Some("remove") => None,
            Some("system") => Some("system"),
            Some(_) => Some("developer"),
            None if self.reasoning => Some("developer"),
            None => Some("system"),
        }
    }

    /// The output format's `strict`: true unless `strictJsonSchema` is false.
    pub(super) fn strict(&self) -> bool {
        self.options.strict_json_schema.unwrap_or(true)
    }

    /// Why a reasoning effort update cannot apply to this request, when it
    /// cannot.
    pub(super) fn update_blocked(&self, body: &Map<String, Value>) -> Option<&'static str> {
        if !self.caps.gpt6 {
            return Some("reasoningEffortUpdate is only supported by GPT-6 and later models");
        }
        let pro = body
            .get("reasoning")
            .and_then(|r| r.get("mode"))
            .is_some_and(|mode| mode == "pro");
        if pro
            || body.contains_key("context_management")
            || body.get("truncation").is_some_and(|t| t == "auto")
        {
            return Some(
                "reasoningEffortUpdate requires standard reasoning mode without automatic compaction or automatic truncation",
            );
        }
        None
    }

    /// Send the options over the core parameters they duplicate, and the
    /// reasoning settings the model takes.
    pub(super) fn apply(&self, model: &str, p: &InferenceRequestParams, body: &mut Map<String, Value>) {
        let o = &self.options;
        let mut set = |key: &str, value: Option<Value>| {
            if let Some(value) = value {
                body.insert(key.into(), value);
            }
        };
        set("conversation", o.conversation.clone().map(Value::from));
        set("instructions", o.instructions.clone().map(Value::from));
        set("max_tool_calls", o.max_tool_calls.clone().map(Value::Number));
        set("metadata", o.metadata.clone());
        set("parallel_tool_calls", o.parallel_tool_calls.map(Value::from));
        set(
            "previous_response_id",
            o.previous_response_id.clone().map(Value::from),
        );
        set("prompt_cache_key", o.prompt_cache_key.clone().map(Value::from));
        set(
            "prompt_cache_options",
            o.prompt_cache_options.as_ref().map(|cache| {
                let mut out = Map::new();
                if let Some(mode) = &cache.mode {
                    out.insert("mode".into(), json!(mode));
                }
                if let Some(ttl) = &cache.ttl {
                    out.insert("ttl".into(), json!(ttl));
                }
                if let Some(prewarm) = cache.prewarm {
                    out.insert("prewarm".into(), json!(prewarm));
                }
                Value::Object(out)
            }),
        );
        set(
            "prompt_cache_retention",
            o.prompt_cache_retention.clone().map(Value::from),
        );
        set("safety_identifier", o.safety_identifier.clone().map(Value::from));
        set("service_tier", o.service_tier.clone().map(Value::from));
        set("store", o.store.map(Value::from));
        set("truncation", o.truncation.clone().map(Value::from));
        set("user", o.user.clone().map(Value::from));
        set("top_logprobs", o.top_logprobs().map(Value::Number));
        set("include", o.include.as_ref().map(|include| json!(include)));
        set(
            "context_management",
            o.context_management.as_ref().map(|entries| {
                entries
                    .iter()
                    .map(|c| json!({"type": c.kind, "compact_threshold": c.compact_threshold}))
                    .collect()
            }),
        );
        if let Some(verbosity) = &o.text_verbosity {
            let text = body.entry("text").or_insert_with(|| json!({}));
            text["verbosity"] = json!(verbosity);
        }
        body.remove("reasoning");
        let core = match &p.reasoning {
            ProviderValue::Value(r) => match &r.effort {
                ProviderValue::Value(effort) => Some(effort.as_deref().unwrap_or("none")),
                ProviderValue::Default => None,
            },
            ProviderValue::Default => None,
        };
        let mut effort = o.reasoning_effort.as_deref().or(core);
        if let Some(e) = effort
            && !self.caps.takes_effort(e)
        {
            dropped(
                "reasoningEffort",
                model,
                &format!(
                    "the model takes only these reasoning efforts: {}",
                    self.caps.efforts.unwrap_or_default().join(", ")
                ),
            );
            effort = None;
        }
        let core_summary = match p.output.as_ref().map(|o| &o.reasoning_summary) {
            Some(ProviderValue::Value(summary)) => Some(summary.as_deref()),
            _ => None,
        };
        let summary = match (&o.reasoning_summary, core_summary) {
            (Some(summary), _) => summary.as_deref(),
            (None, Some(summary)) => summary,
            (None, None) => effort.filter(|e| *e != "none").map(|_| "detailed"),
        };
        if self.reasoning {
            let mut reasoning = Map::new();
            for (key, value) in [
                ("effort", effort),
                ("summary", summary),
                ("mode", o.reasoning_mode.as_deref()),
                ("context", o.reasoning_context.as_deref()),
            ] {
                if let Some(value) = value {
                    reasoning.insert(key.into(), json!(value));
                }
            }
            if !reasoning.is_empty() {
                body.insert("reasoning".into(), reasoning.into());
            }
            return;
        }
        let not_reasoning = "the model is not a reasoning model";
        if effort.is_some_and(|e| e != "none") {
            dropped("reasoningEffort", model, not_reasoning);
        }
        let given_summary =
            matches!(o.reasoning_summary, Some(Some(_))) || core_summary.flatten().is_some();
        for (setting, given) in [
            ("reasoningSummary", given_summary),
            ("reasoningMode", o.reasoning_mode.is_some()),
            ("reasoningContext", o.reasoning_context.is_some()),
        ] {
            if given {
                dropped(setting, model, not_reasoning);
            }
        }
    }

    /// Finish the request once its input is converted: the reasoning effort
    /// updates, the compaction trigger, the `include` values the call needs,
    /// and the settings the model does not take.
    pub(super) fn finish(
        &self,
        model: &str,
        body: &mut Map<String, Value>,
        input: &mut Vec<Value>,
        tools: &[Tool],
    ) -> Result<(), Error> {
        let o = &self.options;
        let efforts = || self.caps.efforts.unwrap_or_default().join(", ");
        for item in input.iter() {
            if item["type"] == "configuration_update"
                && let Some(effort) = item["reasoning"]["effort"].as_str()
                && !self.caps.takes_effort(effort)
            {
                return Err(invalid(format!(
                    "message-level reasoningEffortUpdate {effort:?}: {model} takes only these reasoning efforts: {}",
                    efforts()
                )));
            }
        }
        if let Some(update) = &o.reasoning_effort_update {
            let reason = self.update_blocked(body).map(str::to_owned).or_else(|| {
                (!self.caps.takes_effort(update))
                    .then(|| format!("the model takes only these reasoning efforts: {}", efforts()))
            });
            match reason {
                Some(reason) => dropped("reasoningEffortUpdate", model, &reason),
                None => {
                    let first = input.first();
                    let same = first.is_some_and(|item| {
                        item["type"] == "configuration_update"
                            && item["reasoning"]["effort"] == update.as_str()
                    });
                    if !same {
                        input.insert(
                            0,
                            json!({"type": "configuration_update", "reasoning": {"effort": update}}),
                        );
                    }
                }
            }
        }
        if input
            .windows(2)
            .any(|pair| pair.iter().all(|item| item["type"] == "configuration_update"))
        {
            return Err(invalid(
                "OpenAI does not take adjacent reasoning effort configuration updates",
            ));
        }
        if o.compaction_trigger == Some(true) {
            input.push(json!({"type": "compaction_trigger"}));
        }

        let has = |ids: &[&str]| {
            tools.iter().any(|tool| {
                tool.kind == ToolKind::Provider
                    && tool
                        .tool_config
                        .as_ref()
                        .and_then(|c| c.id.as_deref())
                        .is_some_and(|id| ids.contains(&id))
            })
        };
        if o.top_logprobs().is_some() {
            include(body, "message.output_text.logprobs");
        }
        if has(&["openai.web_search", "openai.web_search_preview"])
            && o.include_web_search_sources != Some(false)
        {
            include(body, "web_search_call.action.sources");
        }
        if has(&["openai.code_interpreter"]) {
            include(body, "code_interpreter_call.outputs");
        }
        if body.get("store") == Some(&Value::Bool(false)) && self.reasoning {
            include(body, "reasoning.encrypted_content");
        }

        if self.caps.gpt6 && body.remove("prompt_cache_retention").is_some() {
            dropped(
                "promptCacheRetention",
                model,
                "GPT-6 and later models take promptCacheOptions instead",
            );
        }
        if self.reasoning {
            // An update in the input sets the effort the model reasons with.
            let effort = input
                .iter()
                .rev()
                .find(|item| item["type"] == "configuration_update")
                .and_then(|item| item["reasoning"]["effort"].as_str())
                .or_else(|| body.get("reasoning")?.get("effort")?.as_str())
                .map(str::to_owned);
            if !(effort.as_deref() == Some("none") && self.caps.non_reasoning_params) {
                let reason = "reasoning models do not take it";
                for (key, setting) in [("temperature", "temperature"), ("top_p", "topP")] {
                    if body.remove(key).is_some() {
                        dropped(setting, model, reason);
                    }
                }
                if self.caps.efforts.is_some() {
                    let top = body.remove("top_logprobs").is_some();
                    let listed = remove_include(body, "message.output_text.logprobs");
                    if top || listed {
                        dropped("logprobs", model, reason);
                    }
                }
            }
        }
        let tier = body.get("service_tier").and_then(Value::as_str);
        let unsupported_tier = match tier {
            Some("flex") if !self.caps.flex => {
                Some("flex processing is only available for o3, o4-mini and gpt-5 models")
            }
            Some("priority" | "fast") if !self.caps.priority => Some(
                "priority processing is only available for gpt-4, gpt-5 other than nano, o3 and o4-mini models",
            ),
            _ => None,
        };
        if let Some(reason) = unsupported_tier {
            body.remove("service_tier");
            dropped("serviceTier", model, reason);
        }
        Ok(())
    }
}

/// Add a value to the body's `include` list once.
pub(super) fn include(body: &mut Map<String, Value>, value: &str) {
    let include = body.entry("include").or_insert_with(|| json!([]));
    if let Some(include) = include.as_array_mut()
        && !include.iter().any(|v| v == value)
    {
        include.push(json!(value));
    }
}

/// Take a value out of the body's `include` list, and the list when it ends
/// up empty; whether the value was there.
fn remove_include(body: &mut Map<String, Value>, value: &str) -> bool {
    let Some(include) = body.get_mut("include").and_then(Value::as_array_mut) else {
        return false;
    };
    let before = include.len();
    include.retain(|v| v != value);
    let removed = include.len() != before;
    if include.is_empty() {
        body.remove("include");
    }
    removed
}
