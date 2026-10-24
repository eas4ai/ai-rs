//! OpenAI's own Chat Completions requests: the call's `openai` options, sent
//! as TypeScript sends them, and the settings left out for the model
//! (`getArgs` in `chat/openai-chat-language-model.ts`, OAI-005 and OAI-014).

use serde_json::{Map, Value, json};

use super::{
    capabilities::{Capabilities, dropped},
    options::{ChatOptions, Logprobs},
    params::invalid,
};
use crate::{
    Error,
    models::params::{InferenceRequestParams, ProviderValue},
};

/// One Chat Completions call to OpenAI: its options and what its model takes.
pub(super) struct ChatCall {
    pub options: ChatOptions,
    caps: Capabilities,
    /// The model is a reasoning model, or `forceReasoning` makes it one.
    pub reasoning: bool,
}

impl ChatCall {
    pub(super) fn new(model: &str, p: &InferenceRequestParams) -> Result<Self, Error> {
        if p.cache
            .as_ref()
            .is_some_and(|c| matches!(c.mode, ProviderValue::Value(_)))
        {
            return Err(invalid(
                "OpenAI takes no cache mode; use the openai option promptCacheOptions",
            ));
        }
        let options = ChatOptions::parse(p)?;
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

    /// The response format's `strict`: true unless `strictJsonSchema` is false.
    pub(super) fn strict(&self) -> bool {
        self.options.strict_json_schema.unwrap_or(true)
    }

    /// Send the options over the core parameters they duplicate, and leave
    /// out what the model does not take.
    pub(super) fn apply(&self, model: &str, body: &mut Map<String, Value>) {
        let o = &self.options;
        let mut set = |key: &str, value: Option<Value>| {
            if let Some(value) = value {
                body.insert(key.into(), value);
            }
        };
        set("logit_bias", o.logit_bias.clone().map(Value::Object));
        set("user", o.user.clone().map(Value::from));
        set(
            "parallel_tool_calls",
            o.parallel_tool_calls.map(Value::from),
        );
        set(
            "max_completion_tokens",
            o.max_completion_tokens.clone().map(Value::Number),
        );
        set("store", o.store.map(Value::from));
        set("metadata", o.metadata.clone().map(Value::Object));
        set("prediction", o.prediction.clone().map(Value::Object));
        set("service_tier", o.service_tier.clone().map(Value::from));
        set("verbosity", o.text_verbosity.clone().map(Value::from));
        set(
            "prompt_cache_key",
            o.prompt_cache_key.clone().map(Value::from),
        );
        set(
            "prompt_cache_options",
            o.prompt_cache_options
                .as_ref()
                .and_then(|cache| serde_json::to_value(cache).ok()),
        );
        set(
            "prompt_cache_retention",
            o.prompt_cache_retention.clone().map(Value::from),
        );
        set(
            "safety_identifier",
            o.safety_identifier.clone().map(Value::from),
        );
        match &o.logprobs {
            Some(Logprobs::Flag(true)) => {
                set("logprobs", Some(json!(true)));
                set("top_logprobs", Some(json!(0)));
            }
            Some(Logprobs::Top(n)) => {
                set("logprobs", Some(json!(true)));
                set("top_logprobs", Some(Value::Number(n.clone())));
            }
            _ => {}
        }

        let core = body
            .remove("reasoning_effort")
            .and_then(|e| e.as_str().map(str::to_owned));
        let mut effort = o.reasoning_effort.clone().or(core);
        if let Some(e) = &effort
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
        match &effort {
            Some(e) if self.reasoning => {
                body.insert("reasoning_effort".into(), json!(e));
            }
            Some(e) if e != "none" => {
                dropped(
                    "reasoningEffort",
                    model,
                    "the model is not a reasoning model",
                );
            }
            _ => {}
        }
        if self.caps.gpt6 && body.remove("prompt_cache_retention").is_some() {
            dropped(
                "promptCacheRetention",
                model,
                "GPT-6 and later models take promptCacheOptions instead",
            );
        }
        let mut leave_out = |key: &str, setting: &str, reason: &str| {
            if body.remove(key).is_some() {
                dropped(setting, model, reason);
            }
        };
        if self.reasoning {
            let reason = "reasoning models do not take it";
            if !(effort.as_deref() == Some("none") && self.caps.non_reasoning_params) {
                leave_out("temperature", "temperature", reason);
                leave_out("top_p", "topP", reason);
                leave_out("logprobs", "logprobs", reason);
            }
            leave_out("frequency_penalty", "frequencyPenalty", reason);
            leave_out("presence_penalty", "presencePenalty", reason);
            leave_out("logit_bias", "logitBias", reason);
            leave_out("top_logprobs", "topLogprobs", reason);
        } else if model.starts_with("gpt-4o-search-preview")
            || model.starts_with("gpt-4o-mini-search-preview")
        {
            leave_out(
                "temperature",
                "temperature",
                "the search preview models do not take it",
            );
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
    }
}
