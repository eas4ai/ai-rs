//! The `openai` provider options of a Responses call, with the TypeScript
//! option names (`openai-responses-language-model-options.ts`). An unknown
//! option name or a value outside an option's schema fails with
//! `Error::Validation` before any request.

use serde::{Deserialize, Deserializer};
use serde_json::{Number, Value};

use super::params::invalid;
use crate::{Error, models::params::InferenceRequestParams};

/// The most logprobs OpenAI returns per token; `logprobs: true` asks for it.
pub(crate) const TOP_LOGPROBS_MAX: u64 = 20;

/// The Responses language model options (`openaiLanguageModelResponsesOptions`).
/// A field that TypeScript marks `optional` and not `nullish` takes no `null`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ResponsesOptions {
    pub conversation: Option<String>,
    pub include: Option<Vec<String>>,
    #[serde(default, deserialize_with = "not_null")]
    pub include_web_search_sources: Option<bool>,
    pub instructions: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub logprobs: Option<Logprobs>,
    pub max_tool_calls: Option<Number>,
    pub metadata: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub previous_response_id: Option<String>,
    pub prompt_cache_key: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub prompt_cache_options: Option<PromptCacheOptions>,
    pub prompt_cache_retention: Option<String>,
    pub reasoning_effort: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub reasoning_effort_update: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub reasoning_mode: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub reasoning_context: Option<String>,
    /// `Some(None)` is an explicit `null`, which sends no summary.
    #[serde(default, deserialize_with = "given")]
    pub reasoning_summary: Option<Option<String>>,
    pub safety_identifier: Option<String>,
    pub service_tier: Option<String>,
    pub store: Option<bool>,
    #[serde(default, deserialize_with = "not_null")]
    pub pass_through_unsupported_files: Option<bool>,
    pub strict_json_schema: Option<bool>,
    pub text_verbosity: Option<String>,
    pub truncation: Option<String>,
    pub user: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub system_message_mode: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub force_reasoning: Option<bool>,
    pub context_management: Option<Vec<ContextManagement>>,
    #[serde(default, deserialize_with = "not_null")]
    pub compaction_trigger: Option<bool>,
    #[serde(default, deserialize_with = "not_null")]
    pub allowed_tools: Option<AllowedTools>,
}

/// `true`, or a number of top logprobs from 1 to 20.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum Logprobs {
    Flag(bool),
    Top(Number),
}

#[derive(Clone, Debug, Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct PromptCacheOptions {
    #[serde(
        default,
        deserialize_with = "not_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub mode: Option<String>,
    #[serde(
        default,
        deserialize_with = "not_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub ttl: Option<String>,
    #[serde(
        default,
        deserialize_with = "not_null",
        skip_serializing_if = "Option::is_none"
    )]
    pub prewarm: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ContextManagement {
    #[serde(rename = "type")]
    pub kind: String,
    pub compact_threshold: Number,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct AllowedTools {
    pub tool_names: Vec<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub mode: Option<String>,
}

/// A present field, which may not be `null`.
fn not_null<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Option<T>, D::Error> {
    T::deserialize(d).map(Some)
}

/// A present field, `null` included.
fn given<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> Result<Option<Option<T>>, D::Error> {
    Option::<T>::deserialize(d).map(Some)
}

pub(crate) const EFFORT_UPDATES: &[&str] = &["none", "low", "medium", "high", "xhigh", "max"];

/// Fail unless `value` is one of `allowed`.
fn one_of(name: &str, value: Option<&str>, allowed: &[&str]) -> Result<(), Error> {
    match value {
        Some(value) if !allowed.contains(&value) => Err(invalid(format!(
            "openai option {name} must be one of {}, not {value:?}",
            allowed.join(", ")
        ))),
        _ => Ok(()),
    }
}

impl ResponsesOptions {
    /// The `openai` entry of the call's provider parameters; none gives the
    /// defaults.
    pub(crate) fn parse(p: &InferenceRequestParams) -> Result<Self, Error> {
        let Some(value) = p.provider_params.as_ref().and_then(|p| p.get("openai")) else {
            return Ok(Self::default());
        };
        let options: Self = serde_json::from_value(value.clone())
            .map_err(|e| invalid(format!("invalid openai provider options: {e}")))?;
        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<(), Error> {
        for value in self.include.iter().flatten() {
            one_of(
                "include",
                Some(value),
                &[
                    "reasoning.encrypted_content",
                    "file_search_call.results",
                    "web_search_call.results",
                    "message.output_text.logprobs",
                ],
            )?;
        }
        if let Some(Logprobs::Top(n)) = &self.logprobs
            && !n
                .as_f64()
                .is_some_and(|n| (1.0..=TOP_LOGPROBS_MAX as f64).contains(&n))
        {
            return Err(invalid(format!(
                "openai option logprobs must be true, false or a number from 1 to {TOP_LOGPROBS_MAX}, not {n}"
            )));
        }
        if let Some(cache) = &self.prompt_cache_options {
            one_of(
                "promptCacheOptions.mode",
                cache.mode.as_deref(),
                &["implicit", "explicit"],
            )?;
            one_of(
                "promptCacheOptions.ttl",
                cache.ttl.as_deref(),
                &["30m", "12h"],
            )?;
        }
        one_of(
            "promptCacheRetention",
            self.prompt_cache_retention.as_deref(),
            &["in_memory", "24h"],
        )?;
        one_of(
            "reasoningEffortUpdate",
            self.reasoning_effort_update.as_deref(),
            EFFORT_UPDATES,
        )?;
        one_of(
            "reasoningMode",
            self.reasoning_mode.as_deref(),
            &["standard", "pro"],
        )?;
        one_of(
            "reasoningContext",
            self.reasoning_context.as_deref(),
            &["auto", "current_turn", "all_turns"],
        )?;
        one_of(
            "serviceTier",
            self.service_tier.as_deref(),
            &["auto", "flex", "priority", "fast", "ultrafast", "default"],
        )?;
        one_of(
            "textVerbosity",
            self.text_verbosity.as_deref(),
            &["low", "medium", "high"],
        )?;
        one_of(
            "truncation",
            self.truncation.as_deref(),
            &["auto", "disabled"],
        )?;
        one_of(
            "systemMessageMode",
            self.system_message_mode.as_deref(),
            &["system", "developer", "remove"],
        )?;
        for entry in self.context_management.iter().flatten() {
            one_of("contextManagement.type", Some(&entry.kind), &["compaction"])?;
        }
        if let Some(allowed) = &self.allowed_tools {
            if allowed.tool_names.is_empty() {
                return Err(invalid(
                    "openai option allowedTools.toolNames must name at least one tool",
                ));
            }
            one_of(
                "allowedTools.mode",
                allowed.mode.as_deref(),
                &["auto", "required"],
            )?;
        }
        if self.conversation.is_some() && self.previous_response_id.is_some() {
            return Err(invalid(
                "openai options conversation and previousResponseId cannot be used together",
            ));
        }
        Ok(())
    }

    /// The `top_logprobs` the options ask for: 20 for `true`.
    pub(crate) fn top_logprobs(&self) -> Option<Number> {
        match &self.logprobs {
            Some(Logprobs::Flag(true)) => Some((TOP_LOGPROBS_MAX - 1).into()),
            Some(Logprobs::Top(n)) => Some(n.clone()),
            _ => None,
        }
    }
}

/// The Chat Completions language model options (`openaiLanguageModelChatOptions`).
/// Every field is `optional`, so none takes `null`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ChatOptions {
    #[serde(default, deserialize_with = "not_null")]
    pub logit_bias: Option<serde_json::Map<String, Value>>,
    #[serde(default, deserialize_with = "not_null")]
    pub logprobs: Option<Logprobs>,
    #[serde(default, deserialize_with = "not_null")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, deserialize_with = "not_null")]
    pub user: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub reasoning_effort: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub max_completion_tokens: Option<Number>,
    #[serde(default, deserialize_with = "not_null")]
    pub store: Option<bool>,
    #[serde(default, deserialize_with = "not_null")]
    pub metadata: Option<serde_json::Map<String, Value>>,
    #[serde(default, deserialize_with = "not_null")]
    pub prediction: Option<serde_json::Map<String, Value>>,
    #[serde(default, deserialize_with = "not_null")]
    pub service_tier: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub strict_json_schema: Option<bool>,
    #[serde(default, deserialize_with = "not_null")]
    pub text_verbosity: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub prompt_cache_key: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub prompt_cache_options: Option<PromptCacheOptions>,
    #[serde(default, deserialize_with = "not_null")]
    pub prompt_cache_retention: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub safety_identifier: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub system_message_mode: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub force_reasoning: Option<bool>,
}

impl ChatOptions {
    /// The `openai` entry of the call's provider parameters; none gives the
    /// defaults.
    pub(crate) fn parse(p: &InferenceRequestParams) -> Result<Self, Error> {
        let Some(value) = p.provider_params.as_ref().and_then(|p| p.get("openai")) else {
            return Ok(Self::default());
        };
        if value.get("reasoningSummary").is_some() {
            return Err(invalid(
                "openai option reasoningSummary is only supported by the Responses API, not Chat Completions",
            ));
        }
        let options: Self = serde_json::from_value(value.clone())
            .map_err(|e| invalid(format!("invalid openai provider options: {e}")))?;
        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<(), Error> {
        for (token, bias) in self.logit_bias.iter().flatten() {
            if token.trim().parse::<f64>().is_err() || !bias.is_number() {
                return Err(invalid(format!(
                    "openai option logitBias maps token ids to numbers, not {token:?} to {bias}"
                )));
            }
        }
        for (key, value) in self.metadata.iter().flatten() {
            let fits = key.chars().count() <= 64
                && value.as_str().is_some_and(|v| v.chars().count() <= 512);
            if !fits {
                return Err(invalid(
                    "openai option metadata maps keys of at most 64 characters to strings of at most 512",
                ));
            }
        }
        one_of(
            "reasoningEffort",
            self.reasoning_effort.as_deref(),
            &["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        )?;
        one_of(
            "serviceTier",
            self.service_tier.as_deref(),
            &["auto", "flex", "priority", "fast", "ultrafast", "default"],
        )?;
        one_of(
            "textVerbosity",
            self.text_verbosity.as_deref(),
            &["low", "medium", "high"],
        )?;
        if let Some(cache) = &self.prompt_cache_options {
            one_of(
                "promptCacheOptions.mode",
                cache.mode.as_deref(),
                &["implicit", "explicit"],
            )?;
            one_of("promptCacheOptions.ttl", cache.ttl.as_deref(), &["30m"])?;
            if cache.prewarm.is_some() {
                return Err(invalid(
                    "openai option promptCacheOptions.prewarm is only supported by the Responses API",
                ));
            }
        }
        one_of(
            "promptCacheRetention",
            self.prompt_cache_retention.as_deref(),
            &["in_memory", "24h"],
        )?;
        one_of(
            "systemMessageMode",
            self.system_message_mode.as_deref(),
            &["system", "developer", "remove"],
        )?;
        Ok(())
    }
}
