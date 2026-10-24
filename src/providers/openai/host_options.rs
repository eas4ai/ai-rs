//! The provider options of a host whose TypeScript chat model is
//! `OpenAICompatibleChatLanguageModel` (HST-026): the `openaiCompatible`
//! entry and the host's own entry, as `getArgs` in
//! `openai-compatible-chat-language-model.ts` reads them, and the options
//! Fireworks renames in `transformRequestBody` (`fireworks-provider.ts`).

use serde_json::{Map, Value, json};

use super::{
    params::invalid,
    profile::{ChatProfile, HostOptions},
};
use crate::{Error, models::params::InferenceRequestParams};

/// The keys the compatible chat model reads itself instead of sending.
const COMPATIBLE_KEYS: [&str; 4] = [
    "user",
    "reasoningEffort",
    "textVerbosity",
    "strictJsonSchema",
];
/// The body fields a host's entry does not replace: TypeScript sets them
/// after it spreads the entry.
const KEPT: [&str; 7] = [
    "messages",
    "tools",
    "tool_choice",
    "reasoning_effort",
    "verbosity",
    "stream",
    "stream_options",
];

/// A chat call's reading of its host's entries.
#[derive(Default)]
pub(super) struct HostEntry {
    user: Option<String>,
    reasoning_effort: Option<String>,
    text_verbosity: Option<String>,
    strict_json_schema: Option<bool>,
    /// The host entry's other keys, under the names the body takes.
    rest: Map<String, Value>,
}

impl HostEntry {
    /// Read the call's entries, or `None` for a host that reads none.
    pub(super) fn read(
        profile: &ChatProfile,
        p: &InferenceRequestParams,
    ) -> Result<Option<Self>, Error> {
        let Some(key) = profile.host_options.key() else {
            return Ok(None);
        };
        let mut entry = Self::default();
        // The host's entry reads after `openaiCompatible`, so it wins.
        for name in ["openaiCompatible", key] {
            let Some(value) = p
                .provider_params
                .as_ref()
                .and_then(|params| params.get(name))
            else {
                continue;
            };
            let Value::Object(options) = value else {
                return Err(invalid(format!(
                    "{name} provider options must be an object"
                )));
            };
            entry.read_compatible(name, options)?;
            if name == key {
                for (option, value) in options {
                    if !COMPATIBLE_KEYS.contains(&option.as_str()) {
                        entry.rest.insert(option.clone(), value.clone());
                    }
                }
            }
        }
        if profile.host_options == HostOptions::Fireworks {
            fireworks(&mut entry.rest)?;
        }
        Ok(Some(entry))
    }

    fn read_compatible(&mut self, name: &str, options: &Map<String, Value>) -> Result<(), Error> {
        let text = |option: &str| match options.get(option) {
            None => Ok(None),
            Some(Value::String(value)) => Ok(Some(value.clone())),
            Some(_) => Err(invalid(format!("{name} option {option} must be a string"))),
        };
        if let Some(user) = text("user")? {
            self.user = Some(user);
        }
        if let Some(effort) = text("reasoningEffort")? {
            self.reasoning_effort = Some(effort);
        }
        if let Some(verbosity) = text("textVerbosity")? {
            self.text_verbosity = Some(verbosity);
        }
        match options.get("strictJsonSchema") {
            None => {}
            Some(Value::Bool(strict)) => self.strict_json_schema = Some(*strict),
            Some(_) => {
                return Err(invalid(format!(
                    "{name} option strictJsonSchema must be a boolean"
                )));
            }
        }
        Ok(())
    }

    /// The `strict` of a JSON schema response format.
    pub(super) fn strict(&self) -> bool {
        self.strict_json_schema.unwrap_or(true)
    }

    /// Put the entries into the body over the core settings they name.
    pub(super) fn apply(&self, profile: &ChatProfile, body: &mut Map<String, Value>) {
        for (option, value) in &self.rest {
            if !KEPT.contains(&option.as_str()) {
                body.insert(option.clone(), value.clone());
            }
        }
        if let Some(user) = &self.user {
            body.insert("user".into(), json!(user));
        }
        if let Some(effort) = &self.reasoning_effort {
            body.insert("reasoning_effort".into(), json!(profile.effort(effort)));
        }
        if let Some(verbosity) = &self.text_verbosity {
            body.insert("verbosity".into(), json!(verbosity));
        }
    }
}

/// Give Fireworks' options their wire names, checking their values.
fn fireworks(rest: &mut Map<String, Value>) -> Result<(), Error> {
    let one_of = |option: &str, value: &Value, allowed: &[&str]| match value.as_str() {
        Some(value) if allowed.contains(&value) => Ok(()),
        _ => Err(invalid(format!(
            "fireworks option {option} must be one of {}, not {value}",
            allowed.join(", ")
        ))),
    };
    if let Some(key) = rest.remove("promptCacheKey") {
        if !key.is_string() {
            return Err(invalid("fireworks option promptCacheKey must be a string"));
        }
        rest.insert("prompt_cache_key".into(), key);
    }
    if let Some(tier) = rest.remove("serviceTier") {
        one_of("serviceTier", &tier, &["priority"])?;
        rest.insert("service_tier".into(), tier);
    }
    if let Some(thinking) = rest.remove("thinking") {
        let Value::Object(thinking) = thinking else {
            return Err(invalid("fireworks option thinking must be an object"));
        };
        let mut sent = Map::new();
        if let Some(kind) = thinking.get("type") {
            one_of("thinking.type", kind, &["enabled", "disabled"])?;
            sent.insert("type".into(), kind.clone());
        }
        if let Some(budget) = thinking.get("budgetTokens") {
            if budget.as_u64().is_none_or(|tokens| tokens < 1024) {
                return Err(invalid(format!(
                    "fireworks option thinking.budgetTokens must be an integer of at least 1024, not {budget}"
                )));
            }
            sent.insert("budget_tokens".into(), budget.clone());
        }
        rest.insert("thinking".into(), Value::Object(sent));
    }
    if let Some(history) = rest.remove("reasoningHistory") {
        one_of(
            "reasoningHistory",
            &history,
            &["disabled", "interleaved", "preserved"],
        )?;
        rest.insert("reasoning_history".into(), history);
    }
    Ok(())
}
