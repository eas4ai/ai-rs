//! DeepSeek's own request rules beyond the dialect's field names, as
//! `@ai-sdk/deepseek` 3.0.70 sends them: the `deepseek` options entry of a
//! chat call (HST-033). A value outside an option's schema, or an option
//! name the entry does not take, fails with `Error::Validation` before any
//! request, as the OpenAI and Anthropic option entries do; TypeScript's
//! schema strips an unknown name.

use serde_json::{Map, Value, json};

use super::{
    dialect::{self, Effort},
    params::invalid,
};
use crate::{Error, models::params::InferenceRequestParams};

/// The `deepseek` entry of a chat call (`deepseekLanguageModelChatOptions`).
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Options {
    logprobs: Option<bool>,
    top_logprobs: Option<u64>,
    user_id: Option<String>,
    /// `enabled` or `disabled`; `adaptive` is read as `enabled`.
    thinking: Option<&'static str>,
    /// The effort as DeepSeek takes it: `low`, `high` or `max`.
    reasoning_effort: Option<&'static str>,
    /// Read as TypeScript reads it, and never sent: DeepSeek's provider has
    /// only JSON mode, which takes no `strict`.
    strict_json_schema: Option<bool>,
}

fn bad(option: &str, expected: &str, value: &Value) -> Error {
    invalid(format!(
        "deepseek option {option} must be {expected}, not {value}"
    ))
}

impl Options {
    /// The `deepseek` entry of the call's provider parameters; none gives the
    /// defaults.
    pub(super) fn parse(p: &InferenceRequestParams) -> Result<Self, Error> {
        let Some(value) = p
            .provider_params
            .as_ref()
            .and_then(|params| params.get("deepseek"))
        else {
            return Ok(Self::default());
        };
        let Value::Object(entry) = value else {
            return Err(invalid("deepseek provider options must be an object"));
        };
        let mut options = Self::default();
        for (option, value) in entry {
            match option.as_str() {
                "logprobs" => {
                    options.logprobs = Some(
                        value
                            .as_bool()
                            .ok_or_else(|| bad(option, "a boolean", value))?,
                    );
                }
                "topLogprobs" => {
                    let top = value
                        .as_u64()
                        .filter(|top| *top <= 20)
                        .ok_or_else(|| bad(option, "an integer from 0 to 20", value))?;
                    options.top_logprobs = Some(top);
                }
                "userId" => {
                    let id = value
                        .as_str()
                        .filter(|id| {
                            !id.is_empty()
                                && id.len() <= 512
                                && id
                                    .bytes()
                                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                        })
                        .ok_or_else(|| {
                            bad(
                                option,
                                "a string of at most 512 ASCII letters, digits, _ and -",
                                value,
                            )
                        })?;
                    options.user_id = Some(id.to_owned());
                }
                "thinking" => {
                    let Value::Object(thinking) = value else {
                        return Err(bad(option, "an object", value));
                    };
                    for (key, value) in thinking {
                        if key != "type" {
                            return Err(invalid(format!(
                                "deepseek option thinking.{key} is not known"
                            )));
                        }
                        options.thinking = Some(match value.as_str() {
                            Some("adaptive" | "enabled") => "enabled",
                            Some("disabled") => "disabled",
                            _ => {
                                return Err(bad(
                                    "thinking.type",
                                    "adaptive, enabled or disabled",
                                    value,
                                ));
                            }
                        });
                    }
                }
                "reasoningEffort" => {
                    options.reasoning_effort = Some(match value.as_str() {
                        Some("low") => "low",
                        Some("medium" | "high") => "high",
                        Some("xhigh" | "max") => "max",
                        _ => return Err(bad(option, "low, medium, high, xhigh or max", value)),
                    });
                }
                "strictJsonSchema" => {
                    options.strict_json_schema = Some(
                        value
                            .as_bool()
                            .ok_or_else(|| bad(option, "a boolean", value))?,
                    );
                }
                other => {
                    return Err(invalid(format!("deepseek option {other} is not known")));
                }
            }
        }
        Ok(options)
    }

    /// The `thinking` the call sends: the option, else the core reasoning
    /// setting, else nothing.
    fn thinking(&self, p: &InferenceRequestParams) -> Option<&'static str> {
        self.thinking.or_else(|| match dialect::effort(p)? {
            Effort::Disabled => Some("disabled"),
            Effort::Level(_) => Some("enabled"),
        })
    }

    /// Whether the model thinks in this call, which decides HST-009's
    /// temperature and top-p rule: the thinking the call sends, else the
    /// model's own behavior.
    pub(super) fn thinking_on(&self, model: &str, p: &InferenceRequestParams) -> bool {
        match self.thinking(p) {
            Some("disabled") => false,
            Some(_) => true,
            None => model == "deepseek-reasoner" || !dialect::deepseek_legacy(model),
        }
    }

    /// Put the options into the body over the core settings they replace. A
    /// core effort is already in the body under DeepSeek's name; the option's
    /// effort replaces it, and neither goes while thinking is disabled.
    pub(super) fn apply(&self, p: &InferenceRequestParams, body: &mut Map<String, Value>) {
        let thinking = self.thinking(p);
        if let Some(kind) = thinking {
            body.insert("thinking".into(), json!({"type": kind}));
        }
        if let Some(effort) = self.reasoning_effort {
            body.insert("reasoning_effort".into(), json!(effort));
        }
        if thinking == Some("disabled") {
            body.remove("reasoning_effort");
        }
        if self.logprobs == Some(true) || self.top_logprobs.is_some() {
            body.insert("logprobs".into(), json!(true));
        }
        if let Some(top) = self.top_logprobs {
            body.insert("top_logprobs".into(), json!(top));
        }
        if let Some(user) = &self.user_id {
            body.insert("user_id".into(), json!(user));
        }
    }
}
