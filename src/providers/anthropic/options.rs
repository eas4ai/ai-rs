//! The `anthropic` provider options of a Messages call, with the TypeScript
//! option names (`anthropic-language-model-options.ts`), and the options of
//! system messages and file parts. An unknown option name or a value
//! outside an option's schema fails with `Error::Validation` before any
//! request (ANT-002).

use serde::{Deserialize, Deserializer};
use serde_json::{Map, Number, Value, json};

use crate::{Error, models::params::InferenceRequestParams};

fn invalid(message: impl Into<String>) -> Error {
    Error::Validation(message.into())
}

/// A field that TypeScript marks `optional` and not `nullish`: absent, or a
/// value other than `null`.
fn not_null<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Option<T>, D::Error> {
    T::deserialize(d).map(Some)
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub(crate) enum StructuredOutputMode {
    #[serde(rename = "outputFormat")]
    OutputFormat,
    #[serde(rename = "jsonTool")]
    JsonTool,
    #[serde(rename = "auto")]
    Auto,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ThinkingType {
    Adaptive,
    Enabled,
    Disabled,
    BetweenTools,
}
impl ThinkingType {
    pub(crate) fn wire(self) -> &'static str {
        match self {
            Self::Adaptive => "adaptive",
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
            Self::BetweenTools => "between_tools",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct BlockBinding {
    pub prefix_mismatch_behavior: PrefixMismatch,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PrefixMismatch {
    Error,
    DropBlock,
}

/// `thinking`: one of TypeScript's union members, each with only its own
/// fields, or a block binding alone.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Thinking {
    #[serde(rename = "type", default, deserialize_with = "not_null")]
    pub kind: Option<ThinkingType>,
    #[serde(default, deserialize_with = "not_null")]
    pub budget_tokens: Option<Number>,
    #[serde(default, deserialize_with = "not_null")]
    pub display: Option<Display>,
    #[serde(default, deserialize_with = "not_null")]
    pub block_binding: Option<BlockBinding>,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Display {
    Omitted,
    Summarized,
    Updates,
}
impl Display {
    pub(crate) fn wire(self) -> &'static str {
        match self {
            Self::Omitted => "omitted",
            Self::Summarized => "summarized",
            Self::Updates => "updates",
        }
    }
    pub(crate) fn parse(text: &str) -> Option<Self> {
        match text {
            "omitted" => Some(Self::Omitted),
            "summarized" => Some(Self::Summarized),
            "updates" => Some(Self::Updates),
            _ => None,
        }
    }
}
impl Thinking {
    pub(crate) fn of(kind: ThinkingType) -> Self {
        Self {
            kind: Some(kind),
            ..Default::default()
        }
    }
    fn validate(&self) -> Result<(), Error> {
        let wrong = |field: &str| {
            Err(invalid(format!(
                "anthropic option thinking does not take {field} with type {:?}",
                self.kind.map_or("none", ThinkingType::wire)
            )))
        };
        match self.kind {
            None if self.block_binding.is_none() => {
                return Err(invalid(
                    "anthropic option thinking needs a type or a blockBinding",
                ));
            }
            Some(ThinkingType::Adaptive) | None => {
                if self.budget_tokens.is_some() {
                    return wrong("budgetTokens");
                }
                if self.kind.is_none() && self.display.is_some() {
                    return wrong("display");
                }
            }
            Some(ThinkingType::Enabled) => {
                if self.display.is_some() {
                    return wrong("display");
                }
                if self.block_binding.is_some() {
                    return wrong("blockBinding");
                }
            }
            Some(ThinkingType::Disabled | ThinkingType::BetweenTools) => {
                if self.budget_tokens.is_some() {
                    return wrong("budgetTokens");
                }
                if self.display.is_some() {
                    return wrong("display");
                }
                if self.block_binding.is_some() {
                    return wrong("blockBinding");
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Effort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}
impl Effort {
    pub(crate) fn wire(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CacheControl {
    #[serde(rename = "type")]
    pub kind: Ephemeral,
    #[serde(default, deserialize_with = "not_null")]
    pub ttl: Option<Ttl>,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Ephemeral {
    Ephemeral,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub(crate) enum Ttl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct MetadataOption {
    #[serde(default, deserialize_with = "not_null")]
    pub user_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct McpServer {
    #[serde(rename = "type")]
    pub kind: Url,
    pub name: String,
    pub url: String,
    #[serde(default)]
    pub authorization_token: Option<Option<String>>,
    #[serde(default)]
    pub tool_configuration: Option<Option<ToolConfiguration>>,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Url {
    Url,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ToolConfiguration {
    #[serde(default)]
    pub enabled: Option<Option<bool>>,
    #[serde(default)]
    pub allowed_tools: Option<Option<Vec<String>>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Container {
    #[serde(default, deserialize_with = "not_null")]
    pub id: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub skills: Option<Vec<Skill>>,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Skill {
    #[serde(rename_all = "camelCase")]
    Anthropic {
        skill_id: String,
        #[serde(default, deserialize_with = "not_null")]
        version: Option<String>,
    },
    #[serde(rename_all = "camelCase")]
    Custom {
        provider_reference: std::collections::BTreeMap<String, String>,
        #[serde(default, deserialize_with = "not_null")]
        version: Option<String>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct TaskBudget {
    #[serde(rename = "type")]
    pub kind: Tokens,
    pub total: u64,
    #[serde(default, deserialize_with = "not_null")]
    pub remaining: Option<u64>,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Tokens {
    Tokens,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Speed {
    Fast,
    Standard,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Fallbacks {
    Default,
    List(Vec<Value>),
}
impl<'de> Deserialize<'de> for Fallbacks {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Entry {
            #[allow(dead_code)]
            model: String,
            #[serde(default, deserialize_with = "not_null")]
            #[allow(dead_code)]
            max_tokens: Option<u64>,
            #[serde(default, deserialize_with = "not_null")]
            #[allow(dead_code)]
            thinking: Option<Map<String, Value>>,
            #[serde(default, deserialize_with = "not_null")]
            #[allow(dead_code)]
            output_config: Option<Map<String, Value>>,
            #[serde(default, deserialize_with = "not_null")]
            #[allow(dead_code)]
            speed: Option<Speed>,
        }
        let value = Value::deserialize(d)?;
        match &value {
            Value::String(text) if text == "default" => Ok(Self::Default),
            Value::Array(entries) => {
                for entry in entries {
                    Entry::deserialize(entry).map_err(serde::de::Error::custom)?;
                }
                Ok(Self::List(entries.clone()))
            }
            _ => Err(serde::de::Error::custom(
                "fallbacks must be \"default\" or a list of fallback models",
            )),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Safeguard {
    #[serde(rename = "type")]
    pub kind: DangerousToolUse,
    #[serde(default, deserialize_with = "not_null")]
    pub classifier_context: Option<Map<String, Value>>,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DangerousToolUse {
    DangerousToolUse,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Compaction {
    #[serde(rename = "type")]
    pub kind: Summarize,
    #[serde(default, deserialize_with = "not_null")]
    pub instructions: Option<String>,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Summarize {
    Summarize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ContextManagement {
    pub edits: Vec<Edit>,
}
/// A `{type, value}` count, sent as given.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Count {
    #[serde(rename = "type")]
    pub kind: String,
    pub value: Number,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", deny_unknown_fields)]
pub(crate) enum Edit {
    #[serde(rename = "clear_tool_uses_20250919", rename_all = "camelCase")]
    ClearToolUses {
        #[serde(default, deserialize_with = "not_null")]
        trigger: Option<Count>,
        #[serde(default, deserialize_with = "not_null")]
        keep: Option<Count>,
        #[serde(default, deserialize_with = "not_null")]
        clear_at_least: Option<Count>,
        #[serde(default, deserialize_with = "not_null")]
        clear_tool_inputs: Option<bool>,
        #[serde(default, deserialize_with = "not_null")]
        exclude_tools: Option<Vec<String>>,
    },
    #[serde(rename = "clear_thinking_20251015")]
    ClearThinking {
        #[serde(default, deserialize_with = "not_null")]
        keep: Option<Value>,
    },
    #[serde(rename = "compact_20260112", rename_all = "camelCase")]
    Compact {
        #[serde(default, deserialize_with = "not_null")]
        trigger: Option<Count>,
        #[serde(default, deserialize_with = "not_null")]
        pause_after_compaction: Option<bool>,
        #[serde(default, deserialize_with = "not_null")]
        instructions: Option<String>,
    },
}
impl Count {
    fn check(&self, edit: &str, field: &str, kinds: &[&str]) -> Result<(), Error> {
        if kinds.contains(&self.kind.as_str()) {
            Ok(())
        } else {
            Err(invalid(format!(
                "anthropic context management {edit} {field} takes type {}, not {:?}",
                kinds.join(" or "),
                self.kind
            )))
        }
    }
    fn wire(&self) -> Value {
        json!({"type": self.kind, "value": self.value})
    }
}
impl Edit {
    fn validate(&self) -> Result<(), Error> {
        match self {
            Self::ClearToolUses {
                trigger,
                keep,
                clear_at_least,
                ..
            } => {
                let edit = "clear_tool_uses_20250919";
                if let Some(t) = trigger {
                    t.check(edit, "trigger", &["input_tokens", "tool_uses"])?;
                }
                if let Some(k) = keep {
                    k.check(edit, "keep", &["tool_uses"])?;
                }
                if let Some(c) = clear_at_least {
                    c.check(edit, "clearAtLeast", &["input_tokens"])?;
                }
            }
            Self::ClearThinking { keep } => {
                let all = keep.as_ref().is_none_or(|k| k == "all");
                let turns = keep.as_ref().is_some_and(|k| {
                    k.as_object().is_some_and(|o| {
                        o.len() == 2
                            && o.get("type") == Some(&json!("thinking_turns"))
                            && o["value"].is_number()
                    })
                });
                if !all && !turns {
                    return Err(invalid(
                        "anthropic context management clear_thinking_20251015 keep takes \"all\" or {type: \"thinking_turns\", value}",
                    ));
                }
            }
            Self::Compact { trigger, .. } => {
                if let Some(t) = trigger {
                    t.check("compact_20260112", "trigger", &["input_tokens"])?;
                }
            }
        }
        Ok(())
    }
    /// The edit with snake-case keys.
    pub(crate) fn wire(&self) -> Value {
        let mut out = Map::new();
        let mut put = |key: &str, value: Option<Value>| {
            if let Some(value) = value {
                out.insert(key.into(), value);
            }
        };
        match self {
            Self::ClearToolUses {
                trigger,
                keep,
                clear_at_least,
                clear_tool_inputs,
                exclude_tools,
            } => {
                put("type", Some(json!("clear_tool_uses_20250919")));
                put("trigger", trigger.as_ref().map(Count::wire));
                put("keep", keep.as_ref().map(Count::wire));
                put("clear_at_least", clear_at_least.as_ref().map(Count::wire));
                put("clear_tool_inputs", clear_tool_inputs.map(Value::Bool));
                put("exclude_tools", exclude_tools.as_ref().map(|t| json!(t)));
            }
            Self::ClearThinking { keep } => {
                put("type", Some(json!("clear_thinking_20251015")));
                put("keep", keep.clone());
            }
            Self::Compact {
                trigger,
                pause_after_compaction,
                instructions,
            } => {
                put("type", Some(json!("compact_20260112")));
                put("trigger", trigger.as_ref().map(Count::wire));
                put(
                    "pause_after_compaction",
                    pause_after_compaction.map(Value::Bool),
                );
                put("instructions", instructions.as_ref().map(|i| json!(i)));
            }
        }
        Value::Object(out)
    }
}

/// The language model options (`anthropicLanguageModelOptions`).
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct AnthropicOptions {
    #[serde(default, deserialize_with = "not_null")]
    pub send_reasoning: Option<bool>,
    #[serde(default, deserialize_with = "not_null")]
    pub structured_output_mode: Option<StructuredOutputMode>,
    #[serde(default, deserialize_with = "not_null")]
    pub thinking: Option<Thinking>,
    #[serde(default, deserialize_with = "not_null")]
    pub disable_parallel_tool_use: Option<bool>,
    #[serde(default, deserialize_with = "not_null")]
    pub cache_control: Option<CacheControl>,
    #[serde(default, deserialize_with = "not_null")]
    pub metadata: Option<MetadataOption>,
    #[serde(default, deserialize_with = "not_null")]
    pub mcp_servers: Option<Vec<McpServer>>,
    #[serde(default, deserialize_with = "not_null")]
    pub container: Option<Container>,
    #[serde(default, deserialize_with = "not_null")]
    pub tool_streaming: Option<bool>,
    #[serde(default, deserialize_with = "not_null")]
    pub effort: Option<Effort>,
    #[serde(default, deserialize_with = "not_null")]
    pub task_budget: Option<TaskBudget>,
    #[serde(default, deserialize_with = "not_null")]
    pub speed: Option<Speed>,
    #[serde(default, deserialize_with = "not_null")]
    pub service_tier: Option<ServiceTier>,
    #[serde(default, deserialize_with = "not_null")]
    pub inference_geo: Option<InferenceGeo>,
    #[serde(default, deserialize_with = "not_null")]
    pub fallbacks: Option<Fallbacks>,
    #[serde(default, deserialize_with = "not_null")]
    pub anthropic_beta: Option<Vec<String>>,
    #[serde(default, deserialize_with = "not_null")]
    pub safeguards: Option<Vec<Safeguard>>,
    #[serde(default, deserialize_with = "not_null")]
    pub compaction: Option<Compaction>,
    #[serde(default, deserialize_with = "not_null")]
    pub context_management: Option<ContextManagement>,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ServiceTier {
    Auto,
    StandardOnly,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InferenceGeo {
    Us,
    Global,
}

impl AnthropicOptions {
    /// The options of a call: its `anthropic` provider parameters, then those
    /// under `extra_key` over them, for a host that reads its own key.
    pub(crate) fn parse(
        p: Option<&InferenceRequestParams>,
        extra_key: Option<&str>,
    ) -> Result<Self, Error> {
        let params = p.and_then(|p| p.provider_params.as_ref());
        let mut merged = Map::new();
        for key in std::iter::once("anthropic").chain(extra_key) {
            match params.and_then(|params| params.get(key)) {
                None | Some(Value::Null) => {}
                Some(Value::Object(options)) => merged.extend(options.clone()),
                Some(_) => {
                    return Err(invalid(format!(
                        "the {key} provider options must be an object"
                    )));
                }
            }
        }
        let options: Self = serde_json::from_value(Value::Object(merged))
            .map_err(|e| invalid(format!("invalid anthropic provider options: {e}")))?;
        options.validate()?;
        Ok(options)
    }

    fn validate(&self) -> Result<(), Error> {
        if self.compaction.is_some() && self.context_management.is_some() {
            return Err(invalid(
                "anthropic options compaction and contextManagement cannot be used together",
            ));
        }
        if let Some(thinking) = &self.thinking {
            thinking.validate()?;
        }
        if let Some(budget) = &self.task_budget
            && budget.total < 20_000
        {
            return Err(invalid(
                "anthropic option taskBudget.total must be at least 20000",
            ));
        }
        for edit in self.context_management.iter().flat_map(|m| &m.edits) {
            edit.validate()?;
        }
        Ok(())
    }
}

/// The options of a system message (`anthropicSystemMessageProviderOptions`).
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SystemOptions {
    #[serde(default, deserialize_with = "not_null")]
    pub clear_at: Option<ClearAt>,
    #[serde(default, deserialize_with = "not_null")]
    pub effort: Option<Effort>,
    #[serde(default, deserialize_with = "not_null")]
    pub tool_changes: Option<Vec<ToolChange>>,
}
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ClearAt {
    NextUserMessage,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ToolChange {
    #[serde(rename_all = "camelCase")]
    ToolAddition { tool_name: String },
    #[serde(rename_all = "camelCase")]
    ToolRemoval { tool_name: String },
}

/// The options of a file part (`anthropicFilePartProviderOptions`).
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct FileOptions {
    #[serde(default, deserialize_with = "not_null")]
    pub container_upload: Option<bool>,
    #[serde(default, deserialize_with = "not_null")]
    pub citations: Option<Citations>,
    #[serde(default, deserialize_with = "not_null")]
    pub title: Option<String>,
    #[serde(default, deserialize_with = "not_null")]
    pub context: Option<String>,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct Citations {
    pub enabled: bool,
}

/// Read a part's `anthropic` metadata as options of type `T`. Other keys in
/// the metadata, such as a signature or cache control, are not options and
/// are passed over, as TypeScript's schema strips them.
pub(crate) fn part_options<T: for<'de> Deserialize<'de> + Default>(
    metadata: Option<&Value>,
    what: &str,
) -> Result<T, Error> {
    match metadata {
        None | Some(Value::Null) => Ok(T::default()),
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|e| invalid(format!("invalid anthropic {what} options: {e}"))),
    }
}
