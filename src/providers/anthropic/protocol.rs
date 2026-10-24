//! The Anthropic Messages protocol: the request of a call, as `prepareRequest`
//! in `anthropic-language-model.ts` builds it (ANT-002 to ANT-006), and its
//! stream.
pub use super::messages::{file_to_anthropic, messages_to_anthropic, tool_result_to_anthropic};
use super::{
    capabilities::{Capabilities, Notes},
    errors, messages,
    options::{
        AnthropicOptions, Display, Effort, Fallbacks, Skill, StructuredOutputMode, Thinking,
        ThinkingType,
    },
    params::{self, Reasoning, add_betas},
    stream::Reading,
    tools::{self, ToolNames, ToolRules},
};
use crate::{
    Error, Model, Provider, ProviderConfig, ProviderProtocol, StreamOptions,
    models::files::DownloadTransport,
    providers::{
        EventStream, http,
        operations::{EvaluationInput, EvaluationParams, Item, Question},
    },
    transport::HttpRequest,
    types::{FileData, Message, MessageRole, Part, messages::Metadata, tools::ToolChoice},
};
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default)]
pub struct AnthropicMessagesProtocol;

/// How a server's Messages API differs from Anthropic's.
#[derive(Debug)]
pub(crate) struct MessagesProfile {
    /// The protocol's class id in snapshots.
    pub protocol_class_id: &'static str,
    pub api: http::ApiProfile,
    /// Whether a model always reasons, so that it takes a reasoning effort
    /// and disabled reasoning becomes the lowest effort.
    pub always_reasons: fn(&str) -> bool,
    /// The configuration `env` value that holds a bearer token sent in
    /// place of the API key, for Anthropic's own API (ANT-001).
    pub auth_token_env: Option<&'static str>,
    /// Whether the server takes `output_config.format`; without it an output
    /// schema goes as the JSON tool.
    pub native_structured_output: bool,
    /// Whether the server takes strict function tools.
    pub strict_tools: bool,
    /// The highest temperature the server takes for a model.
    pub max_temperature: fn(&str) -> f64,
    /// Whether a model keeps temperature and top-p with thinking.
    pub sampling_with_thinking: fn(&str) -> bool,
}

const ANTHROPIC_RULES: MessagesProfile = MessagesProfile {
    protocol_class_id: "anthropic_messages",
    api: errors::API,
    always_reasons: |_| false,
    auth_token_env: None,
    native_structured_output: true,
    strict_tools: true,
    max_temperature: |_| 1.0,
    sampling_with_thinking: |_| false,
};

/// The Messages API of any server, as `AnthropicMessagesProtocol` speaks it:
/// no Anthropic token, whatever the configuration's `env` holds.
pub(crate) static ANTHROPIC: MessagesProfile = ANTHROPIC_RULES;

/// Anthropic's own Messages API, for the provider of Anthropic's API only:
/// an `ANTHROPIC_AUTH_TOKEN` in `env` goes as a bearer token (ANT-001).
pub(crate) static ANTHROPIC_OWN: MessagesProfile = MessagesProfile {
    auth_token_env: Some("ANTHROPIC_AUTH_TOKEN"),
    ..ANTHROPIC_RULES
};

/// The Messages protocol for a host, as its profile describes it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostMessagesProtocol(pub(crate) &'static MessagesProfile);

/// The name of the JSON tool that carries an output schema to a model that
/// answers through it (ANT-005).
pub(crate) const JSON_TOOL: &str = "json";

/// A Messages request before its URL: the body, the headers, the query and
/// the betas it needs, what its stream reader needs to know, and the
/// changes ANT-004 made to it.
pub(crate) struct MessagesRequest {
    pub body: Map<String, Value>,
    pub headers: BTreeMap<String, String>,
    pub query: BTreeMap<String, Value>,
    pub betas: Vec<String>,
    pub reading: Reading,
    pub changes: Vec<(String, String)>,
}

impl MessagesRequest {
    /// The request's headers with one `anthropic-beta` header (ANT-001).
    pub(crate) fn headers(&self, config: &ProviderConfig) -> BTreeMap<String, String> {
        with_betas(config, &self.headers, &self.betas)
    }
}

/// Headers with one `anthropic-beta` header: these betas, then those the
/// configuration's and these headers' own `anthropic-beta` headers name,
/// once each (ANT-001).
pub(crate) fn with_betas(
    config: &ProviderConfig,
    headers: &BTreeMap<String, String>,
    betas: &[String],
) -> BTreeMap<String, String> {
    let named = |headers: &BTreeMap<String, String>| {
        headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
            .flat_map(|(_, value)| value.split(','))
            .map(|beta| beta.trim().to_ascii_lowercase())
            .filter(|beta| !beta.is_empty())
            .collect::<Vec<_>>()
    };
    let mut all = betas.to_vec();
    let user = [named(&config.headers), named(headers)].concat();
    add_betas(&mut all, user.iter().map(String::as_str));
    let mut out: BTreeMap<String, String> = headers
        .iter()
        .filter(|(name, _)| !name.eq_ignore_ascii_case("anthropic-beta"))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    if !all.is_empty() {
        out.insert("anthropic-beta".into(), all.join(","));
    }
    out
}

/// Send the profile's bearer token, when the configuration holds one, in
/// place of the API key; with an API key value too it fails.
pub(crate) fn authorize(
    profile: &MessagesProfile,
    config: &ProviderConfig,
    request: &mut HttpRequest,
) -> Result<(), Error> {
    let Some(token) = profile
        .auth_token_env
        .and_then(|env| config.env.get(env))
        .filter(|token| !token.is_empty())
    else {
        return Ok(());
    };
    if config
        .api_key_value
        .as_deref()
        .is_some_and(|key| !key.is_empty())
    {
        return Err(Error::Configuration(format!(
            "{} takes an API key or an auth token, not both",
            profile.api.label
        )));
    }
    request
        .headers
        .retain(|(name, _)| !name.eq_ignore_ascii_case("x-api-key"));
    if !request
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("authorization"))
    {
        request
            .headers
            .push(("authorization".into(), format!("Bearer {token}")));
    }
    Ok(())
}

/// The thinking and effort a core reasoning level maps to
/// (`resolveAnthropicReasoningConfig`).
fn reasoning_config(
    reasoning: &Reasoning,
    caps: &Capabilities,
    notes: &mut Notes,
) -> (Option<Thinking>, Option<Effort>) {
    match reasoning {
        Reasoning::Default => (None, None),
        Reasoning::Disabled if caps.between_tools_thinking => {
            (Some(Thinking::of(ThinkingType::BetweenTools)), None)
        }
        Reasoning::Disabled if caps.rejects_disabled_thinking => {
            notes.note(
                "reasoning",
                "reasoning none is not taken by this model, which always thinks; effort low is sent instead",
            );
            (None, Some(Effort::Low))
        }
        Reasoning::Disabled => (Some(Thinking::of(ThinkingType::Disabled)), None),
        Reasoning::Level(level) if caps.adaptive_thinking => {
            let effort = match level.as_str() {
                "minimal" | "low" => Effort::Low,
                "medium" => Effort::Medium,
                "high" => Effort::High,
                "xhigh" if caps.xhigh_effort => Effort::Xhigh,
                _ => Effort::Max,
            };
            if effort.wire() != level {
                notes.note(
                    "reasoning",
                    format!(
                        "reasoning {level:?} is not taken directly; effort {:?} is sent",
                        effort.wire()
                    ),
                );
            }
            let thinking = Thinking {
                display: Some(Display::Summarized),
                ..Thinking::of(ThinkingType::Adaptive)
            };
            (Some(thinking), Some(effort))
        }
        Reasoning::Level(level) => {
            let percent = match level.as_str() {
                "minimal" => 0.02,
                "low" => 0.1,
                "medium" => 0.3,
                "high" => 0.6,
                "xhigh" => 0.9,
                _ => 0.95,
            };
            let max = caps.max_output_tokens;
            let budget = ((max as f64 * percent).round() as u64).clamp(1024, max.max(1024));
            let thinking = Thinking {
                budget_tokens: Some(budget.min(max).into()),
                ..Thinking::of(ThinkingType::Enabled)
            };
            (Some(thinking), None)
        }
    }
}

/// The effort of a host model that always reasons: disabled reasoning and
/// `minimal` as `low`, and no thinking setting (HST-007).
fn always_reasoning(reasoning: &Reasoning) -> (Option<Thinking>, Option<Effort>) {
    let effort = match reasoning {
        Reasoning::Default => return (None, None),
        Reasoning::Disabled => Effort::Low,
        Reasoning::Level(level) => match level.as_str() {
            "minimal" | "low" => Effort::Low,
            "medium" => Effort::Medium,
            "high" => Effort::High,
            "xhigh" => Effort::Xhigh,
            _ => Effort::Max,
        },
    };
    (None, Some(effort))
}

/// The body, headers and betas of a Messages call, and what its stream
/// reader needs. A buffered call (`streaming` false) sends no `stream`.
pub(crate) fn request_parts(
    profile: &MessagesProfile,
    model: &str,
    messages: &[Message],
    options: &StreamOptions,
    streaming: bool,
) -> Result<MessagesRequest, Error> {
    let label = profile.api.label;
    let mut core = params::read(options.params.as_ref(), label)?;
    let mut o = AnthropicOptions::parse(options.params.as_ref(), None)?;
    let caps = Capabilities::of(model);
    let mut notes = Notes::new(model);
    let mut betas: Vec<String> = Vec::new();

    let max_temperature = (profile.max_temperature)(model);
    if let Some(t) = core.temperature {
        if t > max_temperature {
            notes.note(
                "temperature",
                format!(
                    "{t} is above the maximum of {max_temperature:.1}; {max_temperature:.1} is sent"
                ),
            );
            core.temperature = Some(max_temperature);
        } else if t < 0.0 {
            notes.note(
                "temperature",
                format!("{t} is below the minimum of 0; 0 is sent"),
            );
            core.temperature = Some(0.0);
        }
    }
    if !caps.known && core.max_tokens.is_none() {
        notes.note(
            "maxOutputTokens",
            format!(
                "the model is unknown, so its output is limited to {} tokens; set an output limit to change it",
                caps.max_output_tokens
            ),
        );
    }
    if caps.rejects_sampling {
        if core.temperature.take().is_some() {
            notes.note(
                "temperature",
                "the model takes no temperature; it is left out",
            );
        }
        if core.top_k.take().is_some() {
            notes.note("topK", "the model takes no top-k; it is left out");
        }
        if core.top_p.take().is_some() {
            notes.note("topP", "the model takes no top-p; it is left out");
        }
    }
    let claude = Capabilities::claude(model);
    let structured_output = profile.native_structured_output && caps.structured_output;
    let strict_tools = profile.strict_tools && caps.structured_output;
    let mode = o
        .structured_output_mode
        .unwrap_or(StructuredOutputMode::Auto);
    let schema = options.output_schema.as_ref();
    let mut use_format = mode == StructuredOutputMode::OutputFormat
        || (mode == StructuredOutputMode::Auto && structured_output);
    if !use_format && caps.rejects_forced_tool_use && structured_output && schema.is_some() {
        notes.note(
            "structuredOutputMode",
            "the JSON tool needs a forced tool choice, which the model rejects; the output format is used instead",
        );
        use_format = true;
    }
    let json_tool = schema.is_some() && !use_format;
    if json_tool && o.disable_parallel_tool_use == Some(false) {
        notes.note(
            "disableParallelToolUse",
            "parallel tool use stays off with the JSON tool, so that one JSON call answers",
        );
    }

    let names = ToolNames::of(&options.tools);
    let send_reasoning = o.send_reasoning.unwrap_or(true);
    let converted = messages::convert(
        messages,
        &messages::Context {
            send_reasoning,
            names: &names,
        },
        &mut notes,
    )?;
    add_betas(&mut betas, converted.betas.iter().map(String::as_str));

    // The core reasoning maps to thinking and effort the options do not set.
    if o.effort.is_none() {
        let (thinking, effort) = if (profile.always_reasons)(model) {
            always_reasoning(&core.reasoning)
        } else {
            reasoning_config(&core.reasoning, &caps, &mut notes)
        };
        if o.thinking.is_none() {
            o.thinking = thinking;
        }
        if effort.is_some()
            && o.thinking.as_ref().and_then(|t| t.kind) != Some(ThinkingType::Disabled)
        {
            o.effort = effort;
        }
    }
    // A core reasoning summary sets the display of adaptive thinking.
    if let Some(summary) = &core.summary {
        let display = match summary.as_deref() {
            None => Display::Omitted,
            Some(text) => Display::parse(text).unwrap_or(Display::Summarized),
        };
        let reasoning_off = core.reasoning == Reasoning::Disabled;
        match o.thinking.as_mut() {
            Some(t) if t.kind == Some(ThinkingType::Adaptive) => {
                if t.display.is_none() || core.reasoning != Reasoning::Default {
                    t.display = Some(display);
                }
            }
            None if !reasoning_off && caps.adaptive_thinking => {
                o.thinking = Some(Thinking {
                    display: Some(display),
                    ..Thinking::of(ThinkingType::Adaptive)
                });
            }
            None if !reasoning_off => notes.note(
                "reasoning summary",
                "the model has no adaptive thinking to display; the summary is left out",
            ),
            _ => {}
        }
    }
    if caps.rejects_disabled_thinking
        && let Some(thinking) = &o.thinking
        && thinking.kind == Some(ThinkingType::Disabled)
    {
        if caps.between_tools_thinking {
            notes.note(
                "thinking",
                "thinking cannot be disabled for this model; between_tools thinking, its lowest setting, is sent",
            );
            o.thinking = Some(Thinking::of(ThinkingType::BetweenTools));
        } else {
            notes.note(
                "thinking",
                "thinking cannot be disabled for this model, which always thinks; the setting is left out",
            );
            o.thinking = None;
        }
    }
    if caps.rejects_budget_thinking
        && o.thinking.as_ref().and_then(|t| t.kind) == Some(ThinkingType::Enabled)
    {
        notes.note(
            "thinking",
            "budget thinking is not taken by this model; adaptive thinking is sent, and effort sets how much it thinks",
        );
        o.thinking = Some(Thinking::of(ThinkingType::Adaptive));
    }
    let high_or_more = matches!(o.effort, Some(Effort::Xhigh | Effort::Max));
    let thinking_type = o.thinking.as_ref().and_then(|t| t.kind);
    if high_or_more
        && ((caps.rejects_disabled_above_high && thinking_type == Some(ThinkingType::Disabled))
            || thinking_type == Some(ThinkingType::BetweenTools))
    {
        notes.note(
            "effort",
            format!(
                "effort {:?} is not taken with this thinking; effort high is sent",
                o.effort.map_or("", Effort::wire)
            ),
        );
        o.effort = Some(Effort::High);
    }

    let thinking_type = o.thinking.as_ref().and_then(|t| t.kind);
    let thinking_on = matches!(
        thinking_type,
        Some(ThinkingType::Enabled | ThinkingType::Adaptive | ThinkingType::BetweenTools)
    );
    let binding = o.thinking.as_ref().and_then(|t| t.block_binding.clone());
    let mut budget = o
        .thinking
        .as_ref()
        .filter(|_| thinking_type == Some(ThinkingType::Enabled))
        .and_then(|t| t.budget_tokens.clone());
    let display = o
        .thinking
        .as_ref()
        .filter(|_| thinking_type == Some(ThinkingType::Adaptive))
        .and_then(|t| t.display);
    let max_tokens = core.max_tokens.unwrap_or(caps.max_output_tokens);

    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    let mut total_tokens = max_tokens;
    if thinking_on {
        if thinking_type == Some(ThinkingType::Enabled) && budget.is_none() {
            notes.note(
                "extended thinking",
                "enabled thinking needs a budget; the default of 1024 tokens is sent",
            );
            budget = Some(1024.into());
        }
        let keeps_sampling = (profile.sampling_with_thinking)(model);
        if core.temperature.is_some() && !keeps_sampling {
            core.temperature = None;
            notes.note(
                "temperature",
                "temperature is not taken with thinking; it is left out",
            );
        }
        if core.top_k.take().is_some() {
            notes.note("topK", "top-k is not taken with thinking; it is left out");
        }
        if core.top_p.is_some() && !keeps_sampling {
            core.top_p = None;
            notes.note("topP", "top-p is not taken with thinking; it is left out");
        }
        let budget_tokens = budget.as_ref().and_then(|b| b.as_u64()).unwrap_or(0);
        total_tokens = max_tokens.saturating_add(budget_tokens);
    } else if claude && core.top_p.is_some() && core.temperature.is_some() {
        notes.note(
            "topP",
            "top-p is not taken together with a temperature; top-p is left out",
        );
        core.top_p = None;
    }
    if caps.known && total_tokens > caps.max_output_tokens {
        if core.max_tokens.is_some() {
            notes.note(
                "maxOutputTokens",
                format!(
                    "{total_tokens} (the output limit and the thinking budget) is above the model's limit of {}; the limit is sent",
                    caps.max_output_tokens
                ),
            );
        }
        total_tokens = caps.max_output_tokens;
    }
    body.insert("max_tokens".into(), json!(total_tokens));
    if let Some(t) = core.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(k) = core.top_k {
        body.insert("top_k".into(), json!(k));
    }
    if let Some(p) = core.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if thinking_on || thinking_type == Some(ThinkingType::Disabled) || binding.is_some() {
        let mut thinking = Map::new();
        if let Some(kind) = thinking_type {
            thinking.insert("type".into(), json!(kind.wire()));
        }
        if let Some(budget) = &budget {
            thinking.insert("budget_tokens".into(), json!(budget));
        }
        if let Some(display) = display {
            thinking.insert("display".into(), json!(display.wire()));
        }
        if let Some(binding) = &binding {
            thinking.insert(
                "block_binding".into(),
                json!({"prefix_mismatch_behavior": match binding.prefix_mismatch_behavior {
                    super::options::PrefixMismatch::Error => "error",
                    super::options::PrefixMismatch::DropBlock => "drop_block",
                }}),
            );
        }
        body.insert("thinking".into(), Value::Object(thinking));
    }
    let mut output = Map::new();
    if let Some(effort) = o.effort {
        output.insert("effort".into(), json!(effort.wire()));
    }
    if let Some(budget) = &o.task_budget {
        let mut task = json!({"type": "tokens", "total": budget.total});
        if let Some(remaining) = budget.remaining {
            task["remaining"] = json!(remaining);
        }
        output.insert("task_budget".into(), task);
    }
    if let (true, Some(schema)) = (use_format, schema) {
        output.insert(
            "format".into(),
            json!({"type": "json_schema", "schema": super::schema::sanitize(schema.as_value())?}),
        );
    }
    if !output.is_empty() {
        body.insert("output_config".into(), Value::Object(output));
    }
    if let Some(speed) = o.speed {
        body.insert(
            "speed".into(),
            json!(match speed {
                super::options::Speed::Fast => "fast",
                super::options::Speed::Standard => "standard",
            }),
        );
    }
    let service_tier = match o.service_tier {
        Some(super::options::ServiceTier::Auto) => Some("auto".to_owned()),
        Some(super::options::ServiceTier::StandardOnly) => Some("standard_only".to_owned()),
        None => core.service_tier.clone(),
    };
    if let Some(tier) = service_tier {
        body.insert("service_tier".into(), json!(tier));
    }
    if let Some(geo) = o.inference_geo {
        body.insert(
            "inference_geo".into(),
            json!(match geo {
                super::options::InferenceGeo::Us => "us",
                super::options::InferenceGeo::Global => "global",
            }),
        );
    }
    match &o.fallbacks {
        Some(Fallbacks::Default) => {
            body.insert("fallbacks".into(), json!("default"));
        }
        Some(Fallbacks::List(list)) if !list.is_empty() => {
            body.insert("fallbacks".into(), json!(list));
        }
        _ => {}
    }
    if let Some(cache) = &o.cache_control {
        let mut control = json!({"type": "ephemeral"});
        if let Some(ttl) = cache.ttl {
            control["ttl"] = json!(match ttl {
                super::options::Ttl::FiveMinutes => "5m",
                super::options::Ttl::OneHour => "1h",
            });
        }
        body.insert("cache_control".into(), control);
    } else if let Some(retention) = &core.cache {
        let mut control = json!({"type": "ephemeral"});
        if let Some(ttl) = retention {
            control["ttl"] = json!(ttl);
        }
        body.insert("cache_control".into(), control);
    }
    let user_id = o
        .metadata
        .as_ref()
        .and_then(|m| m.user_id.clone())
        .or(core.safety_identifier.clone());
    if let Some(user_id) = user_id {
        body.insert("metadata".into(), json!({"user_id": user_id}));
    }
    if let Some(servers) = o.mcp_servers.as_ref().filter(|s| !s.is_empty()) {
        let servers: Vec<Value> = servers
            .iter()
            .map(|server| {
                let mut wire = json!({"type": "url", "name": server.name, "url": server.url});
                if let Some(token) = &server.authorization_token {
                    wire["authorization_token"] = json!(token);
                }
                if let Some(Some(config)) = &server.tool_configuration {
                    let mut tools = Map::new();
                    if let Some(allowed) = &config.allowed_tools {
                        tools.insert("allowed_tools".into(), json!(allowed));
                    }
                    if let Some(enabled) = &config.enabled {
                        tools.insert("enabled".into(), json!(enabled));
                    }
                    wire["tool_configuration"] = Value::Object(tools);
                }
                wire
            })
            .collect();
        body.insert("mcp_servers".into(), json!(servers));
        add_betas(&mut betas, ["mcp-client-2025-04-04"]);
    }
    if let Some(container) = &o.container {
        match container.skills.as_ref().filter(|s| !s.is_empty()) {
            Some(skills) => {
                let skills = skills
                    .iter()
                    .map(|skill| match skill {
                        Skill::Anthropic { skill_id, version } => {
                            let mut wire = json!({"type": "anthropic", "skill_id": skill_id});
                            if let Some(version) = version {
                                wire["version"] = json!(version);
                            }
                            Ok(wire)
                        }
                        Skill::Custom {
                            provider_reference,
                            version,
                        } => {
                            let id = provider_reference.get("anthropic").ok_or_else(|| {
                                Error::Validation(
                                    "a custom container skill needs an anthropic provider reference"
                                        .into(),
                                )
                            })?;
                            let mut wire = json!({"type": "custom", "skill_id": id});
                            if let Some(version) = version {
                                wire["version"] = json!(version);
                            }
                            Ok(wire)
                        }
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let mut wire = json!({"skills": skills});
                if let Some(id) = &container.id {
                    wire["id"] = json!(id);
                }
                body.insert("container".into(), wire);
                add_betas(
                    &mut betas,
                    [
                        "code-execution-2025-08-25",
                        "skills-2025-10-02",
                        "files-api-2025-04-14",
                    ],
                );
                if !tools::has_code_execution(&options.tools) {
                    notes.note("container", "skills need a code execution tool");
                }
            }
            None => {
                if let Some(id) = &container.id {
                    body.insert("container".into(), json!(id));
                }
            }
        }
    }
    if let Some(system) = converted.system.filter(|s| !s.is_empty()) {
        body.insert("system".into(), json!(system));
    }
    body.insert("messages".into(), json!(converted.messages));
    if let Some(safeguards) = o.safeguards.as_ref().filter(|s| !s.is_empty()) {
        let wire: Vec<Value> = safeguards
            .iter()
            .map(|s| {
                let mut wire = json!({"type": "dangerous_tool_use"});
                if let Some(context) = &s.classifier_context {
                    wire["classifier_context"] = json!(context);
                }
                wire
            })
            .collect();
        body.insert("safeguards".into(), json!(wire));
        add_betas(&mut betas, ["dangerous-tool-use-2026-09-03"]);
    }
    if let Some(compaction) = &o.compaction {
        let mut wire = json!({"type": "summarize"});
        if let Some(instructions) = &compaction.instructions {
            wire["instructions"] = json!(instructions);
        }
        body.insert("compaction".into(), wire);
        add_betas(&mut betas, ["compact-2026-09-04"]);
    }
    if let Some(management) = &o.context_management {
        let edits: Vec<Value> = management.edits.iter().map(|e| e.wire()).collect();
        let compact = edits.iter().any(|e| e["type"] == "compact_20260112");
        body.insert("context_management".into(), json!({"edits": edits}));
        add_betas(&mut betas, ["context-management-2025-06-27"]);
        if compact {
            add_betas(&mut betas, ["compact-2026-01-12"]);
        }
    } else if let Some(threshold) = core.compaction {
        body.insert(
            "context_management".into(),
            json!({"edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": threshold}}]}),
        );
        add_betas(
            &mut betas,
            ["context-management-2025-06-27", "compact-2026-01-12"],
        );
    }
    if o.task_budget.is_some() {
        add_betas(&mut betas, ["task-budgets-2026-03-13"]);
    }
    if o.speed == Some(super::options::Speed::Fast) {
        add_betas(&mut betas, ["fast-mode-2026-02-01"]);
    }
    if display == Some(Display::Updates) {
        add_betas(&mut betas, ["thinking-display-updates-2026-08-18"]);
    }
    if binding.is_some() {
        add_betas(&mut betas, ["thinking-binding-controls-2026-08-01"]);
    }
    match &o.fallbacks {
        Some(Fallbacks::Default) => add_betas(&mut betas, ["server-side-fallback-2026-07-01"]),
        Some(Fallbacks::List(list)) if !list.is_empty() => {
            add_betas(&mut betas, ["server-side-fallback-2026-06-01"])
        }
        _ => {}
    }

    // The tools, with the JSON tool after the call's own.
    let mut call_tools = options.tools.clone();
    let choice = options.tool_choice.clone().or(core.tool_choice.clone());
    let disable_parallel = o
        .disable_parallel_tool_use
        .or(core.parallel_tool_calls.map(|parallel| !parallel));
    let rejects_forced =
        caps.rejects_forced_tool_use || thinking_type == Some(ThinkingType::Enabled);
    let (choice, rules) = match (json_tool, schema) {
        (true, Some(schema)) => {
            let spec = crate::types::tools::ToolSpec {
                description: Some("Respond with a JSON object.".into()),
                params: schema.clone(),
            };
            call_tools.push(crate::types::tools::Tool::function(JSON_TOOL, spec));
            (
                Some(ToolChoice::Required),
                ToolRules {
                    structured_output: false,
                    strict_tools,
                    eager_input_streaming: streaming && o.tool_streaming.unwrap_or(true),
                    rejects_forced_tool_use: rejects_forced,
                    disable_parallel_tool_use: Some(true),
                },
            )
        }
        _ => (
            choice,
            ToolRules {
                structured_output,
                strict_tools,
                eager_input_streaming: streaming && o.tool_streaming.unwrap_or(true),
                rejects_forced_tool_use: rejects_forced,
                disable_parallel_tool_use: disable_parallel,
            },
        ),
    };
    let prepared = tools::prepare(&call_tools, choice.as_ref(), &rules, &mut notes, &mut betas)?;
    if !prepared.tools.is_empty() {
        body.insert("tools".into(), json!(prepared.tools));
    }
    if let Some(choice) = prepared.tool_choice {
        body.insert("tool_choice".into(), choice);
    }
    if streaming {
        body.insert("stream".into(), json!(true));
    }
    if let Some(user_betas) = &o.anthropic_beta {
        add_betas(&mut betas, user_betas.iter().map(String::as_str));
    }
    // The Python SDK merges extra_body after all named request parameters.
    body.extend(core.extra_body);
    let documents = messages::citation_documents(messages);
    Ok(MessagesRequest {
        body,
        headers: core.headers,
        query: core.query,
        betas,
        reading: Reading {
            json_tool,
            names,
            documents,
        },
        changes: notes.into_changes(),
    })
}

/// Whether a user file goes to Anthropic by URL: images and PDFs; another
/// file given by URL is downloaded first.
fn sent_by_url(media_type: &str) -> bool {
    media_type.starts_with("image/") || media_type == "application/pdf"
}
fn download_url(part: &Part) -> Option<&str> {
    match part {
        Part::File(file) if !sent_by_url(&file.media_type) => file
            .data
            .as_text()
            .filter(|url| crate::types::media::is_downloadable_url(url)),
        _ => None,
    }
}
/// The messages with each file to download given as empty bytes, to check
/// the request before any download; `None` when there is none.
fn without_downloads(messages: &[Message]) -> Option<Vec<Message>> {
    let mut found = false;
    let mut messages = messages.to_vec();
    for msg in messages.iter_mut().filter(|m| m.role == MessageRole::User) {
        for part in &mut msg.parts {
            if download_url(part).is_some()
                && let Part::File(file) = part
            {
                file.data = FileData::Bytes(Vec::new());
                found = true;
            }
        }
    }
    found.then_some(messages)
}
/// Download each user file that Anthropic does not take by URL.
async fn download_files(
    mut messages: Vec<Message>,
    transport: &dyn DownloadTransport,
    label: &str,
) -> Result<Vec<Message>, Error> {
    for msg in messages.iter_mut().filter(|m| m.role == MessageRole::User) {
        for part in &mut msg.parts {
            let Some(url) = download_url(part).map(str::to_owned) else {
                continue;
            };
            let (bytes, _) = crate::models::files::download(
                transport,
                &url,
                crate::models::files::DEFAULT_MAX_BYTES,
            )
            .await
            .map_err(|e| {
                crate::errors::ProviderError::new(
                    crate::errors::ProviderErrorKind::Response,
                    format!("{label} input download failed"),
                )
                .with_source(e)
            })?;
            if let Part::File(file) = part {
                file.data = FileData::Bytes(bytes.to_vec());
            }
        }
    }
    Ok(messages)
}

/// The HTTP request of a Messages call on Anthropic's own paths.
pub(crate) fn http_request(
    profile: &MessagesProfile,
    config: &ProviderConfig,
    parts: MessagesRequest,
) -> Result<(HttpRequest, Reading), Error> {
    let headers = parts.headers(config);
    let mut request = http::request(
        &profile.api,
        config,
        "POST",
        &["v1", "messages"],
        Some(Value::Object(parts.body)),
        headers,
        parts.query,
    )?;
    authorize(profile, config, &mut request)?;
    Ok((request, parts.reading))
}

fn start(
    profile: &'static MessagesProfile,
    model: &Model,
    messages: Vec<Message>,
    options: StreamOptions,
    provider: &dyn Provider,
) -> EventStream {
    let transport = provider.shared_transport();
    let downloads = provider.download_transport();
    let config = provider.config().clone();
    let name = provider.name().to_owned();
    let model_id = model.id.clone();
    stream::once(async move {
        // Check the request before any download, so that a call that fails
        // validation fetches nothing.
        let messages = match (&downloads, without_downloads(&messages)) {
            (Some(downloads), Some(inline)) => {
                request_parts(profile, &model_id, &inline, &options, true)?;
                download_files(messages, downloads.as_ref(), profile.api.label).await?
            }
            _ => messages,
        };
        let parts = request_parts(profile, &model_id, &messages, &options, true)?;
        let (request, reading) = http_request(profile, &config, parts)?;
        let transport = transport.ok_or_else(|| {
            Error::Configuration(format!(
                "{} streams require an owned HttpTransport handle (Provider::shared_transport)",
                profile.api.label
            ))
        })?;
        let response = http::send(
            &profile.api,
            transport.as_ref(),
            request,
            &name,
            Some(&model_id),
        )
        .await?;
        Ok::<_, Error>(super::stream::parse(
            response.body,
            name,
            &profile.api,
            reading,
        ))
    })
    .try_flatten()
    .boxed()
}

async fn evaluate(
    profile: &'static MessagesProfile,
    model: &Model,
    state: EvaluationInput,
    questions: BTreeMap<String, Question>,
    params: EvaluationParams,
    provider: &dyn Provider,
) -> Result<Item<Metadata>, Error> {
    let name = provider.name().to_owned();
    crate::ops::experimental::evaluate_with_language_model(
        provider.name(),
        state,
        questions,
        params,
        |messages, options| async move {
            let events: Vec<_> = start(profile, model, messages, options, provider)
                .try_collect()
                .await?;
            crate::providers::google::language::hydrate(&name, events)
        },
    )
    .await
}

#[async_trait]
impl ProviderProtocol for AnthropicMessagesProtocol {
    fn protocol_class_id(&self) -> &str {
        ANTHROPIC.protocol_class_id
    }
    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        start(&ANTHROPIC, model, messages, options, provider)
    }
    async fn evaluate(
        &self,
        model: &Model,
        state: EvaluationInput,
        questions: BTreeMap<String, Question>,
        params: EvaluationParams,
        provider: &dyn Provider,
    ) -> Result<Item<Metadata>, Error> {
        evaluate(&ANTHROPIC, model, state, questions, params, provider).await
    }
}

#[async_trait]
impl ProviderProtocol for HostMessagesProtocol {
    fn protocol_class_id(&self) -> &str {
        self.0.protocol_class_id
    }
    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        start(self.0, model, messages, options, provider)
    }
    async fn evaluate(
        &self,
        model: &Model,
        state: EvaluationInput,
        questions: BTreeMap<String, Question>,
        params: EvaluationParams,
        provider: &dyn Provider,
    ) -> Result<Item<Metadata>, Error> {
        evaluate(self.0, model, state, questions, params, provider).await
    }
}
