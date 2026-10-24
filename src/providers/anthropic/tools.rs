//! Anthropic's provider tools, with a function per tool version of
//! `anthropicTools` (`anthropic-tools.ts`), and the request form of the
//! tools and tool choice of a call (`anthropic-prepare-tools.ts`, ANT-006).
//!
//! A provider tool's `tool_config` arguments are TypeScript's arguments in
//! snake case. The tool's own name is the application's; the request names
//! the tool by its Anthropic name, and calls and results come back under the
//! application's name. A function tool takes `defer_loading`,
//! `allowed_callers`, `input_examples`, `eager_input_streaming` and
//! `cache_control` from its `tool_config` arguments.
use crate::{
    Error,
    types::tools::{Tool, ToolChoice, ToolChoiceMode, ToolConfig, ToolKind},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

use super::{capabilities::Notes, params::add_betas};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserLocation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Citations {
    pub enabled: bool,
}
impl From<bool> for Citations {
    fn from(enabled: bool) -> Self {
        Self { enabled }
    }
}
/// Whether a web tool's results stay in the response (`full`) or are left
/// out of it (`excluded`), for the 2026-03-18 web tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseInclusion {
    Full,
    Excluded,
}
#[derive(Clone, Debug, Default)]
pub struct WebSearchOptions {
    pub max_uses: Option<i64>,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
    pub user_location: Option<UserLocation>,
    /// For `web_search_20260318` only.
    pub response_inclusion: Option<ResponseInclusion>,
}
#[derive(Clone, Debug, Default)]
pub struct WebFetchOptions {
    pub max_uses: Option<i64>,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
    pub citations: Option<Citations>,
    pub max_content_tokens: Option<i64>,
    /// For `web_fetch_20260318` only.
    pub use_cache: Option<bool>,
    /// For `web_fetch_20260318` only.
    pub response_inclusion: Option<ResponseInclusion>,
}
#[derive(Clone, Debug)]
pub struct ComputerUseOptions {
    pub display_width_px: i64,
    pub display_height_px: i64,
    pub display_number: Option<i64>,
    /// For `computer_20251124` only.
    pub enable_zoom: Option<bool>,
}
/// A member of the computer toolset: whether it is on and loaded lazily.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolsetMember {
    pub enabled: Option<bool>,
    pub defer_loading: Option<bool>,
}
/// The advisor's prompt caching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdvisorCacheTtl {
    FiveMinutes,
    OneHour,
}
#[derive(Clone, Debug, Default)]
pub struct AdvisorOptions {
    pub model: String,
    pub max_uses: Option<i64>,
    /// At least 1024.
    pub max_tokens: Option<i64>,
    pub caching: Option<AdvisorCacheTtl>,
}

/// The members of `computer_toolset_20260801`.
pub const COMPUTER_TOOLSET_MEMBERS: &[&str] = &[
    "screenshot",
    "zoom",
    "left_click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "left_click_drag",
    "mouse_move",
    "left_mouse_down",
    "left_mouse_up",
    "cursor_position",
    "scroll",
    "type",
    "key",
    "hold_key",
    "wait",
];

fn declaration(name: &str, id: &str, args: Value) -> Tool {
    Tool {
        kind: ToolKind::Provider,
        name: name.into(),
        spec: None,
        tool_config: Some(ToolConfig {
            id: Some(format!("anthropic.{id}")),
            args: args.as_object().cloned().unwrap_or_default(),
        }),
        require_approval: false,
    }
}
fn non_null(value: Value) -> Value {
    Value::Object(
        value
            .as_object()
            .into_iter()
            .flatten()
            .filter(|(_, v)| !v.is_null())
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    )
}
fn check_domains(
    name: &str,
    allowed: &Option<Vec<String>>,
    blocked: &Option<Vec<String>>,
) -> Result<(), Error> {
    if allowed.as_ref().is_some_and(|v| !v.is_empty())
        && blocked.as_ref().is_some_and(|v| !v.is_empty())
    {
        return Err(Error::Validation(format!(
            "anthropic.{name}: pass only one of allowed_domains or blocked_domains"
        )));
    }
    Ok(())
}
fn search(id: &str, options: WebSearchOptions) -> Result<Tool, Error> {
    check_domains(id, &options.allowed_domains, &options.blocked_domains)?;
    let location = options.user_location.map(|loc| {
        let mut v = json!(loc);
        v["type"] = json!("approximate");
        v
    });
    Ok(declaration(
        "web_search",
        id,
        non_null(json!({
            "max_uses": options.max_uses,
            "allowed_domains": options.allowed_domains,
            "blocked_domains": options.blocked_domains,
            "user_location": location,
            "response_inclusion": options.response_inclusion,
        })),
    ))
}
fn fetch(id: &str, options: WebFetchOptions) -> Result<Tool, Error> {
    check_domains(id, &options.allowed_domains, &options.blocked_domains)?;
    Ok(declaration(
        "web_fetch",
        id,
        non_null(json!({
            "max_uses": options.max_uses,
            "allowed_domains": options.allowed_domains,
            "blocked_domains": options.blocked_domains,
            "citations": options.citations,
            "max_content_tokens": options.max_content_tokens,
            "use_cache": options.use_cache,
            "response_inclusion": options.response_inclusion,
        })),
    ))
}
fn computer(id: &str, options: ComputerUseOptions) -> Tool {
    declaration(
        "computer",
        id,
        non_null(json!({
            "display_width_px": options.display_width_px,
            "display_height_px": options.display_height_px,
            "display_number": options.display_number,
            "enable_zoom": options.enable_zoom,
        })),
    )
}

/// The advisor, version 2026-03-01: another model the model can consult.
pub fn advisor_20260301(options: AdvisorOptions) -> Result<Tool, Error> {
    if options.max_tokens.is_some_and(|max| max < 1024) {
        return Err(Error::Validation(
            "anthropic.advisor_20260301: max_tokens must be at least 1024".into(),
        ));
    }
    let caching = options.caching.map(|ttl| {
        json!({"type": "ephemeral", "ttl": match ttl {
            AdvisorCacheTtl::FiveMinutes => "5m",
            AdvisorCacheTtl::OneHour => "1h",
        }})
    });
    Ok(declaration(
        "advisor",
        "advisor_20260301",
        non_null(json!({
            "model": options.model,
            "max_uses": options.max_uses,
            "max_tokens": options.max_tokens,
            "caching": caching,
        })),
    ))
}
/// The bash tool, version 2024-10-22, which the application runs.
pub fn bash_20241022() -> Tool {
    declaration("bash", "bash_20241022", json!({}))
}
/// The bash tool, version 2025-01-24, which the application runs.
pub fn bash_20250124() -> Tool {
    declaration("bash", "bash_20250124", json!({}))
}
/// Code execution, version 2025-05-22, which Anthropic runs.
pub fn code_execution_20250522() -> Tool {
    declaration("code_execution", "code_execution_20250522", json!({}))
}
/// Code execution, version 2025-08-25, with bash and file editing.
pub fn code_execution_20250825() -> Tool {
    declaration("code_execution", "code_execution_20250825", json!({}))
}
/// Code execution, version 2026-01-20, with programmatic tool calling.
pub fn code_execution_20260120() -> Tool {
    declaration("code_execution", "code_execution_20260120", json!({}))
}
/// Computer use, version 2024-10-22, which the application runs.
pub fn computer_20241022(options: ComputerUseOptions) -> Tool {
    computer(
        "computer_20241022",
        ComputerUseOptions {
            enable_zoom: None,
            ..options
        },
    )
}
/// Computer use, version 2025-01-24, which the application runs.
pub fn computer_20250124(options: ComputerUseOptions) -> Tool {
    computer(
        "computer_20250124",
        ComputerUseOptions {
            enable_zoom: None,
            ..options
        },
    )
}
/// Computer use, version 2025-11-24, with zoom.
pub fn computer_20251124(options: ComputerUseOptions) -> Tool {
    computer("computer_20251124", options)
}
/// The computer toolset, version 2026-08-01: its members are separate tools
/// for the model, and their calls come back as calls of this tool with the
/// member's name as `action`.
pub fn computer_toolset_20260801(configs: BTreeMap<String, ToolsetMember>) -> Result<Tool, Error> {
    let mut wire = Map::new();
    for (member, config) in configs {
        if !COMPUTER_TOOLSET_MEMBERS.contains(&member.as_str()) {
            return Err(Error::Validation(format!(
                "anthropic.computer_toolset_20260801 has no member {member:?}"
            )));
        }
        wire.insert(
            member,
            non_null(json!({"enabled": config.enabled, "defer_loading": config.defer_loading})),
        );
    }
    let args = if wire.is_empty() {
        json!({})
    } else {
        json!({"configs": wire})
    };
    Ok(declaration("computer", "computer_toolset_20260801", args))
}
/// The memory tool, version 2025-08-18, which the application runs.
pub fn memory_20250818() -> Tool {
    declaration("memory", "memory_20250818", json!({}))
}
/// The text editor, version 2024-10-22, which the application runs.
pub fn text_editor_20241022() -> Tool {
    declaration("str_replace_editor", "text_editor_20241022", json!({}))
}
/// The text editor, version 2025-01-24, which the application runs.
pub fn text_editor_20250124() -> Tool {
    declaration("str_replace_editor", "text_editor_20250124", json!({}))
}
/// The text editor, version 2025-04-29, which the application runs.
pub fn text_editor_20250429() -> Tool {
    declaration(
        "str_replace_based_edit_tool",
        "text_editor_20250429",
        json!({}),
    )
}
/// The text editor, version 2025-07-28, with an optional view limit.
pub fn text_editor_20250728(max_characters: Option<i64>) -> Tool {
    declaration(
        "str_replace_based_edit_tool",
        "text_editor_20250728",
        non_null(json!({"max_characters": max_characters})),
    )
}
/// Web fetch, version 2025-09-10, which Anthropic runs.
pub fn web_fetch_20250910(options: WebFetchOptions) -> Result<Tool, Error> {
    fetch(
        "web_fetch_20250910",
        WebFetchOptions {
            use_cache: None,
            response_inclusion: None,
            ..options
        },
    )
}
/// Web fetch, version 2026-02-09, with dynamic filtering.
pub fn web_fetch_20260209(options: WebFetchOptions) -> Result<Tool, Error> {
    fetch(
        "web_fetch_20260209",
        WebFetchOptions {
            use_cache: None,
            response_inclusion: None,
            ..options
        },
    )
}
/// Web fetch, version 2026-03-18, with its cache and response inclusion.
pub fn web_fetch_20260318(options: WebFetchOptions) -> Result<Tool, Error> {
    fetch("web_fetch_20260318", options)
}
/// Web search, version 2025-03-05, which Anthropic runs.
pub fn web_search_20250305(options: WebSearchOptions) -> Result<Tool, Error> {
    search(
        "web_search_20250305",
        WebSearchOptions {
            response_inclusion: None,
            ..options
        },
    )
}
/// Web search, version 2026-02-09, with dynamic filtering.
pub fn web_search_20260209(options: WebSearchOptions) -> Result<Tool, Error> {
    search(
        "web_search_20260209",
        WebSearchOptions {
            response_inclusion: None,
            ..options
        },
    )
}
/// Web search, version 2026-03-18, with response inclusion.
pub fn web_search_20260318(options: WebSearchOptions) -> Result<Tool, Error> {
    search("web_search_20260318", options)
}
/// Tool search by regular expression, version 2025-11-19.
pub fn tool_search_regex_20251119() -> Tool {
    declaration(
        "tool_search_tool_regex",
        "tool_search_regex_20251119",
        json!({}),
    )
}
/// Tool search by BM25, version 2025-11-19.
pub fn tool_search_bm25_20251119() -> Tool {
    declaration(
        "tool_search_tool_bm25",
        "tool_search_bm25_20251119",
        json!({}),
    )
}

/// Web search, version 2026-02-09 (`web_search_20260209`).
pub fn web_search(options: WebSearchOptions) -> Result<Tool, Error> {
    web_search_20260209(options)
}
/// Web fetch, version 2026-02-09 (`web_fetch_20260209`).
pub fn web_fetch(options: WebFetchOptions) -> Result<Tool, Error> {
    web_fetch_20260209(options)
}
/// Computer use, version 2025-11-24 (`computer_20251124`).
pub fn computer_use(options: ComputerUseOptions) -> Tool {
    computer_20251124(options)
}
/// The text editor, version 2025-07-28 (`text_editor_20250728`).
pub fn text_editor(max_characters: Option<i64>) -> Tool {
    text_editor_20250728(max_characters)
}
/// Code execution, version 2026-01-20 (`code_execution_20260120`).
pub fn code_execution() -> Tool {
    code_execution_20260120()
}
/// The bash tool, version 2025-01-24 (`bash_20250124`).
pub fn bash() -> Tool {
    bash_20250124()
}
/// The memory tool, version 2025-08-18 (`memory_20250818`).
pub fn memory() -> Tool {
    memory_20250818()
}

/// One provider tool version: its Anthropic tool name, its wire type, its
/// beta, and the arguments it takes in wire order.
struct Version {
    id: &'static str,
    name: &'static str,
    wire_type: &'static str,
    beta: Option<&'static str>,
    args: &'static [&'static str],
    /// Arguments it cannot go without.
    required: &'static [&'static str],
}
const COMPUTER_ARGS: &[&str] = &["display_width_px", "display_height_px", "display_number"];
const DISPLAY: &[&str] = &["display_width_px", "display_height_px"];
const FETCH_ARGS: &[&str] = &[
    "max_uses",
    "allowed_domains",
    "blocked_domains",
    "citations",
    "max_content_tokens",
];
const SEARCH_ARGS: &[&str] = &[
    "max_uses",
    "allowed_domains",
    "blocked_domains",
    "user_location",
];
const VERSIONS: &[Version] = &[
    Version {
        id: "code_execution_20250522",
        name: "code_execution",
        wire_type: "code_execution_20250522",
        beta: Some("code-execution-2025-05-22"),
        args: &[],
        required: &[],
    },
    Version {
        id: "code_execution_20250825",
        name: "code_execution",
        wire_type: "code_execution_20250825",
        beta: Some("code-execution-2025-08-25"),
        args: &[],
        required: &[],
    },
    Version {
        id: "code_execution_20260120",
        name: "code_execution",
        wire_type: "code_execution_20260120",
        beta: None,
        args: &[],
        required: &[],
    },
    Version {
        id: "computer_20241022",
        name: "computer",
        wire_type: "computer_20241022",
        beta: Some("computer-use-2024-10-22"),
        args: COMPUTER_ARGS,
        required: DISPLAY,
    },
    Version {
        id: "computer_20250124",
        name: "computer",
        wire_type: "computer_20250124",
        beta: Some("computer-use-2025-01-24"),
        args: COMPUTER_ARGS,
        required: DISPLAY,
    },
    Version {
        id: "computer_20251124",
        name: "computer",
        wire_type: "computer_20251124",
        beta: Some("computer-use-2025-11-24"),
        args: &[
            "display_width_px",
            "display_height_px",
            "display_number",
            "enable_zoom",
        ],
        required: DISPLAY,
    },
    Version {
        id: "computer_toolset_20260801",
        name: "computer",
        wire_type: "computer_toolset_20260801",
        beta: None,
        args: &["configs"],
        required: &[],
    },
    Version {
        id: "text_editor_20241022",
        name: "str_replace_editor",
        wire_type: "text_editor_20241022",
        beta: Some("computer-use-2024-10-22"),
        args: &[],
        required: &[],
    },
    Version {
        id: "text_editor_20250124",
        name: "str_replace_editor",
        wire_type: "text_editor_20250124",
        beta: Some("computer-use-2025-01-24"),
        args: &[],
        required: &[],
    },
    Version {
        id: "text_editor_20250429",
        name: "str_replace_based_edit_tool",
        wire_type: "text_editor_20250429",
        beta: Some("computer-use-2025-01-24"),
        args: &[],
        required: &[],
    },
    Version {
        id: "text_editor_20250728",
        name: "str_replace_based_edit_tool",
        wire_type: "text_editor_20250728",
        beta: None,
        args: &["max_characters"],
        required: &[],
    },
    Version {
        id: "bash_20241022",
        name: "bash",
        wire_type: "bash_20241022",
        beta: Some("computer-use-2024-10-22"),
        args: &[],
        required: &[],
    },
    Version {
        id: "bash_20250124",
        name: "bash",
        wire_type: "bash_20250124",
        beta: Some("computer-use-2025-01-24"),
        args: &[],
        required: &[],
    },
    Version {
        id: "memory_20250818",
        name: "memory",
        wire_type: "memory_20250818",
        beta: Some("context-management-2025-06-27"),
        args: &[],
        required: &[],
    },
    Version {
        id: "web_fetch_20250910",
        name: "web_fetch",
        wire_type: "web_fetch_20250910",
        beta: Some("web-fetch-2025-09-10"),
        args: FETCH_ARGS,
        required: &[],
    },
    Version {
        id: "web_fetch_20260209",
        name: "web_fetch",
        wire_type: "web_fetch_20260209",
        beta: Some("code-execution-web-tools-2026-02-09"),
        args: FETCH_ARGS,
        required: &[],
    },
    Version {
        id: "web_fetch_20260318",
        name: "web_fetch",
        wire_type: "web_fetch_20260318",
        beta: None,
        args: &[
            "max_uses",
            "allowed_domains",
            "blocked_domains",
            "citations",
            "max_content_tokens",
            "use_cache",
            "response_inclusion",
        ],
        required: &[],
    },
    Version {
        id: "web_search_20250305",
        name: "web_search",
        wire_type: "web_search_20250305",
        beta: None,
        args: SEARCH_ARGS,
        required: &[],
    },
    Version {
        id: "web_search_20260209",
        name: "web_search",
        wire_type: "web_search_20260209",
        beta: Some("code-execution-web-tools-2026-02-09"),
        args: SEARCH_ARGS,
        required: &[],
    },
    Version {
        id: "web_search_20260318",
        name: "web_search",
        wire_type: "web_search_20260318",
        beta: None,
        args: &[
            "max_uses",
            "allowed_domains",
            "blocked_domains",
            "user_location",
            "response_inclusion",
        ],
        required: &[],
    },
    Version {
        id: "tool_search_regex_20251119",
        name: "tool_search_tool_regex",
        wire_type: "tool_search_tool_regex_20251119",
        beta: None,
        args: &[],
        required: &[],
    },
    Version {
        id: "tool_search_bm25_20251119",
        name: "tool_search_tool_bm25",
        wire_type: "tool_search_tool_bm25_20251119",
        beta: None,
        args: &[],
        required: &[],
    },
    Version {
        id: "advisor_20260301",
        name: "advisor",
        wire_type: "advisor_20260301",
        beta: Some("advisor-tool-2026-03-01"),
        args: &["model", "max_uses", "max_tokens", "caching"],
        required: &["model"],
    },
];

/// The version a provider tool's id names: `anthropic.<id>`, or the wire
/// type for the tool search versions, as Vertex AI's helpers name them.
fn version(tool: &Tool) -> Option<&'static Version> {
    let config = tool.tool_config.as_ref()?;
    let id = config.id.as_deref()?.strip_prefix("anthropic.")?;
    VERSIONS.iter().find(|v| v.id == id || v.wire_type == id)
}

/// The Anthropic tool name of each provider tool a call declares, by the
/// application's name (`createToolNameMapping`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ToolNames {
    by_custom: BTreeMap<String, &'static str>,
    /// The custom names of computer toolset tools.
    toolsets: BTreeMap<String, &'static str>,
}
impl ToolNames {
    pub(crate) fn of(tools: &[Tool]) -> Self {
        let mut names = Self::default();
        for tool in tools.iter().filter(|t| t.kind == ToolKind::Provider) {
            if let Some(version) = version(tool) {
                names.by_custom.insert(tool.name.clone(), version.name);
                if version.id == "computer_toolset_20260801" {
                    names.toolsets.insert(tool.name.clone(), "computer");
                }
            }
        }
        names
    }
    /// The Anthropic name of an application tool name.
    pub(crate) fn provider<'a>(&self, custom: &'a str) -> &'a str {
        self.by_custom.get(custom).copied().unwrap_or(custom)
    }
    /// The application's name of an Anthropic tool name.
    pub(crate) fn custom<'a>(&'a self, provider: &'a str) -> &'a str {
        self.by_custom
            .iter()
            .find(|(_, name)| **name == provider)
            .map_or(provider, |(custom, _)| custom.as_str())
    }
    /// The toolset name of an application tool, if it is a toolset.
    pub(crate) fn toolset(&self, custom: &str) -> Option<&'static str> {
        self.toolsets.get(custom).copied()
    }
    /// Whether a provider tool is declared under another name than its
    /// Anthropic name.
    pub(crate) fn aliased(&self) -> Option<&str> {
        self.by_custom
            .iter()
            .find(|(custom, provider)| custom.as_str() != **provider)
            .map(|(custom, _)| custom.as_str())
    }
}

/// How a call's tools go out: the model's limits on them and the defaults
/// that the call's options give.
pub(crate) struct ToolRules {
    pub structured_output: bool,
    pub strict_tools: bool,
    pub eager_input_streaming: bool,
    pub rejects_forced_tool_use: bool,
    pub disable_parallel_tool_use: Option<bool>,
}

/// The tools of a request and their tool choice.
pub(crate) struct Prepared {
    pub tools: Vec<Value>,
    pub tool_choice: Option<Value>,
}

/// The request form of a function tool.
fn function(
    tool: &Tool,
    rules: &ToolRules,
    notes: &mut Notes,
    betas: &mut Vec<String>,
) -> Result<Value, Error> {
    let spec = tool
        .spec
        .as_ref()
        .ok_or_else(|| Error::Validation(format!("function tool {:?} has no spec", tool.name)))?;
    let args = tool.tool_config.as_ref().map(|c| &c.args);
    let arg = |key: &str| args.and_then(|args| args.get(key)).filter(|v| !v.is_null());
    let mut wire = Map::new();
    wire.insert("name".into(), json!(tool.name));
    if let Some(description) = &spec.description {
        wire.insert("description".into(), json!(description));
    }
    wire.insert("input_schema".into(), spec.params.as_value().clone());
    if let Some(control) = notes.cache(arg("cache_control").cloned(), "tool definition", true) {
        wire.insert("cache_control".into(), control);
    }
    let eager = match arg("eager_input_streaming") {
        Some(Value::Bool(eager)) => *eager,
        Some(_) => {
            return Err(Error::Validation(format!(
                "eager_input_streaming of tool {:?} must be a boolean",
                tool.name
            )));
        }
        None => rules.eager_input_streaming,
    };
    if eager {
        wire.insert("eager_input_streaming".into(), json!(true));
    }
    if tool.is_strict() {
        if rules.strict_tools {
            wire.insert("strict".into(), json!(true));
        } else {
            notes.note(
                "strict",
                format!(
                    "tool {:?} is strict, and strict tools are not taken here; the flag is left out",
                    tool.name
                ),
            );
        }
    }
    if let Some(defer) = arg("defer_loading") {
        wire.insert("defer_loading".into(), defer.clone());
    }
    let callers = arg("allowed_callers");
    if let Some(callers) = callers {
        wire.insert("allowed_callers".into(), callers.clone());
    }
    let examples = arg("input_examples");
    if let Some(examples) = examples {
        wire.insert("input_examples".into(), examples.clone());
    }
    if rules.structured_output {
        add_betas(betas, ["structured-outputs-2025-11-13"]);
    }
    if examples.is_some() || callers.is_some() {
        add_betas(betas, ["advanced-tool-use-2025-11-20"]);
    }
    Ok(Value::Object(wire))
}

/// The request form of a provider tool, from its version and arguments.
fn provider(tool: &Tool, betas: &mut Vec<String>) -> Result<Value, Error> {
    let id = tool
        .tool_config
        .as_ref()
        .and_then(|c| c.id.as_deref())
        .unwrap_or(&tool.name);
    let version = version(tool).ok_or_else(|| {
        Error::Validation(format!("Anthropic does not take the provider tool {id:?}"))
    })?;
    let args = tool
        .tool_config
        .as_ref()
        .map(|c| c.args.clone())
        .unwrap_or_default();
    if let Some(key) = args
        .keys()
        .find(|key| !version.args.contains(&key.as_str()))
    {
        return Err(Error::Validation(format!(
            "anthropic.{} takes no argument {key:?}",
            version.id
        )));
    }
    if let Some(key) = version
        .required
        .iter()
        .find(|key| args.get(**key).is_none_or(Value::is_null))
    {
        return Err(Error::Validation(format!(
            "anthropic.{} needs the argument {key:?}",
            version.id
        )));
    }
    let mut wire = Map::new();
    wire.insert("type".into(), json!(version.wire_type));
    if version.id != "computer_toolset_20260801" {
        wire.insert("name".into(), json!(version.name));
    }
    for key in version.args {
        if let Some(value) = args.get(*key).filter(|v| !v.is_null()) {
            wire.insert((*key).into(), value.clone());
        }
    }
    if let Some(beta) = version.beta {
        add_betas(betas, [beta]);
    }
    Ok(Value::Object(wire))
}

/// The tools and tool choice of a request (`prepareTools`).
pub(crate) fn prepare(
    tools: &[Tool],
    choice: Option<&ToolChoice>,
    rules: &ToolRules,
    notes: &mut Notes,
    betas: &mut Vec<String>,
) -> Result<Prepared, Error> {
    let mut wire = Vec::new();
    for tool in tools {
        wire.push(match tool.kind {
            ToolKind::Function => function(tool, rules, notes, betas)?,
            ToolKind::Provider => provider(tool, betas)?,
        });
    }
    if wire.is_empty() {
        // A call without tools sends neither tools nor a tool choice.
        return Ok(Prepared {
            tools: wire,
            tool_choice: None,
        });
    }
    let parallel = rules.disable_parallel_tool_use;
    let with_parallel = |mut choice: Value| {
        if let Some(disable) = parallel {
            choice["disable_parallel_tool_use"] = json!(disable);
        }
        choice
    };
    let mut forced_to_auto = |what: &str| {
        notes.note(
            "toolChoice",
            format!("tool choice {what} is not taken with this model or thinking; auto is sent"),
        );
    };
    let tool_choice = match choice {
        None => parallel
            .filter(|disable| *disable)
            .map(|_| with_parallel(json!({"type": "auto"}))),
        Some(ToolChoice::Auto) => Some(with_parallel(json!({"type": "auto"}))),
        Some(ToolChoice::None) => Some(json!({"type": "none"})),
        Some(ToolChoice::Required) if rules.rejects_forced_tool_use => {
            forced_to_auto("required");
            Some(with_parallel(json!({"type": "auto"})))
        }
        Some(ToolChoice::Required) => Some(with_parallel(json!({"type": "any"}))),
        Some(ToolChoice::Tool(name)) => {
            return Ok(named(
                &name.0,
                wire,
                rules,
                with_parallel,
                &mut forced_to_auto,
            ));
        }
        Some(ToolChoice::Selection(s))
            if s.tools.len() == 1
                && matches!(s.mode, ToolChoiceMode::Auto | ToolChoiceMode::Required) =>
        {
            let name = s.tools.first().map(|t| t.0.clone()).unwrap_or_default();
            return Ok(named(
                &name,
                wire,
                rules,
                with_parallel,
                &mut forced_to_auto,
            ));
        }
        Some(ToolChoice::Selection(_)) => {
            return Err(Error::Validation(
                "Anthropic does not support allowed tool subsets".into(),
            ));
        }
    };
    Ok(Prepared {
        tools: wire,
        tool_choice,
    })
}

/// A named tool choice: the tool, or on a model that rejects forced tool use
/// only that tool with `auto`.
fn named(
    name: &str,
    tools: Vec<Value>,
    rules: &ToolRules,
    with_parallel: impl Fn(Value) -> Value,
    mut forced_to_auto: impl FnMut(&str),
) -> Prepared {
    if rules.rejects_forced_tool_use {
        forced_to_auto(&format!("tool {name:?}"));
        return Prepared {
            tools: tools
                .into_iter()
                .filter(|tool| tool["name"] == name)
                .collect(),
            tool_choice: Some(with_parallel(json!({"type": "auto"}))),
        };
    }
    Prepared {
        tools,
        tool_choice: Some(with_parallel(json!({"type": "tool", "name": name}))),
    }
}

/// Whether a call declares a code execution tool.
pub(crate) fn has_code_execution(tools: &[Tool]) -> bool {
    tools
        .iter()
        .filter_map(version)
        .any(|v| matches!(v.id, "code_execution_20250825" | "code_execution_20260120"))
}
