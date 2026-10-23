//! Provider-executed tool factories. Arguments use the Python wire snake_case keys.
use super::{dialect, params::invalid, protocol::Wire};
use crate::{
    Error,
    types::tools::{Tool, ToolConfig, ToolKind},
};
use serde::Deserialize;
use serde_json::{Map, Value, json};

fn make(name: &str, args: Value) -> Result<Tool, Error> {
    let mut args = args
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("OpenAI tool arguments must be an object"))?;
    args.retain(|_, v| !v.is_null());
    Tool::provider(
        name,
        ToolConfig {
            id: Some(format!("openai.{name}")),
            args,
        },
    )
    .map_err(|e| invalid(e.to_string()))
}
macro_rules! factory {($($name:ident),*)=>{$(
    pub fn $name(args:Value)->Result<Tool,Error> {make(stringify!($name),args)}
)*};}
factory!(
    web_search,
    web_search_preview,
    file_search,
    code_interpreter,
    image_generation,
    shell,
    mcp,
    tool_search
);
pub fn local_shell() -> Tool {
    Tool {
        kind: ToolKind::Provider,
        name: "local_shell".into(),
        spec: None,
        tool_config: Some(ToolConfig {
            id: Some("openai.local_shell".into()),
            args: Map::new(),
        }),
        require_approval: false,
    }
}
/// The computer use tool: the application runs its `computer_call`s and
/// answers each with a screenshot.
pub fn computer() -> Tool {
    Tool {
        name: "computer".into(),
        tool_config: Some(ToolConfig {
            id: Some("openai.computer".into()),
            args: Map::new(),
        }),
        ..local_shell()
    }
}
/// Programmatic tool calling: OpenAI runs code that calls the call's
/// function tools marked with `allowed_callers` `["programmatic"]`.
pub fn programmatic_tool_calling() -> Tool {
    Tool {
        name: "programmatic_tool_calling".into(),
        tool_config: Some(ToolConfig {
            id: Some("openai.programmatic_tool_calling".into()),
            args: Map::new(),
        }),
        ..local_shell()
    }
}
/// A custom tool named `name`, whose calls carry free text that the
/// application runs. `args` may hold a `description`, `async` and a
/// `format`: `{"type": "grammar", "syntax": "regex" | "lark",
/// "definition"}` or `{"type": "text"}` (`tool/custom.ts`).
pub fn custom(name: impl Into<String>, args: Value) -> Result<Tool, Error> {
    let args = args
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("OpenAI tool arguments must be an object"))?;
    CustomArgs::parse(&args)?;
    Tool::provider(
        name,
        ToolConfig {
            id: Some("openai.custom".into()),
            args,
        },
    )
    .map_err(|e| invalid(e.to_string()))
}
/// The arguments of a custom tool (`customArgsSchema`).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct CustomArgs {
    description: Option<String>,
    #[serde(rename = "async")]
    is_async: Option<bool>,
    format: Option<CustomFormat>,
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
#[allow(dead_code)]
enum CustomFormat {
    Grammar {
        syntax: GrammarSyntax,
        definition: String,
    },
    Text,
}
#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum GrammarSyntax {
    Regex,
    Lark,
}
impl CustomArgs {
    fn parse(args: &Map<String, Value>) -> Result<Self, Error> {
        serde_json::from_value(Value::Object(args.clone()))
            .map_err(|e| invalid(format!("invalid OpenAI custom tool arguments: {e}")))
    }
}
pub fn apply_patch() -> Tool {
    Tool {
        name: "apply_patch".into(),
        tool_config: Some(ToolConfig {
            id: Some("openai.apply_patch".into()),
            args: Map::new(),
        }),
        ..local_shell()
    }
}
pub(super) fn convert(tools: &[Tool], wire: Wire) -> Result<Vec<Value>, Error> {
    let responses = matches!(wire, Wire::Responses(_));
    let mut result = vec![];
    for tool in tools {
        tool.validate_shape().map_err(|e| invalid(e.to_string()))?;
        if tool.kind == ToolKind::Function {
            let spec = tool
                .spec
                .as_ref()
                .ok_or_else(|| invalid("function tool has no spec"))?;
            let parameters = match wire {
                Wire::Chat(profile) => {
                    dialect::tool_parameters(profile, serde_json::to_value(&spec.params)?)?
                }
                Wire::Responses(_) => serde_json::to_value(&spec.params)?,
            };
            let function = json!({"name":tool.name,"description":spec.description.as_deref().unwrap_or(""),"parameters":parameters});
            if responses {
                let mut function = function;
                function["type"] = json!("function");
                result.push(function);
            } else {
                result.push(json!({"type":"function","function":function}));
            }
            continue;
        }
        if !responses {
            return Err(Error::UnsupportedOperation(
                "OpenAI built-in tools require the Responses API".into(),
            ));
        }
        if let Wire::Responses(profile) = wire
            && profile.dialect != super::profile::ResponsesDialect::OpenAI
        {
            return Err(Error::UnsupportedOperation(format!(
                "{} does not support provider tools",
                profile.api.label
            )));
        }
        let config = tool
            .tool_config
            .as_ref()
            .ok_or_else(|| invalid("provider tool has no config"))?;
        let id = config.id.as_deref().unwrap_or("");
        let name = id
            .strip_prefix("openai.")
            .ok_or_else(|| invalid("provider tool is not an OpenAI tool"))?;
        if ![
            "web_search",
            "web_search_preview",
            "file_search",
            "code_interpreter",
            "image_generation",
            "local_shell",
            "shell",
            "apply_patch",
            "mcp",
            "tool_search",
        ]
        .contains(&name)
        {
            return Err(Error::UnsupportedOperation(format!(
                "unsupported OpenAI provider tool {id}"
            )));
        }
        let mut args = if matches!(name, "local_shell" | "apply_patch") {
            Map::new()
        } else {
            config.args.clone()
        };
        if name == "file_search"
            && let Some(ranking) = args.remove("ranking").filter(|v| !v.is_null())
        {
            args.insert("ranking_options".into(), ranking);
        }
        if name == "code_interpreter" {
            args.entry("container")
                .or_insert_with(|| json!({"type":"auto"}));
        }
        // Python uses {"type": name, **args}, so an explicit type in args wins.
        args.entry("type").or_insert_with(|| json!(name));
        result.push(args.into());
    }
    Ok(result)
}

/// How a tool of the call can appear in `tool_choice.allowed_tools`.
#[derive(Clone, Debug, PartialEq)]
enum Allowed {
    Entry(Value),
    Unsupported(String),
}

/// A sent tool's entry in `allowed_tools` (`toAllowedToolResolution`).
fn allowed_entry(sent: &Value) -> Allowed {
    let kind = sent["type"].as_str().unwrap_or_default();
    match kind {
        "function" | "custom" => Allowed::Entry(json!({"type": kind, "name": sent["name"]})),
        "mcp" => Allowed::Entry(json!({"type": "mcp", "server_label": sent["server_label"]})),
        "file_search"
        | "web_search"
        | "web_search_preview"
        | "image_generation"
        | "code_interpreter"
        | "computer"
        | "apply_patch"
        | "shell"
        | "local_shell"
        | "programmatic_tool_calling" => Allowed::Entry(json!({"type": kind})),
        _ => Allowed::Unsupported(format!(
            "OpenAI does not support {kind} tools in tool_choice.allowed_tools"
        )),
    }
}

/// The tool choice of the option `allowedTools`: each named tool of the call
/// as OpenAI lists it, by its name or by the OpenAI name of a provider tool,
/// and a name the call does not declare as a function. A tool that cannot be
/// listed is left out and logged; a list left empty fails
/// (`openai-responses-prepare-tools.ts`).
pub(super) fn allowed_tools(
    allowed: &super::options::AllowedTools,
    tools: &[Tool],
    sent: &[Value],
    model: &str,
) -> Result<Value, Error> {
    // By tool name, and by the OpenAI name of a provider tool named otherwise;
    // `None` marks an OpenAI name that several different tools share.
    let mut by_name = std::collections::BTreeMap::new();
    let mut aliases: std::collections::BTreeMap<String, Option<Allowed>> = Default::default();
    for (tool, sent) in tools.iter().zip(sent) {
        let resolution = allowed_entry(sent);
        by_name.insert(tool.name.clone(), resolution.clone());
        let canonical = sent["type"].as_str().unwrap_or_default();
        if tool.kind == ToolKind::Provider && canonical != tool.name {
            aliases
                .entry(canonical.to_owned())
                .and_modify(|alias| {
                    if alias.as_ref() != Some(&resolution) {
                        *alias = None;
                    }
                })
                .or_insert(Some(resolution));
        }
    }
    let mut entries = vec![];
    let mut left_out = vec![];
    for name in &allowed.tool_names {
        let setting = format!("allowedTools entry {name:?}");
        let resolution = match (by_name.get(name), aliases.get(name)) {
            (Some(direct), _) => direct.clone(),
            (None, Some(Some(alias))) => alias.clone(),
            (None, Some(None)) => {
                super::capabilities::dropped(
                    &setting,
                    model,
                    "several tools of the call share this OpenAI tool name",
                );
                left_out.push(name.as_str());
                continue;
            }
            (None, None) => Allowed::Entry(json!({"type": "function", "name": name})),
        };
        match resolution {
            Allowed::Entry(entry) => entries.push(entry),
            Allowed::Unsupported(reason) => {
                super::capabilities::dropped(&setting, model, &reason);
                left_out.push(name.as_str());
            }
        }
    }
    if entries.is_empty() {
        return Err(invalid(format!(
            "allowedTools names only tools that cannot be allow-listed: {}",
            left_out.join(", ")
        )));
    }
    Ok(json!({
        "type": "allowed_tools",
        "mode": allowed.mode.as_deref().unwrap_or("auto"),
        "tools": entries,
    }))
}
