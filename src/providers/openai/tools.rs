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

//! Provider-executed tool factories. Arguments use the Python wire snake_case keys.
use super::{dialect, params::invalid, protocol::Wire};
use crate::{
    Error,
    types::tools::{Tool, ToolConfig, ToolKind},
};
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
    let responses = matches!(wire, Wire::Responses);
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
                Wire::Responses => serde_json::to_value(&spec.params)?,
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
