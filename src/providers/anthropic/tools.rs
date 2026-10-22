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

//! Provider-executed Anthropic tool declarations.
use crate::{
    Error,
    types::tools::{Tool, ToolConfig, ToolKind},
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

pub const BETA_HEADERS: &[(&str, &str)] = &[
    (
        "anthropic.web_search_20260209",
        "code-execution-web-tools-2026-02-09",
    ),
    (
        "anthropic.web_fetch_20260209",
        "code-execution-web-tools-2026-02-09",
    ),
    ("anthropic.computer_20251124", "computer-use-2025-11-24"),
    ("anthropic.bash_20250124", "computer-use-2025-01-24"),
    ("anthropic.memory_20250818", "context-management-2025-06-27"),
];

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
#[derive(Clone, Debug, Default)]
pub struct WebSearchOptions {
    pub max_uses: Option<i64>,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
    pub user_location: Option<UserLocation>,
}
#[derive(Clone, Debug, Default)]
pub struct WebFetchOptions {
    pub max_uses: Option<i64>,
    pub allowed_domains: Option<Vec<String>>,
    pub blocked_domains: Option<Vec<String>>,
    pub citations: Option<Citations>,
    pub max_content_tokens: Option<i64>,
}
#[derive(Clone, Debug)]
pub struct ComputerUseOptions {
    pub display_width_px: i64,
    pub display_height_px: i64,
    pub display_number: Option<i64>,
    pub enable_zoom: Option<bool>,
}
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
pub fn web_search(options: WebSearchOptions) -> Result<Tool, Error> {
    check_domains(
        "web_search",
        &options.allowed_domains,
        &options.blocked_domains,
    )?;
    let location = options.user_location.map(|loc| {
        let mut v = json!(loc);
        v["type"] = json!("approximate");
        v
    });
    Ok(declaration(
        "web_search",
        "web_search_20260209",
        non_null(
            json!({"max_uses":options.max_uses,"allowed_domains":options.allowed_domains,"blocked_domains":options.blocked_domains,"user_location":location}),
        ),
    ))
}
pub fn web_fetch(options: WebFetchOptions) -> Result<Tool, Error> {
    check_domains(
        "web_fetch",
        &options.allowed_domains,
        &options.blocked_domains,
    )?;
    Ok(declaration(
        "web_fetch",
        "web_fetch_20260209",
        non_null(
            json!({"max_uses":options.max_uses,"allowed_domains":options.allowed_domains,"blocked_domains":options.blocked_domains,"citations":options.citations,"max_content_tokens":options.max_content_tokens}),
        ),
    ))
}
pub fn computer_use(options: ComputerUseOptions) -> Tool {
    declaration(
        "computer",
        "computer_20251124",
        non_null(
            json!({"display_width_px":options.display_width_px,"display_height_px":options.display_height_px,"display_number":options.display_number,"enable_zoom":options.enable_zoom}),
        ),
    )
}
pub fn text_editor(max_characters: Option<i64>) -> Tool {
    declaration(
        "str_replace_based_edit_tool",
        "text_editor_20250728",
        non_null(json!({"max_characters":max_characters})),
    )
}
pub fn code_execution() -> Tool {
    declaration("code_execution", "code_execution_20260120", json!({}))
}
pub fn bash() -> Tool {
    declaration("bash", "bash_20250124", json!({}))
}
pub fn memory() -> Tool {
    declaration("memory", "memory_20250818", json!({}))
}

pub(super) fn to_wire(
    tools: &[Tool],
) -> Result<(Vec<Value>, std::collections::BTreeSet<String>), Error> {
    let mut wire = Vec::new();
    let mut betas = std::collections::BTreeSet::new();
    for tool in tools.iter().filter(|t| t.kind == ToolKind::Function) {
        let spec = tool.spec.as_ref().ok_or_else(|| {
            Error::Validation(format!("function tool {:?} has no spec", tool.name))
        })?;
        wire.push(json!({"name":tool.name,"description":spec.description.as_deref().unwrap_or(""),"input_schema":spec.params}));
    }
    for tool in tools.iter().filter(|t| t.kind == ToolKind::Provider) {
        let config = tool.tool_config.as_ref();
        let id = config.and_then(|c| c.id.as_deref()).unwrap_or(&tool.name);
        let native = id
            .strip_prefix("anthropic.")
            .filter(|_| config.is_some())
            .ok_or_else(|| {
                Error::Validation(format!("Anthropic does not support provider tool {id:?}"))
            })?;
        let mut value = Map::from_iter([
            ("type".into(), json!(native)),
            ("name".into(), json!(tool.name)),
        ]);
        if let Some(c) = config {
            value.extend(c.args.clone());
        }
        wire.push(Value::Object(value));
        if let Some((_, beta)) = BETA_HEADERS.iter().find(|(key, _)| *key == id) {
            betas.insert((*beta).into());
        }
    }
    Ok((wire, betas))
}
