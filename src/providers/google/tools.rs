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

//! Gemini's built-in tools. Each factory returns a provider-executed tool
//! with the TypeScript id (`google.<name>`) and arguments in the camel case
//! the API takes, such as `google_search(json!({"timeRangeFilter": {...}}))`.
//! The Interactions API also takes `computer_use`, `mcp_server` and
//! `retrieval`; generate content rejects them.

use serde_json::{Map, Value, json};

use super::{api::Flavor, capabilities};
use crate::{
    Error,
    types::tools::{Tool, ToolChoice, ToolConfig, ToolKind},
};

fn make(name: &str, args: Value) -> Result<Tool, Error> {
    let args = match args {
        Value::Object(args) => args,
        Value::Null => Map::new(),
        _ => {
            return Err(Error::Validation(format!(
                "google.{name} tool arguments must be an object"
            )));
        }
    };
    let args = args
        .into_iter()
        .filter(|(_, value)| !value.is_null())
        .collect();
    Tool::provider(
        name,
        ToolConfig {
            id: Some(format!("google.{name}")),
            args,
        },
    )
    .map_err(|error| Error::Validation(error.to_string()))
}

fn plain(name: &str) -> Tool {
    Tool {
        kind: ToolKind::Provider,
        name: name.into(),
        spec: None,
        tool_config: Some(ToolConfig {
            id: Some(format!("google.{name}")),
            args: Map::new(),
        }),
        require_approval: false,
    }
}

/// Google Search grounding, with the optional `searchTypes` and
/// `timeRangeFilter` (`{"startTime", "endTime"}`).
pub fn google_search(args: Value) -> Result<Tool, Error> {
    make("google_search", args)
}

/// Enterprise Web Search grounding. Vertex AI only.
pub fn enterprise_web_search() -> Tool {
    plain("enterprise_web_search")
}

/// Google Maps grounding.
pub fn google_maps() -> Tool {
    plain("google_maps")
}

/// URL context: Gemini reads the URLs in the prompt.
pub fn url_context() -> Tool {
    plain("url_context")
}

/// Code execution: Gemini writes and runs Python code.
pub fn code_execution() -> Tool {
    plain("code_execution")
}

/// File Search retrieval: `fileSearchStoreNames`, and the optional `topK`
/// and `metadataFilter`.
pub fn file_search(args: Value) -> Result<Tool, Error> {
    let tool = make("file_search", args)?;
    let names = tool
        .tool_config
        .as_ref()
        .and_then(|config| config.args.get("fileSearchStoreNames"));
    if !names.is_some_and(|names| {
        names
            .as_array()
            .is_some_and(|names| names.iter().all(Value::is_string))
    }) {
        return Err(Error::Validation(
            "google.file_search needs fileSearchStoreNames, a list of store names".into(),
        ));
    }
    Ok(tool)
}

/// A Vertex RAG Engine corpus: `ragCorpus` and the optional `topK`. Vertex
/// AI only.
pub fn vertex_rag_store(args: Value) -> Result<Tool, Error> {
    let tool = make("vertex_rag_store", args)?;
    if !tool
        .tool_config
        .as_ref()
        .and_then(|config| config.args.get("ragCorpus"))
        .is_some_and(Value::is_string)
    {
        return Err(Error::Validation(
            "google.vertex_rag_store needs ragCorpus, a corpus name".into(),
        ));
    }
    Ok(tool)
}

/// Computer use, for the Interactions API: `environment` (`browser` by
/// default) and `excludedPredefinedFunctions`.
pub fn computer_use(args: Value) -> Result<Tool, Error> {
    make("computer_use", args)
}

/// A remote MCP server, for the Interactions API: `name`, `url`, `headers`
/// and `allowedTools`.
pub fn mcp_server(args: Value) -> Result<Tool, Error> {
    make("mcp_server", args)
}

/// Retrieval, for the Interactions API: `retrievalTypes` and
/// `vertexAiSearchConfig`.
pub fn retrieval(args: Value) -> Result<Tool, Error> {
    make("retrieval", args)
}

/// The `google.` name of a provider tool, or a validation error for another
/// provider's tool.
pub(crate) fn builtin_name(tool: &Tool) -> Result<&str, Error> {
    let id = tool
        .tool_config
        .as_ref()
        .and_then(|config| config.id.as_deref())
        .unwrap_or_default();
    id.strip_prefix("google.").ok_or_else(|| {
        Error::Validation(format!(
            "the provider tool {:?} ({id}) is not a Google tool",
            tool.name
        ))
    })
}

fn declaration(tool: &Tool) -> Result<Value, Error> {
    let spec = tool
        .spec
        .as_ref()
        .ok_or_else(|| Error::Validation(format!("function tool {:?} has no spec", tool.name)))?;
    Ok(json!({
        "name": tool.name,
        "description": spec.description.as_deref().unwrap_or(""),
        "parametersJsonSchema": serde_json::to_value(&spec.params)?,
    }))
}

/// The `tools` and `toolConfig` of a generate content request
/// (`google-prepare-tools.ts`).
pub(crate) fn prepare(
    tools: &[Tool],
    choice: Option<&ToolChoice>,
    model_id: &str,
    flavor: Flavor,
) -> Result<(Option<Value>, Option<Value>), Error> {
    if let Some(ToolChoice::Selection(_)) = choice {
        return Err(Error::Validation(
            "Google does not support tool selections".into(),
        ));
    }
    if tools.is_empty() {
        return Ok((None, None));
    }
    for tool in tools {
        tool.validate_shape()
            .map_err(|error| Error::Validation(error.to_string()))?;
    }
    let functions: Vec<&Tool> = tools
        .iter()
        .filter(|t| t.kind == ToolKind::Function)
        .collect();
    let builtins: Vec<&Tool> = tools
        .iter()
        .filter(|t| t.kind == ToolKind::Provider)
        .collect();
    let gemini_3 = capabilities::uses_gemini_3_features(model_id);
    if !builtins.is_empty() && !functions.is_empty() && !gemini_3 {
        return Err(Error::Validation(format!(
            "{model_id} does not take function tools and built-in tools together; Gemini 3 models do"
        )));
    }
    let mut wire = Vec::new();
    for tool in &builtins {
        wire.push(builtin(tool, model_id, flavor)?);
    }
    let mode = |strict: bool| match choice {
        None | Some(ToolChoice::Auto) if strict => Some(json!({"mode": "VALIDATED"})),
        None => None,
        Some(ToolChoice::Auto) => Some(json!({"mode": "AUTO"})),
        Some(ToolChoice::None) => Some(json!({"mode": "NONE"})),
        Some(ToolChoice::Required) => Some(json!({"mode": "AUTO"})),
        Some(ToolChoice::Tool(tool)) => {
            Some(json!({"mode": "ANY", "allowedFunctionNames": [tool.0]}))
        }
        Some(ToolChoice::Selection(_)) => None,
    };
    if functions.is_empty() {
        return Ok((Some(Value::Array(wire)), None));
    }
    let declarations = functions
        .iter()
        .map(|tool| declaration(tool))
        .collect::<Result<Vec<_>, _>>()?;
    wire.push(json!({"functionDeclarations": declarations}));
    if !builtins.is_empty() {
        // Gemini 3 combines them: validated calls by default.
        let calling = mode(true).unwrap_or_else(|| json!({"mode": "VALIDATED"}));
        let mut config = json!({"functionCallingConfig": calling});
        if !flavor.is_vertex() {
            config["includeServerSideToolInvocations"] = json!(true);
        }
        return Ok((Some(Value::Array(wire)), Some(config)));
    }
    let strict = functions.iter().any(|tool| tool.is_strict());
    let config = mode(strict).map(|calling| json!({"functionCallingConfig": calling}));
    Ok((Some(Value::Array(wire)), config))
}

fn builtin(tool: &Tool, model_id: &str, flavor: Flavor) -> Result<Value, Error> {
    let name = builtin_name(tool)?;
    let args = tool
        .tool_config
        .as_ref()
        .map(|config| config.args.clone())
        .unwrap_or_default();
    let gemini_2 = capabilities::supports_gemini_2_tools(model_id);
    let refuse = |why: &str| Err(Error::Validation(format!("google.{name}: {why}")));
    match name {
        "google_search" if gemini_2 => Ok(json!({"googleSearch": args})),
        "google_search" => refuse("Google Search requires Gemini 2.0 or newer"),
        "url_context" if gemini_2 => Ok(json!({"urlContext": {}})),
        "url_context" => refuse("the URL context tool requires Gemini 2.0 or newer"),
        "code_execution" if gemini_2 => Ok(json!({"codeExecution": {}})),
        "code_execution" => refuse("the code execution tool requires Gemini 2.0 or newer"),
        "google_maps" if gemini_2 => Ok(json!({"googleMaps": {}})),
        "google_maps" => refuse("Google Maps grounding requires Gemini 2.0 or newer"),
        "file_search" if capabilities::supports_file_search(model_id) => {
            Ok(json!({"fileSearch": args}))
        }
        "file_search" => refuse("the file search tool requires a Gemini 2.5 or Gemini 3 model"),
        "enterprise_web_search" | "vertex_rag_store" if !flavor.is_vertex() => {
            refuse("this tool is Vertex AI only")
        }
        "enterprise_web_search" if gemini_2 => Ok(json!({"enterpriseWebSearch": {}})),
        "enterprise_web_search" => refuse("Enterprise Web Search requires Gemini 2.0 or newer"),
        "vertex_rag_store" if gemini_2 => {
            let mut store = json!({"rag_resources": {"rag_corpus": args.get("ragCorpus")}});
            if let Some(top_k) = args.get("topK") {
                store["similarity_top_k"] = top_k.clone();
            }
            Ok(json!({"retrieval": {"vertex_rag_store": store}}))
        }
        "vertex_rag_store" => refuse("the RAG store tool requires Gemini 2.0 or newer"),
        "computer_use" | "mcp_server" | "retrieval" => {
            refuse("this tool is for the Interactions API only")
        }
        other => Err(Error::Validation(format!(
            "unknown Google tool google.{other}"
        ))),
    }
}
