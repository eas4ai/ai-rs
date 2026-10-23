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

//! Request conversion and the Anthropic Messages wire protocol.
pub use super::messages::{file_to_anthropic, messages_to_anthropic, tool_result_to_anthropic};
use super::{errors, params, tools};
use crate::{
    Error, Model, Provider, ProviderProtocol, StreamOptions,
    providers::{EventStream, http},
    transport::HttpRequest,
    types::Message,
};
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream};
use serde_json::{Value, json};

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
}

pub(crate) static ANTHROPIC: MessagesProfile = MessagesProfile {
    protocol_class_id: "anthropic_messages",
    api: errors::API,
    always_reasons: |_| false,
};

/// The Messages protocol for a host, as its profile describes it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostMessagesProtocol(pub(crate) &'static MessagesProfile);

/// How a Messages request sends an output schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SchemaMode {
    /// Native structured output in `output_config.format`.
    OutputFormat,
    /// A forced tool named `json` whose input is the answer, for servers
    /// without native structured output (`jsonResponseTool`). The stream
    /// gives its input as text.
    JsonTool,
}

/// The name of the JSON tool of [`SchemaMode::JsonTool`].
pub(crate) const JSON_TOOL: &str = "json";

/// Models that reject a forced tool choice; the JSON tool goes to them with
/// automatic tool choice (`getModelCapabilities`).
fn rejects_forced_tool_use(model: &str) -> bool {
    ["claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"]
        .iter()
        .any(|id| model.contains(id))
}

/// A Messages request before its URL: the body with `stream: true`, the
/// headers and the query.
pub(crate) struct MessagesRequest {
    pub body: serde_json::Map<String, Value>,
    pub headers: std::collections::BTreeMap<String, String>,
    pub query: std::collections::BTreeMap<String, Value>,
}

pub(crate) fn request_parts(
    profile: &MessagesProfile,
    model: &str,
    messages: &[Message],
    options: &StreamOptions,
    schema_mode: SchemaMode,
) -> Result<MessagesRequest, Error> {
    let mut params = params::translate(options.params.as_ref(), profile, model)?;
    if let Some(choice) = &options.tool_choice {
        params
            .body
            .insert("tool_choice".into(), params::tool_choice(choice)?);
    }
    let (system, messages) = messages_to_anthropic(messages)?;
    let (mut wire_tools, betas) = tools::to_wire(&options.tools)?;
    if let (SchemaMode::JsonTool, Some(schema)) = (schema_mode, &options.output_schema) {
        wire_tools.push(json!({
            "name": JSON_TOOL,
            "description": "Respond with a JSON object.",
            "input_schema": schema.as_value(),
        }));
        let kind = if rejects_forced_tool_use(model) {
            "auto"
        } else {
            "any"
        };
        params.body.insert(
            "tool_choice".into(),
            json!({"type": kind, "disable_parallel_tool_use": true}),
        );
    }
    params.body.entry("max_tokens").or_insert(json!(8192));
    params.body.insert("model".into(), json!(model));
    params.body.insert("messages".into(), json!(messages));
    params.body.insert("stream".into(), json!(true));
    if let Some(system) = system {
        params.body.insert("system".into(), json!(system));
    }
    if !wire_tools.is_empty() {
        params.body.insert("tools".into(), json!(wire_tools));
    }
    params::add_betas(&mut params.headers, &betas);
    if let (SchemaMode::OutputFormat, Some(schema)) = (schema_mode, &options.output_schema) {
        let format =
            json!({"type":"json_schema","schema":super::schema::transform(schema.as_value(),0)?});
        let output = params.body.entry("output_config").or_insert(json!({}));
        output["format"] = format;
    }
    // The Python SDK merges extra_body after all named request parameters.
    params.body.extend(params.extra_body);
    Ok(MessagesRequest {
        body: params.body,
        headers: params.headers,
        query: params.query,
    })
}

fn build_request(
    profile: &MessagesProfile,
    model: &Model,
    messages: &[Message],
    options: &StreamOptions,
    provider: &dyn Provider,
) -> Result<HttpRequest, Error> {
    let parts = request_parts(
        profile,
        &model.id,
        messages,
        options,
        SchemaMode::OutputFormat,
    )?;
    http::request(
        &profile.api,
        provider.config(),
        "POST",
        &["v1", "messages"],
        Some(Value::Object(parts.body)),
        parts.headers,
        parts.query,
    )
}

fn start(
    profile: &'static MessagesProfile,
    model: &Model,
    messages: Vec<Message>,
    options: StreamOptions,
    provider: &dyn Provider,
) -> EventStream {
    let request = build_request(profile, model, &messages, &options, provider);
    let transport = provider.shared_transport();
    let name = provider.name().to_owned();
    let model_id = model.id.clone();
    stream::once(async move {
        let request = request?;
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
            false,
        ))
    })
    .try_flatten()
    .boxed()
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
}
