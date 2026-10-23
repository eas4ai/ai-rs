use super::{
    dialect, messages, params,
    profile::{self, ChatProfile, Dialect, ResponsesDialect, ResponsesProfile},
    streaming,
};
use crate::{
    Error, Model, Provider, ProviderConfig, ProviderProtocol, StreamOptions,
    models::params::{ExtraHeaderValue, InferenceRequestParams},
    providers::{EventStream, http},
    transport::HttpRequest,
    types::{Message, ToolChoice},
};
use futures::{StreamExt, TryStreamExt, stream};
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAIChatCompletionsProtocol;
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAIResponsesProtocol;
/// Responses for the name `openai`, Chat Completions for every other name.
/// This is not the protocol a provider uses: `OpenAICompatibleProvider`
/// decides when it is built, from the API it targets.
#[deprecated(
    note = "choose by the API, not the name: use `Provider::default_protocol` of the provider"
)]
pub fn default_protocol(provider: &str) -> Arc<dyn ProviderProtocol> {
    if provider == "openai" {
        Arc::new(OpenAIResponsesProtocol)
    } else {
        Arc::new(OpenAIChatCompletionsProtocol)
    }
}

/// The API a request uses: Responses or Chat Completions, as a server's
/// profile describes it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Wire {
    Responses(&'static ResponsesProfile),
    Chat(&'static ChatProfile),
}
impl Wire {
    pub(crate) fn api(self) -> &'static http::ApiProfile {
        match self {
            Self::Responses(profile) => &profile.api,
            Self::Chat(profile) => &profile.api,
        }
    }
    fn responses(self) -> bool {
        matches!(self, Self::Responses(_))
    }
    /// The Responses dialect, or `None` for Chat Completions.
    pub(crate) fn responses_dialect(self) -> Option<ResponsesDialect> {
        match self {
            Self::Responses(profile) => Some(profile.dialect),
            Self::Chat(_) => None,
        }
    }
}

/// Responses for a host, as its profile describes it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostResponsesProtocol(pub(crate) &'static ResponsesProfile);
impl ProviderProtocol for HostResponsesProtocol {
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
        start(
            model,
            messages,
            options,
            provider,
            Wire::Responses(self.0),
            None,
        )
    }
}

/// Chat Completions for a host, as its profile describes it, with the
/// transport that downloads the media URLs the host cannot fetch, if any.
#[derive(Clone)]
pub(crate) struct HostChatProtocol(
    pub(crate) &'static ChatProfile,
    pub(crate) Option<Arc<dyn crate::models::files::DownloadTransport>>,
);
impl ProviderProtocol for HostChatProtocol {
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
        start(
            model,
            messages,
            options,
            provider,
            Wire::Chat(self.0),
            self.1.clone(),
        )
    }
}
impl ProviderProtocol for OpenAIChatCompletionsProtocol {
    fn protocol_class_id(&self) -> &str {
        profile::OPENAI.protocol_class_id
    }
    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        start(
            model,
            messages,
            options,
            provider,
            Wire::Chat(&profile::OPENAI),
            None,
        )
    }
}
impl ProviderProtocol for OpenAIResponsesProtocol {
    fn protocol_class_id(&self) -> &str {
        "openai_responses"
    }
    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        start(
            model,
            messages,
            options,
            provider,
            Wire::Responses(&profile::OPENAI_RESPONSES),
            None,
        )
    }
}
/// Chat Completions with explicit public-network media downloads.
/// Download transports must honor the core DNS-pinning contract. Runtime transports
/// are not serialized; configure downloads again after restoring a snapshot.
#[derive(Clone)]
pub struct OpenAIChatCompletionsWithDownloads {
    downloads: Arc<dyn crate::models::files::DownloadTransport>,
}
impl OpenAIChatCompletionsProtocol {
    pub fn with_download_transport(
        self,
        downloads: Arc<dyn crate::models::files::DownloadTransport>,
    ) -> OpenAIChatCompletionsWithDownloads {
        OpenAIChatCompletionsWithDownloads { downloads }
    }
}
impl ProviderProtocol for OpenAIChatCompletionsWithDownloads {
    fn protocol_class_id(&self) -> &str {
        profile::OPENAI.protocol_class_id
    }
    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        start(
            model,
            messages,
            options,
            provider,
            Wire::Chat(&profile::OPENAI),
            Some(self.downloads.clone()),
        )
    }
}
fn start(
    model: &Model,
    messages: Vec<Message>,
    options: StreamOptions,
    provider: &dyn Provider,
    wire: Wire,
    downloads: Option<Arc<dyn crate::models::files::DownloadTransport>>,
) -> EventStream {
    let config = provider.config().clone();
    let provider_name = provider.name().to_owned();
    let model_id = model.id.clone();
    let transport = provider.shared_transport();
    let downloads = downloads.or_else(|| provider.download_transport());
    stream::once(async move {
        // Check the request before any download, with each file to download
        // standing in as bytes, so a call that fails validation sends nothing.
        if downloads.is_some()
            && let Some(inline) = messages::without_downloads(&messages, wire)
        {
            wire_request(wire, &model_id, &config, inline, &options)?;
        }
        let messages = messages::download_files(messages, wire, downloads.as_deref()).await?;
        let request = wire_request(wire, &model_id, &config, messages, &options)?;
        let transport = transport.ok_or_else(|| {
            Error::Configuration(format!(
                "{} protocols require Provider::shared_transport",
                wire.api().label
            ))
        })?;
        let response = http::send(
            wire.api(),
            transport.as_ref(),
            request,
            &provider_name,
            Some(&model_id),
        )
        .await?;
        let image_type = options
            .tools
            .iter()
            .find_map(|t| {
                t.tool_config
                    .as_ref()
                    .filter(|c| c.id.as_deref() == Some("openai.image_generation"))
                    .map(|c| {
                        c.args
                            .get("output_format")
                            .and_then(Value::as_str)
                            .unwrap_or("png")
                    })
            })
            .unwrap_or("png");
        let hold_reasoning = wire.responses_dialect() == Some(ResponsesDialect::OpenAI)
            && options
                .params
                .as_ref()
                .is_some_and(|p| params::unstored(p, wire));
        Ok::<_, Error>(streaming::parse(
            response.body,
            provider_name,
            wire,
            format!("image/{image_type}"),
            options.output_schema.is_some(),
            hold_reasoning,
        ))
    })
    .try_flatten()
    .boxed()
}
/// Build a complete HTTP request without sending it.
pub fn build_request(
    model: &str,
    config: &ProviderConfig,
    messages: Vec<Message>,
    options: &StreamOptions,
    responses: bool,
) -> Result<HttpRequest, Error> {
    let wire = if responses {
        Wire::Responses(&profile::OPENAI_RESPONSES)
    } else {
        Wire::Chat(&profile::OPENAI)
    };
    wire_request(wire, model, config, messages, options)
}
pub(crate) fn wire_request(
    wire: Wire,
    model: &str,
    config: &ProviderConfig,
    messages: Vec<Message>,
    options: &StreamOptions,
) -> Result<HttpRequest, Error> {
    let responses = wire.responses();
    let p = options.params.clone().unwrap_or_default();
    let chat = match wire {
        Wire::Chat(profile) => Some(profile),
        Wire::Responses(_) => None,
    };
    match wire {
        Wire::Chat(profile) => dialect::check(profile, model, &p, options)?,
        Wire::Responses(profile) => dialect::check_responses(profile, model, &p, options)?,
    }
    let mut body = params::build(&p, wire)?;
    match wire {
        Wire::Chat(profile) => dialect::request(profile, model, &p, &mut body),
        Wire::Responses(profile) => dialect::responses_request(profile, model, &mut body),
    }
    let use_references = body.get("store") != Some(&Value::Bool(false));
    body.insert("model".into(), json!(model));
    body.insert("stream".into(), json!(true));
    let mut converted = messages::convert(messages, wire, model, use_references)?;
    if let (Some(profile), Some(schema)) = (chat, &options.output_schema)
        && let Some(message) = dialect::schema_message(profile, schema)?
    {
        converted.insert(0, message);
    }
    body.insert(
        if responses { "input" } else { "messages" }.into(),
        converted.into(),
    );
    if let Wire::Chat(profile) = wire
        && profile.include_usage
    {
        body.insert("stream_options".into(), json!({"include_usage":true}));
    }
    // A server that takes only automatic tool choice gets no tools for none.
    let mut send_tools = true;
    if let Wire::Chat(profile) = wire
        && profile.auto_tool_choice_only
    {
        let choice = options
            .tool_choice
            .as_ref()
            .or(p.tool_calling.as_ref().map(|t| &t.tool_choice));
        match choice {
            None | Some(ToolChoice::Auto) => {}
            Some(ToolChoice::None) => {
                send_tools = false;
                for key in ["tool_choice", "parallel_tool_calls"] {
                    body.remove(key);
                }
            }
            Some(other) => {
                return Err(params::invalid(format!(
                    "{} supports only automatic tool choice, not {}",
                    profile.api.label,
                    tool_choice_name(other)
                )));
            }
        }
    }
    if send_tools && !options.tools.is_empty() {
        body.insert(
            "tools".into(),
            super::tools::convert(&options.tools, wire)?.into(),
        );
    }
    if send_tools && let Some(choice) = &options.tool_choice {
        body.insert("tool_choice".into(), params::tool_choice(choice, wire)?);
    }
    if let Some(profile) = chat
        && profile.dialect == Dialect::Mistral
    {
        mistral_tools(&mut body, dialect::tool_choice(&p, options))?;
    }
    if wire.responses_dialect() == Some(ResponsesDialect::HuggingFace) {
        huggingface_tools(&mut body, dialect::tool_choice(&p, options));
    }
    if let Some(schema) = &options.output_schema {
        let name = options.output_type_name.as_deref().unwrap_or("output");
        let host_format = match chat {
            Some(profile) => dialect::response_format(profile, name, schema)?,
            None => None,
        };
        match wire {
            _ if host_format.is_some() => {
                body.insert("response_format".into(), host_format.into());
            }
            Wire::Responses(profile) if profile.dialect == ResponsesDialect::HuggingFace => {
                let text = body.entry("text").or_insert_with(|| json!({}));
                text["format"] =
                    json!({"type":"json_schema","name":name,"schema":schema,"strict":false});
            }
            Wire::Responses(_) => {
                let schema = strict_output_schema(schema)?;
                let text = body.entry("text").or_insert_with(|| json!({}));
                text["format"] =
                    json!({"type":"json_schema","name":name,"schema":schema,"strict":true});
            }
            Wire::Chat(profile) if profile.sends_json_schema(model) => {
                let schema = strict_output_schema(schema)?;
                body.insert("response_format".into(),json!({"type":"json_schema","json_schema":{"name":name,"schema":schema,"strict":true}}));
            }
            Wire::Chat(_) => {
                body.insert("response_format".into(), json!({"type":"json_object"}));
            }
        }
    }
    // Match SDK extra_body merging: explicit escape-hatch keys win at the wire boundary.
    if let Some(extra) = &p.extra_body {
        body.extend(extra.clone());
    }
    let suffix: &[&str] = if responses {
        &["responses"]
    } else {
        &["chat", "completions"]
    };
    http_request(
        wire.api(),
        "POST",
        config,
        suffix,
        Some(Value::Object(body)),
        Some(&p),
    )
}
/// Mistral's tool choice: `any` for required or a named tool, whose name
/// filters the tools, and neither `tool_choice` nor `parallel_tool_calls`
/// without tools (`mistral-prepare-tools.ts`).
fn mistral_tools(
    body: &mut serde_json::Map<String, Value>,
    choice: Option<&ToolChoice>,
) -> Result<(), Error> {
    if let Some(ToolChoice::Tool(tool)) = choice {
        let tools = body.get_mut("tools").and_then(Value::as_array_mut);
        let found = tools.is_some_and(|tools| {
            tools.retain(|t| t["function"]["name"].as_str() == Some(tool.0.as_str()));
            !tools.is_empty()
        });
        if !found {
            return Err(params::invalid(format!(
                "Mistral tool choice names the tool {tool}, which the call does not declare"
            )));
        }
    }
    let has_tools = body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty());
    if !has_tools {
        body.remove("tools");
        body.remove("tool_choice");
        body.remove("parallel_tool_calls");
        return Ok(());
    }
    if matches!(choice, Some(ToolChoice::Required | ToolChoice::Tool(_))) {
        body.insert("tool_choice".into(), json!("any"));
    }
    Ok(())
}
/// Hugging Face's tool choice: a named tool in the Chat Completions shape,
/// and neither tools nor `tool_choice` for `none`
/// (`huggingface-responses-prepare-tools.ts`).
fn huggingface_tools(body: &mut serde_json::Map<String, Value>, choice: Option<&ToolChoice>) {
    match choice {
        Some(ToolChoice::None) => {
            body.remove("tools");
            body.remove("tool_choice");
        }
        Some(ToolChoice::Tool(tool)) => {
            body.insert(
                "tool_choice".into(),
                json!({"type":"function","function":{"name":tool.0}}),
            );
        }
        _ => {}
    }
}
fn tool_choice_name(choice: &ToolChoice) -> &'static str {
    match choice {
        ToolChoice::Auto => "auto",
        ToolChoice::None => "none",
        ToolChoice::Required => "required",
        ToolChoice::Tool(_) => "a named tool",
        ToolChoice::Selection(_) => "a tool selection",
    }
}
pub(super) fn http_request(
    api: &http::ApiProfile,
    method: &str,
    config: &ProviderConfig,
    path: &[&str],
    body: Option<Value>,
    params: Option<&InferenceRequestParams>,
) -> Result<HttpRequest, Error> {
    let headers = params
        .and_then(|p| p.extra_headers.as_ref())
        .into_iter()
        .flatten()
        .filter_map(|(name, value)| match value {
            ExtraHeaderValue::Value(value) => Some((name.clone(), value.clone())),
            _ => None,
        })
        .collect();
    let query = params
        .and_then(|p| p.extra_query.clone())
        .unwrap_or_default();
    http::request(api, config, method, path, body, headers, query)
}
fn strict_output_schema(schema: &schemars::Schema) -> Result<Value, Error> {
    let mut schema = serde_json::to_value(schema)?;
    strict_schema(&mut schema, 0)?;
    Ok(schema)
}
fn strict_schema(schema: &mut Value, _depth: usize) -> Result<(), Error> {
    let root = schema.clone();
    strict_node(schema, &root, 0)
}
fn strict_node(schema: &mut Value, root: &Value, depth: usize) -> Result<(), Error> {
    if depth > 128 {
        return Err(params::invalid("OpenAI output schema exceeds 128 levels"));
    }
    let Some(object) = schema.as_object_mut() else {
        return Ok(());
    };
    if object.len() > 1
        && let Some(reference) = object.get("$ref").and_then(Value::as_str)
    {
        let resolved = reference
            .strip_prefix('#')
            .and_then(|p| root.pointer(p))
            .and_then(Value::as_object)
            .ok_or_else(|| params::invalid("OpenAI schema references must resolve locally"))?;
        let mut expanded = resolved.clone();
        expanded.extend(object.clone());
        expanded.remove("$ref");
        *object = expanded;
    }
    if object.get("default") == Some(&Value::Null) {
        object.remove("default");
    }
    if object.get("type").and_then(Value::as_str) == Some("object")
        && !object.contains_key("additionalProperties")
    {
        object.insert("additionalProperties".into(), json!(false));
    }
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        object.insert(
            "required".into(),
            json!(properties.keys().collect::<Vec<_>>()),
        );
    }
    for key in ["properties", "$defs", "definitions"] {
        if let Some(map) = object.get_mut(key).and_then(Value::as_object_mut) {
            for value in map.values_mut() {
                strict_node(value, root, depth + 1)?;
            }
        }
    }
    for key in ["items", "additionalProperties"] {
        if let Some(value) = object.get_mut(key) {
            strict_node(value, root, depth + 1)?;
        }
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(values) = object.get_mut(key).and_then(Value::as_array_mut) {
            for value in values {
                strict_node(value, root, depth + 1)?;
            }
        }
    }
    if let Some(all) = object
        .get("allOf")
        .and_then(Value::as_array)
        .filter(|a| a.len() == 1)
        && let Some(only) = all[0].as_object()
    {
        let mut flattened = object.clone();
        flattened.remove("allOf");
        flattened.extend(only.clone());
        *object = flattened;
        return strict_node(schema, root, depth + 1);
    }
    Ok(())
}
