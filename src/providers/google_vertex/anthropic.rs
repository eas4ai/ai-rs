//! Anthropic's Claude models on Vertex AI, over the Anthropic Messages
//! protocol at
//! `https://<host>/v1/projects/<project>/locations/<location>/publishers/anthropic/models/<id>:streamRawPredict`
//! (`:rawPredict` for a buffered call), as
//! `anthropic/google-vertex-anthropic-provider.ts` sends it.
//!
//! The body has no `model` and holds `anthropic_version: "vertex-2023-10-16"`.
//! The option `labels` under `vertex` goes out as the base64 JSON header
//! `x-vertex-ai-labels`, unless the configuration or the call sets that
//! header. Vertex AI takes no URL sources, so every file URL is downloaded
//! first, which needs a download transport. An output schema goes out as a
//! forced tool named `json`, whose input is the answer's text. Only the
//! Anthropic tools of [`tools`] are taken; another fails with
//! `Error::Validation`.

use std::{fmt, sync::Arc};

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use futures::{StreamExt, TryStreamExt, stream};
use serde_json::{Value, json};

use super::auth::{self, Auth, AuthTransport, Credentials};
use crate::{
    Error, Model, Provider, ProviderConfig, ProviderProtocol, ProviderRegistry, StreamOptions,
    models::files::DownloadTransport,
    providers::{
        EventStream,
        anthropic::{
            errors,
            protocol::{MessagesProfile, request_parts},
            stream::{self as messages_stream, Reading},
        },
        http::{self, ApiProfile},
    },
    transport::{HttpRequest, HttpTransport},
    types::{
        Message, MessageRole, Part,
        messages::{ContentPart, FileData, FilePart, SpecialToolResult, ToolResultValue},
        tools::ToolKind,
    },
};

/// The Anthropic API version Vertex AI takes.
const ANTHROPIC_VERSION: &str = "vertex-2023-10-16";
const LABELS_HEADER: &str = "x-vertex-ai-labels";

/// The Anthropic tool types Vertex AI takes (`googleVertexAnthropicTools`).
const TOOL_TYPES: [&str; 10] = [
    "bash_20241022",
    "bash_20250124",
    "text_editor_20241022",
    "text_editor_20250124",
    "text_editor_20250429",
    "text_editor_20250728",
    "computer_20241022",
    "web_search_20250305",
    "tool_search_tool_regex_20251119",
    "tool_search_tool_bm25_20251119",
];

static API: ApiProfile = ApiProfile {
    label: "Google Vertex AI Anthropic",
    ..errors::API
};

static PROFILE: MessagesProfile = MessagesProfile {
    protocol_class_id: "google_vertex_anthropic_messages",
    api: ApiProfile {
        label: "Google Vertex AI Anthropic",
        ..errors::API
    },
    always_reasons: |_| false,
    auth_token_env: None,
    // Vertex AI takes no output format and no strict tools (VTX-013).
    native_structured_output: false,
    strict_tools: false,
    max_temperature: |_| 1.0,
    sampling_with_thinking: |_| false,
    options_key: None,
    host_media: |_| false,
};

/// The Anthropic tools Vertex AI takes, in the versions it lists.
pub mod tools {
    use serde_json::{Map, Value, json};

    use crate::types::tools::{Tool, ToolConfig, ToolKind};

    fn declaration(name: &str, wire_type: &str, args: Value) -> Tool {
        Tool {
            kind: ToolKind::Provider,
            name: name.into(),
            spec: None,
            tool_config: Some(ToolConfig {
                id: Some(format!("anthropic.{wire_type}")),
                args: args.as_object().cloned().unwrap_or_default(),
            }),
            require_approval: false,
        }
    }

    fn without_nulls(args: Value) -> Value {
        match args {
            Value::Object(fields) => Value::Object(
                fields
                    .into_iter()
                    .filter(|(_, value)| !value.is_null())
                    .collect::<Map<_, _>>(),
            ),
            other => other,
        }
    }

    /// The bash tool, version 2024-10-22.
    pub fn bash_20241022() -> Tool {
        declaration("bash", "bash_20241022", json!({}))
    }

    /// The bash tool, version 2025-01-24.
    pub fn bash_20250124() -> Tool {
        declaration("bash", "bash_20250124", json!({}))
    }

    /// The text editor, version 2024-10-22.
    pub fn text_editor_20241022() -> Tool {
        declaration("str_replace_editor", "text_editor_20241022", json!({}))
    }

    /// The text editor, version 2025-01-24.
    pub fn text_editor_20250124() -> Tool {
        declaration("str_replace_editor", "text_editor_20250124", json!({}))
    }

    /// The text editor, version 2025-04-29.
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
            without_nulls(json!({"max_characters": max_characters})),
        )
    }

    /// Computer use, version 2024-10-22, for a display of this size.
    pub fn computer_20241022(
        display_width_px: i64,
        display_height_px: i64,
        display_number: Option<i64>,
    ) -> Tool {
        declaration(
            "computer",
            "computer_20241022",
            without_nulls(json!({
                "display_width_px": display_width_px,
                "display_height_px": display_height_px,
                "display_number": display_number,
            })),
        )
    }

    /// Web search, version 2025-03-05, with optional limits on its uses
    /// and domains.
    pub fn web_search_20250305(
        max_uses: Option<i64>,
        allowed_domains: Option<Vec<String>>,
        blocked_domains: Option<Vec<String>>,
    ) -> Tool {
        declaration(
            "web_search",
            "web_search_20250305",
            without_nulls(json!({
                "max_uses": max_uses,
                "allowed_domains": allowed_domains,
                "blocked_domains": blocked_domains,
            })),
        )
    }

    /// Tool search by regular expression, version 2025-11-19.
    pub fn tool_search_regex_20251119() -> Tool {
        declaration(
            "tool_search_tool_regex",
            "tool_search_tool_regex_20251119",
            json!({}),
        )
    }

    /// Tool search by BM25, version 2025-11-19.
    pub fn tool_search_bm25_20251119() -> Tool {
        declaration(
            "tool_search_tool_bm25",
            "tool_search_tool_bm25_20251119",
            json!({}),
        )
    }
}

fn base_url(config: &ProviderConfig) -> Result<String, Error> {
    let configured = config.base_url();
    if !configured.is_empty() {
        return Ok(configured);
    }
    let project = super::project(config)?;
    let location = super::location(config, None)?;
    Ok(format!(
        "https://{}/v1/projects/{project}/locations/{location}/publishers/anthropic/models",
        super::host(location)
    ))
}

/// The Anthropic provider on Vertex AI. Build one with [`provider`]. It
/// signs in only with credentials: an API key in the configuration is not
/// sent.
#[derive(Clone)]
pub struct VertexAnthropicProvider {
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
    auth: Auth,
    signed: Arc<AuthTransport>,
    downloads: Option<Arc<dyn DownloadTransport>>,
}

impl fmt::Debug for VertexAnthropicProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VertexAnthropicProvider")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

fn signer(
    transport: &Arc<dyn HttpTransport>,
    auth: &Auth,
    config: &ProviderConfig,
) -> Arc<AuthTransport> {
    Arc::new(AuthTransport {
        inner: transport.clone(),
        auth: auth.clone(),
        config: config.clone(),
        auth_wins: false,
        base_url,
    })
}

impl VertexAnthropicProvider {
    /// Sign in with OAuth access tokens from this source.
    #[must_use]
    pub fn with_access_token_source(mut self, source: Arc<dyn super::AccessTokenSource>) -> Self {
        self.auth = Auth::with(Credentials::Token(source));
        self.signed = signer(&self.transport, &self.auth, &self.config);
        self
    }

    /// Sign in with google-cloud-auth credentials.
    #[cfg(feature = "google-auth")]
    #[must_use]
    pub fn with_credentials(
        mut self,
        credentials: google_cloud_auth::credentials::Credentials,
    ) -> Self {
        self.auth = Auth::with(Credentials::Google(credentials));
        self.signed = signer(&self.transport, &self.auth, &self.config);
        self
    }

    /// Download file URLs before each request, as Vertex AI takes no URL
    /// sources. Without one, a file URL fails with `Error::Validation`.
    /// Download transports must honor the core DNS-pinning contract.
    #[must_use]
    pub fn with_download_transport(mut self, downloads: Arc<dyn DownloadTransport>) -> Self {
        self.downloads = Some(downloads);
        self
    }

    fn ready(&self) -> bool {
        self.auth.available(&self.config)
    }
}

/// Build the Anthropic provider on Vertex AI, at the `env` project and
/// location or the configured base URL.
pub fn provider(
    mut config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> VertexAnthropicProvider {
    config.api_key_value = None;
    config.api_key_env = None;
    let auth = Auth::default();
    VertexAnthropicProvider {
        signed: signer(&transport, &auth, &config),
        config,
        transport,
        auth,
        downloads: None,
    }
}

#[async_trait]
impl Provider for VertexAnthropicProvider {
    fn name(&self) -> &str {
        "google-vertex-anthropic"
    }
    fn config(&self) -> &ProviderConfig {
        &self.config
    }
    fn provider_class_id(&self) -> &str {
        "google-vertex-anthropic"
    }
    fn base_url(&self) -> String {
        base_url(&self.config).unwrap_or_default()
    }
    fn default_protocol(&self) -> Option<Arc<dyn ProviderProtocol>> {
        Some(Arc::new(VertexAnthropicProtocol {
            downloads: self.downloads.clone(),
        }))
    }
    fn transport(&self) -> Option<&dyn HttpTransport> {
        self.ready()
            .then(|| self.signed.as_ref() as &dyn HttpTransport)
    }
    fn shared_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        self.ready()
            .then(|| self.signed.clone() as Arc<dyn HttpTransport>)
    }
    fn is_configured(&self) -> bool {
        base_url(&self.config).is_ok() && self.ready()
    }
}

/// The Anthropic Messages protocol on Vertex AI.
#[derive(Clone)]
pub struct VertexAnthropicProtocol {
    downloads: Option<Arc<dyn DownloadTransport>>,
}

impl fmt::Debug for VertexAnthropicProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VertexAnthropicProtocol")
            .finish_non_exhaustive()
    }
}

/// Each file of a history that a request sends: the user's files and the
/// files in tool results.
fn files_mut(messages: &mut [Message]) -> impl Iterator<Item = &mut FilePart> {
    messages.iter_mut().flat_map(|message| {
        let user = message.role == MessageRole::User;
        message.parts.iter_mut().flat_map(move |part| {
            let files: Vec<&mut FilePart> = match part {
                Part::File(file) if user => vec![file],
                Part::ToolResult(result) => {
                    let mut files: Vec<&mut FilePart> = Vec::new();
                    if let ToolResultValue::Special(SpecialToolResult::Content(content)) =
                        &mut result.result
                    {
                        files.extend(content.value.iter_mut().filter_map(|item| match item {
                            ContentPart::File(file) => Some(file),
                            ContentPart::Text(_) => None,
                        }));
                    }
                    if let Some(ToolResultValue::Special(SpecialToolResult::Content(content))) =
                        &mut result.model_input
                    {
                        files.extend(content.value.iter_mut().filter_map(|item| match item {
                            ContentPart::File(file) => Some(file),
                            ContentPart::Text(_) => None,
                        }));
                    }
                    files
                }
                _ => Vec::new(),
            };
            files
        })
    })
}

fn file_url(file: &FilePart) -> Option<&str> {
    file.data
        .as_text()
        .filter(|text| text.starts_with("http://") || text.starts_with("https://"))
}

/// The history with each file URL replaced by its downloaded bytes.
async fn download_files(
    mut messages: Vec<Message>,
    downloads: Option<&dyn DownloadTransport>,
) -> Result<Vec<Message>, Error> {
    for file in files_mut(&mut messages) {
        let Some(url) = file_url(file).map(str::to_owned) else {
            continue;
        };
        let Some(downloads) = downloads else {
            return Err(Error::Validation(
                "Google Vertex AI Anthropic does not fetch file URLs; supply the file bytes or build the provider with a download transport".into(),
            ));
        };
        let (bytes, _) = crate::models::files::download(
            downloads,
            &url,
            crate::models::files::DEFAULT_MAX_BYTES,
        )
        .await
        .map_err(|error| {
            crate::errors::ProviderError::new(
                crate::errors::ProviderErrorKind::Response,
                format!("{} input download failed", API.label),
            )
            .with_source(error)
        })?;
        file.data = FileData::Bytes(bytes.to_vec());
    }
    Ok(messages)
}

/// The history with each file URL replaced by empty bytes, to check a
/// request before any download.
fn without_urls(messages: &[Message]) -> Vec<Message> {
    let mut messages = messages.to_vec();
    for file in files_mut(&mut messages) {
        if file_url(file).is_some() {
            file.data = FileData::Bytes(Vec::new());
        }
    }
    messages
}

/// Reject an Anthropic tool that Vertex AI does not list.
fn check_tools(options: &StreamOptions) -> Result<(), Error> {
    for tool in options
        .tools
        .iter()
        .filter(|t| t.kind == ToolKind::Provider)
    {
        let id = tool
            .tool_config
            .as_ref()
            .and_then(|config| config.id.as_deref())
            .unwrap_or(&tool.name);
        let wire_type = id.strip_prefix("anthropic.").unwrap_or(id);
        if !TOOL_TYPES.contains(&wire_type) {
            return Err(Error::Validation(format!(
                "Google Vertex AI Anthropic does not take the tool {id:?}; use the versions in google_vertex::anthropic::tools"
            )));
        }
    }
    Ok(())
}

/// The base64 JSON of the option `labels` under `vertex`, or `None`
/// without it.
fn labels(options: &StreamOptions) -> Result<Option<String>, Error> {
    let Some(vertex) = options
        .params
        .as_ref()
        .and_then(|p| p.provider_params.as_ref())
        .and_then(|params| params.get("vertex"))
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    let invalid = || {
        Error::Validation(
            "the vertex option of Google Vertex AI Anthropic takes only labels, a map of strings"
                .into(),
        )
    };
    let fields = vertex.as_object().ok_or_else(invalid)?;
    if fields.keys().any(|key| key != "labels") {
        return Err(invalid());
    }
    let Some(labels) = fields.get("labels").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let entries = labels.as_object().ok_or_else(invalid)?;
    if !entries.values().all(Value::is_string) {
        return Err(invalid());
    }
    Ok(Some(STANDARD.encode(labels.to_string())))
}

fn has_header<'a>(mut names: impl Iterator<Item = &'a String>, name: &str) -> bool {
    names.any(|n| n.eq_ignore_ascii_case(name))
}

/// The request of a call, after the checks every call makes before it is
/// sent.
fn build(
    model: &Model,
    messages: &[Message],
    options: &StreamOptions,
    config: &ProviderConfig,
    streaming: bool,
) -> Result<(HttpRequest, Reading), Error> {
    let base = base_url(config)?;
    check_tools(options)?;
    let mut parts = request_parts(&PROFILE, &model.id, messages, options, streaming)?;
    parts.body.remove("model");
    parts
        .body
        .insert("anthropic_version".into(), json!(ANTHROPIC_VERSION));
    let mut headers = parts.headers(config);
    if let Some(labels) = labels(options)?
        && !has_header(config.headers.keys(), LABELS_HEADER)
        && !has_header(headers.keys(), LABELS_HEADER)
    {
        headers.insert(LABELS_HEADER.into(), labels);
    }
    let method = if streaming {
        "streamRawPredict"
    } else {
        "rawPredict"
    };
    let request = http::request_to(
        &API,
        &base,
        config,
        "POST",
        &[&format!("{}:{method}", model.id)],
        Some(Value::Object(parts.body)),
        headers,
        parts.query,
    )?;
    Ok((request, parts.reading))
}

impl VertexAnthropicProtocol {
    /// Check the call, download its files and build its request.
    async fn prepare(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: &StreamOptions,
        config: &ProviderConfig,
        streaming: bool,
    ) -> Result<(HttpRequest, Reading), Error> {
        // Check before any download, with each file URL as empty bytes, so
        // that a call that fails validation fetches nothing.
        build(model, &without_urls(&messages), options, config, streaming)?;
        let messages = download_files(messages, self.downloads.as_deref()).await?;
        build(model, &messages, options, config, streaming)
    }
}

fn transport_of(provider: &dyn Provider) -> Result<Arc<dyn HttpTransport>, Error> {
    provider
        .shared_transport()
        .ok_or_else(|| auth::unavailable(provider.config()))
}

#[async_trait]
impl ProviderProtocol for VertexAnthropicProtocol {
    fn protocol_class_id(&self) -> &str {
        PROFILE.protocol_class_id
    }

    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        let protocol = self.clone();
        let model = model.clone();
        let config = provider.config().clone();
        let name = provider.name().to_owned();
        let transport = transport_of(provider);
        stream::once(async move {
            let transport = transport?;
            let (request, reading) = protocol
                .prepare(&model, messages, &options, &config, true)
                .await?;
            let response =
                http::send(&API, transport.as_ref(), request, &name, Some(&model.id)).await?;
            Ok::<_, Error>(messages_stream::parse(response.body, name, &API, reading))
        })
        .try_flatten()
        .boxed()
    }

    async fn generate(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> Result<Message, Error> {
        let transport = transport_of(provider)?;
        let config = provider.config().clone();
        let (request, reading) = self
            .prepare(model, messages, &options, &config, false)
            .await?;
        let response = http::json(
            &API,
            transport.as_ref(),
            request,
            provider.name(),
            Some(&model.id),
        )
        .await?;
        let events: Vec<_> =
            messages_stream::parse_message(&response, provider.name().to_owned(), &API, reading)
                .try_collect()
                .await?;
        let (message, _) = crate::providers::google::language::hydrate(provider.name(), events)?;
        Ok(message)
    }
}

/// Register the Anthropic provider on Vertex AI under
/// `google-vertex-anthropic` and `@ai-sdk/google-vertex/anthropic`, with
/// snapshot restoration that uses this transport.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    register_provider(registry, provider(config, transport))
}

/// Register a built provider, with its credentials and download transport.
/// A restored provider gets them back; a snapshot holds none of them.
pub fn register_provider(
    registry: &mut ProviderRegistry,
    provider: VertexAnthropicProvider,
) -> Result<(), Error> {
    let mut next = registry.clone();
    let shared = Arc::new(provider.clone());
    for handle in ["google-vertex-anthropic", "@ai-sdk/google-vertex/anthropic"] {
        next.register(handle, shared.clone())?;
    }
    let template = provider.clone();
    next.register_provider_factory("google-vertex-anthropic", move |value: Value, _| {
        let config: ProviderConfig = serde_json::from_value(value)?;
        let mut restored = self::provider(config, template.transport.clone());
        restored.auth = template.auth.clone();
        restored.signed = signer(&restored.transport, &restored.auth, &restored.config);
        restored.downloads.clone_from(&template.downloads);
        Ok(Arc::new(restored))
    })?;
    let downloads = provider.downloads.clone();
    next.register_protocol_factory(PROFILE.protocol_class_id, move |_| {
        Ok(Arc::new(VertexAnthropicProtocol {
            downloads: downloads.clone(),
        }))
    })?;
    *registry = next;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_listed_tools_are_taken() {
        let mut options = StreamOptions {
            tools: vec![
                tools::bash_20241022(),
                tools::text_editor_20250728(Some(100)),
                tools::computer_20241022(1024, 768, None),
                tools::web_search_20250305(Some(2), None, None),
                tools::tool_search_bm25_20251119(),
            ],
            ..Default::default()
        };
        check_tools(&options).unwrap();
        options.tools =
            vec![crate::providers::anthropic::tools::web_fetch(Default::default()).unwrap()];
        assert!(matches!(check_tools(&options), Err(Error::Validation(_))));
    }

    #[test]
    fn labels_are_base64_json() {
        let options = StreamOptions {
            params: Some(crate::models::params::InferenceRequestParams {
                provider_params: Some(
                    [("vertex".to_owned(), json!({"labels": {"team": "ai"}}))].into(),
                ),
                ..Default::default()
            }),
            ..Default::default()
        };
        let encoded = labels(&options).unwrap().unwrap();
        assert_eq!(
            STANDARD.decode(encoded).unwrap(),
            br#"{"team":"ai"}"#.to_vec()
        );
    }
}
