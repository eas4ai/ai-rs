//! Provider ownership, explicit configuration, and protocol dispatch.

use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use futures::{
    StreamExt,
    stream::{self, BoxStream},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::operations::*;
use crate::{
    errors::Error,
    models::{Model, StreamOptions},
    transport::{HttpTransport, WebSocketRequest, WebSocketTransport},
    types::{
        events::Event,
        messages::{FilePart, Message},
    },
};

pub type EventStream = BoxStream<'static, Result<Event, Error>>;

/// Configuration fields that can hold a credential. Snapshots leave them out.
const CREDENTIAL_FIELDS: [&str; 3] = ["api_key_value", "env", "headers"];

/// Configuration fields that decide where a provider sends its requests. A
/// restore takes them from the registered provider, with the credentials.
const ENDPOINT_FIELDS: [&str; 4] = [
    "default_base_url",
    "base_url_env",
    "api_key_env",
    "config_envs",
];

/// An explicit configuration snapshot. The library never reads the process environment.
#[derive(Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderConfig {
    #[serde(default)]
    pub default_base_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_value: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub base_url_env: Option<String>,
    #[serde(default, deserialize_with = "null_default")]
    pub config_envs: Vec<String>,
    #[serde(default, deserialize_with = "null_default")]
    pub headers: BTreeMap<String, String>,
    #[serde(default, deserialize_with = "null_default")]
    pub env: BTreeMap<String, String>,
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("api_key_configured", &self.api_key().is_some())
            .field("config_envs", &self.config_envs)
            .finish_non_exhaustive()
    }
}

impl ProviderConfig {
    pub fn base_url(&self) -> String {
        let mut url = self
            .base_url_env
            .as_ref()
            .and_then(|key| self.value(key))
            .unwrap_or(&self.default_base_url)
            .to_owned();
        for key in &self.config_envs {
            if let Some(value) = self.env.get(key) {
                url = url.replace(&format!("${{{key}}}"), value);
                url = url.replace(&format!("${key}"), value);
            }
        }
        url
    }

    pub fn api_key(&self) -> Option<&str> {
        self.api_key_value
            .as_deref()
            .or_else(|| self.api_key_env.as_ref().and_then(|key| self.value(key)))
    }

    pub fn is_configured(&self) -> bool {
        (self.api_key_env.is_none() || self.api_key().is_some_and(|v| !v.is_empty()))
            && self.config_envs.iter().all(|key| self.value(key).is_some())
    }

    /// Serialize the configuration without its credentials: the API key value,
    /// the `env` map and the headers. A restore takes them from the registry.
    pub fn snapshot(&self) -> Result<Value, Error> {
        let mut value = serde_json::to_value(self)?;
        if let Value::Object(fields) = &mut value {
            for field in CREDENTIAL_FIELDS {
                fields.remove(field);
            }
        }
        Ok(value)
    }

    fn value(&self, key: &str) -> Option<&str> {
        self.env
            .get(key)
            .map(String::as_str)
            .filter(|value| !value.is_empty())
    }
}

/// Minimal models.dev metadata supplied by the caller, without a bundled catalog.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelsDevProvider {
    pub id: String,
    pub npm: Option<String>,
    pub api: Option<String>,
    #[serde(default, deserialize_with = "null_default")]
    pub env: Vec<String>,
}

#[async_trait]
pub trait ProviderProtocol: Send + Sync {
    fn protocol_class_id(&self) -> &str;

    /// Include protocol-specific settings when a custom implementation has them.
    fn to_value(&self) -> Result<Value, Error> {
        Ok(serde_json::json!({"protocol_class_id": self.protocol_class_id()}))
    }

    fn stream(
        &self,
        _model: &Model,
        _messages: Vec<Message>,
        _options: StreamOptions,
        _provider: &dyn Provider,
    ) -> EventStream {
        unsupported_stream("stream")
    }

    async fn generate(
        &self,
        _model: &Model,
        _messages: Vec<Message>,
        _options: StreamOptions,
        _provider: &dyn Provider,
    ) -> Result<Message, Error> {
        Err(unsupported("generate"))
    }

    async fn evaluate(
        &self,
        _model: &Model,
        _state: EvaluationInput,
        _questions: BTreeMap<String, Question>,
        _params: EvaluationParams,
        _provider: &dyn Provider,
    ) -> Result<Item<Metadata>, Error> {
        Err(unsupported("evaluate"))
    }

    async fn generate_image(
        &self,
        _model: &Model,
        _prompt: ImagePrompt,
        _params: ImageParams,
        _provider: &dyn Provider,
    ) -> Result<Item<Vec<FilePart>>, Error> {
        Err(unsupported("generate_image"))
    }

    async fn generate_video(
        &self,
        _model: &Model,
        _prompt: VideoPrompt,
        _params: VideoParams,
        _provider: &dyn Provider,
    ) -> Result<Item<Vec<FilePart>>, Error> {
        Err(unsupported("generate_video"))
    }

    async fn generate_audio(
        &self,
        _model: &Model,
        _prompt: AudioPrompt,
        _params: AudioParams,
        _provider: &dyn Provider,
    ) -> Result<Item<Vec<FilePart>>, Error> {
        Err(unsupported("generate_audio"))
    }

    async fn embed(
        &self,
        _model: &Model,
        _values: Vec<String>,
        _params: EmbedParams,
        _provider: &dyn Provider,
    ) -> Result<Item<Vec<Vec<f64>>>, Error> {
        Err(unsupported("embed"))
    }

    async fn transcribe(
        &self,
        _model: &Model,
        _audio: TranscribeInput,
        _params: TranscribeParams,
        _provider: &dyn Provider,
    ) -> Result<Item<Transcription>, Error> {
        Err(unsupported("transcribe"))
    }

    async fn rerank(
        &self,
        _model: &Model,
        _documents: Documents,
        _query: String,
        _params: RerankParams,
        _provider: &dyn Provider,
    ) -> Result<Item<Vec<RankedDocument>>, Error> {
        Err(unsupported("rerank"))
    }

    /// The values one `embed` call takes for the model.
    fn embed_limits(&self, _model: &Model) -> CallLimits {
        CallLimits::default()
    }

    /// The parameters of the call that embeds `range` of a split list of
    /// `values` values. A protocol whose options hold an entry per value
    /// slices them here.
    fn embed_params_for(
        &self,
        _model: &Model,
        params: &EmbedParams,
        _values: usize,
        _range: std::ops::Range<usize>,
    ) -> Result<EmbedParams, Error> {
        Ok(params.clone())
    }

    /// The images one `generate_image` call returns for the model.
    fn image_limits(&self, _model: &Model) -> CallLimits {
        CallLimits::default()
    }

    /// The videos one video call or job returns for the model.
    fn video_limits(&self, _model: &Model) -> CallLimits {
        CallLimits::default()
    }

    /// Start a video job and return its operation value, which
    /// `video_status` takes.
    async fn start_video(
        &self,
        _model: &Model,
        _prompt: VideoPrompt,
        _params: VideoParams,
        _provider: &dyn Provider,
    ) -> Result<Item<Value>, Error> {
        Err(unsupported("start_video"))
    }

    async fn video_status(
        &self,
        _model: &Model,
        _operation: &Value,
        _provider: &dyn Provider,
    ) -> Result<VideoStatus, Error> {
        Err(unsupported("video_status"))
    }

    /// Fetch a video that a job returned as a URL, with the credentials the
    /// protocol attaches. A video already in bytes comes back unchanged.
    async fn download_video(
        &self,
        _model: &Model,
        _video: FilePart,
        _provider: &dyn Provider,
    ) -> Result<FilePart, Error> {
        Err(unsupported("download_video"))
    }

    async fn stream_transcription(
        &self,
        _model: &Model,
        _audio: AudioChunks,
        _format: AudioFormat,
        _params: StreamTranscribeParams,
        _provider: &dyn Provider,
    ) -> Result<TranscriptionStream, Error> {
        Err(unsupported("stream_transcription"))
    }

    async fn stream_translation(
        &self,
        _model: &Model,
        _audio: AudioChunks,
        _format: AudioFormat,
        _params: StreamTranslateParams,
        _provider: &dyn Provider,
    ) -> Result<TranslationStream, Error> {
        Err(unsupported("stream_translation"))
    }

    /// An ephemeral token for a client to open a realtime session with.
    async fn create_realtime_token(
        &self,
        _model: &Model,
        _params: RealtimeTokenParams,
        _provider: &dyn Provider,
    ) -> Result<RealtimeToken, Error> {
        Err(unsupported("realtime"))
    }

    /// The provider's session setup for a session configuration.
    fn realtime_session(
        &self,
        _model: &Model,
        _config: &RealtimeSessionConfig,
    ) -> Result<Value, Error> {
        Err(unsupported("realtime"))
    }

    /// A codec for one realtime connection.
    fn realtime_codec(&self, _model: &Model) -> Result<Box<dyn RealtimeCodec>, Error> {
        Err(unsupported("realtime"))
    }

    /// The WebSocket to open for a server-side connection with a token, and
    /// whether the connection waits for the session-created event before it
    /// sends a client event.
    fn realtime_connection(
        &self,
        _model: &Model,
        _token: &RealtimeToken,
    ) -> Result<(WebSocketRequest, bool), Error> {
        Err(unsupported("realtime"))
    }

    /// The WebSocket a server opens with the provider's own credentials,
    /// and how a session starts on it (OPS-014).
    fn realtime_server_connection(
        &self,
        _model: &Model,
        _provider: &dyn Provider,
    ) -> Result<(WebSocketRequest, RealtimeStartup), Error> {
        Err(unsupported("realtime server connection"))
    }

    /// Create a WebRTC session from a client's SDP offer (OPS-014).
    async fn create_webrtc_session(
        &self,
        _model: &Model,
        _offer_sdp: String,
        _config: RealtimeSessionConfig,
        _provider: &dyn Provider,
    ) -> Result<WebRtcSession, Error> {
        Err(unsupported("webrtc session"))
    }
}

/// Implement a provider to own a transport and select a default wire protocol.
/// Methods return UnsupportedOperation when an optional capability is absent.
#[async_trait]
pub trait Provider: Send + Sync + AsProvider {
    fn name(&self) -> &str;
    fn config(&self) -> &ProviderConfig;
    fn provider_class_id(&self) -> &str {
        self.name()
    }
    /// Serialize configuration only. Credentials, transports and open
    /// connections are excluded; see `ProviderConfig::snapshot`.
    fn to_value(&self) -> Result<Value, Error> {
        let mut value = self.config().snapshot()?;
        value["name"] = Value::String(self.name().to_owned());
        value["provider_class_id"] = Value::String(self.provider_class_id().to_owned());
        if let Some(protocol) = self.protocol_override() {
            value["protocol_override"] = protocol.to_value()?;
        }
        Ok(value)
    }
    fn default_protocol(&self) -> Option<Arc<dyn ProviderProtocol>> {
        None
    }
    fn protocol_override(&self) -> Option<Arc<dyn ProviderProtocol>> {
        None
    }
    fn protocol(&self) -> Option<Arc<dyn ProviderProtocol>> {
        self.protocol_override().or_else(|| self.default_protocol())
    }
    fn transport(&self) -> Option<&dyn HttpTransport> {
        None
    }
    /// Share an owned transport with a provider adapter's static event stream.
    /// Implement this alongside `transport` when using a concrete HTTP protocol.
    fn shared_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        None
    }
    fn client(&self) -> Result<&dyn HttpTransport, Error> {
        self.transport()
            .ok_or_else(|| Error::Configuration("provider client has not been initialized".into()))
    }
    fn base_url(&self) -> String {
        self.config().base_url()
    }
    fn api_key(&self) -> Option<&str> {
        self.config().api_key()
    }
    fn is_configured(&self) -> bool {
        self.config().is_configured()
    }
    async fn aclose(&self) -> Result<(), Error> {
        Ok(())
    }

    fn stream(&self, model: &Model, messages: Vec<Message>, options: StreamOptions) -> EventStream {
        match model.protocol.clone().or_else(|| self.protocol()) {
            Some(protocol) => protocol.stream(model, messages, options, self.as_provider()),
            None => unsupported_stream("stream"),
        }
    }

    async fn generate(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
    ) -> Result<Message, Error> {
        match model.protocol.clone().or_else(|| self.protocol()) {
            Some(protocol) => {
                protocol
                    .generate(model, messages, options, self.as_provider())
                    .await
            }
            None => Err(unsupported("generate")),
        }
    }

    async fn evaluate(
        &self,
        model: &Model,
        state: EvaluationInput,
        questions: BTreeMap<String, Question>,
        params: EvaluationParams,
    ) -> Result<Item<Metadata>, Error> {
        for question in questions.values() {
            question.validate()?;
        }
        match model.protocol.clone().or_else(|| self.protocol()) {
            Some(protocol) => {
                protocol
                    .evaluate(model, state, questions, params, self.as_provider())
                    .await
            }
            None => Err(unsupported("evaluate")),
        }
    }

    async fn generate_image(
        &self,
        model: &Model,
        prompt: ImagePrompt,
        params: ImageParams,
    ) -> Result<Item<Vec<FilePart>>, Error> {
        match model.protocol.clone().or_else(|| self.protocol()) {
            Some(protocol) => {
                protocol
                    .generate_image(model, prompt, params, self.as_provider())
                    .await
            }
            None => Err(unsupported("generate_image")),
        }
    }

    async fn generate_video(
        &self,
        model: &Model,
        prompt: VideoPrompt,
        params: VideoParams,
    ) -> Result<Item<Vec<FilePart>>, Error> {
        match model.protocol.clone().or_else(|| self.protocol()) {
            Some(protocol) => {
                protocol
                    .generate_video(model, prompt, params, self.as_provider())
                    .await
            }
            None => Err(unsupported("generate_video")),
        }
    }

    async fn generate_audio(
        &self,
        model: &Model,
        prompt: AudioPrompt,
        params: AudioParams,
    ) -> Result<Item<Vec<FilePart>>, Error> {
        match model.protocol.clone().or_else(|| self.protocol()) {
            Some(protocol) => {
                protocol
                    .generate_audio(model, prompt, params, self.as_provider())
                    .await
            }
            None => Err(unsupported("generate_audio")),
        }
    }

    async fn embed(
        &self,
        model: &Model,
        values: Vec<String>,
        params: EmbedParams,
    ) -> Result<Item<Vec<Vec<f64>>>, Error> {
        match model.protocol.clone().or_else(|| self.protocol()) {
            Some(protocol) => {
                protocol
                    .embed(model, values, params, self.as_provider())
                    .await
            }
            None => Err(unsupported("embed")),
        }
    }

    async fn transcribe(
        &self,
        model: &Model,
        audio: TranscribeInput,
        params: TranscribeParams,
    ) -> Result<Item<Transcription>, Error> {
        match model.protocol.clone().or_else(|| self.protocol()) {
            Some(protocol) => {
                protocol
                    .transcribe(model, audio, params, self.as_provider())
                    .await
            }
            None => Err(unsupported("transcribe")),
        }
    }

    async fn rerank(
        &self,
        model: &Model,
        documents: Documents,
        query: String,
        params: RerankParams,
    ) -> Result<Item<Vec<RankedDocument>>, Error> {
        match model.protocol.clone().or_else(|| self.protocol()) {
            Some(protocol) => {
                protocol
                    .rerank(model, documents, query, params, self.as_provider())
                    .await
            }
            None => Err(unsupported("rerank")),
        }
    }

    /// The WebSocket transport for live and realtime sessions, when the
    /// provider has one.
    fn websocket(&self) -> Option<Arc<dyn WebSocketTransport>> {
        None
    }

    /// The transport that fetches the media URLs a protocol must download
    /// itself, when the provider has one. It honors the core DNS-pinning
    /// contract of [`crate::models::files::DownloadTransport`].
    fn download_transport(&self) -> Option<Arc<dyn crate::models::files::DownloadTransport>> {
        None
    }

    /// Upload a file through the provider's files operation.
    async fn upload_file(
        &self,
        _file: UploadFile,
        _params: UploadParams,
    ) -> Result<Item<UploadedFile>, Error> {
        Err(unsupported("upload_file"))
    }

    /// A file's metadata, by the provider reference its upload returned.
    async fn file_info(
        &self,
        _reference: &BTreeMap<String, String>,
    ) -> Result<Item<UploadedFile>, Error> {
        Err(unsupported("file_info"))
    }

    /// A file's bytes, by its provider reference, reading at most
    /// `max_bytes`.
    async fn download_file(
        &self,
        _reference: &BTreeMap<String, String>,
        _max_bytes: usize,
    ) -> Result<Item<DownloadedFile>, Error> {
        Err(unsupported("download_file"))
    }

    /// Delete a file by its provider reference; whether the provider deleted it.
    async fn delete_file(
        &self,
        _reference: &BTreeMap<String, String>,
    ) -> Result<Item<bool>, Error> {
        Err(unsupported("delete_file"))
    }

    /// Upload a skill through the provider's skills operation.
    async fn upload_skill(
        &self,
        _files: Vec<SkillFile>,
        _params: SkillParams,
    ) -> Result<Item<UploadedSkill>, Error> {
        Err(unsupported("upload_skill"))
    }

    async fn batch_start(
        &self,
        _requests: Vec<BatchRequest>,
        _params: BatchParams,
    ) -> Result<BatchStarted, Error> {
        Err(unsupported("batch_start"))
    }

    async fn batch_status(&self, _batch_id: &str) -> Result<BatchStatus, Error> {
        Err(unsupported("batch_status"))
    }

    async fn batch_cancel(&self, _batch_id: &str) -> Result<(), Error> {
        Err(unsupported("batch_cancel"))
    }

    async fn batch_list(
        &self,
        _page_size: Option<u32>,
        _cursor: Option<String>,
    ) -> Result<BatchPage, Error> {
        Err(unsupported("batch_list"))
    }

    async fn batch_results(&self, _batch_id: &str) -> Result<BatchResults, Error> {
        Err(unsupported("batch_results"))
    }

    async fn list_models(&self) -> Result<Vec<String>, Error> {
        Err(unsupported("list_models"))
    }
    async fn probe(&self, _model: &Model) -> Result<(), Error> {
        Err(unsupported("probe"))
    }

    fn from_id(id: &str, registry: &ProviderRegistry) -> Result<Arc<dyn Provider>, Error>
    where
        Self: Sized,
    {
        registry.get_provider(id)
    }
    fn from_modelsdev_provider(
        _metadata: &ModelsDevProvider,
        _config: ProviderConfig,
    ) -> Result<Self, Error>
    where
        Self: Sized,
    {
        Err(unsupported("from_modelsdev_provider"))
    }
}

/// A provider that delegates to a registered protocol using an explicit transport.
#[derive(Clone)]
pub struct ConfiguredProvider {
    name: String,
    config: ProviderConfig,
    default_protocol: Arc<dyn ProviderProtocol>,
    protocol_override: Option<Arc<dyn ProviderProtocol>>,
    transport: Option<Arc<dyn HttpTransport>>,
}
impl std::fmt::Debug for ConfiguredProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfiguredProvider")
            .field("name", &self.name)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}
impl ConfiguredProvider {
    pub fn new(
        name: impl Into<String>,
        config: ProviderConfig,
        protocol: Arc<dyn ProviderProtocol>,
        transport: Option<Arc<dyn HttpTransport>>,
    ) -> Self {
        Self {
            name: name.into(),
            config,
            default_protocol: protocol,
            protocol_override: None,
            transport,
        }
    }
    pub fn with_protocol(mut self, protocol: Arc<dyn ProviderProtocol>) -> Self {
        self.protocol_override = Some(protocol);
        self
    }
    pub fn from_modelsdev_provider(
        metadata: &ModelsDevProvider,
        mut config: ProviderConfig,
        protocol: Arc<dyn ProviderProtocol>,
        transport: Option<Arc<dyn HttpTransport>>,
    ) -> Self {
        if config.default_base_url.is_empty() {
            config.default_base_url = metadata.api.clone().unwrap_or_default();
        }
        if config.api_key_env.is_none() {
            // Template variables describe endpoint configuration, not credentials.
            let (key, envs) = provider_config(metadata);
            config.api_key_env = key;
            config.config_envs = envs;
        }
        Self::new(metadata.id.clone(), config, protocol, transport)
    }
}
#[async_trait]
impl Provider for ConfiguredProvider {
    fn name(&self) -> &str {
        &self.name
    }
    fn config(&self) -> &ProviderConfig {
        &self.config
    }
    fn default_protocol(&self) -> Option<Arc<dyn ProviderProtocol>> {
        Some(self.default_protocol.clone())
    }
    fn protocol_override(&self) -> Option<Arc<dyn ProviderProtocol>> {
        self.protocol_override.clone()
    }
    fn transport(&self) -> Option<&dyn HttpTransport> {
        self.transport.as_deref()
    }
    fn shared_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        self.transport.clone()
    }
}

/// Explicit registry. Clones share providers but registration requires exclusive access.
#[derive(Clone, Default)]
pub struct ProviderRegistry {
    providers: BTreeMap<String, Arc<dyn Provider>>,
    provider_factories: BTreeMap<String, ProviderFactory>,
    protocol_factories: BTreeMap<String, ProtocolFactory>,
}

type ProviderFactory =
    Arc<dyn Fn(Value, &ProviderRegistry) -> Result<Arc<dyn Provider>, Error> + Send + Sync>;
type ProtocolFactory = Arc<dyn Fn(Value) -> Result<Arc<dyn ProviderProtocol>, Error> + Send + Sync>;
impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register(
        &mut self,
        id: impl Into<String>,
        provider: Arc<dyn Provider>,
    ) -> Result<(), Error> {
        let id = id.into();
        if id.is_empty() {
            return Err(Error::Configuration(
                "provider handle must not be empty".into(),
            ));
        }
        if self
            .providers
            .get(&id)
            .is_some_and(|existing| !Arc::ptr_eq(existing, &provider))
        {
            return Err(Error::Configuration(format!(
                "duplicate provider handle: {id:?}"
            )));
        }
        self.providers.insert(id, provider);
        Ok(())
    }
    /// Register restoration for a concrete provider class, including its custom fields.
    pub fn register_provider_factory<F>(
        &mut self,
        class_id: impl Into<String>,
        factory: F,
    ) -> Result<(), Error>
    where
        F: Fn(Value, &ProviderRegistry) -> Result<Arc<dyn Provider>, Error> + Send + Sync + 'static,
    {
        let class_id = class_id.into();
        if class_id.is_empty() || self.provider_factories.contains_key(&class_id) {
            return Err(Error::Configuration(format!(
                "duplicate or empty provider class id: {class_id:?}"
            )));
        }
        self.provider_factories.insert(class_id, Arc::new(factory));
        Ok(())
    }
    /// Register restoration for a concrete wire protocol class.
    pub fn register_protocol_factory<F>(
        &mut self,
        class_id: impl Into<String>,
        factory: F,
    ) -> Result<(), Error>
    where
        F: Fn(Value) -> Result<Arc<dyn ProviderProtocol>, Error> + Send + Sync + 'static,
    {
        let class_id = class_id.into();
        if class_id.is_empty() || self.protocol_factories.contains_key(&class_id) {
            return Err(Error::Configuration(format!(
                "duplicate or empty provider protocol class id: {class_id:?}"
            )));
        }
        self.protocol_factories.insert(class_id, Arc::new(factory));
        Ok(())
    }
    /// Restore a provider snapshot. The credentials (API key value, `env` map
    /// and headers) and the endpoint fields (`default_base_url`,
    /// `base_url_env`, `api_key_env` and `config_envs`) come from the provider
    /// registered under the snapshot's name, never from the snapshot. A
    /// snapshot of another provider class is refused.
    pub fn restore_provider(&self, mut value: Value) -> Result<Arc<dyn Provider>, Error> {
        let id = value
            .get("provider_class_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Error::Configuration("provider data must include provider_class_id".into())
            })?
            .to_owned();
        let name = value
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| Error::Configuration("provider data must include a name".into()))?
            .to_owned();
        // Look the provider up first, so every failure to restore it names it.
        let registered = self.providers.get(&name).ok_or_else(|| {
            Error::Configuration(format!(
                "cannot restore provider {name:?}: the registry holds no provider under that name to supply its credentials"
            ))
        })?;
        let factory = self.provider_factories.get(&id).ok_or_else(|| {
            Error::Configuration(format!(
                "cannot restore provider {name:?}: unknown provider_class_id {id:?}"
            ))
        })?;
        // The registered provider decides where requests go and how they
        // authenticate: a changed snapshot cannot send its credentials to
        // another host, and an endpoint change applies to older snapshots.
        if registered.provider_class_id() != id {
            return Err(Error::Configuration(format!(
                "cannot restore provider {name:?}: its snapshot class {id:?} differs from the registered provider's class {:?}",
                registered.provider_class_id()
            )));
        }
        let registered_config = serde_json::to_value(registered.config())?;
        let fields = value
            .as_object_mut()
            .ok_or_else(|| Error::Configuration("provider data must be a JSON object".into()))?;
        for field in CREDENTIAL_FIELDS.into_iter().chain(ENDPOINT_FIELDS) {
            match registered_config
                .get(field)
                .filter(|value| !value.is_null())
            {
                Some(setting) => fields.insert(field.to_owned(), setting.clone()),
                None => fields.remove(field),
            };
        }
        factory(value, self)
    }
    pub fn restore_protocol(&self, value: Value) -> Result<Arc<dyn ProviderProtocol>, Error> {
        let id = value
            .get("protocol_class_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                Error::Configuration("provider protocol data must include protocol_class_id".into())
            })?;
        let factory = self.protocol_factories.get(id).ok_or_else(|| {
            Error::Configuration(format!("unknown provider protocol_class_id: {id:?}"))
        })?;
        factory(value)
    }
    pub fn get_provider(&self, id: &str) -> Result<Arc<dyn Provider>, Error> {
        self.providers
            .get(id)
            .cloned()
            .ok_or_else(|| Error::UnsupportedProvider(id.to_owned()))
    }
    /// Match a provider's models.dev identity first, then its adapter package handle.
    pub fn from_modelsdev_provider(
        &self,
        metadata: &ModelsDevProvider,
    ) -> Result<Arc<dyn Provider>, Error> {
        self.get_provider(&metadata.id).or_else(|_| {
            metadata
                .npm
                .as_ref()
                .ok_or_else(|| Error::UnsupportedProvider(metadata.id.clone()))
                .and_then(|npm| self.get_provider(npm))
        })
    }
}

pub fn get_provider(id: &str, registry: &ProviderRegistry) -> Result<Arc<dyn Provider>, Error> {
    registry.get_provider(id)
}
pub fn provider_base_url(metadata: &ModelsDevProvider) -> Option<&str> {
    metadata.api.as_deref()
}
pub fn provider_config(metadata: &ModelsDevProvider) -> (Option<String>, Vec<String>) {
    let references = template_envs(metadata.api.as_deref().unwrap_or(""));
    let mut envs = metadata.env.clone();
    for name in &references {
        if !envs.contains(name) {
            envs.push(name.clone());
        }
    }
    let mut candidates: Vec<_> = envs
        .iter()
        .filter(|name| !references.contains(name))
        .collect();
    if candidates.is_empty() {
        candidates = envs.iter().collect();
    }
    let key = ["API_KEY", "TOKEN", "SECRET", "BEARER"]
        .into_iter()
        .find_map(|marker| {
            candidates
                .iter()
                .find(|name| name.contains(marker))
                .map(|name| (*name).clone())
        })
        .or_else(|| candidates.first().map(|name| (*name).clone()));
    let config = envs
        .into_iter()
        .filter(|name| Some(name) != key.as_ref())
        .collect();
    (key, config)
}

fn template_envs(url: &str) -> Vec<String> {
    let mut names = Vec::new();
    for tail in url.split('$').skip(1) {
        let tail = tail.strip_prefix('{').unwrap_or(tail);
        let name: String = tail
            .chars()
            .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
            .collect();
        if name.starts_with(|c: char| c.is_ascii_uppercase() || c == '_') && !names.contains(&name)
        {
            names.push(name);
        }
    }
    names
}
fn unsupported(operation: &str) -> Error {
    Error::UnsupportedOperation(operation.to_owned())
}
fn unsupported_stream(operation: &str) -> EventStream {
    let error = unsupported(operation);
    stream::once(async move { Err(error) }).boxed()
}

/// Object-safe provider reference for the trait's forwarding defaults.
#[doc(hidden)]
pub trait AsProvider {
    fn as_provider(&self) -> &dyn Provider;
}
impl<T: Provider> AsProvider for T {
    fn as_provider(&self) -> &dyn Provider {
        self
    }
}

fn null_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}
