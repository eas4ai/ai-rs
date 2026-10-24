use super::{
    OpenAIResponsesProtocol, errors, protocol,
    protocol::{HostChatProtocol, HostResponsesProtocol},
};
use crate::{
    Error, Model, Provider, ProviderConfig, ProviderProtocol, ProviderRegistry,
    models::files::DownloadTransport,
    providers::{
        ModelsDevProvider,
        base::provider_config,
        hosts, http,
        operations::{
            BatchPage, BatchParams, BatchRequest, BatchResults, BatchStarted, BatchStatus,
            DownloadedFile, Item, SkillFile, SkillParams, UploadFile, UploadParams, UploadedFile,
            UploadedSkill,
        },
    },
    transport::{HttpTransport, WebSocketTransport},
};
use async_trait::async_trait;
use serde_json::Value;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone)]
pub struct OpenAICompatibleProvider {
    name: String,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
    /// Chosen when the provider is built, from the API it targets.
    default_protocol: Arc<dyn ProviderProtocol>,
    protocol_override: Option<Arc<dyn ProviderProtocol>>,
    downloads: Option<Arc<dyn DownloadTransport>>,
    websocket: Option<Arc<dyn WebSocketTransport>>,
}
impl std::fmt::Debug for OpenAICompatibleProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAICompatibleProvider")
            .field("name", &self.name)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}
impl OpenAICompatibleProvider {
    pub const HANDLES: [&'static str; 3] =
        ["openai", "@ai-sdk/openai", "@ai-sdk/openai-compatible"];
    pub fn new(mut config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> Self {
        if config.default_base_url.is_empty() {
            config.default_base_url = "https://api.openai.com/v1".into();
            if config.base_url_env.is_none() {
                config.base_url_env = Some("OPENAI_BASE_URL".into());
            }
        }
        if config.api_key_env.is_none() {
            config.api_key_env = Some("OPENAI_API_KEY".into());
        }
        Self {
            name: "openai".into(),
            config,
            transport,
            default_protocol: Arc::new(OpenAIResponsesProtocol),
            protocol_override: None,
            downloads: None,
            websocket: None,
        }
    }
    /// Build a provider from models.dev metadata. Metadata that names a
    /// Responses host of `providers::hosts`, xAI or Hugging Face, by package
    /// or id, gets that host's Responses rules. Other metadata speaks
    /// Responses when it names OpenAI's own adapter package,
    /// `@ai-sdk/openai`, and Chat Completions otherwise; metadata that names
    /// a Chat Completions host, by package or id, gets that host's request
    /// and response rules and the other model kinds it serves, such as
    /// Together AI's embeddings, images and reranking.
    pub fn from_modelsdev_provider(
        metadata: &ModelsDevProvider,
        mut config: ProviderConfig,
        transport: Arc<dyn HttpTransport>,
    ) -> Result<Self, Error> {
        let explicit_url = !config.default_base_url.is_empty();
        if !explicit_url {
            config.default_base_url = metadata.api.clone().unwrap_or_default();
            if metadata.id == "openai" && config.default_base_url.is_empty() {
                config.default_base_url = "https://api.openai.com/v1".into();
            }
        }
        if config.default_base_url.is_empty() {
            return Err(Error::Configuration(format!(
                "provider {:?} does not declare an API URL",
                metadata.id
            )));
        }
        let (key, envs) = provider_config(metadata);
        if config.api_key_env.is_none() {
            config.api_key_env = key;
        }
        config.config_envs = envs;
        if metadata.id == "openai" && !explicit_url && config.base_url_env.is_none() {
            config.base_url_env = Some("OPENAI_BASE_URL".into());
        }
        // A Responses host keeps its rules even under OpenAI's package, whose
        // own Responses rules would send its item ids and settings wrongly.
        let default_protocol: Arc<dyn ProviderProtocol> =
            if let Some(profile) = hosts::responses_profile_for(metadata) {
                Arc::new(HostResponsesProtocol(profile))
            } else if metadata.npm.as_deref() == Some("@ai-sdk/openai") {
                Arc::new(OpenAIResponsesProtocol)
            } else if let Some(protocol) = hosts::chat_protocol_for(metadata) {
                protocol
            } else {
                // A server that only speaks OpenAI's wire format gets the
                // OpenAI-compatible rules, not OpenAI's own.
                Arc::new(HostChatProtocol(&super::profile::COMPATIBLE, None))
            };
        Ok(Self {
            name: metadata.id.clone(),
            config,
            transport,
            default_protocol,
            protocol_override: None,
            downloads: None,
            websocket: None,
        })
    }
    #[must_use]
    pub fn with_protocol(mut self, protocol: Arc<dyn ProviderProtocol>) -> Self {
        self.protocol_override = Some(protocol);
        self
    }
    /// Download the media URLs an API cannot fetch itself before each
    /// request. Without one, such a URL fails with `Error::Validation`.
    /// Download transports must honor the core DNS-pinning contract.
    /// Runtime transports are not serialized; a restore through a registry
    /// takes the download transport of the provider registered under the
    /// same name.
    #[must_use]
    pub fn with_download_transport(mut self, downloads: Arc<dyn DownloadTransport>) -> Self {
        self.downloads = Some(downloads);
        self
    }
    /// Open realtime and live WebSockets with this transport. Runtime
    /// transports are not serialized; set this again after restoring a
    /// snapshot.
    #[must_use]
    pub fn with_websocket_transport(mut self, websocket: Arc<dyn WebSocketTransport>) -> Self {
        self.websocket = Some(websocket);
        self
    }
    /// OpenAI's own API, whose files, skills and batches the provider
    /// serves; an OpenAI-compatible server has none of them.
    fn openai_api(&self) -> bool {
        self.default_protocol.protocol_class_id() == "openai_responses"
    }
    /// The files operation of OpenAI's own API.
    fn files(&self, operation: &str) -> Result<super::files::Files<'_>, Error> {
        if !self.openai_api() {
            return Err(Error::UnsupportedOperation(operation.to_owned()));
        }
        Ok(super::files::Files {
            config: &self.config,
            transport: self.transport.as_ref(),
            provider: &self.name,
        })
    }
    async fn get(&self, path: &[&str], model: Option<&str>) -> Result<Value, Error> {
        let request = protocol::http_request(&errors::API, "GET", &self.config, path, None, None)?;
        http::json(
            &errors::API,
            self.transport.as_ref(),
            request,
            &self.name,
            model,
        )
        .await
    }
}
#[async_trait]
impl Provider for OpenAICompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }
    fn config(&self) -> &ProviderConfig {
        &self.config
    }
    fn provider_class_id(&self) -> &str {
        "openai-compatible"
    }
    fn transport(&self) -> Option<&dyn HttpTransport> {
        Some(self.transport.as_ref())
    }
    fn shared_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        Some(self.transport.clone())
    }
    fn default_protocol(&self) -> Option<Arc<dyn ProviderProtocol>> {
        Some(self.default_protocol.clone())
    }
    fn protocol_override(&self) -> Option<Arc<dyn ProviderProtocol>> {
        self.protocol_override.clone()
    }
    fn websocket(&self) -> Option<Arc<dyn WebSocketTransport>> {
        self.websocket.clone()
    }
    fn download_transport(&self) -> Option<Arc<dyn DownloadTransport>> {
        self.downloads.clone()
    }
    fn is_configured(&self) -> bool {
        self.api_key().is_some_and(|k| !k.is_empty()) && self.config.is_configured()
    }
    async fn list_models(&self) -> Result<Vec<String>, Error> {
        let value = self.get(&["models"], None).await?;
        let data = value
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| http::response_error(&self.name, "OpenAI model list is missing data"))?;
        let mut ids = data
            .iter()
            .map(|m| {
                m.get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        http::response_error(&self.name, "OpenAI model entry is missing id")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        ids.sort();
        Ok(ids)
    }
    async fn upload_file(
        &self,
        file: UploadFile,
        params: UploadParams,
    ) -> Result<Item<UploadedFile>, Error> {
        self.files("upload_file")?.upload(file, params).await
    }
    async fn file_info(
        &self,
        reference: &BTreeMap<String, String>,
    ) -> Result<Item<UploadedFile>, Error> {
        self.files("file_info")?.info(reference).await
    }
    async fn download_file(
        &self,
        reference: &BTreeMap<String, String>,
        max_bytes: usize,
    ) -> Result<Item<DownloadedFile>, Error> {
        self.files("download_file")?
            .download(reference, max_bytes)
            .await
    }
    async fn delete_file(&self, reference: &BTreeMap<String, String>) -> Result<Item<bool>, Error> {
        self.files("delete_file")?.delete(reference).await
    }
    async fn upload_skill(
        &self,
        files: Vec<SkillFile>,
        params: SkillParams,
    ) -> Result<Item<UploadedSkill>, Error> {
        self.files("upload_skill")?
            .upload_skill(files, params)
            .await
    }
    async fn batch_start(
        &self,
        requests: Vec<BatchRequest>,
        params: BatchParams,
    ) -> Result<BatchStarted, Error> {
        self.files("batch_start")?
            .batch_start(requests, params)
            .await
    }
    async fn batch_status(&self, batch_id: &str) -> Result<BatchStatus, Error> {
        self.files("batch_status")?.batch_status(batch_id).await
    }
    async fn batch_cancel(&self, batch_id: &str) -> Result<(), Error> {
        self.files("batch_cancel")?.batch_cancel(batch_id).await
    }
    async fn batch_list(
        &self,
        page_size: Option<u32>,
        cursor: Option<String>,
    ) -> Result<BatchPage, Error> {
        self.files("batch_list")?
            .batch_list(page_size, cursor)
            .await
    }
    async fn batch_results(&self, batch_id: &str) -> Result<BatchResults, Error> {
        self.files("batch_results")?
            .batch_results(batch_id, self.transport.clone())
            .await
    }
    async fn probe(&self, model: &Model) -> Result<(), Error> {
        if !self.is_configured() {
            return Err(crate::errors::ProviderError::new(
                crate::errors::ProviderErrorKind::NotConfigured,
                format!("provider {:?} is not configured", self.name),
            )
            .with_provider(&self.name)
            .into());
        }
        self.get(&["models", &model.id], Some(&model.id)).await?;
        Ok(())
    }
}
/// Register this instance under its provider id and, for OpenAI, its package handles.
pub fn register(
    registry: &mut ProviderRegistry,
    provider: Arc<OpenAICompatibleProvider>,
) -> Result<(), Error> {
    registry.register(provider.name(), provider.clone())?;
    if provider.name() == "openai" {
        for handle in OpenAICompatibleProvider::HANDLES {
            registry.register(handle, provider.clone())?;
        }
    }
    Ok(())
}
/// Snapshot restoration captures a caller-owned transport; snapshots contain no clients.
pub fn register_factories(
    registry: &mut ProviderRegistry,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    registry.register_protocol_factory("openai_responses", |_| {
        Ok(Arc::new(super::OpenAIResponsesProtocol))
    })?;
    registry.register_protocol_factory("openai_chat_completions", |_| {
        Ok(Arc::new(super::OpenAIChatCompletionsProtocol))
    })?;
    registry.register_protocol_factory("openai_decisions", |_| {
        Ok(Arc::new(super::OpenAIDecisionsProtocol))
    })?;
    for class in ["openai_embedding", "openai_compatible_embedding"] {
        registry.register_protocol_factory(class, |value| {
            Ok(Arc::new(super::embedding::restore(&value)))
        })?;
    }
    for class in ["openai_completions", "openai_compatible_completions"] {
        registry.register_protocol_factory(class, |value| {
            Ok(Arc::new(super::completions::restore(&value)))
        })?;
    }
    registry.register_provider_factory("openai-compatible", move |value, registry| {
        let name = value
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| !n.is_empty())
            .ok_or_else(|| Error::Configuration("OpenAI snapshot is missing name".into()))?
            .to_owned();
        let config = serde_json::from_value(value.clone())?;
        let protocol_override = value
            .get("protocol_override")
            .filter(|p| !p.is_null())
            .map(|p| registry.restore_protocol(p.clone()))
            .transpose()?;
        // The registered provider of this name decides the API and the
        // download transport, as it decides the endpoint.
        let registered = registry.get_provider(&name)?;
        let default_protocol = registered.default_protocol().ok_or_else(|| {
            Error::Configuration(format!(
                "cannot restore provider {name:?}: the registered provider has no default protocol"
            ))
        })?;
        Ok(Arc::new(OpenAICompatibleProvider {
            name,
            config,
            transport: transport.clone(),
            default_protocol,
            protocol_override,
            downloads: registered.download_transport(),
            websocket: None,
        }))
    })
}
