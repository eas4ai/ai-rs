use super::{
    AnthropicMessagesProtocol,
    errors::API,
    protocol::{self, HostMessagesProtocol, MessagesProfile},
};
use crate::{
    Error, Model, Provider, ProviderConfig, ProviderProtocol, ProviderRegistry,
    errors::{ProviderError, ProviderErrorKind},
    models::files::DownloadTransport,
    providers::{ModelsDevProvider, base::provider_config, http},
    transport::HttpTransport,
};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::{fmt, sync::Arc};

#[derive(Clone)]
pub struct AnthropicCompatibleProvider {
    name: String,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
    protocol_override: Option<Arc<dyn ProviderProtocol>>,
    anthropic_version: String,
    /// Whether this provider is for Anthropic's own API, which alone gets
    /// Anthropic's key and token names by default (ANT-001).
    anthropic_api: bool,
    downloads: Option<Arc<dyn DownloadTransport>>,
}
impl fmt::Debug for AnthropicCompatibleProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicCompatibleProvider")
            .field("name", &self.name)
            .field("config", &self.config)
            .field("anthropic_version", &self.anthropic_version)
            .finish_non_exhaustive()
    }
}
impl AnthropicCompatibleProvider {
    pub const HANDLES: &[&str] = &["anthropic", "@ai-sdk/anthropic"];
    /// Build a provider of Anthropic's API without reading process
    /// environment variables.
    pub fn new(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> Self {
        Self::build(config, transport, true)
    }
    /// A provider of Anthropic's own API defaults to Anthropic's base URL
    /// and key names and takes its auth token; another server gets none of
    /// them.
    fn build(
        mut config: ProviderConfig,
        transport: Arc<dyn HttpTransport>,
        anthropic_api: bool,
    ) -> Self {
        if anthropic_api {
            if config.default_base_url.is_empty() {
                config.default_base_url = "https://api.anthropic.com".into();
                if config.base_url_env.is_none() {
                    config.base_url_env = Some("ANTHROPIC_BASE_URL".into());
                }
            }
            if config.api_key_env.is_none() {
                config.api_key_env = Some("ANTHROPIC_API_KEY".into());
            }
        }
        Self {
            name: "anthropic".into(),
            config,
            transport,
            protocol_override: None,
            anthropic_version: "2023-06-01".into(),
            anthropic_api,
            downloads: None,
        }
        .with_anthropic_version("2023-06-01")
    }
    /// The Messages rules of this provider's API.
    fn profile(&self) -> &'static MessagesProfile {
        if self.anthropic_api {
            &protocol::ANTHROPIC_OWN
        } else {
            &protocol::ANTHROPIC
        }
    }
    /// Download the user files that Anthropic does not take by URL, such as
    /// a text file, before each request (ANT-010). Without one, such a URL
    /// fails with `Error::Validation`. Download transports must honor the
    /// core DNS-pinning contract. Runtime transports are not serialized; set
    /// this again after restoring a snapshot.
    #[must_use]
    pub fn with_download_transport(mut self, downloads: Arc<dyn DownloadTransport>) -> Self {
        self.downloads = Some(downloads);
        self
    }
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
    #[must_use]
    pub fn with_anthropic_version(mut self, version: impl Into<String>) -> Self {
        self.anthropic_version = version.into();
        self.config
            .headers
            .retain(|k, _| !k.eq_ignore_ascii_case("anthropic-version"));
        self.config
            .headers
            .insert("anthropic-version".into(), self.anthropic_version.clone());
        self
    }
    #[must_use]
    pub fn with_protocol(mut self, protocol: Arc<dyn ProviderProtocol>) -> Self {
        self.protocol_override = Some(protocol);
        self
    }
    pub fn from_modelsdev(
        metadata: &ModelsDevProvider,
        mut config: ProviderConfig,
        transport: Arc<dyn HttpTransport>,
    ) -> Result<Self, Error> {
        if config.default_base_url.is_empty() {
            config.default_base_url = metadata
                .api
                .clone()
                .or_else(|| {
                    (metadata.id == "anthropic").then(|| "https://api.anthropic.com".into())
                })
                .ok_or_else(|| {
                    Error::Configuration(format!(
                        "provider {:?} does not declare an API URL",
                        metadata.id
                    ))
                })?;
            if metadata.id == "anthropic" && config.base_url_env.is_none() {
                config.base_url_env = Some("ANTHROPIC_BASE_URL".into());
            }
        }
        if config.api_key_env.is_none() {
            let (key, envs) = provider_config(metadata);
            config.api_key_env = key;
            config.config_envs = envs;
        }
        Ok(Self::build(config, transport, metadata.id == "anthropic").with_name(&metadata.id))
    }
}
#[async_trait]
impl Provider for AnthropicCompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }
    fn config(&self) -> &ProviderConfig {
        &self.config
    }
    fn provider_class_id(&self) -> &str {
        "anthropic-compatible"
    }
    fn default_protocol(&self) -> Option<Arc<dyn ProviderProtocol>> {
        Some(if self.anthropic_api {
            Arc::new(HostMessagesProtocol(&protocol::ANTHROPIC_OWN))
        } else {
            Arc::new(AnthropicMessagesProtocol)
        })
    }
    fn protocol_override(&self) -> Option<Arc<dyn ProviderProtocol>> {
        self.protocol_override.clone()
    }
    fn transport(&self) -> Option<&dyn HttpTransport> {
        Some(self.transport.as_ref())
    }
    fn shared_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        Some(self.transport.clone())
    }
    fn download_transport(&self) -> Option<Arc<dyn DownloadTransport>> {
        self.downloads.clone()
    }
    fn is_configured(&self) -> bool {
        let token = self
            .profile()
            .auth_token_env
            .and_then(|env| self.config.env.get(env))
            .is_some_and(|token| !token.is_empty());
        (token || self.api_key().is_some_and(|key| !key.is_empty())) && self.config.is_configured()
    }
    fn to_value(&self) -> Result<Value, Error> {
        let mut value = self.config.snapshot()?;
        value["name"] = json!(self.name);
        value["provider_class_id"] = json!(self.provider_class_id());
        value["anthropic_version"] = json!(self.anthropic_version);
        value["anthropic_api"] = json!(self.anthropic_api);
        if let Some(protocol) = &self.protocol_override {
            value["protocol_override"] = protocol.to_value()?;
        }
        Ok(value)
    }
    async fn list_models(&self) -> Result<Vec<String>, Error> {
        let mut request = http::request(
            &API,
            &self.config,
            "GET",
            &["v1", "models"],
            None,
            Default::default(),
            Default::default(),
        )?;
        protocol::authorize(self.profile(), &self.config, &mut request)?;
        let data = http::json(&API, self.transport.as_ref(), request, &self.name, None).await?;
        let models = data.get("data").and_then(Value::as_array).ok_or_else(|| {
            http::response_error(&self.name, "Anthropic model list is missing data")
        })?;
        let mut ids = models
            .iter()
            .map(|m| {
                m.get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        http::response_error(&self.name, "Anthropic model list entry is missing id")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        ids.sort();
        Ok(ids)
    }
    async fn probe(&self, model: &Model) -> Result<(), Error> {
        if !self.is_configured() {
            return Err(ProviderError::new(
                ProviderErrorKind::NotConfigured,
                format!("provider {:?} is not configured", self.name),
            )
            .with_provider(&self.name)
            .into());
        }
        let mut request = http::request(
            &API,
            &self.config,
            "GET",
            &["v1", "models", &model.id],
            None,
            Default::default(),
            Default::default(),
        )?;
        protocol::authorize(self.profile(), &self.config, &mut request)?;
        http::json(
            &API,
            self.transport.as_ref(),
            request,
            &self.name,
            Some(&model.id),
        )
        .await?;
        Ok(())
    }
}
/// Register aliases and snapshot factories using this transport for restored models.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    // Commit registrations together so an existing handle cannot leave half a setup.
    let mut next = registry.clone();
    let provider = Arc::new(AnthropicCompatibleProvider::new(config, transport.clone()));
    for handle in AnthropicCompatibleProvider::HANDLES {
        next.register(*handle, provider.clone())?;
    }
    next.register_protocol_factory("anthropic_messages", |_| {
        Ok(Arc::new(AnthropicMessagesProtocol))
    })?;
    next.register_provider_factory("anthropic-compatible", move |value, registry| {
        let config = serde_json::from_value(value.clone())?;
        let name = value
            .get("name")
            .and_then(Value::as_str)
            .filter(|n| !n.is_empty())
            .ok_or_else(|| {
                Error::Configuration("Anthropic provider snapshot must include name".into())
            })?;
        let version = value
            .get("anthropic_version")
            .and_then(Value::as_str)
            .unwrap_or("2023-06-01");
        // A snapshot from before the field was kept is Anthropic's own API
        // only when it has Anthropic's name.
        let anthropic_api = value
            .get("anthropic_api")
            .and_then(Value::as_bool)
            .unwrap_or(name == "anthropic");
        let mut provider =
            AnthropicCompatibleProvider::build(config, transport.clone(), anthropic_api)
                .with_name(name)
                .with_anthropic_version(version);
        if let Some(protocol) = value.get("protocol_override").filter(|v| !v.is_null()) {
            provider = provider.with_protocol(registry.restore_protocol(protocol.clone())?);
        }
        Ok(Arc::new(provider))
    })?;
    *registry = next;
    Ok(())
}
