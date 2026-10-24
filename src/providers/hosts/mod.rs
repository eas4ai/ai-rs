//! Hosted providers that speak a protocol the crate implements: Cerebras,
//! DeepInfra, DeepSeek, Fireworks, Mistral, Moonshot AI, Together AI and Z.ai
//! over OpenAI Chat Completions, xAI and Hugging Face over OpenAI Responses,
//! and MiniMax over the Anthropic Messages protocol.
//!
//! Each host is the protocol's adapter with a host profile: its endpoint, its
//! key and the request and response differences the AI SDK for TypeScript
//! 7.0.148 records for it. Each host module has `provider`, which builds the
//! provider, and `register`, which registers it under the host's name and
//! package handle with snapshot restoration. No process environment is read:
//! the key comes from `ProviderConfig::api_key_value`, or from the `env` value
//! named by the host's key variable.
//!
//! ```
//! use std::sync::Arc;
//! use suprnova_ai::{ProviderConfig, ProviderRegistry};
//! use suprnova_ai::providers::hosts::cerebras;
//! use suprnova_ai::transport::HttpTransport;
//!
//! fn registry(transport: Arc<dyn HttpTransport>, key: String) -> Result<ProviderRegistry, suprnova_ai::Error> {
//!     let mut registry = ProviderRegistry::new();
//!     let config = ProviderConfig { api_key_value: Some(key), ..Default::default() };
//!     cerebras::register(&mut registry, config, transport)?;
//!     Ok(registry)
//! }
//! ```
//!
//! Options a host profile marks unsupported, such as Z.ai's penalties and
//! seed, fail with `Error::Validation` before any request, as the OpenAI and
//! Anthropic adapters reject theirs; the TypeScript SDK drops them with a
//! warning. An output schema goes to a host without JSON schema support as JSON
//! mode, which does not enforce the schema.
//!
//! Together AI, DeepInfra and Fireworks serve every model kind of their
//! TypeScript packages. A chat call reads the host's own options entry as
//! the compatible chat model does (HST-026); each module's
//! `completion_protocol` gives its completion models; and `ops::embed`,
//! `ops::generate_image` and, for Together AI, `ops::rerank` work with the
//! provider's models. `list_models` and `probe` return
//! `Error::UnsupportedOperation` for every host.
use crate::{
    Error, Provider, ProviderConfig, ProviderProtocol, ProviderRegistry,
    models::files::DownloadTransport,
    providers::{
        ModelsDevProvider,
        anthropic::protocol::{HostMessagesProtocol, MessagesProfile},
        http::ApiProfile,
        openai::{
            profile::{ChatProfile, ResponsesProfile},
            protocol::{HostChatProtocol, HostResponsesProtocol},
        },
    },
    transport::HttpTransport,
};
use async_trait::async_trait;
use std::{fmt, sync::Arc};

pub mod cerebras;
pub mod deepinfra;
pub mod deepseek;
pub mod fireworks;
pub mod huggingface;
mod images;
pub mod minimax;
pub mod mistral;
mod models;
pub mod moonshotai;
mod rerank;
pub mod togetherai;
pub mod xai;
pub mod zai;

/// Every host, for lookups by package or name.
static HOSTS: [&Host; 11] = [
    &cerebras::HOST,
    &deepinfra::HOST,
    &fireworks::HOST,
    &togetherai::HOST,
    &zai::HOST,
    &minimax::HOST,
    &deepseek::HOST,
    &moonshotai::HOST,
    &mistral::HOST,
    &xai::HOST,
    &huggingface::HOST,
];

/// The host that models.dev metadata names, by its package handle first and
/// then by its id.
fn host_for(metadata: &ModelsDevProvider) -> Option<&'static Host> {
    HOSTS
        .iter()
        .find(|host| metadata.npm.as_deref() == Some(host.handle))
        .or_else(|| HOSTS.iter().find(|host| host.name == metadata.id))
        .copied()
}
/// The error rules of the Chat Completions host a provider is, for the
/// completion and embedding calls it makes on the compatible protocols
/// (HST-027, HST-028).
pub(crate) fn chat_api(provider: &dyn Provider) -> Option<&'static ApiProfile> {
    HOSTS
        .iter()
        .find(|host| host.name == provider.provider_class_id())
        .and_then(|host| match host.protocol {
            HostProtocol::Chat(profile) => Some(&profile.api),
            HostProtocol::Responses(_) | HostProtocol::Messages(_) => None,
        })
}
/// The Chat Completions profile of the host that models.dev metadata names.
pub(crate) fn chat_profile_for(metadata: &ModelsDevProvider) -> Option<&'static ChatProfile> {
    match host_for(metadata)?.protocol {
        HostProtocol::Chat(profile) => Some(profile),
        HostProtocol::Responses(_) | HostProtocol::Messages(_) => None,
    }
}
/// The Responses profile of the host that models.dev metadata names.
pub(crate) fn responses_profile_for(
    metadata: &ModelsDevProvider,
) -> Option<&'static ResponsesProfile> {
    match host_for(metadata)?.protocol {
        HostProtocol::Responses(profile) => Some(profile),
        HostProtocol::Chat(_) | HostProtocol::Messages(_) => None,
    }
}

/// A host: its names, endpoint, key variable and protocol profile.
#[derive(Debug)]
pub(crate) struct Host {
    pub name: &'static str,
    /// The package handle, such as `@ai-sdk/cerebras`.
    pub handle: &'static str,
    pub default_base_url: &'static str,
    pub api_key_env: &'static str,
    pub protocol: HostProtocol,
    /// The model kinds beside the language model.
    pub models: Models,
}

/// The model kinds a host's TypeScript package ships beside its language
/// model, which the host provider's default protocol serves.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Models {
    /// Embeddings on OAI-016's compatible protocol (HST-028).
    pub embeddings: bool,
    /// The host's image model (HST-029 to HST-031).
    pub images: Option<images::Images>,
    /// Together AI's reranking (HST-032).
    pub reranking: bool,
}
impl Models {
    /// The language model alone.
    pub(crate) const CHAT: Self = Self {
        embeddings: false,
        images: None,
        reranking: false,
    };
    /// DeepInfra's: embeddings and images.
    pub(crate) const DEEPINFRA: Self = Self {
        embeddings: true,
        images: Some(images::Images::DeepInfra),
        reranking: false,
    };
    /// Fireworks': embeddings and images.
    pub(crate) const FIREWORKS: Self = Self {
        embeddings: true,
        images: Some(images::Images::Fireworks),
        reranking: false,
    };
    /// Together AI's: embeddings, images and reranking.
    pub(crate) const TOGETHER_AI: Self = Self {
        embeddings: true,
        images: Some(images::Images::TogetherAI),
        reranking: true,
    };
    /// Whether the host serves a model kind beside its language model.
    fn beside_chat(self) -> bool {
        self.embeddings || self.images.is_some() || self.reranking
    }
}
impl Host {
    /// The protocol of the host's models: its language model's, with the
    /// other model kinds it serves.
    fn protocol(
        &'static self,
        downloads: Option<Arc<dyn DownloadTransport>>,
    ) -> Arc<dyn ProviderProtocol> {
        match self.protocol {
            HostProtocol::Chat(profile) if self.models.beside_chat() => Arc::new(
                models::HostModels::new(HostChatProtocol(profile, downloads), self.models),
            ),
            _ => self.protocol.protocol(downloads),
        }
    }
}

#[derive(Debug)]
pub(crate) enum HostProtocol {
    Chat(&'static ChatProfile),
    Responses(&'static ResponsesProfile),
    Messages(&'static MessagesProfile),
}
impl HostProtocol {
    fn protocol(&self, downloads: Option<Arc<dyn DownloadTransport>>) -> Arc<dyn ProviderProtocol> {
        match self {
            Self::Chat(profile) => Arc::new(HostChatProtocol(profile, downloads)),
            Self::Responses(profile) => Arc::new(HostResponsesProtocol(profile)),
            Self::Messages(profile) => Arc::new(HostMessagesProtocol(profile)),
        }
    }
    fn class_id(&self) -> &'static str {
        match self {
            Self::Chat(profile) => profile.protocol_class_id,
            Self::Responses(profile) => profile.protocol_class_id,
            Self::Messages(profile) => profile.protocol_class_id,
        }
    }
}

/// A hosted provider. Build one with a host module's `provider` function.
/// Its name and its snapshot class id are the host's name.
#[derive(Clone)]
pub struct HostProvider {
    host: &'static Host,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
    downloads: Option<Arc<dyn DownloadTransport>>,
}
impl fmt::Debug for HostProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostProvider")
            .field("name", &self.host.name)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}
impl HostProvider {
    /// Fill in the host's default base URL and key variable where the
    /// configuration leaves them unset.
    fn new(
        host: &'static Host,
        mut config: ProviderConfig,
        transport: Arc<dyn HttpTransport>,
    ) -> Self {
        if config.default_base_url.is_empty() {
            config.default_base_url = host.default_base_url.into();
        }
        if config.api_key_env.is_none() {
            config.api_key_env = Some(host.api_key_env.into());
        }
        Self {
            host,
            config,
            transport,
            downloads: None,
        }
    }
    /// Download the media URLs that a Chat Completions host cannot fetch
    /// before each request: images, videos and text for Moonshot AI, audio
    /// and PDFs for the hosts that take them as OpenAI does. Without one, such
    /// a URL fails with `Error::Validation`. Responses and Messages hosts send
    /// no downloaded media and ignore it. Download transports must honor the
    /// core DNS-pinning contract. A snapshot does not keep the transport; a
    /// restore through a registry made with `register_with_downloads` gets it
    /// back.
    #[must_use]
    pub fn with_download_transport(mut self, downloads: Arc<dyn DownloadTransport>) -> Self {
        self.downloads = Some(downloads);
        self
    }
}
#[async_trait]
impl Provider for HostProvider {
    fn name(&self) -> &str {
        self.host.name
    }
    fn config(&self) -> &ProviderConfig {
        &self.config
    }
    fn provider_class_id(&self) -> &str {
        self.host.name
    }
    fn default_protocol(&self) -> Option<Arc<dyn ProviderProtocol>> {
        Some(self.host.protocol(self.downloads.clone()))
    }
    fn transport(&self) -> Option<&dyn HttpTransport> {
        Some(self.transport.as_ref())
    }
    fn shared_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        Some(self.transport.clone())
    }
    fn is_configured(&self) -> bool {
        self.api_key().is_some_and(|key| !key.is_empty()) && self.config.is_configured()
    }
}

/// Register the host's provider under its name and handle, with its provider
/// and protocol snapshot factories. Nothing is registered when a handle or
/// factory is already taken.
fn register(
    host: &'static Host,
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
    downloads: Option<Arc<dyn DownloadTransport>>,
) -> Result<(), Error> {
    let mut next = registry.clone();
    let build = move |config| {
        let provider = HostProvider::new(host, config, transport.clone());
        match &downloads {
            Some(downloads) => provider.with_download_transport(downloads.clone()),
            None => provider,
        }
    };
    let provider = Arc::new(build(config));
    for handle in [host.name, host.handle] {
        next.register(handle, provider.clone())?;
    }
    // A restore takes the endpoint and the credentials from the registered
    // provider; the snapshot keeps only the rest of the configuration.
    next.register_provider_factory(host.name, move |value, _| {
        let config = serde_json::from_value(value)?;
        Ok(Arc::new(build(config)))
    })?;
    next.register_protocol_factory(host.protocol.class_id(), move |_| Ok(host.protocol(None)))?;
    *registry = next;
    Ok(())
}
