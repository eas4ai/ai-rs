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

//! Hosted providers that speak a protocol the crate implements: Cerebras,
//! DeepInfra, Fireworks, Together AI and Z.ai over OpenAI Chat Completions,
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
//! mode, which does not enforce the schema. Hosts stream and generate only:
//! `list_models` and `probe` return `Error::UnsupportedOperation`.
use crate::{
    Error, Provider, ProviderConfig, ProviderProtocol, ProviderRegistry,
    providers::{
        ModelsDevProvider,
        anthropic::protocol::{HostMessagesProtocol, MessagesProfile},
        openai::{profile::ChatProfile, protocol::HostChatProtocol},
    },
    transport::HttpTransport,
};
use async_trait::async_trait;
use std::{fmt, sync::Arc};

pub mod cerebras;
pub mod deepinfra;
pub mod fireworks;
pub mod minimax;
pub mod togetherai;
pub mod zai;

/// Every host, for lookups by package or name.
static HOSTS: [&Host; 6] = [
    &cerebras::HOST,
    &deepinfra::HOST,
    &fireworks::HOST,
    &togetherai::HOST,
    &zai::HOST,
    &minimax::HOST,
];

/// The Chat Completions profile of the host that models.dev metadata names,
/// by its package handle first and then by its id.
pub(crate) fn chat_profile_for(metadata: &ModelsDevProvider) -> Option<&'static ChatProfile> {
    let host = HOSTS
        .iter()
        .find(|host| metadata.npm.as_deref() == Some(host.handle))
        .or_else(|| HOSTS.iter().find(|host| host.name == metadata.id))?;
    match host.protocol {
        HostProtocol::Chat(profile) => Some(profile),
        HostProtocol::Messages(_) => None,
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
}

#[derive(Debug)]
pub(crate) enum HostProtocol {
    Chat(&'static ChatProfile),
    Messages(&'static MessagesProfile),
}
impl HostProtocol {
    fn protocol(&self) -> Arc<dyn ProviderProtocol> {
        match self {
            Self::Chat(profile) => Arc::new(HostChatProtocol(profile)),
            Self::Messages(profile) => Arc::new(HostMessagesProtocol(profile)),
        }
    }
    fn class_id(&self) -> &'static str {
        match self {
            Self::Chat(profile) => profile.protocol_class_id,
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
        }
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
        Some(self.host.protocol.protocol())
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
) -> Result<(), Error> {
    let mut next = registry.clone();
    let provider = Arc::new(HostProvider::new(host, config, transport.clone()));
    for handle in [host.name, host.handle] {
        next.register(handle, provider.clone())?;
    }
    // A restore takes the endpoint and the credentials from the registered
    // provider; the snapshot keeps only the rest of the configuration.
    next.register_provider_factory(host.name, move |value, _| {
        let config = serde_json::from_value(value)?;
        Ok(Arc::new(HostProvider::new(host, config, transport.clone())))
    })?;
    next.register_protocol_factory(host.protocol.class_id(), move |_| {
        Ok(host.protocol.protocol())
    })?;
    *registry = next;
    Ok(())
}
