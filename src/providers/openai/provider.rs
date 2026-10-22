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

use super::{
    OpenAIChatCompletionsProtocol, OpenAIResponsesProtocol, errors, protocol,
    protocol::{HostChatProtocol, HostResponsesProtocol},
};
use crate::{
    Error, Model, Provider, ProviderConfig, ProviderProtocol, ProviderRegistry,
    providers::{ModelsDevProvider, base::provider_config, hosts, http},
    transport::HttpTransport,
};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone)]
pub struct OpenAICompatibleProvider {
    name: String,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
    /// Chosen when the provider is built, from the API it targets.
    default_protocol: Arc<dyn ProviderProtocol>,
    protocol_override: Option<Arc<dyn ProviderProtocol>>,
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
        }
    }
    /// Build a provider from models.dev metadata. It speaks Responses when
    /// the metadata names OpenAI's own adapter package, `@ai-sdk/openai`, and
    /// Chat Completions otherwise. Metadata that names a Chat Completions
    /// host of `providers::hosts`, by package or id, gets that host's
    /// request and response rules.
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
        let default_protocol: Arc<dyn ProviderProtocol> =
            if metadata.npm.as_deref() == Some("@ai-sdk/openai") {
                Arc::new(OpenAIResponsesProtocol)
            } else if let Some(profile) = hosts::chat_profile_for(metadata) {
                Arc::new(HostChatProtocol(profile, None))
            } else if let Some(profile) = hosts::responses_profile_for(metadata) {
                Arc::new(HostResponsesProtocol(profile))
            } else {
                Arc::new(OpenAIChatCompletionsProtocol)
            };
        Ok(Self {
            name: metadata.id.clone(),
            config,
            transport,
            default_protocol,
            protocol_override: None,
        })
    }
    #[must_use]
    pub fn with_protocol(mut self, protocol: Arc<dyn ProviderProtocol>) -> Self {
        self.protocol_override = Some(protocol);
        self
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
        // The registered provider of this name decides the API, as it
        // decides the endpoint.
        let default_protocol = registry
            .get_provider(&name)?
            .default_protocol()
            .ok_or_else(|| {
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
        }))
    })
}
