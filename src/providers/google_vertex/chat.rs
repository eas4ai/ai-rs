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

//! The partner models on Vertex AI's OpenAI-compatible endpoint, the open
//! models (MaaS) and xAI: Chat Completions at
//! `<base>/chat/completions`, with the Vertex auth headers after the
//! configured headers.

use std::{fmt, sync::Arc};

use async_trait::async_trait;
use futures::{StreamExt, stream};
use serde_json::Value;

use super::auth::{self, Auth, AuthTransport, Credentials};
use crate::{
    Error, Model, Provider, ProviderConfig, ProviderProtocol, ProviderRegistry, StreamOptions,
    models::files::DownloadTransport,
    providers::{
        EventStream,
        openai::{profile::ChatProfile, protocol::HostChatProtocol},
    },
    transport::HttpTransport,
    types::Message,
};

/// A Chat Completions host on Vertex AI.
pub(crate) struct ChatHost {
    /// The provider's name and snapshot class id.
    pub name: &'static str,
    /// The package handle, such as `@ai-sdk/google-vertex/maas`.
    pub handle: &'static str,
    pub profile: &'static ChatProfile,
    /// The base URL of the configuration, the location checked.
    pub base_url: fn(&ProviderConfig) -> Result<String, Error>,
}

/// A partner model provider on Vertex AI's OpenAI-compatible endpoint.
/// Build one with [`maas::provider`](super::maas::provider) or
/// [`xai::provider`](super::xai::provider). It signs in only with
/// credentials: an API key in the configuration is not sent.
#[derive(Clone)]
pub struct VertexChatProvider {
    host: &'static ChatHost,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
    auth: Auth,
    signed: Arc<AuthTransport>,
    downloads: Option<Arc<dyn DownloadTransport>>,
}

impl fmt::Debug for VertexChatProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VertexChatProvider")
            .field("name", &self.host.name)
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
        auth_wins: true,
    })
}

impl VertexChatProvider {
    pub(crate) fn new(
        host: &'static ChatHost,
        mut config: ProviderConfig,
        transport: Arc<dyn HttpTransport>,
    ) -> Self {
        // These hosts take no Express Mode key.
        config.api_key_value = None;
        config.api_key_env = None;
        let auth = Auth::default();
        Self {
            host,
            signed: signer(&transport, &auth, &config),
            config,
            transport,
            auth,
            downloads: None,
        }
    }

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

    /// Download the file URLs the host does not fetch before each request.
    /// Without one, such a URL fails with `Error::Validation`. Download
    /// transports must honor the core DNS-pinning contract.
    #[must_use]
    pub fn with_download_transport(mut self, downloads: Arc<dyn DownloadTransport>) -> Self {
        self.downloads = Some(downloads);
        self
    }

    fn ready(&self) -> bool {
        self.auth.available(&self.config)
    }
}

/// Chat Completions on Vertex AI: the host's protocol, after the location
/// and the credentials are checked.
#[derive(Clone)]
pub(crate) struct VertexChatProtocol {
    host: &'static ChatHost,
    downloads: Option<Arc<dyn DownloadTransport>>,
}

#[async_trait]
impl ProviderProtocol for VertexChatProtocol {
    fn protocol_class_id(&self) -> &str {
        self.host.profile.protocol_class_id
    }

    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        let config = provider.config();
        let checked = (self.host.base_url)(config).and_then(|_| {
            provider
                .shared_transport()
                .map(|_| ())
                .ok_or_else(|| auth::unavailable(config))
        });
        if let Err(error) = checked {
            return stream::once(async { Err(error) }).boxed();
        }
        HostChatProtocol(self.host.profile, self.downloads.clone())
            .stream(model, messages, options, provider)
    }
}

#[async_trait]
impl Provider for VertexChatProvider {
    fn name(&self) -> &str {
        self.host.name
    }
    fn config(&self) -> &ProviderConfig {
        &self.config
    }
    fn provider_class_id(&self) -> &str {
        self.host.name
    }
    fn base_url(&self) -> String {
        (self.host.base_url)(&self.config).unwrap_or_default()
    }
    fn default_protocol(&self) -> Option<Arc<dyn ProviderProtocol>> {
        Some(Arc::new(VertexChatProtocol {
            host: self.host,
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
        (self.host.base_url)(&self.config).is_ok() && self.ready()
    }
}

/// Register a built provider under its name and handle, with snapshot
/// restoration that keeps its credentials and transports.
pub(crate) fn register(
    registry: &mut ProviderRegistry,
    provider: VertexChatProvider,
) -> Result<(), Error> {
    let host = provider.host;
    let mut next = registry.clone();
    let shared = Arc::new(provider.clone());
    for handle in [host.name, host.handle] {
        next.register(handle, shared.clone())?;
    }
    let template = provider.clone();
    next.register_provider_factory(host.name, move |value: Value, _| {
        let config: ProviderConfig = serde_json::from_value(value)?;
        let mut restored =
            VertexChatProvider::new(template.host, config, template.transport.clone());
        restored.auth = template.auth.clone();
        restored.signed = signer(&restored.transport, &restored.auth, &restored.config);
        restored.downloads.clone_from(&template.downloads);
        Ok(Arc::new(restored))
    })?;
    let downloads = provider.downloads.clone();
    next.register_protocol_factory(host.profile.protocol_class_id, move |_| {
        Ok(Arc::new(VertexChatProtocol {
            host,
            downloads: downloads.clone(),
        }))
    })?;
    *registry = next;
    Ok(())
}
