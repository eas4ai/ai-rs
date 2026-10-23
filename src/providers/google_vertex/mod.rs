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

//! Google Vertex AI, as the AI SDK for TypeScript 7.0.148 package
//! `@ai-sdk/google-vertex` sends and reads it.
//!
//! [`register`] installs the provider under `google-vertex` and
//! `@ai-sdk/google-vertex`. It runs the Gemini protocol of
//! [`google`] on Vertex AI: Gemini language models,
//! embeddings, Gemini image generation, Veo video, Gemini and Chirp
//! text-to-speech, Cloud Speech-to-Text and Gemini transcription, and the
//! Interactions API. [`anthropic`], [`maas`] and [`xai`] install the partner
//! models of Anthropic, the open models and xAI.
//!
//! No process environment and no credential file is read. The
//! configuration's `env` map holds the values the TypeScript package reads
//! from the environment: `GOOGLE_VERTEX_PROJECT`, `GOOGLE_VERTEX_LOCATION`,
//! an Express Mode key in `GOOGLE_VERTEX_API_KEY` (or the API key value), and,
//! with the `google-auth` feature, a service account in
//! `GOOGLE_CLIENT_EMAIL`, `GOOGLE_PRIVATE_KEY` and `GOOGLE_PRIVATE_KEY_ID`.
//! An Express Mode key goes out as `x-goog-api-key`. Otherwise each request
//! carries the Vertex auth headers of the provider's credentials: an
//! [`AccessTokenSource`], google-cloud-auth `Credentials` (feature
//! `google-auth`), or the `env` service account; configured headers keep
//! their values. A service account needs a process rustls crypto provider,
//! because google-cloud-auth signs with the process default.
//!
//! ```
//! use std::sync::Arc;
//! use suprnova_ai::{Error, Model, ProviderConfig, ProviderRegistry, get_model};
//! use suprnova_ai::providers::google_vertex::{self, AccessTokenSource};
//! use suprnova_ai::transport::HttpTransport;
//!
//! fn gemini(
//!     transport: Arc<dyn HttpTransport>,
//!     tokens: Arc<dyn AccessTokenSource>,
//! ) -> Result<Model, Error> {
//!     let config = ProviderConfig {
//!         env: [
//!             ("GOOGLE_VERTEX_PROJECT".to_owned(), "my-project".to_owned()),
//!             ("GOOGLE_VERTEX_LOCATION".to_owned(), "us-central1".to_owned()),
//!         ]
//!         .into(),
//!         ..Default::default()
//!     };
//!     let provider = google_vertex::provider(config, transport).with_access_token_source(tokens);
//!     let mut registry = ProviderRegistry::new();
//!     google_vertex::register_provider(&mut registry, provider)?;
//!     get_model("google-vertex:gemini-2.5-flash", &registry)
//! }
//! ```

pub mod anthropic;
mod auth;
mod chat;
pub mod maas;
pub mod xai;

pub use auth::AccessTokenSource;
pub use chat::VertexChatProvider;

use std::{fmt, sync::Arc};

use async_trait::async_trait;
use serde_json::Value;

use self::auth::{Auth, AuthTransport, AuthWebSocket, Credentials};
use crate::{
    Error, Provider, ProviderConfig, ProviderProtocol, ProviderRegistry,
    models::files::DownloadTransport,
    providers::google::{self, GeminiProtocol, api::Flavor},
    transport::{HttpTransport, WebSocketTransport},
};

/// The env value of the Express Mode API key.
pub(crate) const API_KEY_ENV: &str = "GOOGLE_VERTEX_API_KEY";
/// The env value of the Google Cloud project.
pub(crate) const PROJECT_ENV: &str = "GOOGLE_VERTEX_PROJECT";
/// The env value of the Vertex AI location.
pub(crate) const LOCATION_ENV: &str = "GOOGLE_VERTEX_LOCATION";
/// The base URL of Express Mode.
const EXPRESS_BASE_URL: &str = "https://aiplatform.googleapis.com/v1/publishers/google";
/// The default size limit of a tool result file a Vertex call downloads.
pub(crate) const TOOL_RESULT_LIMIT: usize = 7 * 1024 * 1024;

/// Whether a value is one DNS label: letters, digits and inner hyphens, at
/// most 63 characters (`isValidHostnamePart`).
pub(crate) fn is_dns_label(value: &str) -> bool {
    let bytes = value.as_bytes();
    !bytes.is_empty()
        && bytes.len() <= 63
        && bytes
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || *b == b'-')
        && bytes[0] != b'-'
        && bytes[bytes.len() - 1] != b'-'
}

/// The Vertex AI host of a location.
pub(crate) fn host(location: &str) -> String {
    match location {
        "global" => "aiplatform.googleapis.com".into(),
        "eu" | "us" => format!("aiplatform.{location}.rep.googleapis.com"),
        other => format!("{other}-aiplatform.googleapis.com"),
    }
}

fn env<'a>(config: &'a ProviderConfig, name: &str) -> Option<&'a str> {
    config
        .env
        .get(name)
        .map(String::as_str)
        .filter(|value| !value.is_empty())
}

/// The configuration's Google Cloud project.
pub(crate) fn project(config: &ProviderConfig) -> Result<&str, Error> {
    env(config, PROJECT_ENV).ok_or_else(|| {
        Error::Configuration(format!(
            "Google Vertex AI needs a project: set {PROJECT_ENV} in the configuration's env"
        ))
    })
}

/// The configuration's location, or `default`, checked to be one DNS label.
pub(crate) fn location<'a>(
    config: &'a ProviderConfig,
    default: Option<&'a str>,
) -> Result<&'a str, Error> {
    let location = env(config, LOCATION_ENV).or(default).ok_or_else(|| {
        Error::Configuration(format!(
            "Google Vertex AI needs a location: set {LOCATION_ENV} in the configuration's env"
        ))
    })?;
    if !is_dns_label(location) {
        return Err(Error::Configuration(
            "Invalid Google Vertex location. Expected a single DNS label (letters, digits, and hyphens). Use a base URL for custom endpoints.".into(),
        ));
    }
    Ok(location)
}

/// The base URL of the Google publisher's models (VTX-001): the configured
/// base URL; in Express Mode the global publisher URL; otherwise the
/// project's location.
pub(crate) fn base_url(config: &ProviderConfig) -> Result<String, Error> {
    let configured = config.base_url();
    if !configured.is_empty() {
        return Ok(configured);
    }
    if auth::express(config) {
        return Ok(EXPRESS_BASE_URL.into());
    }
    let project = project(config)?;
    let location = location(config, None)?;
    Ok(format!(
        "https://{}/v1beta1/projects/{project}/locations/{location}/publishers/google",
        host(location)
    ))
}

/// Whether the configuration is in Express Mode.
pub(crate) fn express(config: &ProviderConfig) -> bool {
    auth::express(config)
}

/// The error of an operation that Express Mode keys cannot run.
pub(crate) fn express_error(what: &str) -> Error {
    Error::Configuration(format!(
        "Google Vertex {what} do not support Express Mode API keys. Use standard Google Cloud credentials instead."
    ))
}

/// The error of a call whose provider cannot sign in.
pub(crate) fn unavailable(config: &ProviderConfig) -> Error {
    auth::unavailable(config)
}

/// The Vertex AI provider. Build one with [`provider`].
#[derive(Clone)]
pub struct VertexProvider {
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
    auth: Auth,
    /// The transport that adds the Vertex auth headers to `transport`.
    signed: Arc<AuthTransport>,
    downloads: Option<Arc<dyn DownloadTransport>>,
    websocket: Option<Arc<dyn WebSocketTransport>>,
    tool_result_limit: usize,
}

impl fmt::Debug for VertexProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VertexProvider")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl VertexProvider {
    /// Sign in with OAuth access tokens from this source.
    #[must_use]
    pub fn with_access_token_source(mut self, source: Arc<dyn AccessTokenSource>) -> Self {
        self.auth = Auth::with(Credentials::Token(source));
        self.sign()
    }

    /// Sign in with google-cloud-auth credentials, such as application
    /// default credentials; every header they give goes out, the quota
    /// project's included. Service-account credentials sign with the
    /// process rustls crypto provider: install one first, such as
    /// `rustls::crypto::ring::default_provider().install_default()`, because
    /// without one google-cloud-auth panics when it signs, unless another
    /// crate in the build turns on its `default-rustls-provider` feature.
    #[cfg(feature = "google-auth")]
    #[must_use]
    pub fn with_credentials(
        mut self,
        credentials: google_cloud_auth::credentials::Credentials,
    ) -> Self {
        self.auth = Auth::with(Credentials::Google(credentials));
        self.sign()
    }

    /// Download the tool result file URLs that Vertex AI cannot reference
    /// before each request. Without one, such a URL fails with
    /// `Error::Validation`. Download transports must honor the core
    /// DNS-pinning contract.
    #[must_use]
    pub fn with_download_transport(mut self, downloads: Arc<dyn DownloadTransport>) -> Self {
        self.downloads = Some(downloads);
        self
    }

    /// Open the Live API WebSockets of live transcription with this
    /// transport; the Vertex auth headers are added to each connection.
    #[must_use]
    pub fn with_websocket_transport(mut self, websocket: Arc<dyn WebSocketTransport>) -> Self {
        self.websocket = Some(websocket);
        self
    }

    /// The size limit of a downloaded tool result file; 7 MiB by default.
    #[must_use]
    pub fn with_tool_result_download_limit(mut self, bytes: usize) -> Self {
        self.tool_result_limit = bytes;
        self
    }

    fn protocol(&self) -> GeminiProtocol {
        GeminiProtocol {
            flavor: Flavor::Vertex,
            downloads: self.downloads.clone(),
            tool_result_limit: Some(self.tool_result_limit),
        }
    }

    /// Rebuild the signing transport after the configuration or the
    /// credentials changed.
    fn sign(mut self) -> Self {
        self.signed = Arc::new(AuthTransport {
            inner: self.transport.clone(),
            auth: self.auth.clone(),
            config: self.config.clone(),
            auth_wins: false,
            base_url,
        });
        self
    }

    /// Whether a call can sign in. It is asked on each call, because an
    /// application can install a crypto provider after building this one.
    fn ready(&self) -> bool {
        express(&self.config) || self.auth.available(&self.config)
    }
}

/// Build the Vertex AI provider. An unset key variable becomes
/// `GOOGLE_VERTEX_API_KEY`; the base URL is the configured one or follows
/// from the `env` project and location.
pub fn provider(mut config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> VertexProvider {
    if config.api_key_env.is_none() {
        config.api_key_env = Some(API_KEY_ENV.into());
    }
    let auth = Auth::default();
    VertexProvider {
        signed: Arc::new(AuthTransport {
            inner: transport.clone(),
            auth: auth.clone(),
            config: config.clone(),
            auth_wins: false,
            base_url,
        }),
        config,
        transport,
        auth,
        downloads: None,
        websocket: None,
        tool_result_limit: TOOL_RESULT_LIMIT,
    }
}

#[async_trait]
impl Provider for VertexProvider {
    fn name(&self) -> &str {
        "google-vertex"
    }
    fn config(&self) -> &ProviderConfig {
        &self.config
    }
    fn provider_class_id(&self) -> &str {
        "google-vertex"
    }
    fn base_url(&self) -> String {
        base_url(&self.config).unwrap_or_default()
    }
    fn default_protocol(&self) -> Option<Arc<dyn ProviderProtocol>> {
        Some(Arc::new(self.protocol()))
    }
    /// The signing transport, or `None` when the provider cannot sign in,
    /// so that every call fails before a request.
    fn transport(&self) -> Option<&dyn HttpTransport> {
        self.ready()
            .then(|| self.signed.as_ref() as &dyn HttpTransport)
    }
    fn shared_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        self.ready()
            .then(|| self.signed.clone() as Arc<dyn HttpTransport>)
    }
    fn websocket(&self) -> Option<Arc<dyn WebSocketTransport>> {
        let inner = self.websocket.clone()?;
        Some(Arc::new(AuthWebSocket {
            inner,
            auth: self.auth.clone(),
            config: self.config.clone(),
            base_url,
        }))
    }
    fn is_configured(&self) -> bool {
        base_url(&self.config).is_ok() && self.ready()
    }
}

/// Register the Vertex AI provider under `google-vertex` and
/// `@ai-sdk/google-vertex`, with snapshot restoration that uses this
/// transport. The provider signs in with an Express Mode key or the `env`
/// service account; use [`register_provider`] for other credentials.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    register_provider(registry, provider(config, transport))
}

/// Register a built provider, with its credentials and its download and
/// WebSocket transports, under `google-vertex` and `@ai-sdk/google-vertex`.
/// A restored provider gets the registered provider's credentials and
/// transports; a snapshot holds none of them. Nothing is registered when a
/// handle or factory is already taken.
pub fn register_provider(
    registry: &mut ProviderRegistry,
    provider: VertexProvider,
) -> Result<(), Error> {
    let mut next = registry.clone();
    let shared = Arc::new(provider.clone());
    for handle in ["google-vertex", "@ai-sdk/google-vertex"] {
        next.register(handle, shared.clone())?;
    }
    let template = provider.clone();
    next.register_provider_factory("google-vertex", move |value: Value, _| {
        let config: ProviderConfig = serde_json::from_value(value)?;
        Ok(Arc::new(
            VertexProvider {
                config,
                ..template.clone()
            }
            .sign(),
        ))
    })?;
    let protocol = provider.protocol();
    next.register_protocol_factory(GeminiProtocol::class_id(Flavor::Vertex), move |_| {
        Ok(Arc::new(protocol.clone()))
    })?;
    google::register_interactions(&mut next)?;
    *registry = next;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locations_are_dns_labels() {
        for good in ["us-central1", "global", "eu", "a", &"x".repeat(63)] {
            assert!(is_dns_label(good), "{good}");
        }
        for bad in [
            "",
            "-us",
            "us-",
            "us-central1/x",
            "us.central",
            &"x".repeat(64),
            "ü",
        ] {
            assert!(!is_dns_label(bad), "{bad}");
        }
    }

    #[test]
    fn hosts_follow_the_location() {
        assert_eq!(host("global"), "aiplatform.googleapis.com");
        assert_eq!(host("eu"), "aiplatform.eu.rep.googleapis.com");
        assert_eq!(host("us"), "aiplatform.us.rep.googleapis.com");
        assert_eq!(
            host("europe-west4"),
            "europe-west4-aiplatform.googleapis.com"
        );
    }
}
