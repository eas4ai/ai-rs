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

//! The Gemini API, as the AI SDK for TypeScript 7.0.148 package
//! `@ai-sdk/google` sends and reads it, with every model kind it ships.
//!
//! [`register`] installs the provider under `google` and `@ai-sdk/google`.
//! Its protocol streams and buffers language model calls
//! (`streamGenerateContent` and `generateContent`) and runs the operations
//! of [`crate::ops`] for the model: embeddings, Gemini image generation,
//! speech, transcription and live transcription, speech translation, Veo
//! video jobs, realtime tokens and the Live API codec, and evaluation. The
//! provider uploads files and runs batches. [`interactions::protocol`]
//! gives the Interactions API protocol for a model or an agent.
//!
//! The key is the configuration's API key value, or the `env` value named
//! `GOOGLE_GENERATIVE_AI_API_KEY`, and goes out in the `x-goog-api-key`
//! header on every request and WebSocket; it is never put in a URL. No
//! process environment is read.
//!
//! ```
//! use std::sync::Arc;
//! use suprnova_ai::{Model, ProviderConfig, ProviderRegistry, get_model};
//! use suprnova_ai::providers::google;
//! use suprnova_ai::transport::HttpTransport;
//!
//! fn gemini(transport: Arc<dyn HttpTransport>, key: String) -> Result<Model, suprnova_ai::Error> {
//!     let mut registry = ProviderRegistry::new();
//!     let config = ProviderConfig { api_key_value: Some(key), ..Default::default() };
//!     google::register(&mut registry, config, transport)?;
//!     get_model("google:gemini-2.5-flash", &registry)
//! }
//! ```
//!
//! Provider options are the `google` entry of the call's provider
//! parameters, with the TypeScript option names (`thinkingConfig`,
//! `safetySettings`, `cachedContent` and the rest); an unknown name, or a
//! setting the TypeScript provider drops with a warning, fails with
//! `Error::Validation` before any request. Thought signatures are kept in
//! part metadata under `google` and sent back with the history. Grounding
//! sources are in the final message metadata under `google.sources`.

pub(crate) mod accumulator;
pub(crate) mod api;
pub(crate) mod capabilities;
pub(crate) mod language;
pub(crate) mod messages;
pub(crate) mod options;
pub(crate) mod request;
pub(crate) mod response;
pub(crate) mod tool_results;
pub mod tools;

use std::{collections::BTreeMap, fmt, sync::Arc};

use async_trait::async_trait;
use serde_json::Value;

use self::{
    api::{Endpoint, Flavor},
    language::Call,
};
use crate::{
    Error, Model, Provider, ProviderConfig, ProviderProtocol, ProviderRegistry, StreamOptions,
    models::files::DownloadTransport,
    providers::{EventStream, operations::*},
    transport::{HttpTransport, WebSocketTransport},
    types::Message,
};

/// The key variable the TypeScript provider reads.
const API_KEY_ENV: &str = "GOOGLE_GENERATIVE_AI_API_KEY";

/// The Gemini protocol: generate content and the model operations, on the
/// Gemini API or on Vertex AI.
#[derive(Clone)]
pub struct GeminiProtocol {
    pub(crate) flavor: Flavor,
    pub(crate) downloads: Option<Arc<dyn DownloadTransport>>,
    /// The size limit of tool result files downloaded on Vertex AI.
    pub(crate) tool_result_limit: Option<usize>,
}

impl fmt::Debug for GeminiProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeminiProtocol")
            .field("flavor", &self.flavor)
            .finish_non_exhaustive()
    }
}

impl GeminiProtocol {
    pub(crate) fn class_id(flavor: Flavor) -> &'static str {
        match flavor {
            Flavor::Gemini => "google_generative_ai",
            Flavor::Vertex => "google_vertex_generative_ai",
        }
    }

    pub(crate) fn call(&self, model: &Model, provider: &dyn Provider) -> Result<Call, Error> {
        let mut endpoint = Endpoint::of(self.flavor, provider)?;
        if self.flavor.is_vertex() && model.id.starts_with("endpoints/") {
            // A tuned model's endpoint is under the location, not the publisher.
            if let Some(base) = endpoint.base.strip_suffix("/publishers/google") {
                endpoint.base = base.to_owned();
            }
        }
        Ok(Call {
            endpoint,
            model_id: model.id.clone(),
            downloads: self.downloads.clone(),
            tool_result_limit: self.tool_result_limit,
        })
    }
}

#[async_trait]
impl ProviderProtocol for GeminiProtocol {
    fn protocol_class_id(&self) -> &str {
        Self::class_id(self.flavor)
    }

    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        match self.call(model, provider) {
            Ok(call) => call.stream(messages, options),
            Err(error) => futures::StreamExt::boxed(futures::stream::once(async { Err(error) })),
        }
    }

    async fn generate(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> Result<Message, Error> {
        let call = self.call(model, provider)?;
        call.generate(messages, &options)
            .await
            .map(|(message, _)| message)
    }

    async fn evaluate(
        &self,
        model: &Model,
        state: EvaluationInput,
        questions: BTreeMap<String, Question>,
        params: EvaluationParams,
        provider: &dyn Provider,
    ) -> Result<Item<Metadata>, Error> {
        let call = self.call(model, provider)?;
        crate::ops::experimental::evaluate_with_language_model(
            provider.name(),
            state,
            questions,
            params,
            |messages, options| async move { call.generate(messages, &options).await },
        )
        .await
    }
}

/// The Gemini API provider. Build one with [`provider`].
#[derive(Clone)]
pub struct GoogleProvider {
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
    downloads: Option<Arc<dyn DownloadTransport>>,
    websocket: Option<Arc<dyn WebSocketTransport>>,
}

impl fmt::Debug for GoogleProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GoogleProvider")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl GoogleProvider {
    /// Download the user file URLs that Gemini does not fetch before each
    /// request. Without one, such a URL fails with `Error::Validation`.
    /// Download transports must honor the core DNS-pinning contract.
    #[must_use]
    pub fn with_download_transport(mut self, downloads: Arc<dyn DownloadTransport>) -> Self {
        self.downloads = Some(downloads);
        self
    }

    /// Open the Live API WebSockets of live transcription, speech
    /// translation and realtime sessions with this transport.
    #[must_use]
    pub fn with_websocket_transport(mut self, websocket: Arc<dyn WebSocketTransport>) -> Self {
        self.websocket = Some(websocket);
        self
    }

    pub(crate) fn endpoint(&self) -> Endpoint {
        Endpoint {
            flavor: Flavor::Gemini,
            base: self.config.base_url(),
            config: self.config.clone(),
            provider: "google".into(),
            transport: self.transport.clone(),
        }
    }
}

/// Build the Gemini API provider. An unset base URL becomes
/// `https://generativelanguage.googleapis.com/v1beta` and an unset key
/// variable `GOOGLE_GENERATIVE_AI_API_KEY`.
pub fn provider(mut config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> GoogleProvider {
    if config.default_base_url.is_empty() {
        config.default_base_url = api::DEFAULT_BASE_URL.into();
    }
    if config.api_key_env.is_none() {
        config.api_key_env = Some(API_KEY_ENV.into());
    }
    GoogleProvider {
        config,
        transport,
        downloads: None,
        websocket: None,
    }
}

#[async_trait]
impl Provider for GoogleProvider {
    fn name(&self) -> &str {
        "google"
    }
    fn config(&self) -> &ProviderConfig {
        &self.config
    }
    fn provider_class_id(&self) -> &str {
        "google"
    }
    fn default_protocol(&self) -> Option<Arc<dyn ProviderProtocol>> {
        Some(Arc::new(GeminiProtocol {
            flavor: Flavor::Gemini,
            downloads: self.downloads.clone(),
            tool_result_limit: None,
        }))
    }
    fn transport(&self) -> Option<&dyn HttpTransport> {
        Some(self.transport.as_ref())
    }
    fn shared_transport(&self) -> Option<Arc<dyn HttpTransport>> {
        Some(self.transport.clone())
    }
    fn websocket(&self) -> Option<Arc<dyn WebSocketTransport>> {
        self.websocket.clone()
    }
    fn is_configured(&self) -> bool {
        self.api_key().is_some_and(|key| !key.is_empty()) && self.config.is_configured()
    }
}

/// Register the Gemini API provider under `google` and `@ai-sdk/google`,
/// with snapshot restoration that uses this transport.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    register_provider(registry, provider(config, transport))
}

/// Register a built provider, with its download and WebSocket transports,
/// under `google` and `@ai-sdk/google`. Nothing is registered when a handle
/// or factory is already taken.
pub fn register_provider(
    registry: &mut ProviderRegistry,
    provider: GoogleProvider,
) -> Result<(), Error> {
    let mut next = registry.clone();
    let shared = Arc::new(provider.clone());
    for handle in ["google", "@ai-sdk/google"] {
        next.register(handle, shared.clone())?;
    }
    let template = provider.clone();
    next.register_provider_factory("google", move |value: Value, _| {
        let config: ProviderConfig = serde_json::from_value(value)?;
        let mut restored = self::provider(config, template.transport.clone());
        restored.downloads.clone_from(&template.downloads);
        restored.websocket.clone_from(&template.websocket);
        Ok(Arc::new(restored))
    })?;
    let downloads = provider.downloads.clone();
    next.register_protocol_factory(GeminiProtocol::class_id(Flavor::Gemini), move |_| {
        Ok(Arc::new(GeminiProtocol {
            flavor: Flavor::Gemini,
            downloads: downloads.clone(),
            tool_result_limit: None,
        }))
    })?;
    *registry = next;
    Ok(())
}
