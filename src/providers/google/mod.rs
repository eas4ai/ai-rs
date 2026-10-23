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
pub(crate) mod batch;
pub(crate) mod capabilities;
pub(crate) mod embed;
pub(crate) mod files;
pub(crate) mod image;
pub mod interactions;
pub(crate) mod language;
pub(crate) mod live;
pub(crate) mod messages;
pub(crate) mod options;
pub(crate) mod realtime;
pub(crate) mod request;
pub(crate) mod response;
pub(crate) mod speech;
pub(crate) mod tool_results;
pub mod tools;
pub(crate) mod transcription;
pub(crate) mod translation;
pub(crate) mod video;

pub use realtime::browser_url as realtime_browser_url;

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

    fn embed_limits(&self, _model: &Model) -> CallLimits {
        CallLimits {
            max_per_call: Some(embed::MAX_VALUES),
            parallel: true,
        }
    }

    fn embed_params_for(
        &self,
        _model: &Model,
        params: &EmbedParams,
        values: usize,
        range: std::ops::Range<usize>,
    ) -> Result<EmbedParams, Error> {
        embed::params_for(params, values, range)
    }

    async fn embed(
        &self,
        model: &Model,
        values: Vec<String>,
        params: EmbedParams,
        provider: &dyn Provider,
    ) -> Result<Item<Vec<Vec<f64>>>, Error> {
        let call = self.call(model, provider)?;
        embed::embed(&call.endpoint, &model.id, values, &params).await
    }

    fn image_limits(&self, _model: &Model) -> CallLimits {
        CallLimits {
            max_per_call: Some(1),
            parallel: true,
        }
    }

    async fn generate_image(
        &self,
        model: &Model,
        prompt: ImagePrompt,
        params: ImageParams,
        provider: &dyn Provider,
    ) -> Result<Item<Vec<crate::types::messages::FilePart>>, Error> {
        let call = self.call(model, provider)?;
        image::generate(&call, &prompt, &params).await
    }

    async fn generate_audio(
        &self,
        model: &Model,
        prompt: AudioPrompt,
        params: AudioParams,
        provider: &dyn Provider,
    ) -> Result<Item<Vec<crate::types::messages::FilePart>>, Error> {
        let call = self.call(model, provider)?;
        speech::generate(&call.endpoint, &model.id, &prompt, &params).await
    }

    async fn transcribe(
        &self,
        model: &Model,
        audio: TranscribeInput,
        params: TranscribeParams,
        provider: &dyn Provider,
    ) -> Result<Item<Transcription>, Error> {
        let call = self.call(model, provider)?;
        transcription::transcribe(&call.endpoint, &model.id, &audio, &params).await
    }

    async fn stream_transcription(
        &self,
        model: &Model,
        audio: AudioChunks,
        format: AudioFormat,
        params: StreamTranscribeParams,
        provider: &dyn Provider,
    ) -> Result<TranscriptionStream, Error> {
        let call = self.call(model, provider)?;
        let socket = websocket_of(provider)?;
        transcription::stream(&call.endpoint, &model.id, socket, audio, &format, &params)
    }

    async fn stream_translation(
        &self,
        model: &Model,
        audio: AudioChunks,
        format: AudioFormat,
        params: StreamTranslateParams,
        provider: &dyn Provider,
    ) -> Result<TranslationStream, Error> {
        if self.flavor.is_vertex() {
            return Err(Error::UnsupportedOperation("stream_translation".into()));
        }
        let call = self.call(model, provider)?;
        let socket = websocket_of(provider)?;
        translation::stream(&call.endpoint, &model.id, socket, audio, &format, &params)
    }

    fn video_limits(&self, _model: &Model) -> CallLimits {
        CallLimits {
            max_per_call: Some(video::MAX_VIDEOS),
            parallel: true,
        }
    }

    async fn start_video(
        &self,
        model: &Model,
        prompt: VideoPrompt,
        params: VideoParams,
        provider: &dyn Provider,
    ) -> Result<Item<Value>, Error> {
        let call = self.call(model, provider)?;
        video::start(&call.endpoint, &model.id, &prompt, &params).await
    }

    async fn video_status(
        &self,
        model: &Model,
        operation: &Value,
        provider: &dyn Provider,
    ) -> Result<VideoStatus, Error> {
        let call = self.call(model, provider)?;
        video::status(&call.endpoint, &model.id, operation).await
    }

    async fn download_video(
        &self,
        model: &Model,
        video_part: crate::types::messages::FilePart,
        provider: &dyn Provider,
    ) -> Result<crate::types::messages::FilePart, Error> {
        let call = self.call(model, provider)?;
        video::download(&call.endpoint, video_part).await
    }

    async fn create_realtime_token(
        &self,
        model: &Model,
        params: RealtimeTokenParams,
        provider: &dyn Provider,
    ) -> Result<RealtimeToken, Error> {
        if self.flavor.is_vertex() {
            return Err(Error::UnsupportedOperation("realtime".into()));
        }
        let call = self.call(model, provider)?;
        realtime::create_token(&call.endpoint, &model.id, &params).await
    }

    fn realtime_session(
        &self,
        model: &Model,
        config: &RealtimeSessionConfig,
    ) -> Result<Value, Error> {
        if self.flavor.is_vertex() {
            return Err(Error::UnsupportedOperation("realtime".into()));
        }
        realtime::session_setup(config, &model.id)
    }

    fn realtime_codec(&self, model: &Model) -> Result<Box<dyn RealtimeCodec>, Error> {
        if self.flavor.is_vertex() {
            return Err(Error::UnsupportedOperation("realtime".into()));
        }
        Ok(Box::new(realtime::Codec::new(&model.id)))
    }

    fn realtime_connection(
        &self,
        _model: &Model,
        token: &RealtimeToken,
    ) -> Result<(crate::transport::WebSocketRequest, bool), Error> {
        if self.flavor.is_vertex() {
            return Err(Error::UnsupportedOperation("realtime".into()));
        }
        realtime::connection(token)
    }
}

/// The provider's WebSocket transport, which live sessions need.
pub(crate) fn websocket_of(provider: &dyn Provider) -> Result<Arc<dyn WebSocketTransport>, Error> {
    provider.websocket().ok_or_else(|| {
        Error::Configuration(format!(
            "provider {:?} has no WebSocket transport for live sessions; build it with with_websocket_transport",
            provider.name()
        ))
    })
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

    async fn upload_file(
        &self,
        file: UploadFile,
        params: UploadParams,
    ) -> Result<Item<UploadedFile>, Error> {
        files::upload(&self.endpoint(), file, &params).await
    }

    async fn batch_start(
        &self,
        requests: Vec<BatchRequest>,
        params: BatchParams,
    ) -> Result<BatchStarted, Error> {
        batch::start(&self.endpoint(), self.downloads.clone(), &requests, &params).await
    }

    async fn batch_status(&self, batch_id: &str) -> Result<BatchStatus, Error> {
        batch::status(&self.endpoint(), batch_id).await
    }

    async fn batch_cancel(&self, batch_id: &str) -> Result<(), Error> {
        batch::cancel(&self.endpoint(), batch_id).await
    }

    async fn batch_list(
        &self,
        page_size: Option<u32>,
        cursor: Option<String>,
    ) -> Result<BatchPage, Error> {
        batch::list(&self.endpoint(), page_size, cursor).await
    }

    async fn batch_results(&self, batch_id: &str) -> Result<BatchResults, Error> {
        batch::results(&self.endpoint(), batch_id).await
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
    register_interactions(&mut next)?;
    *registry = next;
    Ok(())
}

/// Register the Interactions protocol's snapshot factory once, for the
/// Gemini API and Vertex AI providers alike.
pub(crate) fn register_interactions(registry: &mut ProviderRegistry) -> Result<(), Error> {
    match registry.register_protocol_factory(interactions::CLASS_ID, |value| {
        Ok(interactions::restore(&value))
    }) {
        Ok(()) => Ok(()),
        // The other Google provider registered it already.
        Err(Error::Configuration(message)) if message.contains(interactions::CLASS_ID) => Ok(()),
        Err(error) => Err(error),
    }
}
