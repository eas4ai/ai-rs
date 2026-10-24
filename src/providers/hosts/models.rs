//! The model kinds a host serves beside chat, on the protocol its provider
//! gives every model: Together AI's, DeepInfra's and Fireworks' embeddings
//! on the compatible embedding protocol (HST-028).
use super::Models;
use crate::{
    Error, Model, Provider, ProviderProtocol, StreamOptions,
    providers::{
        EventStream,
        openai::{OpenAIEmbeddingProtocol, protocol::HostChatProtocol},
        operations::{CallLimits, EmbedParams, Item},
    },
    types::Message,
};
use async_trait::async_trait;

/// The values per call and the parallel calls of the hosts' embeddings,
/// `OpenAICompatibleEmbeddingModel`'s defaults.
const EMBEDDINGS_PER_CALL: usize = 2048;

/// A host's chat protocol with the other model kinds it serves.
pub(super) struct HostModels {
    chat: HostChatProtocol,
    models: Models,
}
impl HostModels {
    pub(super) fn new(chat: HostChatProtocol, models: Models) -> Self {
        Self { chat, models }
    }
    fn embeddings(&self) -> Result<OpenAIEmbeddingProtocol, Error> {
        if self.models.embeddings {
            Ok(OpenAIEmbeddingProtocol::compatible(
                EMBEDDINGS_PER_CALL,
                true,
            ))
        } else {
            Err(Error::UnsupportedOperation("embed".into()))
        }
    }
}

#[async_trait]
impl ProviderProtocol for HostModels {
    fn protocol_class_id(&self) -> &str {
        self.chat.protocol_class_id()
    }
    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        self.chat.stream(model, messages, options, provider)
    }
    async fn embed(
        &self,
        model: &Model,
        values: Vec<String>,
        params: EmbedParams,
        provider: &dyn Provider,
    ) -> Result<Item<Vec<Vec<f64>>>, Error> {
        self.embeddings()?
            .embed(model, values, params, provider)
            .await
    }
    fn embed_limits(&self, model: &Model) -> CallLimits {
        self.embeddings()
            .map_or_else(|_| CallLimits::default(), |e| e.embed_limits(model))
    }
}
