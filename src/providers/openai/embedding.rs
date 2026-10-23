//! Embeddings: OpenAI's and an OpenAI-compatible server's, OAI-016.

use serde_json::{Value, json};

use crate::{Error, ProviderProtocol};

/// Embeds with `POST <base>/embeddings`.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAIEmbeddingProtocol {
    compatible: bool,
}

impl OpenAIEmbeddingProtocol {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn compatible(_max_per_call: usize, _parallel: bool) -> Self {
        Self { compatible: true }
    }
}

/// Restore the protocol from its snapshot value.
pub(super) fn restore(_value: &Value) -> OpenAIEmbeddingProtocol {
    OpenAIEmbeddingProtocol::new()
}

impl ProviderProtocol for OpenAIEmbeddingProtocol {
    fn protocol_class_id(&self) -> &str {
        if self.compatible {
            "openai_compatible_embedding"
        } else {
            "openai_embedding"
        }
    }

    fn to_value(&self) -> Result<Value, Error> {
        Ok(json!({"protocol_class_id": self.protocol_class_id()}))
    }
}
