//! OpenAI's native evaluation with `POST <base>/decisions`, OAI-025. Not
//! built yet.

use crate::ProviderProtocol;

/// Evaluates with OpenAI's Decisions API.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAIDecisionsProtocol;

impl ProviderProtocol for OpenAIDecisionsProtocol {
    fn protocol_class_id(&self) -> &str {
        "openai_decisions"
    }
}
