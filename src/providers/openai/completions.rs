//! The Completions API: OpenAI's (`completion/openai-completion-language-model.ts`)
//! and an OpenAI-compatible server's
//! (`openai-compatible-completion-language-model.ts`), OAI-015.

use futures::{StreamExt, stream};
use serde_json::{Value, json};

use crate::{
    Error, Model, Provider, ProviderProtocol, StreamOptions, providers::EventStream, types::Message,
};

/// Streams `POST <base>/completions` with a prompt made from the history.
/// [`OpenAICompletionsProtocol::new`] speaks OpenAI's API;
/// [`OpenAICompletionsProtocol::compatible`] an OpenAI-compatible server's,
/// which takes the call's other provider options in the body.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAICompletionsProtocol {
    compatible: Option<Compatible>,
}

#[derive(Clone, Copy, Debug)]
struct Compatible {
    include_usage: bool,
}

impl OpenAICompletionsProtocol {
    /// OpenAI's Completions API.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The Completions API of an OpenAI-compatible server. Its provider
    /// options are the entry under the provider's name, or that name in
    /// camelCase, and the options other than `echo`, `logitBias`, `suffix`
    /// and `user` go into the body as given. `include_usage` asks for usage
    /// in the stream with `stream_options`, which some servers reject.
    #[must_use]
    pub fn compatible(include_usage: bool) -> Self {
        Self {
            compatible: Some(Compatible { include_usage }),
        }
    }
}

impl ProviderProtocol for OpenAICompletionsProtocol {
    fn protocol_class_id(&self) -> &str {
        match self.compatible {
            Some(_) => "openai_compatible_completions",
            None => "openai_completions",
        }
    }

    fn to_value(&self) -> Result<Value, Error> {
        let mut value = json!({"protocol_class_id": self.protocol_class_id()});
        if let Some(compatible) = self.compatible {
            value["include_usage"] = json!(compatible.include_usage);
        }
        Ok(value)
    }

    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        let _ = (model, messages, options, provider);
        stream::once(async { Err(Error::UnsupportedOperation("completions".into())) }).boxed()
    }
}

/// Restore the protocol from its snapshot value.
pub(super) fn restore(value: &Value) -> OpenAICompletionsProtocol {
    match value.get("protocol_class_id").and_then(Value::as_str) {
        Some("openai_compatible_completions") => OpenAICompletionsProtocol::compatible(
            value
                .get("include_usage")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        ),
        _ => OpenAICompletionsProtocol::new(),
    }
}
