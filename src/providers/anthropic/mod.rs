//! Anthropic's Messages API over the core HTTP transport, as the TypeScript
//! `@ai-sdk/anthropic` 4.0.85 package sends and reads it, with every model
//! kind it ships: the language model, files, skills, batches and evaluation.
//!
//! Construct an [`AnthropicCompatibleProvider`] with explicit configuration and
//! a shared transport. [`register`] installs the standard handles and snapshot
//! restoration in an explicit provider registry.
//!
//! ```
//! use std::sync::Arc;
//! use suprnova_ai::{Model, ProviderConfig};
//! use suprnova_ai::providers::anthropic::AnthropicCompatibleProvider;
//! use suprnova_ai::transport::HttpTransport;
//!
//! fn configured_model(transport: Arc<dyn HttpTransport>, api_key: String) -> Model {
//!     let provider = AnthropicCompatibleProvider::new(
//!         ProviderConfig { api_key_value: Some(api_key), ..Default::default() },
//!         transport,
//!     );
//!     Model::new("claude-sonnet-4-6", Arc::new(provider))
//! }
//! ```
//!
//! Configuration uses the core's explicit `env` snapshot. A custom base URL
//! overrides the default `ANTHROPIC_BASE_URL` lookup. An `ANTHROPIC_AUTH_TOKEN`
//! in `env` goes as `authorization: Bearer` in place of the key, to Anthropic's
//! own API only: a provider of another server, and `AnthropicMessagesProtocol`
//! on its own, send neither Anthropic's token nor its default key. Transport
//! ownership is shared; dropping a stream releases its body without closing the
//! caller's client. `list_models` returns the first API page, as in Python 0.8.0.
//!
//! Provider options are the `anthropic` provider parameter, with the
//! TypeScript option names; an unknown option fails with `Error::Validation`. A
//! part's `anthropic` metadata carries what TypeScript reads from its provider
//! options, such as `cacheControl` or a document's `citations`, and a tool's
//! `tool_config` arguments carry its own in snake case. Where TypeScript changes
//! or leaves out a setting with a warning, the request does the same and logs a
//! `tracing` debug event with target `suprnova_ai::anthropic` that names the
//! setting and the model. Without an output limit a call asks for the model's
//! own limit.
//!
//! An output schema goes out as `output_config.format`, with TypeScript's
//! sanitized schema, or as the forced `json` tool where the model has no
//! structured output; the core decodes the answer through
//! `StreamHandle::output`. [`tools`] builds every tool version of
//! `anthropicTools`. A tool the application runs, such as bash or the computer
//! toolset, arrives as a tool call under the application's tool name; give an
//! agent one with `AgentTool::with_provider_declaration`. Tools Anthropic runs
//! arrive as built-in calls and results in TypeScript's shapes. Sources are in
//! the final message metadata under `anthropic.sources`, and
//! [`container_id`] gives the container to pass on in the next call's
//! `container` option. `with_download_transport` fetches the user files
//! Anthropic does not take by URL, such as a text file. Files, skills and
//! Message Batches need Anthropic's own API.
mod batch;
mod capabilities;
pub(crate) mod errors;
mod files;
pub(crate) mod messages;
mod options;
mod params;
pub mod protocol;
mod provider;
mod schema;
pub(crate) mod stream;
pub mod tools;
pub use protocol::AnthropicMessagesProtocol;
pub use provider::{AnthropicCompatibleProvider, register};

use crate::types::Message;

/// The container id of the latest message whose final metadata holds one,
/// to pass on as the next call's `container` option
/// (`forwardAnthropicContainerIdFromLastStep`, ANT-014).
pub fn container_id(messages: &[Message]) -> Option<String> {
    messages.iter().rev().find_map(|message| {
        message
            .provider_metadata
            .as_ref()?
            .get("anthropic")?
            .get("container")?
            .get("id")?
            .as_str()
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
    })
}
