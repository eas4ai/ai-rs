//! Anthropic Messages API over the core HTTP transport.
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
//! overrides the default `ANTHROPIC_BASE_URL` lookup. Transport ownership is
//! shared; dropping a stream releases its body without closing the caller's client.
//! `list_models` returns the first API page, as in Python 0.8.0.
//!
//! Structured output schemas are lowered to Anthropic's supported subset, with
//! object schemas closed using `additionalProperties: false`. Unsupported
//! constraints become description text, matching the Python SDK's schema helper.
//! Schemars nullable type arrays become `anyOf`. The core decodes the resulting
//! assistant text through `StreamHandle::output`.
mod capabilities;
pub(crate) mod errors;
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
