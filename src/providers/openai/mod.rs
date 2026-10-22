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

//! OpenAI Responses and OpenAI-compatible Chat Completions over `HttpTransport`.
//!
//! Construct a provider with explicit configuration and a shared transport. Register
//! handles and snapshot factories explicitly; no process environment is read.
//!
//! ```
//! use std::sync::Arc;
//! use suprnova_ai::{Model, ProviderConfig};
//! use suprnova_ai::providers::openai::OpenAICompatibleProvider;
//! use suprnova_ai::transport::HttpTransport;
//!
//! fn model(transport: Arc<dyn HttpTransport>, api_key: String) -> Model {
//!     let config = ProviderConfig { api_key_value: Some(api_key), ..Default::default() };
//!     Model::new("gpt-5.4", Arc::new(OpenAICompatibleProvider::new(config, transport)))
//! }
//! ```
//!
//! Use `OpenAICompatibleProvider::from_modelsdev_provider` for xai, ollama,
//! or custom provider metadata. Metadata that names OpenAI's own package,
//! `@ai-sdk/openai`, selects Responses, as `OpenAICompatibleProvider::new` does;
//! other metadata selects Chat Completions. Metadata for Cerebras, DeepInfra,
//! Fireworks, Together AI or Z.ai gets that host's rules from
//! `providers::hosts`, which also registers those hosts directly. An explicit
//! protocol overrides the default. Custom providers using
//! these protocols implement `Provider::shared_transport` for owned streams.
//! `params::OpenAIParams::apply` stores options under the stable `openai` key.
//! Built-in tool arguments use snake_case JSON objects, as in the Python wire format.
pub mod errors;
mod messages;
pub mod params;
pub(crate) mod profile;
pub mod protocol;
mod provider;
mod streaming;
pub mod tools;
pub use params::OpenAIParams;
// Kept for callers of the old name-based choice; it is deprecated.
#[allow(deprecated)]
pub use protocol::default_protocol;
pub use protocol::{
    OpenAIChatCompletionsProtocol, OpenAIChatCompletionsWithDownloads, OpenAIResponsesProtocol,
};
pub use provider::{OpenAICompatibleProvider, register, register_factories};
