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

//! Hugging Face Inference Providers over the Responses API. An output
//! schema goes out unchanged with `strict: false`; a named tool choice goes
//! out in the Chat Completions shape, and tool choice none sends no tools.
//! Only temperature, top-p, the output token limit, metadata, tool choice and
//! reasoning effort are sent: every other request setting, such as top-k, a
//! seed, penalties or a service tier, fails with `Error::Validation`, and
//! built-in tools fail with `Error::UnsupportedOperation`. History goes out
//! as output text messages, with tool calls and results as `function_call`
//! and `function_call_output` items, and user files must be images.
use super::{Host, HostProtocol, HostProvider};
use crate::{
    Error, ProviderConfig, ProviderRegistry,
    providers::{
        http::{ApiProfile, Auth},
        openai::profile::{ResponsesDialect, ResponsesProfile, Setting},
    },
    transport::HttpTransport,
};
use std::sync::Arc;

static PROFILE: ResponsesProfile = ResponsesProfile {
    protocol_class_id: "huggingface_responses",
    api: ApiProfile {
        label: "Hugging Face",
        auth: Auth::Bearer,
        default_headers: &[],
        request_id_header: "x-request-id",
        collapse_server_errors: false,
        keep_error_object: false,
        top_level_first: false,
        code_prefixes_error: false,
    },
    metadata_key: "huggingface",
    params_key: None,
    unsupported: &[
        Setting::TopK,
        Setting::MinP,
        Setting::Seed,
        Setting::FrequencyPenalty,
        Setting::PresencePenalty,
        Setting::ReasoningSummary,
        Setting::TextVerbosity,
        Setting::OutputInclude,
        Setting::MaxToolCalls,
        Setting::ParallelToolCalls,
        Setting::ServiceTier,
        Setting::SafetyIdentifier,
        Setting::CacheKey,
        Setting::CacheRetention,
        Setting::ContextManagement,
        Setting::ToolSelection,
    ],
    dialect: ResponsesDialect::HuggingFace,
};

pub(super) static HOST: Host = Host {
    name: "huggingface",
    handle: "@ai-sdk/huggingface",
    default_base_url: "https://router.huggingface.co/v1",
    api_key_env: "HUGGINGFACE_API_KEY",
    protocol: HostProtocol::Responses(&PROFILE),
};

/// Build the Hugging Face provider. An unset base URL becomes
/// `https://router.huggingface.co/v1` and an unset key variable
/// `HUGGINGFACE_API_KEY`.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> HostProvider {
    HostProvider::new(&HOST, config, transport)
}

/// Register the Hugging Face provider under `huggingface` and
/// `@ai-sdk/huggingface`, with snapshot restoration that uses this
/// transport.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport, None)
}
