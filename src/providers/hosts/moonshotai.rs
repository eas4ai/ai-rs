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

//! Moonshot AI over Chat Completions. Reasoning follows the model family:
//! `kimi-k3` and unknown models take a reasoning effort as `reasoning_effort`
//! and cannot disable reasoning; `kimi-k2.6` and `kimi-k2.5` take
//! `thinking` enabled or disabled; `kimi-k2.7-code` always reasons; and
//! `moonshot-v1-*` models do not reason. The Kimi models have fixed sampling,
//! so a temperature, top-p or penalty fails with `Error::Validation`, as do a
//! seed and the settings Moonshot AI does not take, such as a service tier.
//! Tool and output schemas go out as Moonshot's MFJS subset of JSON schema.
//! Moonshot AI does not fetch media URLs: build the provider with a download
//! transport to send image and video URLs, which otherwise fail.
use super::{Host, HostProtocol, HostProvider};
use crate::{
    Error, ProviderConfig, ProviderRegistry,
    models::files::DownloadTransport,
    providers::{
        http::{ApiProfile, Auth},
        openai::profile::{ChatProfile, Dialect, JsonSchemaModels, Setting},
    },
    transport::HttpTransport,
};
use std::sync::Arc;

static PROFILE: ChatProfile = ChatProfile {
    protocol_class_id: "moonshotai_chat_completions",
    api: ApiProfile {
        label: "Moonshot AI",
        auth: Auth::Bearer,
        default_headers: &[],
        request_id_header: "x-request-id",
        collapse_server_errors: false,
        keep_error_object: false,
        top_level_first: false,
        code_prefixes_error: false,
    },
    include_usage: true,
    max_tokens_field: "max_completion_tokens",
    reasoning_field: "reasoning_content",
    effort_aliases: &[],
    json_schema: JsonSchemaModels::All,
    openai_params: false,
    auto_tool_choice_only: false,
    rejects_penalties_and_seed: false,
    finish_reasons: &[],
    reasoning_outside_completion: false,
    json_drops_tool_calls_after_text: false,
    unsupported: &[
        Setting::ServiceTier,
        Setting::Metadata,
        Setting::CacheRetention,
        Setting::ParallelToolCalls,
        Setting::ToolSelection,
    ],
    dialect: Dialect::Moonshot,
    stream_error_kinds: true,
};

pub(super) static HOST: Host = Host {
    name: "moonshotai",
    handle: "@ai-sdk/moonshotai",
    default_base_url: "https://api.moonshot.ai/v1",
    api_key_env: "MOONSHOT_API_KEY",
    protocol: HostProtocol::Chat(&PROFILE),
};

/// Build the Moonshot AI provider. An unset base URL becomes
/// `https://api.moonshot.ai/v1` and an unset key variable `MOONSHOT_API_KEY`.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> HostProvider {
    HostProvider::new(&HOST, config, transport)
}

/// Register the Moonshot AI provider under `moonshotai` and
/// `@ai-sdk/moonshotai`, with snapshot restoration that uses this transport.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport, None)
}

/// Register the Moonshot AI provider as `register` does, with a download
/// transport that fetches image and video URLs before each request, also for
/// a provider restored from a snapshot.
pub fn register_with_downloads(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
    downloads: Arc<dyn DownloadTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport, Some(downloads))
}
