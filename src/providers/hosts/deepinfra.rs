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

//! DeepInfra over Chat Completions. Requests send no `stream_options`. Some
//! DeepInfra models report reasoning tokens apart from completion tokens;
//! when reasoning tokens exceed completion tokens, output tokens are the sum.
use super::{Host, HostProtocol, HostProvider};
use crate::{
    Error, ProviderConfig, ProviderRegistry,
    providers::{
        http::{ApiProfile, Auth},
        openai::profile::{ChatProfile, Dialect, JsonSchemaModels},
    },
    transport::HttpTransport,
};
use std::sync::Arc;

static PROFILE: ChatProfile = ChatProfile {
    protocol_class_id: "deepinfra_chat_completions",
    api: ApiProfile {
        label: "DeepInfra",
        auth: Auth::Bearer,
        default_headers: &[],
        request_id_header: "x-request-id",
        collapse_server_errors: false,
        keep_error_object: false,
        top_level_first: false,
    },
    include_usage: false,
    max_tokens_field: "max_tokens",
    reasoning_field: "reasoning_content",
    effort_aliases: &[],
    json_schema: JsonSchemaModels::All,
    openai_params: false,
    auto_tool_choice_only: false,
    rejects_penalties_and_seed: false,
    finish_reasons: &[],
    reasoning_outside_completion: true,
    json_drops_tool_calls_after_text: false,
    unsupported: &[],
    dialect: Dialect::OpenAI,
    stream_error_kinds: false,
};

pub(super) static HOST: Host = Host {
    name: "deepinfra",
    handle: "@ai-sdk/deepinfra",
    default_base_url: "https://api.deepinfra.com/v1/openai",
    api_key_env: "DEEPINFRA_API_KEY",
    protocol: HostProtocol::Chat(&PROFILE),
};

/// Build the DeepInfra provider. An unset base URL becomes `https://api.deepinfra.com/v1/openai` and an
/// unset key variable `DEEPINFRA_API_KEY`.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> HostProvider {
    HostProvider::new(&HOST, config, transport)
}

/// Register the DeepInfra provider under `deepinfra` and `@ai-sdk/deepinfra`, with
/// snapshot restoration that uses this transport.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport, None)
}
