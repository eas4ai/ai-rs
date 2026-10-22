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

//! Fireworks over Chat Completions. Fireworks takes the reasoning efforts
//! low, medium and high, so `minimal` goes out as `low` and `xhigh` and `max`
//! as `high`. Fireworks-only fields, such as a thinking budget, go in
//! `InferenceRequestParams::extra_body`.
use super::{Host, HostProtocol, HostProvider};
use crate::{
    Error, ProviderConfig, ProviderRegistry,
    providers::{
        http::{ApiProfile, Auth},
        openai::profile::{ChatProfile, JsonSchemaModels},
    },
    transport::HttpTransport,
};
use std::sync::Arc;

static PROFILE: ChatProfile = ChatProfile {
    protocol_class_id: "fireworks_chat_completions",
    api: ApiProfile {
        label: "Fireworks",
        auth: Auth::Bearer,
        default_headers: &[],
        request_id_header: "x-request-id",
        collapse_server_errors: false,
        keep_error_object: false,
    },
    include_usage: true,
    max_tokens_field: "max_tokens",
    reasoning_field: "reasoning_content",
    effort_aliases: &[("minimal", "low"), ("xhigh", "high"), ("max", "high")],
    json_schema: JsonSchemaModels::All,
    openai_params: false,
    auto_tool_choice_only: false,
    rejects_penalties_and_seed: false,
    finish_reasons: &[],
    reasoning_outside_completion: false,
    json_drops_tool_calls_after_text: false,
};

static HOST: Host = Host {
    name: "fireworks",
    handle: "@ai-sdk/fireworks",
    default_base_url: "https://api.fireworks.ai/v1",
    api_key_env: "FIREWORKS_API_KEY",
    protocol: HostProtocol::Chat(&PROFILE),
};

/// Build the Fireworks provider. An unset base URL becomes `https://api.fireworks.ai/inference/v1` and an
/// unset key variable `FIREWORKS_API_KEY`.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> HostProvider {
    HostProvider::new(&HOST, config, transport)
}

/// Register the Fireworks provider under `fireworks` and `@ai-sdk/fireworks`, with
/// snapshot restoration that uses this transport.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport)
}
