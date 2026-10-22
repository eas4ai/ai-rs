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

//! Cerebras over Chat Completions. Requests send the output token limit as
//! `max_completion_tokens` and reasoning history as `reasoning`, and no
//! `stream_options`. Under an output schema, Cerebras can repeat its answer as
//! tool calls next to the text, so tool events wait for the finish: when text
//! came, every tool call of the response is dropped and a `tool_calls` finish
//! reports `stop`.
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
    protocol_class_id: "cerebras_chat_completions",
    api: ApiProfile {
        label: "Cerebras",
        auth: Auth::Bearer,
        default_headers: &[],
        request_id_header: "x-request-id",
        collapse_server_errors: false,
        keep_error_object: false,
        top_level_first: false,
    },
    include_usage: false,
    max_tokens_field: "max_completion_tokens",
    reasoning_field: "reasoning",
    effort_aliases: &[],
    json_schema: JsonSchemaModels::All,
    openai_params: false,
    auto_tool_choice_only: false,
    rejects_penalties_and_seed: false,
    finish_reasons: &[],
    reasoning_outside_completion: false,
    json_drops_tool_calls_after_text: true,
};

pub(super) static HOST: Host = Host {
    name: "cerebras",
    handle: "@ai-sdk/cerebras",
    default_base_url: "https://api.cerebras.ai/v1",
    api_key_env: "CEREBRAS_API_KEY",
    protocol: HostProtocol::Chat(&PROFILE),
};

/// Build the Cerebras provider. An unset base URL becomes `https://api.cerebras.ai/v1` and an
/// unset key variable `CEREBRAS_API_KEY`.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> HostProvider {
    HostProvider::new(&HOST, config, transport)
}

/// Register the Cerebras provider under `cerebras` and `@ai-sdk/cerebras`, with
/// snapshot restoration that uses this transport.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport)
}
