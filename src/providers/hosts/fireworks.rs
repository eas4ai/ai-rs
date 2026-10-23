//! Fireworks over Chat Completions. Fireworks takes the reasoning efforts
//! low, medium and high, so `minimal` goes out as `low` and `xhigh` and `max`
//! as `high`. Fireworks-only fields, such as a thinking budget, go in
//! `InferenceRequestParams::extra_body`.
use super::{Host, HostProtocol, HostProvider};
use crate::{
    Error, ProviderConfig, ProviderRegistry,
    providers::{
        http::{ApiProfile, Auth},
        openai::profile::{ChatProfile, Dialect, JsonSchemaModels, ReasoningTokens},
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
        top_level_first: false,
        code_prefixes_error: false,
        config_headers: &[],
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
    reasoning_tokens: ReasoningTokens::InCompletion,
    json_drops_tool_calls_after_text: false,
    unsupported: &[],
    dialect: Dialect::OpenAI,
    stream_error_kinds: false,
};

pub(super) static HOST: Host = Host {
    name: "fireworks",
    handle: "@ai-sdk/fireworks",
    default_base_url: "https://api.fireworks.ai/inference/v1",
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
    super::register(&HOST, registry, config, transport, None)
}
