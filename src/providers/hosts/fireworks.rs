//! Fireworks over Chat Completions. Fireworks takes the reasoning efforts
//! low, medium and high, so `minimal` goes out as `low` and `xhigh` and `max`
//! as `high`. The `fireworks` provider options entry takes `promptCacheKey`,
//! `serviceTier`, `thinking` and `reasoningHistory`, sent under Fireworks'
//! names, and any other Fireworks field as it is. Its models also complete
//! text (`completion_protocol`), embed, and generate images, polling the
//! Kontext models' jobs.
use super::{Host, HostProtocol, HostProvider, Models};
use crate::{
    Error, ProviderConfig, ProviderRegistry,
    providers::{
        http::{ApiProfile, Auth},
        openai::{
            OpenAICompletionsProtocol,
            profile::{ChatProfile, Dialect, HostOptions, JsonSchemaModels, ReasoningTokens},
        },
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
    openai_own: false,
    host_options: HostOptions::Fireworks,
};

pub(super) static HOST: Host = Host {
    name: "fireworks",
    handle: "@ai-sdk/fireworks",
    default_base_url: "https://api.fireworks.ai/inference/v1",
    api_key_env: "FIREWORKS_API_KEY",
    protocol: HostProtocol::Chat(&PROFILE),
    models: Models::FIREWORKS,
};

/// Build the Fireworks provider. An unset base URL becomes `https://api.fireworks.ai/inference/v1` and an
/// unset key variable `FIREWORKS_API_KEY`.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> HostProvider {
    HostProvider::new(&HOST, config, transport)
}

/// The protocol of a Fireworks completion model: OAI-015's compatible
/// completions, with streamed usage. Set it on a model of this provider with
/// `Model::with_protocol`; the call reads its options from the `fireworks`
/// entry.
pub fn completion_protocol() -> OpenAICompletionsProtocol {
    OpenAICompletionsProtocol::compatible(true)
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
