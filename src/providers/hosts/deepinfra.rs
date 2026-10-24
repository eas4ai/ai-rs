//! DeepInfra over Chat Completions. Requests send no `stream_options`. Some
//! DeepInfra models report reasoning tokens apart from completion tokens;
//! when reasoning tokens exceed completion tokens, output tokens are the sum.
//!
//! Its models also complete text (`completion_protocol`), embed, and
//! generate and edit images at `https://api.deepinfra.com/v1/inference`,
//! the base URL without its last segment `openai`.
use super::{Host, HostProtocol, HostProvider, Models};
use crate::{
    Error, ProviderConfig, ProviderRegistry,
    models::files::DownloadTransport,
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
    protocol_class_id: "deepinfra_chat_completions",
    api: ApiProfile {
        label: "DeepInfra",
        auth: Auth::Bearer,
        default_headers: &[],
        request_id_header: "x-request-id",
        collapse_server_errors: false,
        keep_error_object: false,
        top_level_first: false,
        code_prefixes_error: false,
        config_headers: &[],
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
    reasoning_tokens: ReasoningTokens::OutsideWhenLarger,
    json_drops_tool_calls_after_text: false,
    unsupported: &[],
    dialect: Dialect::OpenAI,
    stream_error_kinds: false,
    openai_own: false,
    host_options: HostOptions::Compatible("deepinfra"),
};

pub(super) static HOST: Host = Host {
    name: "deepinfra",
    handle: "@ai-sdk/deepinfra",
    default_base_url: "https://api.deepinfra.com/v1/openai",
    api_key_env: "DEEPINFRA_API_KEY",
    protocol: HostProtocol::Chat(&PROFILE),
    models: Models::DEEPINFRA,
};

/// Build the DeepInfra provider. An unset base URL becomes `https://api.deepinfra.com/v1/openai` and an
/// unset key variable `DEEPINFRA_API_KEY`.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> HostProvider {
    HostProvider::new(&HOST, config, transport)
}

/// The protocol of a DeepInfra completion model: OAI-015's compatible
/// completions, without streamed usage, which DeepInfra does not take. Set it on a model of this provider with
/// `Model::with_protocol`; the call reads its options from the `deepinfra`
/// entry.
pub fn completion_protocol() -> OpenAICompletionsProtocol {
    OpenAICompletionsProtocol::compatible(false)
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

/// Register the DeepInfra provider as `register` does, with a download
/// transport that fetches image URLs to edit, also for a provider restored from a
/// snapshot.
pub fn register_with_downloads(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
    downloads: Arc<dyn DownloadTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport, Some(downloads))
}
