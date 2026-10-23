//! Z.ai over Chat Completions. Z.ai takes only automatic tool choice: tool
//! choice none sends no tools, and a required, named or selected tool choice
//! fails with `Error::Validation`, as do frequency and presence penalties and
//! seeds. An output schema goes out as JSON mode, which does not enforce it.
//! The finish reasons `sensitive`, `model_context_window_exceeded` and
//! `network_error` report `content_filter`, `length` and `error`.
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
    protocol_class_id: "zai_chat_completions",
    api: ApiProfile {
        label: "Z.ai",
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
    json_schema: JsonSchemaModels::None,
    openai_params: false,
    auto_tool_choice_only: true,
    rejects_penalties_and_seed: true,
    finish_reasons: &[
        ("sensitive", "content_filter"),
        ("model_context_window_exceeded", "length"),
        ("network_error", "error"),
    ],
    reasoning_tokens: ReasoningTokens::InCompletion,
    json_drops_tool_calls_after_text: false,
    unsupported: &[],
    dialect: Dialect::OpenAI,
    stream_error_kinds: false,
};

pub(super) static HOST: Host = Host {
    name: "zai",
    handle: "@ai-sdk/zai",
    default_base_url: "https://api.z.ai/api/paas/v4",
    api_key_env: "ZAI_API_KEY",
    protocol: HostProtocol::Chat(&PROFILE),
};

/// Build the Z.ai provider. An unset base URL becomes `https://api.z.ai/api/paas/v4` and an
/// unset key variable `ZAI_API_KEY`.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> HostProvider {
    HostProvider::new(&HOST, config, transport)
}

/// Register the Z.ai provider under `zai` and `@ai-sdk/zai`, with
/// snapshot restoration that uses this transport.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport, None)
}
