//! DeepSeek over Chat Completions. A reasoning effort goes out as
//! `thinking: {"type": "enabled"}` with `reasoning_effort` (`minimal` as
//! `low`, `medium` as `high`, `xhigh` as `max`), and disabled reasoning as
//! `thinking: {"type": "disabled"}`. V4 models reason unless reasoning is
//! disabled, and so does `deepseek-reasoner`; while a model reasons, a
//! temperature or top-p fails with `Error::Validation`, as do penalties,
//! seeds and the settings DeepSeek does not take, such as a service tier. An
//! output schema goes out as JSON mode with the schema in a first system
//! message. V4 models get `reasoning_content` back on every assistant turn;
//! earlier models get none from before the last user turn. User files must be
//! GIF, JPEG, PNG or WebP images.
use super::{Host, HostProtocol, HostProvider};
use crate::{
    Error, ProviderConfig, ProviderRegistry,
    providers::{
        http::{ApiProfile, Auth},
        openai::profile::{ChatProfile, Dialect, JsonSchemaModels, ReasoningTokens, Setting},
    },
    transport::HttpTransport,
};
use std::sync::Arc;

static PROFILE: ChatProfile = ChatProfile {
    protocol_class_id: "deepseek_chat_completions",
    api: ApiProfile {
        label: "DeepSeek",
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
    effort_aliases: &[("minimal", "low"), ("medium", "high"), ("xhigh", "max")],
    json_schema: JsonSchemaModels::None,
    openai_params: false,
    auto_tool_choice_only: false,
    rejects_penalties_and_seed: true,
    finish_reasons: &[("insufficient_system_resource", "error")],
    reasoning_tokens: ReasoningTokens::InCompletion,
    json_drops_tool_calls_after_text: false,
    unsupported: &[
        Setting::ServiceTier,
        Setting::SafetyIdentifier,
        Setting::Metadata,
        Setting::CacheKey,
        Setting::CacheRetention,
        Setting::ParallelToolCalls,
        Setting::ToolSelection,
    ],
    dialect: Dialect::DeepSeek,
    stream_error_kinds: true,
};

pub(super) static HOST: Host = Host {
    name: "deepseek",
    handle: "@ai-sdk/deepseek",
    default_base_url: "https://api.deepseek.com",
    api_key_env: "DEEPSEEK_API_KEY",
    protocol: HostProtocol::Chat(&PROFILE),
};

/// Build the DeepSeek provider. An unset base URL becomes
/// `https://api.deepseek.com` and an unset key variable `DEEPSEEK_API_KEY`.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> HostProvider {
    HostProvider::new(&HOST, config, transport)
}

/// Register the DeepSeek provider under `deepseek` and `@ai-sdk/deepseek`,
/// with snapshot restoration that uses this transport.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport, None)
}
