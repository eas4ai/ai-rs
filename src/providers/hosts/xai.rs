//! xAI over the Responses API, for text: `@ai-sdk/xai` 5.0.0 removed its
//! Chat Completions model. Top-k, min-p and seeds are sent; penalties, a
//! reasoning summary, text verbosity, a maximum tool call count, context
//! management, a cache retention, metadata and tool selections fail with
//! `Error::Validation`, as do reasoning settings for the `grok-4.20`
//! reasoning and non-reasoning models. Efforts follow xAI's levels:
//! `minimal` goes out as `low`, and `xhigh` and `max` as `xhigh` for
//! `grok-4.6` and `high` otherwise. `store` is read from the provider
//! parameters under `xai` (`{"store": false}`), and an unstored response
//! asks for its reasoning as encrypted content. History goes out without
//! references to stored items, with the item ids this provider keeps in part
//! metadata under `xai`. Built-in tools, such as web and X search, fail with
//! `Error::UnsupportedOperation`.
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
    protocol_class_id: "xai_responses",
    api: ApiProfile {
        label: "xAI",
        auth: Auth::Bearer,
        default_headers: &[],
        request_id_header: "x-request-id",
        collapse_server_errors: false,
        keep_error_object: false,
        top_level_first: false,
        code_prefixes_error: true,
    },
    metadata_key: "xai",
    params_key: Some("xai"),
    unsupported: &[
        Setting::FrequencyPenalty,
        Setting::PresencePenalty,
        Setting::ReasoningSummary,
        Setting::TextVerbosity,
        Setting::MaxToolCalls,
        Setting::ContextManagement,
        Setting::CacheRetention,
        Setting::Metadata,
        Setting::ToolSelection,
    ],
    dialect: ResponsesDialect::Xai,
};

pub(super) static HOST: Host = Host {
    name: "xai",
    handle: "@ai-sdk/xai",
    default_base_url: "https://api.x.ai/v1",
    api_key_env: "XAI_API_KEY",
    protocol: HostProtocol::Responses(&PROFILE),
};

/// Build the xAI provider. An unset base URL becomes `https://api.x.ai/v1`
/// and an unset key variable `XAI_API_KEY`.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> HostProvider {
    HostProvider::new(&HOST, config, transport)
}

/// Register the xAI provider under `xai` and `@ai-sdk/xai`, with snapshot
/// restoration that uses this transport.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport, None)
}
