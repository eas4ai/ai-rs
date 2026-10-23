//! The open models on Vertex AI (MaaS), such as
//! `deepseek-ai/deepseek-v3.2-maas`, `openai/gpt-oss-120b-maas` and
//! `meta/llama-4-scout-17b-16e-instruct-maas`, over Chat Completions at
//! `https://<host>/v1/projects/<project>/locations/<location>/endpoints/openapi`,
//! the location `global` by default (`maas/google-vertex-maas-provider.ts`).
//! Llama 4 gets `max_tokens: 8192` when the call sets no limit. An output
//! schema goes out as JSON mode, which does not enforce the schema. Every
//! file URL is downloaded first, which needs a download transport.

use std::sync::Arc;

use super::chat::{self, ChatHost, VertexChatProvider};
use crate::{
    Error, ProviderConfig, ProviderRegistry,
    providers::{
        http::{ApiProfile, Auth},
        openai::profile::{ChatProfile, Dialect, JsonSchemaModels, ReasoningTokens},
    },
    transport::HttpTransport,
};

static PROFILE: ChatProfile = ChatProfile {
    protocol_class_id: "google_vertex_maas_chat_completions",
    api: ApiProfile {
        label: "Google Vertex AI MaaS",
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
    auto_tool_choice_only: false,
    rejects_penalties_and_seed: false,
    finish_reasons: &[],
    reasoning_tokens: ReasoningTokens::InCompletion,
    json_drops_tool_calls_after_text: false,
    unsupported: &[],
    dialect: Dialect::VertexMaas,
    stream_error_kinds: false,
};

fn base_url(config: &ProviderConfig) -> Result<String, Error> {
    let configured = config.base_url();
    if !configured.is_empty() {
        return Ok(configured);
    }
    let project = super::project(config)?;
    let location = super::location(config, Some("global"))?;
    Ok(format!(
        "https://{}/v1/projects/{project}/locations/{location}/endpoints/openapi",
        super::host(location)
    ))
}

static HOST: ChatHost = ChatHost {
    name: "google-vertex-maas",
    handle: "@ai-sdk/google-vertex/maas",
    profile: &PROFILE,
    base_url,
};

/// Build the MaaS provider. Set its credentials with
/// [`VertexChatProvider::with_access_token_source`], or, with the
/// `google-auth` feature, `with_credentials` or the `env` service account.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> VertexChatProvider {
    VertexChatProvider::new(&HOST, config, transport)
}

/// Register the MaaS provider under `google-vertex-maas` and
/// `@ai-sdk/google-vertex/maas`, with snapshot restoration that uses this
/// transport.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    register_provider(registry, provider(config, transport))
}

/// Register a built provider, with its credentials and download transport.
pub fn register_provider(
    registry: &mut ProviderRegistry,
    provider: VertexChatProvider,
) -> Result<(), Error> {
    chat::register(registry, provider)
}
