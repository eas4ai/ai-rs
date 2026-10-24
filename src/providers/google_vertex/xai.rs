//! xAI on Vertex AI, such as `xai/grok-4.20-reasoning`, over Chat
//! Completions at
//! `https://aiplatform.googleapis.com/v1/projects/<project>/locations/<location>/endpoints/openapi`,
//! the location `global` by default (`xai/google-vertex-xai-provider.ts`).
//! The stream reports usage, an output schema goes out as a strict JSON
//! schema, no reasoning effort is sent, and only image URLs go out as URLs:
//! other file URLs are downloaded first, which needs a download transport.
//! The output tokens are the completion tokens plus the reasoning tokens.

use std::sync::Arc;

use super::chat::{self, ChatHost, VertexChatProvider};
use crate::{
    Error, ProviderConfig, ProviderRegistry,
    providers::{
        http::{ApiProfile, Auth},
        openai::profile::{ChatProfile, Dialect, HostOptions, JsonSchemaModels, ReasoningTokens},
    },
    transport::HttpTransport,
};

static PROFILE: ChatProfile = ChatProfile {
    protocol_class_id: "google_vertex_xai_chat_completions",
    api: ApiProfile {
        label: "Google Vertex AI xAI",
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
    effort_aliases: &[],
    json_schema: JsonSchemaModels::All,
    openai_params: false,
    auto_tool_choice_only: false,
    rejects_penalties_and_seed: false,
    finish_reasons: &[],
    reasoning_tokens: ReasoningTokens::Outside,
    json_drops_tool_calls_after_text: false,
    unsupported: &[],
    dialect: Dialect::VertexXai,
    stream_error_kinds: false,
    openai_own: false,
    host_options: HostOptions::NotRead,
};

fn base_url(config: &ProviderConfig) -> Result<String, Error> {
    let configured = config.base_url();
    if !configured.is_empty() {
        return Ok(configured);
    }
    let project = super::project(config)?;
    let location = super::location(config, Some("global"))?;
    Ok(format!(
        "https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi"
    ))
}

static HOST: ChatHost = ChatHost {
    name: "google-vertex-xai",
    handle: "@ai-sdk/google-vertex/xai",
    profile: &PROFILE,
    base_url,
};

/// Build the xAI on Vertex AI provider. Set its credentials with
/// [`VertexChatProvider::with_access_token_source`], or, with the
/// `google-auth` feature, `with_credentials` or the `env` service account.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> VertexChatProvider {
    VertexChatProvider::new(&HOST, config, transport)
}

/// Register the xAI on Vertex AI provider under `google-vertex-xai` and
/// `@ai-sdk/google-vertex/xai`, with snapshot restoration that uses this
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
