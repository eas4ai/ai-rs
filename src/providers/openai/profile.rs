//! What differs between servers that speak the Chat Completions protocol.
use super::errors;
use crate::providers::http::ApiProfile;

/// How a server's Chat Completions API differs from OpenAI's. The host
/// profiles follow the AI SDK for TypeScript 7.0.148 host packages.
#[derive(Debug)]
pub(crate) struct ChatProfile {
    /// The protocol's class id in snapshots.
    pub protocol_class_id: &'static str,
    pub api: ApiProfile,
    /// Ask for usage in the stream with `stream_options.include_usage`,
    /// which some servers reject.
    pub include_usage: bool,
    /// The request field of the output token limit.
    pub max_tokens_field: &'static str,
    /// The assistant message field that carries reasoning history.
    pub reasoning_field: &'static str,
    /// Reasoning efforts the server takes under another name.
    pub effort_aliases: &'static [(&'static str, &'static str)],
    /// The models whose output schema goes out as a JSON schema; the other
    /// models get JSON mode, which does not enforce the schema.
    pub json_schema: JsonSchemaModels,
    /// Read the `openai` provider parameters (`OpenAIParams`).
    pub openai_params: bool,
    /// Take only automatic tool choice; tool choice none sends no tools.
    pub auto_tool_choice_only: bool,
    /// Reject frequency and presence penalties and seeds.
    pub rejects_penalties_and_seed: bool,
    /// Finish reasons the server adds, with their normalized names.
    pub finish_reasons: &'static [(&'static str, &'static str)],
    /// Whether `completion_tokens` counts the reasoning tokens.
    pub reasoning_tokens: ReasoningTokens,
    /// Under an output schema, drop tool calls that follow text.
    pub json_drops_tool_calls_after_text: bool,
    /// Request settings the server's TypeScript provider never sends; a call
    /// that sets one fails with `Error::Validation`.
    pub unsupported: &'static [Setting],
    /// How the server writes messages, schemas and stream content.
    pub dialect: Dialect,
    /// Give errors inside the stream a kind from their code and type.
    pub stream_error_kinds: bool,
    /// OpenAI's own Chat Completions API, which takes OpenAI's file shapes
    /// (OAI-007); other servers take the OpenAI-compatible ones.
    pub openai_own: bool,
    /// The provider options entries the host reads (HST-026).
    pub host_options: HostOptions,
}

/// The provider options entries a host reads beside the core settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HostOptions {
    /// None.
    NotRead,
    /// The `openaiCompatible` entry and the entry under this name, as
    /// `OpenAICompatibleChatLanguageModel` reads them.
    Compatible(&'static str),
    /// The compatible reading of the `fireworks` entry, with the options
    /// Fireworks renames (`fireworks-provider.ts`).
    Fireworks,
}
impl HostOptions {
    /// The name of the host's own entry.
    pub(crate) fn key(self) -> Option<&'static str> {
        match self {
            Self::NotRead => None,
            Self::Compatible(key) => Some(key),
            Self::Fireworks => Some("fireworks"),
        }
    }
}

/// How a server's `completion_tokens` relates to its reasoning tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReasoningTokens {
    /// The completion tokens include the reasoning tokens, as OpenAI's do.
    InCompletion,
    /// The server may count reasoning apart from the completion; reasoning
    /// above the completion count is added to it.
    OutsideWhenLarger,
    /// The completion tokens never include the reasoning tokens.
    Outside,
}

/// A request setting that a host profile can mark unsupported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Setting {
    ServiceTier,
    SafetyIdentifier,
    Metadata,
    CacheKey,
    CacheRetention,
    ParallelToolCalls,
    ToolSelection,
    TopK,
    MinP,
    Seed,
    FrequencyPenalty,
    PresencePenalty,
    ReasoningSummary,
    TextVerbosity,
    OutputInclude,
    MaxToolCalls,
    ContextManagement,
}

/// A server whose messages, schemas or stream content differ from OpenAI's
/// in more than field names. Each one's rules are in `dialect.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Dialect {
    OpenAI,
    DeepSeek,
    Moonshot,
    Mistral,
    /// The open models on Vertex AI: every file URL is downloaded, and Llama
    /// 4 gets an output limit.
    VertexMaas,
    /// xAI on Vertex AI: no reasoning effort goes out.
    VertexXai,
}

#[derive(Debug)]
pub(crate) enum JsonSchemaModels {
    All,
    Only(&'static [&'static str]),
    None,
}

impl ChatProfile {
    pub(crate) fn sends_json_schema(&self, model: &str) -> bool {
        match self.json_schema {
            JsonSchemaModels::All => true,
            JsonSchemaModels::Only(models) => models.contains(&model),
            JsonSchemaModels::None => false,
        }
    }

    pub(crate) fn effort<'a>(&self, effort: &'a str) -> &'a str {
        self.effort_aliases
            .iter()
            .find(|(from, _)| *from == effort)
            .map_or(effort, |(_, to)| to)
    }

    /// The normalized name of a finish reason this server adds.
    pub(crate) fn finish_reason(&self, raw: &str) -> Option<&'static str> {
        self.finish_reasons
            .iter()
            .find(|(from, _)| *from == raw)
            .map(|(_, to)| *to)
    }
}

/// How a server's Responses API differs from OpenAI's. The host profiles
/// follow the AI SDK for TypeScript 7.0.148 host packages.
#[derive(Debug)]
pub(crate) struct ResponsesProfile {
    /// The protocol's class id in snapshots.
    pub protocol_class_id: &'static str,
    pub api: ApiProfile,
    /// The key of the part metadata that carries the server's item ids and
    /// encrypted reasoning.
    pub metadata_key: &'static str,
    /// The key of the provider parameters that hold `store`, if the server
    /// reads them.
    pub params_key: Option<&'static str>,
    /// Request settings the server's TypeScript provider never sends; a call
    /// that sets one fails with `Error::Validation`.
    pub unsupported: &'static [Setting],
    /// How the server writes requests and history.
    pub dialect: ResponsesDialect,
}

/// A Responses server whose requests or history differ from OpenAI's. Each
/// one's rules are in `dialect.rs` and `messages.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResponsesDialect {
    OpenAI,
    Xai,
    HuggingFace,
}

/// OpenAI's Responses API.
pub(crate) static OPENAI_RESPONSES: ResponsesProfile = ResponsesProfile {
    protocol_class_id: "openai_responses",
    api: errors::API,
    metadata_key: "openai",
    params_key: Some("openai"),
    unsupported: &[],
    dialect: ResponsesDialect::OpenAI,
};

/// OpenAI's own Chat Completions.
pub(crate) static OPENAI: ChatProfile = ChatProfile {
    api: errors::API,
    openai_own: true,
    // HST-018's rules, as a Responses stream has them (OAI-012).
    stream_error_kinds: true,
    ..COMPATIBLE_RULES
};

/// The Chat Completions of an OpenAI-compatible server without a host
/// profile.
pub(crate) static COMPATIBLE: ChatProfile = COMPATIBLE_RULES;

const COMPATIBLE_RULES: ChatProfile = ChatProfile {
    protocol_class_id: "openai_chat_completions",
    api: errors::COMPATIBLE_API,
    include_usage: true,
    max_tokens_field: "max_completion_tokens",
    reasoning_field: "reasoning",
    effort_aliases: &[],
    json_schema: JsonSchemaModels::All,
    openai_params: true,
    auto_tool_choice_only: false,
    rejects_penalties_and_seed: false,
    finish_reasons: &[],
    reasoning_tokens: ReasoningTokens::InCompletion,
    json_drops_tool_calls_after_text: false,
    unsupported: &[],
    dialect: Dialect::OpenAI,
    stream_error_kinds: false,
    openai_own: false,
    host_options: HostOptions::NotRead,
};

/// Whether OpenAI treats a model as a reasoning model: `o` and digits,
/// followed by `-` or the end, or `gpt-<n>[.<m>][-<variant>]` with `n` at
/// least 5, except a variant starting with `chat` when there is no `.<m>`
/// (`openai-language-model-capabilities.ts`).
pub(crate) fn reasoning_model(model: &str) -> bool {
    super::capabilities::Capabilities::of(model).reasoning
}
