//! MiniMax over the Anthropic Messages protocol, at `<base>/v1/messages` with
//! `x-api-key` and `anthropic-version: 2023-06-01`. MiniMax 3.1 and later
//! (`MiniMax-M3.1`, `minimax-m4`) always reason: a reasoning effort goes out as
//! `output_config.effort` with `minimal` as `low`, and disabled reasoning
//! becomes effort `low`. Older models keep the Anthropic mapping.
use super::{Host, HostProtocol, HostProvider, Models};
use crate::{
    Error, ProviderConfig, ProviderRegistry,
    providers::{
        anthropic::protocol::MessagesProfile,
        http::{ApiProfile, Auth},
    },
    transport::HttpTransport,
};
use std::sync::Arc;

static PROFILE: MessagesProfile = MessagesProfile {
    protocol_class_id: "minimax_messages",
    api: ApiProfile {
        label: "MiniMax",
        auth: Auth::Header("x-api-key"),
        default_headers: &[("anthropic-version", "2023-06-01")],
        request_id_header: "request-id",
        collapse_server_errors: false,
        keep_error_object: false,
        top_level_first: true,
        code_prefixes_error: false,
        config_headers: &[],
    },
    always_reasons: names_version_3_1_or_later,
    auth_token_env: None,
    native_structured_output: true,
    strict_tools: true,
    // MiniMax 3.1 and later take a temperature up to 2 and keep sampling
    // with thinking (ANT-016).
    max_temperature: |model| {
        if names_version_3_1_or_later(model) {
            2.0
        } else {
            1.0
        }
    },
    sampling_with_thinking: names_version_3_1_or_later,
    // The `minimax` entry is read over the `anthropic` one, and a call that
    // used it gets its final metadata under `minimax` too (HST-038).
    options_key: Some("minimax"),
};

pub(super) static HOST: Host = Host {
    name: "minimax",
    handle: "@ai-sdk/minimax",
    default_base_url: "https://api.minimax.io/anthropic",
    api_key_env: "MINIMAX_API_KEY",
    protocol: HostProtocol::Messages(&PROFILE),
    models: Models::CHAT,
};

/// Whether a model id names MiniMax 3.1 or later, read as the TypeScript SDK
/// reads it: `minimax-m<major>[.<minor>][.<more>...]`, in any case, then `-` or
/// the end of the id.
fn names_version_3_1_or_later(model: &str) -> bool {
    let id = model.to_ascii_lowercase();
    let Some(rest) = id.strip_prefix("minimax-m") else {
        return false;
    };
    let version = rest.split('-').next().unwrap_or_default();
    let mut numbers = version.split('.').map(|part| {
        (!part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
            .then(|| part.parse::<u64>().unwrap_or(u64::MAX))
    });
    let Some(Some(major)) = numbers.next() else {
        return false;
    };
    let minor = match numbers.next() {
        None => 0,
        Some(Some(minor)) => minor,
        Some(None) => return false,
    };
    if numbers.any(|part| part.is_none()) {
        return false;
    }
    major > 3 || (major == 3 && minor >= 1)
}

/// Build the MiniMax provider. An unset base URL becomes `https://api.minimax.io/anthropic` and an
/// unset key variable `MINIMAX_API_KEY`.
pub fn provider(config: ProviderConfig, transport: Arc<dyn HttpTransport>) -> HostProvider {
    HostProvider::new(&HOST, config, transport)
}

/// Register the MiniMax provider under `minimax` and `@ai-sdk/minimax`, with
/// snapshot restoration that uses this transport.
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport, None)
}

#[cfg(test)]
mod tests {
    use super::names_version_3_1_or_later;

    #[test]
    fn versions_read_as_the_typescript_sdk_reads_them() {
        for id in [
            "MiniMax-M3.1",
            "minimax-m3.1-highspeed",
            "minimax-m3.123",
            "MINIMAX-M4",
            "minimax-m3.2.5",
            "minimax-m10",
        ] {
            assert!(names_version_3_1_or_later(id), "{id}");
        }
        for id in [
            "MiniMax-M2",
            "MiniMax-M2.5",
            "minimax-m3",
            "minimax-m3.0",
            "minimax-m3x",
            "minimax-m",
            "minimax-m3.",
            "abab6.5-chat",
            "x-minimax-m4",
        ] {
            assert!(!names_version_3_1_or_later(id), "{id}");
        }
    }
}
