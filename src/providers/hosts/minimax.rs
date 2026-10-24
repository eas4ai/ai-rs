//! MiniMax over the Anthropic Messages protocol, at `<base>/v1/messages` with
//! `x-api-key` and `anthropic-version: 2023-06-01`, as `@ai-sdk/minimax`
//! 3.0.62 sends it. The `minimax` options entry is read with the `anthropic`
//! entry's rules and over it, and a call that used it gets its final
//! metadata under `minimax` too. MiniMax 3.1 and later (`MiniMax-M3.1`,
//! `minimax-m4`) always reason: a reasoning effort goes out as
//! `output_config.effort` with `minimal` as `low`, and disabled reasoning
//! becomes effort `low`; an `effort` in either entry wins. They take user
//! images and videos as `image` and `video` blocks with the part's `minimax`
//! `detail` or `fps`, a video `mm_file://` id as a URL, and cache control
//! from `minimax` metadata before `anthropic`. Older models keep the
//! Anthropic mapping and conversion. The video models (`MiniMax-H3`,
//! `MiniMax-H3-Max`) run as jobs through `ops::generate_video` at the root
//! of the base URL, and need a download transport for the finished video
//! (`register_with_downloads`); a call without one, or without a Tokio
//! timer to poll with, fails before its job starts and bills. The video is
//! read up to 256 MiB. `ops::generate_video` gives each job one video and
//! merges the start's `resolvedInputs` into the completed item's `minimax`
//! metadata; `ops::start_video` with an `n` above 1 warns and starts one
//! job, and a status checked later from its `VideoJob` through
//! `ops::video_status` completes without `resolvedInputs`.
use super::{Host, HostProtocol, HostProvider, Models};
use crate::{
    Error, ProviderConfig, ProviderRegistry,
    models::files::DownloadTransport,
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
    // MiniMax 3.1 and later take image and video blocks with `detail` and
    // `fps`, and read cache control under `minimax` first (HST-039).
    host_media: names_version_3_1_or_later,
};

pub(super) static HOST: Host = Host {
    name: "minimax",
    handle: "@ai-sdk/minimax",
    default_base_url: "https://api.minimax.io/anthropic",
    api_key_env: "MINIMAX_API_KEY",
    protocol: HostProtocol::Messages(&PROFILE),
    models: Models::MINIMAX,
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
/// snapshot restoration that uses this transport. Its video models need a
/// download transport: use [`register_with_downloads`].
pub fn register(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport, None)
}

/// [`register`] with the download transport that fetches a finished video,
/// which comes from another origin than the base URL; a video call on a
/// provider without one fails with `Error::Validation` before its job
/// starts.
pub fn register_with_downloads(
    registry: &mut ProviderRegistry,
    config: ProviderConfig,
    transport: Arc<dyn HttpTransport>,
    downloads: Arc<dyn DownloadTransport>,
) -> Result<(), Error> {
    super::register(&HOST, registry, config, transport, Some(downloads))
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
