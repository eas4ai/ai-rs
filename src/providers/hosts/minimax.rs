// Copyright 2026 Vercel, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! MiniMax over the Anthropic Messages protocol, at `<base>/v1/messages` with
//! `x-api-key` and `anthropic-version: 2023-06-01`. MiniMax 3.1 and later
//! (`MiniMax-M3.1`, `minimax-m4`) always reason: a reasoning effort goes out as
//! `output_config.effort` with `minimal` as `low`, and disabled reasoning
//! becomes effort `low`. Older models keep the Anthropic mapping.
use super::{Host, HostProtocol, HostProvider};
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
    },
    always_reasons: |_| false,
};

static HOST: Host = Host {
    name: "minimax",
    handle: "@ai-sdk/minimax",
    default_base_url: "https://api.minimax.io/anthropic",
    api_key_env: "MINIMAX_API_KEY",
    protocol: HostProtocol::Messages(&PROFILE),
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
    super::register(&HOST, registry, config, transport)
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
