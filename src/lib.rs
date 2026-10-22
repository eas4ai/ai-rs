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

#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![doc(test(attr(deny(warnings), allow(dead_code))))]

pub mod agents;
pub mod errors;
pub mod models;
pub mod providers;
#[cfg(feature = "testing")]
pub mod testing;
pub mod transport;
pub mod types;
pub mod ui;
pub mod util;

pub use agents::{Agent, AgentStream, AgentTool, Context, HookRegistry, RunOptions};
pub use errors::Error;
pub use models::{
    Model, RetryPolicy, StreamHandle, StreamOptions, experimental_generate, generate, get_model,
    probe, stream,
};
pub use providers::{Provider, ProviderConfig, ProviderProtocol, ProviderRegistry, get_provider};
pub use types::builders::{assistant_message, system_message, tool_message, user_message};
pub use types::{events, messages, tools};
