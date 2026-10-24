#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![doc(test(attr(deny(warnings), allow(dead_code))))]

pub mod agents;
pub mod errors;
pub mod models;
pub mod ops;
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
// Copyright 2026 Shawn McAllister
