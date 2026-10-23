//! Provider and protocol extension points. Concrete adapters register explicitly.

pub mod anthropic;
pub mod base;
pub mod google;
pub mod google_vertex;
pub mod hosts;
pub(crate) mod http;
pub mod openai;
pub mod operations;
pub use base::{
    ConfiguredProvider, EventStream, ModelsDevProvider, Provider, ProviderConfig, ProviderProtocol,
    ProviderRegistry, get_provider,
};
