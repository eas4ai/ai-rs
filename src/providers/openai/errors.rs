//! OpenAI error mapping into the shared provider hierarchy.
use crate::{Error, providers::http, transport::TransportError};

/// How the OpenAI API authenticates and reports errors.
pub(crate) const API: http::ApiProfile = http::ApiProfile {
    label: "OpenAI",
    auth: http::Auth::Bearer,
    default_headers: &[],
    request_id_header: "x-request-id",
    collapse_server_errors: true,
    keep_error_object: true,
    top_level_first: false,
    code_prefixes_error: false,
};

/// Map a transport failure to a provider error that keeps it as the source.
pub fn map_transport(error: TransportError, provider: &str) -> Error {
    http::transport_error(&API, provider, error)
}
