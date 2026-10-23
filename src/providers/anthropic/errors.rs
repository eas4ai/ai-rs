use crate::{
    Error,
    errors::{ProviderError, ProviderErrorKind},
    providers::http,
};
use serde_json::Value;

/// How the Anthropic API authenticates and reports errors.
pub(crate) const API: http::ApiProfile = http::ApiProfile {
    label: "Anthropic",
    auth: http::Auth::Header("x-api-key"),
    default_headers: &[("anthropic-version", "2023-06-01")],
    request_id_header: "request-id",
    collapse_server_errors: false,
    keep_error_object: false,
    top_level_first: true,
    code_prefixes_error: false,
    config_headers: &[],
};

/// An `error` event inside a Messages stream.
pub(super) fn stream_error(api: &http::ApiProfile, provider: &str, body: Value) -> Error {
    let mut error = ProviderError::new(
        ProviderErrorKind::Api,
        http::error_message(api, &body)
            .map(str::to_owned)
            .unwrap_or_else(|| format!("{} stream failed", api.label)),
    )
    .with_provider(provider);
    error.code = http::error_field(api, &body, "code");
    error.param = http::error_field(api, &body, "param");
    error.error_type = http::error_field(api, &body, "type");
    error.body = Some(body);
    error.into()
}
