# Test fixtures

- `vertex-test-service-account-key.pem` is an RSA private key made only for
  `vtx_002_a_service_account_signs_a_jwt_with_its_key_id_email_and_scope` in
  `tests/spec_vertex.rs`, which signs a token with it and reads the token
  back. The key belongs to no Google account, and the test's service-account
  email is made up, so it signs in to nothing. Secret scanners still report a
  private key here; `.gitleaks.toml` allows this file. The other Vertex AI
  tests build their stand-in keys at run time with `stand_in_key`, around
  bodies that are base64 for `secret` and `secret-key-bytes`.
- `python-0.8.0.json` and `ui-python-0.8.0.json` are written by
  `scripts/generate_python_fixtures.py` and `scripts/generate_ui_fixtures.py`
  from the AI SDK for Python 0.8.0, and read by `tests/python_compat.rs` and
  `tests/ui_python_compat.rs`.
- `openai-streams.json` holds Responses stream events, keyed by the names of
  the Python 0.8.0 tests in `tests/providers/openai/test_adapter.py`; the Rust
  tests `responses_streams_text_usage_and_function_calls` and
  `responses_builtin_tool_call_and_result` in `tests/providers_openai.rs`
  replay them. `body.gz` is a gzip response body for
  `tests/transport_reqwest.rs`. No script or document records where these two
  files came from (`docs/recon.md`, D3).
