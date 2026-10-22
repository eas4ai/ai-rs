# Roadmap

Current: core-hardening

## rename-to-suprnova-ai

Requirements: PKG-001, PKG-002, PKG-003, PKG-004, PKG-005, PKG-006

Rename the crate from `ai` to `suprnova-ai` (library `suprnova_ai`) and the
repository from `eas4ai/ai-rs` to `eas4ai/suprnova-ai-rs`. Change the package
name in `Cargo.toml`, rewrite every `ai::` crate path in `src/`, `tests/`,
`README.md` and `docs/PORT.md`, retitle `README.md`, set the `Cargo.toml`
repository URL, reword `NOTICE`, move the `scripts/check.sh` build directory
to a `suprnova-ai-rs` path, and regenerate `Cargo.lock`. The public API does
not change apart from the crate name. suprchat is not changed: it pins an
older revision by Git URL, and GitHub redirects that URL.

Done when the identity, remote and checks mechanisms pass for every
requirement above, their receipts are current, and the review and the
adversary report have no open findings.

## core-hardening

Requirements: MOD-001, MOD-002, MOD-003, MOD-004, MOD-005, MOD-006, MOD-007, MOD-008, AGT-001, AGT-002, AGT-003, AGT-004, AGT-005, PKG-002, PKG-006

Harden the core before the provider port builds on it. Remove the gateway
remnants: the default `gateway` route in `get_model`, the alias folding in
`canonical_provider_id`, and the gateway-only `routing` and `tags` request
parameters with their types. Make model snapshots credential-free and
restore credentials from the registered provider of the same name. Add
call retries with backoff that honors `retry-after`, optional idle and
total timeouts, and drop-safe retry waits to `stream` and `generate`. Add
the `testing` feature with a scripted provider. In the agent loop, add
async step checks that continue, steer or stop, turn calls to unknown tools
into error results the model sees, add an optional tool concurrency cap,
total run usage and the run's stop reason. Update `README.md`,
`src/agents/README.md` and `docs/PORT.md` to state the departures from
Python. Each requirement's tests in `tests/spec_models.rs` and
`tests/spec_agents.rs` are named after it, and `scripts/check-core.sh`
reports one result per requirement.

Done when the core, identity and checks mechanisms pass for every
requirement above, their receipts are current, and the review and the
adversary report have no open findings.
