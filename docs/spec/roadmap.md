# Roadmap

Current: dialect-hosts

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

## chat-hosts

Requirements: HST-001, HST-002, HST-003, HST-004, HST-005, HST-006, HST-007, AGT-001, AGT-006, PKG-002, PKG-006

Add the first hosted providers on one shared provider layer. Move the parts
the OpenAI and Anthropic adapters each carry today into shared code under
`src/providers/`: the HTTP request builder (URL joins, query, auth and
header order, the bounded body read), the error mapping, and the SSE to JSON
reader. Their behavior does not change, so today's tests keep passing.
Give the Chat Completions protocol a host profile for the request and
response differences in `docs/spec/hosts.md`, and stop choosing the default
protocol by the provider name. Add `providers::hosts` with Cerebras,
DeepInfra, Fireworks, Together AI and Z.ai on Chat Completions and MiniMax on
the Anthropic Messages protocol, each restorable from a snapshot. Send
`stream_options.include_usage` only where HST-002 says. In the agent loop,
skip step checks after a step that ends on a deferred hook, count progress
across every run of the loop, and add `AgentStream::interrupted_messages()`.
Update `README.md`, `src/agents/README.md` and `docs/PORT.md`. Each
requirement's tests in `tests/spec_hosts.rs` and `tests/spec_agents.rs` are
named after it; `scripts/check-hosts.sh` and `scripts/check-core.sh` report
one result per requirement through the shared `scripts/spec-tests.sh`.

Done when the hosts, core, identity and checks mechanisms pass for every
requirement above, their receipts are current, and the review and the
adversary report have no open findings.

## dialect-hosts

Requirements: HST-001, HST-002, HST-008, HST-009, HST-010, HST-011, HST-012, HST-013, HST-014, HST-015, HST-016, HST-017, HST-018, MOD-009, AGT-006, PKG-002, PKG-006

Add DeepSeek, Moonshot AI and Mistral to `providers::hosts` on the Chat
Completions protocol, each restorable from a snapshot. Their profiles also
change how messages, schemas and stream content are written and read: the
DeepSeek `thinking` field, V4 reasoning history and schema system message;
the Moonshot AI model families, MFJS schemas, video input and image
downloads through a provider built with a download transport; and the
Mistral seed, tool choice, thinking chunks, document parts and content
arrays. Each rejects the settings its TypeScript provider never sends. Key
Chat Completions tool call deltas without an index by their position, and
give the three hosts' stream errors a kind and a retry flag. Give
`ReqwestTransport::new()` only a 10 s connect limit and TCP keepalive. Make
`AgentStream::interrupted_messages()` keep the loop's copy of a response up
to its last received tool call, with the results of the tools that
finished, and add `AgentStream::stop()`. Update `README.md`,
`src/agents/README.md` and `docs/PORT.md`. Each requirement's tests in
`tests/spec_hosts.rs`, `tests/spec_models.rs` and `tests/spec_agents.rs`
are named after it; `scripts/check-hosts.sh` and `scripts/check-core.sh`
report one result per requirement through the shared
`scripts/spec-tests.sh`.

Done when the hosts, core, identity and checks mechanisms pass for every
requirement above, their receipts are current, and the review and the
adversary report have no open findings.
