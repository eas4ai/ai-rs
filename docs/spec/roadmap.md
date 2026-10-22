# Roadmap

Current: rename-to-suprnova-ai

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
