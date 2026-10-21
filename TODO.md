# Core port work

- [x] Implement the Python inference contracts and their Rust test ports.
- [x] Verify the integrated crate, transport features, wire compatibility, and quality gates.
- [ ] In progress: finish the port guide, self-audit, commit on main, and write the delivery report.

The implementation follows Brief P1. Types, tools/errors, and models have separate
file ownership. The integration task owns providers, transport, crate metadata,
and cross-module checks. No remote push is authorized.
