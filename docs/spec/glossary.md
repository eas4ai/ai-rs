# Glossary

**Package.** The Cargo package that `Cargo.toml` defines: `suprnova-ai`.
Before the rename it was `ai`.

**Library crate.** The crate other Rust code imports: `suprnova_ai`. Cargo
derives this name from the package name by changing `-` to `_`.

**Crate path.** A Rust path that starts with the library crate's name, such
as `suprnova_ai::Model`. Before the rename these paths started with `ai::`.

**Repository.** The Git repository on GitHub,
`https://github.com/eas4ai/suprnova-ai-rs`. Before the rename it was
`eas4ai/ai-rs`; GitHub redirects the old URL.

**Origin remote.** The Git remote named `origin` in a clone. Sudus pushes
its records to it, because it is this project's authority remote.

**Upstream.** The AI SDK for Python 0.8.0 at
`https://github.com/vercel-labs/ai-python`, which this crate ports. Upstream
names such as the Python package `ai` and the module `ai.types` keep their
spelling here.

**Model id.** A `provider:model` string that `Model` parses. It splits at the
first colon; an id with no colon routes to the `gateway` provider.

**Provider.** An implementation of the `Provider` trait for one vendor API.
The `ProviderRegistry` maps provider names to providers and starts empty.

**Event.** One item of a model stream, a variant of the `Event` enum. It
serializes with the upstream `kind` discriminator.

**Agent.** The `Agent` type: it streams a model, runs the requested tools,
appends the results to the history, and repeats until the model gives a
final answer.

**Transport.** An implementation of the `HttpTransport` trait. It sends the
HTTP requests of a provider. The `reqwest` feature adds `ReqwestTransport`.

**UI adapter.** The `ui::ai_sdk` module. It converts Vercel AI SDK v6 UI
messages to core messages, and core events back to UI messages and
server-sent events.

**Check script.** `scripts/check.sh`. It runs the format check, the builds
and tests with no features and with all features, clippy and rustdoc, with
warnings denied.
