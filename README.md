# ai

Rust port of the inference core in AI SDK for Python 0.8.0.

The crate provides messages, events, tool schemas, stream aggregation, provider
contracts, and streaming HTTP transport. Concrete provider adapters and agent loops
will be added separately. The provider registry starts empty.

```rust
use std::sync::Arc;
use ai::{Model, ProviderConfig, ProviderProtocol, StreamOptions, stream};
use ai::providers::{ConfiguredProvider, EventStream};
use ai::types::{Event, Message};
use ai::types::events::{StreamStart, StreamEnd, TextStart, TextDelta};
use futures::StreamExt;

struct ExampleProtocol;

#[async_trait::async_trait]
impl ProviderProtocol for ExampleProtocol {
    fn protocol_class_id(&self) -> &str { "example" }

    fn stream(
        &self,
        _model: &Model,
        _messages: Vec<Message>,
        _options: StreamOptions,
        _provider: &dyn ai::Provider,
    ) -> EventStream {
        futures::stream::iter([
            Ok(Event::StreamStart(StreamStart::default())),
            Ok(Event::TextStart(TextStart::default())),
            Ok(Event::TextDelta(TextDelta::new("Hello"))),
            Ok(Event::StreamEnd(StreamEnd::default())),
        ]).boxed()
    }
}

# async fn example() -> Result<(), ai::Error> {
let provider = Arc::new(ConfiguredProvider::new(
    "example", ProviderConfig::default(), Arc::new(ExampleProtocol), None,
));
let model = Model::new("example-model", provider);
let mut response = stream(&model, vec![ai::user_message("Hello")], StreamOptions::default());
while let Some(event) = response.next().await {
    event?;
}
assert_eq!(response.text(), "Hello");
# Ok(())
# }
```

Use `StreamOptions::with_output::<T>()` with a `schemars::JsonSchema` type to send
an output schema. Read the result with `response.output::<T>()` for a
`serde::Deserialize` type. `generate` and `experimental_generate` return an
assistant message. They use native provider generation when available and fall
back to streaming when that operation is unsupported.

Provider configuration is explicit. The library does not read process environment
variables. Model IDs use `provider:model`; IDs without a colon use the registered
`gateway` provider. The aliases `gateway`, `ai-gateway`, and `vercel` share a
registry entry.

The default feature set contains no HTTP client. Implement `HttpTransport` to
use an existing client, or enable the optional `reqwest` feature. Media downloads
use a separate `DownloadTransport` contract that pins validated DNS addresses.

Run `./scripts/check.sh` for the pinned toolchain build, test, lint, and documentation checks.

See [docs/PORT.md](docs/PORT.md) for compatibility, test coverage, and deferred work.
