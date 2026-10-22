// Copyright 2026 Vercel, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Tests for MOD-001 to MOD-008 in docs/spec/models.md. Each test name starts
//! with the requirement it checks; scripts/check-core.sh selects them by it.

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

use bytes::Bytes;
use futures::{FutureExt, StreamExt, future::BoxFuture, stream};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use suprnova_ai::{
    Agent, AgentTool, Error, Model, Provider, ProviderConfig, ProviderProtocol, ProviderRegistry,
    RetryPolicy, RunOptions, StreamOptions,
    errors::{HttpErrorContext, ProviderError, ProviderErrorKind},
    generate, get_model,
    providers::{
        ConfiguredProvider, EventStream,
        anthropic::{self, AnthropicCompatibleProvider},
        openai::{self, OpenAICompatibleProvider},
    },
    stream as model_stream,
    testing::{ScriptedProvider, ScriptedResponse},
    transport::{HttpRequest, HttpResponse, HttpTransport, TransportError},
    types::{
        Event, Message, MessageRole,
        events::{StreamEnd, StreamStart, TextDelta, TextStart},
    },
    user_message,
};
use tokio::time::Instant;

// Shared helpers.

/// Records every request and answers each with HTTP 400, so a test can read
/// the headers a provider sent without a real response.
#[derive(Default)]
struct RecordingTransport {
    requests: Mutex<Vec<HttpRequest>>,
}
impl RecordingTransport {
    fn header(&self, index: usize, name: &str) -> Option<String> {
        self.requests.lock().unwrap()[index]
            .headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    }
}
impl HttpTransport for RecordingTransport {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        self.requests.lock().unwrap().push(request);
        async move {
            Ok(HttpResponse {
                status: 400,
                headers: vec![],
                body: stream::iter([Ok(Bytes::from_static(b"{\"error\":{\"message\":\"no\"}}"))])
                    .boxed(),
            })
        }
        .boxed()
    }
}

struct NoProtocol;
impl ProviderProtocol for NoProtocol {
    fn protocol_class_id(&self) -> &str {
        "none"
    }
}

fn configured(name: &str) -> Arc<dyn Provider> {
    Arc::new(ConfiguredProvider::new(
        name,
        ProviderConfig::default(),
        Arc::new(NoProtocol),
        None,
    ))
}

fn no_retry() -> StreamOptions {
    StreamOptions {
        retry: RetryPolicy::none(),
        ..StreamOptions::default()
    }
}

fn status_error(status: u16, message: &str, headers: &[(&str, &str)]) -> Error {
    ProviderError::from_status(message, status, Some("scripted".into()))
        .with_http_context(
            HttpErrorContext::new(status).with_response_headers(
                headers
                    .iter()
                    .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
                    .collect(),
            ),
        )
        .into()
}

fn provider_error(error: &Error) -> &ProviderError {
    match error {
        Error::Provider(error) => error,
        other => panic!("expected a provider error, got {other:?}"),
    }
}

/// Sets its flag when dropped, so a test can see that a stream or future was dropped.
struct DropFlag(Arc<AtomicBool>);
impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// A text response streamed one event at a time, waiting before each event.
fn slow_text_events(delay: Duration, deltas: usize, dropped: Arc<AtomicBool>) -> EventStream {
    let mut events = vec![
        Event::StreamStart(StreamStart::default()),
        Event::TextStart(TextStart::default()),
    ];
    events.extend((0..deltas).map(|_| Event::TextDelta(TextDelta::new("."))));
    events.push(Event::StreamEnd(StreamEnd::default()));
    let guard = DropFlag(dropped);
    stream::unfold(
        (events.into_iter(), guard),
        move |(mut events, guard)| async move {
            tokio::time::sleep(delay).await;
            let event = events.next()?;
            Some((Ok(event), (events, guard)))
        },
    )
    .boxed()
}

/// A provider whose attempts follow a script of delays: attempt n waits
/// `delays[n]` before each event. It counts attempts and records when each
/// attempt's stream is dropped.
struct SlowProvider {
    config: ProviderConfig,
    delays: Vec<Duration>,
    attempts: AtomicUsize,
    dropped: Arc<AtomicBool>,
}
impl SlowProvider {
    fn new(delays: Vec<Duration>) -> Arc<Self> {
        Arc::new(Self {
            config: ProviderConfig::default(),
            delays,
            attempts: AtomicUsize::new(0),
            dropped: Arc::new(AtomicBool::new(false)),
        })
    }
}
#[async_trait::async_trait]
impl Provider for SlowProvider {
    fn name(&self) -> &str {
        "slow"
    }
    fn config(&self) -> &ProviderConfig {
        &self.config
    }
    fn stream(
        &self,
        _model: &Model,
        _messages: Vec<Message>,
        _options: StreamOptions,
    ) -> EventStream {
        let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
        let delay = self.delays[attempt.min(self.delays.len() - 1)];
        self.dropped.store(false, Ordering::SeqCst);
        slow_text_events(delay, 10, self.dropped.clone())
    }
    async fn generate(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
    ) -> Result<Message, Error> {
        let mut events = self.stream(model, messages, options);
        let mut text = String::new();
        while let Some(event) = events.next().await {
            if let Event::TextDelta(delta) = event? {
                text.push_str(&delta.chunk);
            }
        }
        Ok(suprnova_ai::assistant_message(text))
    }
}

async fn drain(model: &Model, options: StreamOptions) -> (Vec<Event>, Option<Error>) {
    let mut response = model_stream(model, vec![user_message("hi")], options);
    let mut events = Vec::new();
    while let Some(item) = response.next().await {
        match item {
            Ok(event) => events.push(event),
            Err(error) => return (events, Some(error)),
        }
    }
    (events, None)
}

// MOD-001

#[test]
fn mod_001_ids_without_a_provider_are_rejected() {
    let mut registry = ProviderRegistry::new();
    registry.register("gateway", configured("gateway")).unwrap();
    match get_model("gpt-5", &registry) {
        Err(Error::Configuration(message)) => {
            assert!(message.contains("gpt-5"), "{message}");
            assert!(message.contains("provider:model"), "{message}");
        }
        other => panic!("expected a configuration error, got {other:?}"),
    }
    let model = get_model("gateway:gpt-5", &registry).unwrap();
    assert_eq!(model.id, "gpt-5");
}

// MOD-002

#[test]
fn mod_002_gateway_handles_are_separate_names() {
    let handles = ["gateway", "ai-gateway", "vercel"];
    let mut registry = ProviderRegistry::new();
    let providers: Vec<_> = handles.iter().map(|handle| configured(handle)).collect();
    for (handle, provider) in handles.iter().zip(&providers) {
        registry.register(*handle, provider.clone()).unwrap();
    }
    for (handle, provider) in handles.iter().zip(&providers) {
        assert!(Arc::ptr_eq(
            &registry.get_provider(handle).unwrap(),
            provider
        ));
    }
    for handle in handles {
        let mut registry = ProviderRegistry::new();
        registry.register(handle, configured(handle)).unwrap();
        for other in handles.iter().filter(|other| **other != handle) {
            assert!(
                matches!(
                    registry.get_provider(other),
                    Err(Error::UnsupportedProvider(_))
                ),
                "{handle} resolved under {other}"
            );
        }
    }
}

// MOD-003

fn secret_config() -> ProviderConfig {
    ProviderConfig {
        api_key_value: Some("sk-direct-secret".into()),
        api_key_env: Some("KEY_VAR".into()),
        env: BTreeMap::from([("KEY_VAR".into(), "sk-env-secret".into())]),
        headers: BTreeMap::from([("authorization".into(), "Bearer header-secret".into())]),
        ..ProviderConfig::default()
    }
}

fn assert_credential_free(model: &Model) {
    let snapshot = model.to_value().unwrap();
    let text = snapshot.to_string();
    for secret in ["sk-direct-secret", "sk-env-secret", "header-secret"] {
        assert!(!text.contains(secret), "{secret} in {text}");
    }
    let provider = snapshot["provider"].as_object().unwrap();
    for field in ["api_key_value", "env", "headers"] {
        assert!(!provider.contains_key(field), "{field} in {text}");
    }
    assert_eq!(snapshot["provider"]["api_key_env"], "KEY_VAR");
}

#[test]
fn mod_003_snapshots_hold_no_credentials() {
    let transport: Arc<dyn HttpTransport> = Arc::new(RecordingTransport::default());
    let configured = ConfiguredProvider::new("custom", secret_config(), Arc::new(NoProtocol), None);
    assert_credential_free(&Model::new("m", Arc::new(configured)));
    let openai = OpenAICompatibleProvider::new(secret_config(), transport.clone());
    assert_credential_free(&Model::new("m", Arc::new(openai)));
    let anthropic = AnthropicCompatibleProvider::new(secret_config(), transport);
    assert_credential_free(&Model::new("m", Arc::new(anthropic)));
}

// MOD-004

fn keyed(key: &str) -> ProviderConfig {
    ProviderConfig {
        api_key_value: Some(key.into()),
        ..ProviderConfig::default()
    }
}

async fn send_once(model: &Model) {
    let mut response = model_stream(model, vec![user_message("hi")], no_retry());
    while response.next().await.is_some() {}
}

#[tokio::test]
async fn mod_004_openai_restore_takes_the_registered_key() {
    let transport = Arc::new(RecordingTransport::default());
    let mut registry = ProviderRegistry::new();
    let live = Arc::new(OpenAICompatibleProvider::new(
        keyed("sk-live"),
        transport.clone(),
    ));
    openai::register(&mut registry, live).unwrap();
    openai::register_factories(&mut registry, transport.clone()).unwrap();

    let old = OpenAICompatibleProvider::new(keyed("sk-old"), transport.clone());
    let snapshot = Model::new("gpt", Arc::new(old)).to_value().unwrap();
    let restored = Model::from_value(snapshot, &registry).unwrap();
    send_once(&restored).await;
    assert_eq!(
        transport.header(0, "authorization").as_deref(),
        Some("Bearer sk-live")
    );
}

#[tokio::test]
async fn mod_004_anthropic_restore_takes_the_registered_key() {
    let transport = Arc::new(RecordingTransport::default());
    let mut registry = ProviderRegistry::new();
    anthropic::register(&mut registry, keyed("sk-live"), transport.clone()).unwrap();

    let old = AnthropicCompatibleProvider::new(keyed("sk-old"), transport.clone());
    let snapshot = Model::new("claude", Arc::new(old)).to_value().unwrap();
    let restored = Model::from_value(snapshot, &registry).unwrap();
    send_once(&restored).await;
    assert_eq!(transport.header(0, "x-api-key").as_deref(), Some("sk-live"));
}

#[test]
fn mod_004_restore_without_a_registered_provider_fails() {
    let transport = Arc::new(RecordingTransport::default());
    let mut registry = ProviderRegistry::new();
    openai::register_factories(&mut registry, transport.clone()).unwrap();
    let old = OpenAICompatibleProvider::new(keyed("sk-old"), transport);
    let snapshot = Model::new("gpt", Arc::new(old)).to_value().unwrap();
    match Model::from_value(snapshot, &registry) {
        Err(Error::Configuration(message)) => assert!(message.contains("\"openai\""), "{message}"),
        other => panic!("expected a configuration error, got {other:?}"),
    }
}

#[test]
fn mod_004_restore_into_an_empty_registry_names_the_provider() {
    let transport: Arc<dyn HttpTransport> = Arc::new(RecordingTransport::default());
    let custom =
        AnthropicCompatibleProvider::new(keyed("sk-old"), transport).with_name("custom-anthropic");
    let snapshot = Model::new("claude", Arc::new(custom)).to_value().unwrap();
    match Model::from_value(snapshot, &ProviderRegistry::new()) {
        Err(Error::Configuration(message)) => {
            assert!(message.contains("\"custom-anthropic\""), "{message}")
        }
        other => panic!("expected a configuration error, got {other:?}"),
    }
}

// MOD-005

fn two_rate_limits_then_text() -> ScriptedProvider {
    ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::error(status_error(429, "first", &[])),
            ScriptedResponse::error(status_error(429, "second", &[])),
            ScriptedResponse::text("third"),
        ],
    )
}

fn assert_waits_2_then_4_seconds(provider: &ScriptedProvider) {
    let requests = provider.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[1].at - requests[0].at, Duration::from_secs(2));
    assert_eq!(requests[2].at - requests[1].at, Duration::from_secs(4));
}

#[tokio::test(start_paused = true)]
async fn mod_005_stream_retries_twice_waiting_2_then_4_seconds() {
    let provider = two_rate_limits_then_text();
    let model = provider.model("m");
    let mut response = model_stream(&model, vec![user_message("hi")], StreamOptions::default());
    while let Some(event) = response.next().await {
        event.unwrap();
    }
    assert_eq!(response.text(), "third");
    assert_waits_2_then_4_seconds(&provider);
}

#[tokio::test(start_paused = true)]
async fn mod_005_generate_retries_twice_waiting_2_then_4_seconds() {
    let provider = two_rate_limits_then_text();
    let message = generate(
        &provider.model("m"),
        vec![user_message("hi")],
        StreamOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(message.text(), "third");
    assert_waits_2_then_4_seconds(&provider);
}

#[tokio::test(start_paused = true)]
async fn mod_005_retry_after_replaces_the_wait_only_within_limits() {
    for (header, wait) in [("1", 1), ("120", 2)] {
        let provider = ScriptedProvider::new(
            "scripted",
            [
                ScriptedResponse::error(status_error(429, "limited", &[("retry-after", header)])),
                ScriptedResponse::text("ok"),
            ],
        );
        let message = generate(
            &provider.model("m"),
            vec![user_message("hi")],
            StreamOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(message.text(), "ok");
        let requests = provider.requests();
        assert_eq!(
            requests[1].at - requests[0].at,
            Duration::from_secs(wait),
            "retry-after {header}"
        );
    }
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::error(status_error(503, "busy", &[("retry-after-ms", "250")])),
            ScriptedResponse::text("ok"),
        ],
    );
    let (_, error) = drain(&provider.model("m"), StreamOptions::default()).await;
    assert!(error.is_none(), "{error:?}");
    let requests = provider.requests();
    assert_eq!(requests[1].at - requests[0].at, Duration::from_millis(250));
}

#[tokio::test(start_paused = true)]
async fn mod_005_non_retryable_errors_are_returned_at_once() {
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::error(status_error(400, "bad request", &[])),
            ScriptedResponse::text("unused"),
        ],
    );
    let (events, error) = drain(&provider.model("m"), StreamOptions::default()).await;
    assert!(events.is_empty());
    assert_eq!(
        provider_error(&error.unwrap()).kind,
        ProviderErrorKind::BadRequest
    );
    assert_eq!(provider.requests().len(), 1);
    assert_eq!(provider.remaining(), 1);
}

#[tokio::test(start_paused = true)]
async fn mod_005_errors_after_the_first_event_are_not_retried() {
    let partial = ScriptedResponse::events(vec![
        Event::StreamStart(StreamStart::default()),
        Event::TextStart(TextStart::default()),
        Event::TextDelta(TextDelta::new("par")),
    ])
    .then_error(status_error(503, "cut off", &[]));
    let provider = ScriptedProvider::new("scripted", [partial, ScriptedResponse::text("unused")]);
    let (events, error) = drain(&provider.model("m"), StreamOptions::default()).await;
    assert_eq!(events.len(), 3);
    assert_eq!(provider_error(&error.unwrap()).message, "cut off");
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn mod_005_exhausted_retries_return_the_last_error() {
    let provider = ScriptedProvider::new(
        "scripted",
        ["first", "second", "third", "fourth"]
            .map(|message| ScriptedResponse::error(status_error(503, message, &[]))),
    );
    let (_, error) = drain(&provider.model("m"), StreamOptions::default()).await;
    assert_eq!(provider_error(&error.unwrap()).message, "third");
    assert_eq!(provider.requests().len(), 3);
    let error = generate(&provider.model("m"), vec![], no_retry())
        .await
        .unwrap_err();
    assert_eq!(provider_error(&error).message, "fourth");
}

#[tokio::test(start_paused = true)]
async fn mod_005_zero_retries_make_one_attempt() {
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::error(status_error(429, "limited", &[])),
            ScriptedResponse::text("unused"),
        ],
    );
    let error = generate(&provider.model("m"), vec![user_message("hi")], no_retry())
        .await
        .unwrap_err();
    assert_eq!(provider_error(&error).kind, ProviderErrorKind::RateLimit);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn mod_005_transport_connection_failures_are_retried() {
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::error(TransportError::Connection("reset".into())),
            ScriptedResponse::text("ok"),
        ],
    );
    let message = generate(&provider.model("m"), vec![], StreamOptions::default())
        .await
        .unwrap();
    assert_eq!(message.text(), "ok");
    assert_eq!(provider.requests().len(), 2);
}

#[tokio::test(start_paused = true)]
async fn mod_005_an_empty_provider_stream_is_retried() {
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::events(vec![]),
            ScriptedResponse::text("ok"),
        ],
    );
    let model = provider.model("m");
    let mut response = model_stream(&model, vec![user_message("hi")], StreamOptions::default());
    while let Some(event) = response.next().await {
        event.unwrap();
    }
    assert_eq!(response.text(), "ok");
    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].at - requests[0].at, Duration::from_secs(2));

    let provider = ScriptedProvider::new("scripted", [ScriptedResponse::events(vec![])]);
    let (events, error) = drain(&provider.model("m"), no_retry()).await;
    assert!(events.is_empty());
    assert_eq!(
        provider_error(&error.unwrap()).kind,
        ProviderErrorKind::IncompleteResponse
    );
    assert_eq!(provider.requests().len(), 1);
}

#[test]
fn mod_005_without_a_tokio_runtime_a_retryable_error_is_returned_not_retried() {
    let provider = rate_limit_then_text();
    let model = provider.model("m");
    let error = futures::executor::block_on(generate(&model, vec![], StreamOptions::default()))
        .unwrap_err();
    assert_eq!(provider_error(&error).kind, ProviderErrorKind::RateLimit);
    assert_eq!(provider.requests().len(), 1);
}

// MOD-006

fn timeout_kind(error: &Error) -> ProviderErrorKind {
    provider_error(error).kind
}

#[tokio::test(start_paused = true)]
async fn mod_006_idle_timeout_ends_a_silent_attempt_and_drops_its_stream() {
    let provider = SlowProvider::new(vec![Duration::from_secs(31)]);
    let model = Model::new("m", provider.clone());
    let options = StreamOptions {
        idle_timeout: Some(Duration::from_secs(30)),
        ..no_retry()
    };
    let start = Instant::now();
    let (events, error) = drain(&model, options.clone()).await;
    assert!(events.is_empty());
    assert_eq!(timeout_kind(&error.unwrap()), ProviderErrorKind::Timeout);
    assert_eq!(start.elapsed(), Duration::from_secs(30));
    assert!(
        provider.dropped.load(Ordering::SeqCst),
        "the silent stream was not dropped"
    );

    let start = Instant::now();
    let error = generate(&model, vec![], options).await.unwrap_err();
    assert_eq!(timeout_kind(&error), ProviderErrorKind::Timeout);
    assert_eq!(start.elapsed(), Duration::from_secs(30));
}

#[tokio::test(start_paused = true)]
async fn mod_006_total_timeout_ends_the_call() {
    let provider = SlowProvider::new(vec![Duration::from_secs(10)]);
    let model = Model::new("m", provider.clone());
    let options = StreamOptions {
        timeout: Some(Duration::from_secs(60)),
        ..no_retry()
    };
    let start = Instant::now();
    let (events, error) = drain(&model, options).await;
    assert!(!events.is_empty());
    assert_eq!(timeout_kind(&error.unwrap()), ProviderErrorKind::Timeout);
    assert_eq!(start.elapsed(), Duration::from_secs(60));
    assert!(provider.dropped.load(Ordering::SeqCst));
}

#[tokio::test(start_paused = true)]
async fn mod_006_total_timeout_ends_a_buffered_call() {
    let provider = SlowProvider::new(vec![Duration::from_secs(10)]);
    let model = Model::new("m", provider);
    let options = StreamOptions {
        timeout: Some(Duration::from_secs(60)),
        ..no_retry()
    };
    let start = Instant::now();
    let error = generate(&model, vec![], options).await.unwrap_err();
    assert_eq!(timeout_kind(&error), ProviderErrorKind::Timeout);
    assert_eq!(start.elapsed(), Duration::from_secs(60));
}

#[test]
fn mod_006_without_a_tokio_runtime_timeouts_are_a_configuration_error() {
    let provider = ScriptedProvider::new("scripted", [ScriptedResponse::text("unused")]);
    let model = provider.model("m");
    let timed = StreamOptions {
        timeout: Some(Duration::from_secs(1)),
        ..StreamOptions::default()
    };
    let idle = StreamOptions {
        idle_timeout: Some(Duration::from_secs(1)),
        ..StreamOptions::default()
    };
    assert!(matches!(
        futures::executor::block_on(generate(&model, vec![], timed)),
        Err(Error::Configuration(_))
    ));
    let (_, error) = futures::executor::block_on(drain(&model, idle));
    assert!(matches!(error, Some(Error::Configuration(_))), "{error:?}");
    assert!(provider.requests().is_empty());
}

#[tokio::test(start_paused = true)]
async fn mod_006_idle_timeout_before_the_first_event_is_retried() {
    let provider = SlowProvider::new(vec![Duration::from_secs(31), Duration::ZERO]);
    let model = Model::new("m", provider.clone());
    let options = StreamOptions {
        idle_timeout: Some(Duration::from_secs(30)),
        ..StreamOptions::default()
    };
    let start = Instant::now();
    let (events, error) = drain(&model, options).await;
    assert!(error.is_none(), "{error:?}");
    assert!(matches!(events.last(), Some(Event::StreamEnd(_))));
    assert_eq!(provider.attempts.load(Ordering::SeqCst), 2);
    // 30 s of silence, then the 2 s wait before the first retry.
    assert_eq!(start.elapsed(), Duration::from_secs(32));
}

#[tokio::test(start_paused = true)]
async fn mod_006_calls_have_no_timeout_by_default() {
    let provider = SlowProvider::new(vec![Duration::from_secs(7200)]);
    let model = Model::new("m", provider);
    let mut response = model_stream(&model, vec![], StreamOptions::default());
    let waited = tokio::time::timeout(Duration::from_secs(3600), response.next()).await;
    assert!(waited.is_err(), "the call ended by itself: {waited:?}");
}

// MOD-007

fn rate_limit_then_text() -> ScriptedProvider {
    ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::error(status_error(429, "limited", &[])),
            ScriptedResponse::text("never"),
        ],
    )
}

#[tokio::test(start_paused = true)]
async fn mod_007_dropping_a_stream_in_its_retry_wait_sends_no_more_requests() {
    let provider = rate_limit_then_text();
    let mut response = model_stream(&provider.model("m"), vec![], StreamOptions::default());
    // The first attempt fails at once, and the call then waits 2 s before retrying.
    assert!(
        tokio::time::timeout(Duration::from_secs(1), response.next())
            .await
            .is_err()
    );
    drop(response);
    tokio::time::sleep(Duration::from_secs(10)).await;
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn mod_007_dropping_a_generate_future_in_its_retry_wait_sends_no_more_requests() {
    let provider = rate_limit_then_text();
    let model = provider.model("m");
    let pending = generate(&model, vec![], StreamOptions::default());
    // The timeout drops the pending future after 1 s of its 2 s wait.
    assert!(
        tokio::time::timeout(Duration::from_secs(1), pending)
            .await
            .is_err()
    );
    tokio::time::sleep(Duration::from_secs(10)).await;
    assert_eq!(provider.requests().len(), 1);
}

#[derive(Deserialize, JsonSchema)]
struct NoInput {}

/// A tool that marks when it starts and when its future is dropped, and never finishes.
fn hanging_tool(started: Arc<AtomicBool>, dropped: Arc<AtomicBool>) -> AgentTool {
    AgentTool::new::<NoInput, u8, _, _>("hang", "Never finishes", move |_, _| {
        let started = started.clone();
        let guard = DropFlag(dropped.clone());
        async move {
            let _guard = guard;
            started.store(true, Ordering::SeqCst);
            futures::future::pending::<()>().await;
            Ok(0)
        }
    })
    .unwrap()
}

#[tokio::test]
async fn mod_007_dropping_an_agent_run_drops_its_running_tools() {
    for loop_buffer in [Some(0), None] {
        let started = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        let provider = ScriptedProvider::new(
            "scripted",
            [ScriptedResponse::tool_call("c1", "hang", json!({}))],
        );
        let agent = Agent::new(vec![hanging_tool(started.clone(), dropped.clone())]);
        let mut run = agent.run_with_options(
            provider.model("m"),
            vec![user_message("go")],
            RunOptions {
                loop_buffer,
                ..RunOptions::default()
            },
        );
        while !started.load(Ordering::SeqCst) {
            let _ = tokio::time::timeout(Duration::from_millis(5), run.next()).await;
        }
        assert!(!dropped.load(Ordering::SeqCst));
        drop(run);
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert!(
            dropped.load(Ordering::SeqCst),
            "tool kept running, buffer {loop_buffer:?}"
        );
    }
}

// MOD-008

#[derive(Deserialize, JsonSchema)]
struct Double {
    x: i64,
}

#[tokio::test]
async fn mod_008_scripted_provider_drives_an_agent_and_records_requests() {
    let provider = ScriptedProvider::new(
        "scripted",
        [
            ScriptedResponse::tool_call("c1", "double", json!({"x": 7})),
            ScriptedResponse::text("14"),
        ],
    );
    let double =
        AgentTool::new::<Double, i64, _, _>("double", "Double a number", |input, _| async move {
            Ok(input.x * 2)
        })
        .unwrap();
    let mut run = Agent::new(vec![double]).run(provider.model("m"), vec![user_message("Double 7")]);
    while let Some(event) = run.next().await {
        event.unwrap();
    }
    assert_eq!(run.text().unwrap(), "14");
    let history = run.messages();
    let requests = provider.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].model_id, "m");
    assert_eq!(requests[0].messages, history[..1]);
    assert_eq!(requests[1].messages, history[..3]);
    assert_eq!(history[2].role, MessageRole::Tool);
    let tools: Vec<Value> = requests[1]
        .options
        .tools
        .iter()
        .map(|tool| serde_json::to_value(tool).unwrap())
        .collect();
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], "double", "{tools:?}");
}

#[tokio::test]
async fn mod_008_scripted_errors_reach_the_caller_unchanged() {
    let provider = ScriptedProvider::new(
        "scripted",
        [ScriptedResponse::error(status_error(400, "bad input", &[]))],
    );
    let error = generate(&provider.model("m"), vec![], StreamOptions::default())
        .await
        .unwrap_err();
    let error = provider_error(&error);
    assert_eq!(error.kind, ProviderErrorKind::BadRequest);
    assert_eq!(error.message, "bad input");
}

#[tokio::test]
async fn mod_008_retryable_scripted_errors_reach_the_caller_with_retries_off() {
    for error in [
        status_error(429, "limited", &[]),
        status_error(503, "busy", &[]),
    ] {
        let message = provider_error(&error).message.clone();
        let provider = ScriptedProvider::new("scripted", [ScriptedResponse::error(error)]);
        let failure = generate(&provider.model("m"), vec![], no_retry())
            .await
            .unwrap_err();
        assert_eq!(provider_error(&failure).message, message);
        assert_eq!(provider.requests().len(), 1);
    }
    let provider = ScriptedProvider::new(
        "scripted",
        [ScriptedResponse::error(TransportError::Connection(
            "reset".into(),
        ))],
    );
    let (_, failure) = drain(&provider.model("m"), no_retry()).await;
    assert!(
        matches!(
            failure,
            Some(Error::Transport(TransportError::Connection(_)))
        ),
        "{failure:?}"
    );
}

#[tokio::test]
async fn mod_008_calls_after_the_script_fail_with_a_configuration_error() {
    let provider = ScriptedProvider::new("scripted", [ScriptedResponse::text("only")]);
    let model = provider.model("m");
    generate(&model, vec![], StreamOptions::default())
        .await
        .unwrap();
    assert!(matches!(
        generate(&model, vec![], StreamOptions::default()).await,
        Err(Error::Configuration(_))
    ));
    let (_, error) = drain(&model, StreamOptions::default()).await;
    assert!(matches!(error, Some(Error::Configuration(_))));
}
