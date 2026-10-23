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

//! The Interactions API: Gemini's stateful generation API, which runs
//! models and agents, keeps a server-side record of each interaction and
//! can run in the background.
//!
//! [`protocol`] runs the model the `Model` names; [`agent_protocol`] runs
//! the agent it names, such as `deep-research-pro-preview-12-2025`. Set one
//! on a model of the `google` or `google-vertex` provider:
//!
//! ```
//! use std::sync::Arc;
//! use suprnova_ai::{Model, Provider};
//! use suprnova_ai::providers::google::interactions;
//!
//! fn deep_research(provider: Arc<dyn Provider>) -> Model {
//!     Model::new("deep-research-pro-preview-12-2025", provider)
//!         .with_protocol(interactions::agent_protocol())
//! }
//! ```
//!
//! An agent call, or a call with the option `background`, runs in the
//! background: a buffered call polls the interaction until it ends, and a
//! stream follows it, reconnecting after a dropped connection. Dropping the
//! call before the interaction ends cancels it on Google's side. Every
//! request names the API revision `2026-05-20` unless the configured headers
//! set `Api-Revision`.

mod output;
mod request;
mod stream;

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream as futures_stream};
use serde_json::{Value, json};

use self::stream::Reader;
use super::{
    api::{Endpoint, Flavor, MEDIA_RESPONSE_BYTES},
    transcription::{API_REVISION, has_header},
};
use crate::{
    Error, Model, Provider, ProviderProtocol, StreamOptions,
    errors::{ProviderError, ProviderErrorKind},
    models::{
        call::require_timer,
        params::{ExtraHeaderValue, InferenceRequestParams},
    },
    providers::{
        EventStream,
        http::{self, SseJson},
    },
    transport::{Headers, HttpRequest},
    types::{Event, Message},
};

/// The protocol class id of the Interactions protocols in snapshots.
pub(crate) const CLASS_ID: &str = "google_interactions";

const FIRST_POLL: Duration = Duration::from_secs(1);
const MAX_POLL: Duration = Duration::from_secs(10);
const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_secs(30 * 60);
/// Consecutive failed attempts after which a background stream gives up.
const STREAM_ATTEMPTS: u32 = 3;
const STREAM_RETRY: Duration = Duration::from_millis(500);

/// The Interactions protocol for a model or an agent.
#[derive(Clone, Copy, Debug, Default)]
pub struct InteractionsProtocol {
    agent: bool,
}

/// The Interactions protocol for the model the `Model` names.
#[must_use]
pub fn protocol() -> Arc<dyn ProviderProtocol> {
    Arc::new(InteractionsProtocol { agent: false })
}

/// The Interactions protocol for the agent the `Model` names, built-in or
/// managed.
#[must_use]
pub fn agent_protocol() -> Arc<dyn ProviderProtocol> {
    Arc::new(InteractionsProtocol { agent: true })
}

/// Restore an Interactions protocol from its snapshot.
pub(crate) fn restore(value: &Value) -> Arc<dyn ProviderProtocol> {
    Arc::new(InteractionsProtocol {
        agent: value.get("agent") == Some(&Value::Bool(true)),
    })
}

/// What an Interactions call needs: the endpoint at the Interactions base,
/// and the headers and query every request of the call adds, the call's
/// own included, as TypeScript sends its call headers on each request.
#[derive(Clone)]
struct Ctx {
    endpoint: Endpoint,
    headers: BTreeMap<String, String>,
    query: BTreeMap<String, Value>,
}

impl Ctx {
    fn new(
        provider: &dyn Provider,
        params: Option<&InferenceRequestParams>,
    ) -> Result<Self, Error> {
        let flavor = if provider.provider_class_id() == "google-vertex" {
            Flavor::Vertex
        } else {
            Flavor::Gemini
        };
        if flavor.is_vertex() {
            if crate::providers::google_vertex::express(provider.config()) {
                return Err(crate::providers::google_vertex::express_error(
                    "Interactions models",
                ));
            }
            crate::providers::google_vertex::base_url(provider.config())?;
        }
        let mut endpoint = Endpoint::of(flavor, provider)?;
        if flavor.is_vertex() {
            // Interactions are under the location, not the publisher.
            let base = endpoint.base.trim_end_matches('/');
            let base = base.strip_suffix("/publishers/google").unwrap_or(base);
            endpoint.base = base.to_owned();
        }
        let mut headers = BTreeMap::new();
        if !has_header(&endpoint.config.headers, "api-revision") {
            headers.insert("Api-Revision".to_owned(), API_REVISION.to_owned());
        }
        let params = params.cloned().unwrap_or_default();
        for (name, value) in params.extra_headers.into_iter().flatten() {
            if let ExtraHeaderValue::Value(value) = value {
                headers.retain(|existing, _| !existing.eq_ignore_ascii_case(&name));
                headers.insert(name, value);
            }
        }
        let query = params.extra_query.unwrap_or_default();
        Ok(Self {
            endpoint,
            headers,
            query,
        })
    }

    fn request(
        &self,
        method: &str,
        path: Vec<String>,
        body: Option<Value>,
        query: BTreeMap<String, Value>,
        accept: Option<&str>,
    ) -> Result<HttpRequest, Error> {
        let mut headers = self.headers.clone();
        if let Some(accept) = accept {
            headers.retain(|existing, _| !existing.eq_ignore_ascii_case("accept"));
            headers.insert("accept".into(), accept.into());
        }
        let mut all = self.query.clone();
        all.extend(query);
        self.endpoint.request(method, &path, body, headers, all)
    }

    fn interaction_path(id: &str) -> Vec<String> {
        vec!["interactions".into(), id.into()]
    }

    async fn create(&self, body: Value, model_id: &str) -> Result<(Headers, Value), Error> {
        let request = self.request(
            "POST",
            vec!["interactions".into()],
            Some(body),
            BTreeMap::new(),
            None,
        )?;
        self.endpoint
            .json(request, Some(model_id), MEDIA_RESPONSE_BYTES)
            .await
    }

    async fn get(&self, id: &str) -> Result<(Headers, Value), Error> {
        let request = self.request(
            "GET",
            Self::interaction_path(id),
            None,
            BTreeMap::new(),
            None,
        )?;
        self.endpoint
            .json(request, None, MEDIA_RESPONSE_BYTES)
            .await
    }

    /// Best-effort cancellation: errors are ignored, so that a failed
    /// cancel cannot hide the error or drop that led to it.
    async fn cancel(&self, id: &str) {
        if id.is_empty() {
            return;
        }
        let mut path = Self::interaction_path(id);
        path.push("cancel".into());
        if let Ok(request) = self.request("POST", path, Some(json!({})), BTreeMap::new(), None) {
            let _ = self.endpoint.send(request, None).await;
        }
    }
}

/// Cancels the interaction when dropped before it ends.
struct CancelGuard {
    ctx: Option<Ctx>,
    id: String,
}

impl CancelGuard {
    fn new(ctx: Ctx, id: String) -> Self {
        Self { ctx: Some(ctx), id }
    }
    fn disarm(&mut self) {
        self.ctx = None;
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx.take()
            && let Ok(handle) = tokio::runtime::Handle::try_current()
        {
            let id = std::mem::take(&mut self.id);
            handle.spawn(async move { ctx.cancel(&id).await });
        }
    }
}

fn header_tier(headers: &Headers) -> Option<String> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("x-gemini-service-tier"))
        .map(|(_, value)| value.clone())
}

fn status(response: &Value) -> Option<&str> {
    response.get("status").and_then(Value::as_str)
}

fn interaction_id(ctx: &Ctx, response: &Value, what: &str) -> Result<String, Error> {
    response
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| {
            http::response_error(
                &ctx.endpoint.provider,
                format!("google.interactions: the background POST response did not include an interaction id; cannot {what}."),
            )
        })
}

/// Poll an interaction after 1 s, doubling to at most 10 s, until it ends or
/// the timeout passes; a timeout cancels it.
async fn poll(ctx: &Ctx, id: &str, timeout: Duration) -> Result<(Headers, Value), Error> {
    require_timer()?;
    let mut guard = CancelGuard::new(ctx.clone(), id.to_owned());
    let started = tokio::time::Instant::now();
    let mut wait = FIRST_POLL;
    loop {
        if started.elapsed() > timeout {
            guard.disarm();
            ctx.cancel(id).await;
            // A retry would start a new interaction and wait the whole
            // timeout again, so the call must not retry this error.
            return Err(ProviderError::new(
                ProviderErrorKind::Timeout,
                format!(
                    "google.interactions: timed out polling interaction {id} after {}ms.",
                    timeout.as_millis()
                ),
            )
            .with_provider(&ctx.endpoint.provider)
            .with_retryable(false)
            .into());
        }
        tokio::time::sleep(wait).await;
        let (headers, response) = ctx.get(id).await?;
        if output::is_terminal(status(&response)) {
            guard.disarm();
            return Ok((headers, response));
        }
        wait = (wait * 2).min(MAX_POLL);
    }
}

impl InteractionsProtocol {
    async fn generate_message(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: &StreamOptions,
        provider: &dyn Provider,
    ) -> Result<Message, Error> {
        let ctx = Ctx::new(provider, options.params.as_ref())?;
        let built = request::build(
            ctx.endpoint.flavor,
            &model.id,
            self.agent,
            &messages,
            options,
        )?;
        let background = self.agent || built.background;
        let (mut headers, mut response) = ctx.create(json!(built.body), &model.id).await?;
        if background && !output::is_terminal(status(&response)) {
            let id = interaction_id(&ctx, &response, "poll the interaction")?;
            let timeout = built
                .polling_timeout_ms
                .map_or(DEFAULT_POLL_TIMEOUT, Duration::from_millis);
            let (polled_headers, polled) = poll(&ctx, &id, timeout).await?;
            if header_tier(&polled_headers).is_some() {
                headers = polled_headers;
            }
            response = polled;
        }
        Ok(output::parse(&response, header_tier(&headers).as_deref()).message)
    }
}

fn sse(
    ctx: &Ctx,
    body: crate::transport::Body,
) -> futures::stream::BoxStream<'static, Result<SseJson, Error>> {
    http::sse_json_with_limit(
        *ctx.endpoint.flavor.api(),
        ctx.endpoint.provider.clone(),
        body,
        MEDIA_RESPONSE_BYTES,
    )
}

/// A `stream: true` create: read its events to the end.
fn plain_stream(ctx: Ctx, body: crate::transport::Body, tier: Option<String>) -> EventStream {
    struct State {
        source: futures::stream::BoxStream<'static, Result<SseJson, Error>>,
        reader: Reader,
        done: bool,
    }
    let state = State {
        source: sse(&ctx, body),
        reader: Reader::new(ctx.endpoint.provider.clone(), tier),
        done: false,
    };
    futures_stream::try_unfold(state, |mut state| async move {
        loop {
            if let Some(event) = state.reader.events.pop_front() {
                return Ok(Some((event, state)));
            }
            if state.done {
                return Ok(None);
            }
            match state.source.next().await {
                Some(Ok(SseJson::Object(value))) => state.reader.event(&value)?,
                Some(Ok(SseJson::Done)) => {}
                Some(Err(error)) => return Err(error),
                None => {
                    state.reader.end();
                    state.done = true;
                }
            }
        }
    })
    .boxed()
}

/// Follow a background interaction with `GET …?stream=true`, reconnecting
/// with `last_event_id` after a dropped connection.
fn background_stream(ctx: Ctx, id: String, tier: Option<String>) -> EventStream {
    struct State {
        ctx: Ctx,
        id: String,
        last_event_id: Option<String>,
        attempt: u32,
        any_event: bool,
        source: Option<futures::stream::BoxStream<'static, Result<SseJson, Error>>>,
        reader: Reader,
        done: bool,
        guard: CancelGuard,
    }
    impl State {
        async fn open(&mut self) -> Result<(), Error> {
            let mut query = BTreeMap::from([("stream".to_owned(), json!("true"))]);
            if let Some(last) = &self.last_event_id {
                query.insert("last_event_id".into(), json!(last));
            }
            let request = self.ctx.request(
                "GET",
                Ctx::interaction_path(&self.id),
                None,
                query,
                Some("text/event-stream"),
            )?;
            let response = self.ctx.endpoint.send(request, None).await?;
            self.source = Some(sse(&self.ctx, response.body));
            self.any_event = false;
            Ok(())
        }

        /// Count a failed attempt; past the limit the stream fails.
        async fn failed(&mut self, error: Error) -> Result<(), Error> {
            self.source = None;
            self.attempt += 1;
            if self.attempt >= STREAM_ATTEMPTS {
                self.done = true;
                return Err(error);
            }
            require_timer()?;
            tokio::time::sleep(STREAM_RETRY * self.attempt).await;
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<Event>, Error> {
            loop {
                if let Some(event) = self.reader.events.pop_front() {
                    return Ok(Some(event));
                }
                if self.done {
                    return Ok(None);
                }
                let Some(source) = self.source.as_mut() else {
                    if let Err(error) = self.open().await {
                        self.failed(error).await?;
                    }
                    continue;
                };
                match source.next().await {
                    Some(Ok(SseJson::Object(value))) => {
                        self.any_event = true;
                        if let Some(id) = value
                            .get("event_id")
                            .and_then(Value::as_str)
                            .filter(|id| !id.is_empty())
                        {
                            self.last_event_id = Some(id.to_owned());
                        }
                        if let Err(error) = self.reader.event(&value) {
                            self.guard.disarm();
                            self.done = true;
                            return Err(error);
                        }
                        if self.reader.done() {
                            self.guard.disarm();
                        }
                    }
                    Some(Ok(SseJson::Done)) => {}
                    Some(Err(error)) => self.failed(error).await?,
                    None => {
                        self.source = None;
                        if self.reader.done() {
                            self.reader.end();
                            self.done = true;
                        } else if !self.any_event {
                            let error = http::response_error(
                                &self.ctx.endpoint.provider,
                                "google.interactions: SSE stream closed without producing any events.",
                            );
                            self.failed(error).await?;
                        } else {
                            self.attempt = 0;
                        }
                    }
                }
            }
        }
    }
    let guard = CancelGuard::new(ctx.clone(), id.clone());
    let state = State {
        reader: Reader::new(ctx.endpoint.provider.clone(), tier),
        ctx,
        id,
        last_event_id: None,
        attempt: 0,
        any_event: false,
        source: None,
        done: false,
        guard,
    };
    futures_stream::try_unfold(state, |mut state| async move {
        let event = state.next().await?;
        Ok(event.map(|event| (event, state)))
    })
    .boxed()
}

#[async_trait]
impl ProviderProtocol for InteractionsProtocol {
    fn protocol_class_id(&self) -> &str {
        CLASS_ID
    }

    fn to_value(&self) -> Result<Value, Error> {
        Ok(json!({"protocol_class_id": CLASS_ID, "agent": self.agent}))
    }

    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        let ctx = match Ctx::new(provider, options.params.as_ref()) {
            Ok(ctx) => ctx,
            Err(error) => return futures_stream::once(async { Err(error) }).boxed(),
        };
        let agent = self.agent;
        let model_id = model.id.clone();
        futures_stream::once(async move {
            let built = request::build(ctx.endpoint.flavor, &model_id, agent, &messages, &options)?;
            if !(agent || built.background) {
                let mut body = built.body;
                body.insert("stream".into(), json!(true));
                let request = ctx.request(
                    "POST",
                    vec!["interactions".into()],
                    Some(json!(body)),
                    BTreeMap::new(),
                    Some("text/event-stream"),
                )?;
                let response = ctx.endpoint.send(request, Some(&model_id)).await?;
                let tier = header_tier(&response.headers);
                return Ok::<_, Error>(plain_stream(ctx, response.body, tier));
            }
            let (headers, response) = ctx.create(json!(built.body), &model_id).await?;
            let tier = header_tier(&headers);
            let id = interaction_id(&ctx, &response, "stream the result")?;
            if output::is_terminal(status(&response)) {
                let events = output::events(output::parse(&response, tier.as_deref()));
                return Ok(futures_stream::iter(events.into_iter().map(Ok)).boxed());
            }
            Ok(background_stream(ctx, id, tier))
        })
        .try_flatten()
        .boxed()
    }

    async fn generate(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> Result<Message, Error> {
        self.generate_message(model, messages, &options, provider)
            .await
    }
}
