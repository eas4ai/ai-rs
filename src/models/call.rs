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

//! Retries and timeouts around one model call.
//!
//! A call retries only before the provider stream yields its first event, so a
//! consumer never receives the same output twice. Waits and timeouts use the
//! Tokio timer; dropping the call drops any pending wait.

use std::time::{Duration, SystemTime};

use futures::{StreamExt, future::BoxFuture, stream};
use tokio::time::Instant;

use crate::{
    errors::{Error, ProviderError, ProviderErrorKind, Result},
    providers::EventStream,
    transport::TransportError,
    types::{Event, Message},
};

use super::{Model, StreamOptions};

/// Header values above this wait are ignored unless they are shorter than the
/// computed backoff, as in the TypeScript SDK.
const MAX_HEADER_DELAY: Duration = Duration::from_secs(60);

/// How a call retries a retryable error that arrives before its first event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Retries after the first attempt. Zero disables retries.
    pub max_retries: u32,
    /// Wait before the first retry.
    pub initial_delay: Duration,
    /// Multiplier applied to the wait before each later retry.
    pub backoff_factor: u32,
}

impl Default for RetryPolicy {
    /// Two retries, waiting 2 s and then 4 s.
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_delay: Duration::from_secs(2),
            backoff_factor: 2,
        }
    }
}

impl RetryPolicy {
    /// A policy that never retries.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            max_retries: 0,
            initial_delay: Duration::ZERO,
            backoff_factor: 1,
        }
    }

    /// The wait before retry `retry` (zero-based), ignoring response headers.
    #[must_use]
    pub fn backoff(&self, retry: u32) -> Duration {
        self.initial_delay
            .saturating_mul(self.backoff_factor.saturating_pow(retry))
    }
}

/// Whether a call may retry this error.
#[must_use]
pub fn is_retryable(error: &Error) -> bool {
    match error {
        Error::Provider(error) => error.is_retryable,
        Error::Transport(TransportError::Connection(_) | TransportError::Timeout) => true,
        _ => false,
    }
}

/// The wait before retry `retry`, honoring `retry-after-ms` and `retry-after`.
fn retry_delay(policy: &RetryPolicy, retry: u32, error: &Error, now: SystemTime) -> Duration {
    let backoff = policy.backoff(retry);
    match header_delay(error, now) {
        Some(delay) if delay < MAX_HEADER_DELAY || delay < backoff => delay,
        _ => backoff,
    }
}

/// The wait a provider asked for, or `None` when it asked for none or for a
/// negative one.
fn header_delay(error: &Error, now: SystemTime) -> Option<Duration> {
    let Error::Provider(error) = error else {
        return None;
    };
    let headers = &error.http_context.as_ref()?.response_headers;
    let header = |name: &str| {
        headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.trim())
    };
    if let Some(milliseconds) = header("retry-after-ms").and_then(finite) {
        return Duration::try_from_secs_f64(milliseconds / 1000.0).ok();
    }
    let value = header("retry-after")?;
    if let Some(seconds) = finite(value) {
        return Duration::try_from_secs_f64(seconds).ok();
    }
    httpdate::parse_http_date(value)
        .ok()?
        .duration_since(now)
        .ok()
}

fn finite(value: &str) -> Option<f64> {
    value.parse::<f64>().ok().filter(|value| value.is_finite())
}

fn timeout_error(model: &Model, message: String) -> Error {
    ProviderError::new(ProviderErrorKind::Timeout, message)
        .with_provider(model.provider.name())
        .into()
}

fn total_timeout_error(model: &Model, timeout: Duration) -> Error {
    timeout_error(
        model,
        format!("the model call exceeded its {timeout:?} timeout"),
    )
}

/// Retries and timeouts need the Tokio timer. Report a missing runtime instead
/// of letting Tokio panic. Tokio offers no check for a runtime built without
/// its time driver; in such a runtime the first wait panics, as every Tokio
/// timer does, so the README requires a runtime with the time driver enabled.
fn require_timer() -> Result<()> {
    tokio::runtime::Handle::try_current()
        .map(|_| ())
        .map_err(|_| {
            Error::Configuration(
                "model call retries and timeouts require a Tokio runtime".to_owned(),
            )
        })
}

enum Wait<T> {
    Ready(T),
    Idle,
    Deadline,
}

/// Await `future` until the idle timeout or the call deadline, whichever ends first.
async fn bounded<T>(
    future: impl Future<Output = T>,
    idle: Option<Duration>,
    deadline: Option<Instant>,
) -> Wait<T> {
    let idle_deadline = idle.map(|idle| Instant::now() + idle);
    let limit = match (idle_deadline, deadline) {
        (Some(idle), Some(total)) => Some((idle.min(total), idle < total)),
        (Some(idle), None) => Some((idle, true)),
        (None, Some(total)) => Some((total, false)),
        (None, None) => None,
    };
    match limit {
        None => Wait::Ready(future.await),
        Some((at, is_idle)) => match tokio::time::timeout_at(at, future).await {
            Ok(value) => Wait::Ready(value),
            Err(_) if is_idle => Wait::Idle,
            Err(_) => Wait::Deadline,
        },
    }
}

/// Sleep for the retry wait, or until the call deadline. Returns false when the
/// deadline ended the wait.
async fn wait_for_retry(wait: Duration, deadline: Option<Instant>) -> bool {
    let wake = Instant::now() + wait;
    match deadline {
        Some(deadline) if deadline <= wake => {
            tokio::time::sleep_until(deadline).await;
            false
        }
        _ => {
            tokio::time::sleep_until(wake).await;
            true
        }
    }
}

struct StreamCall {
    model: Model,
    messages: Vec<Message>,
    options: StreamOptions,
    attempt: Option<EventStream>,
    retries: u32,
    started: bool,
    deadline: Option<Instant>,
    finished: bool,
}

impl StreamCall {
    fn new(
        model: Model,
        messages: Vec<Message>,
        options: StreamOptions,
        deadline: Option<Instant>,
    ) -> Self {
        Self {
            model,
            messages,
            options,
            attempt: None,
            retries: 0,
            started: false,
            deadline,
            finished: false,
        }
    }

    fn uses_timer(&self) -> bool {
        self.options.timeout.is_some() || self.options.idle_timeout.is_some()
    }

    async fn next(&mut self) -> Option<Result<Event>> {
        if self.finished {
            return None;
        }
        if self.deadline.is_none()
            && let Some(timeout) = self.options.timeout
        {
            if let Err(error) = require_timer() {
                self.finished = true;
                return Some(Err(error));
            }
            self.deadline = Some(Instant::now() + timeout);
        } else if self.uses_timer()
            && let Err(error) = require_timer()
        {
            self.finished = true;
            return Some(Err(error));
        }
        loop {
            let attempt = self.attempt.get_or_insert_with(|| {
                self.model
                    .provider
                    .stream(&self.model, self.messages.clone(), self.options.clone())
            });
            let item = match bounded(attempt.next(), self.options.idle_timeout, self.deadline).await
            {
                Wait::Ready(item) => item,
                Wait::Idle => {
                    // An idle attempt is over; a retry starts a fresh one.
                    self.attempt = None;
                    let idle = self.options.idle_timeout.unwrap_or_default();
                    let error = timeout_error(
                        &self.model,
                        format!("the provider sent no event for {idle:?}"),
                    );
                    if self.started {
                        self.finished = true;
                        return Some(Err(error));
                    }
                    Some(Err(error))
                }
                Wait::Deadline => {
                    self.attempt = None;
                    self.finished = true;
                    let timeout = self.options.timeout.unwrap_or_default();
                    return Some(Err(total_timeout_error(&self.model, timeout)));
                }
            };
            let item = match item {
                // An attempt that ends before any event is an incomplete
                // response that arrived before the first event: retry it.
                None if !self.started => {
                    self.attempt = None;
                    Some(Err(ProviderError::incomplete_response(
                        "the provider stream ended without any event; the response is incomplete",
                        Some(self.model.provider.name().to_owned()),
                    )
                    .into()))
                }
                item => item,
            };
            match item {
                None => {
                    self.attempt = None;
                    self.finished = true;
                    return None;
                }
                Some(Ok(event)) => {
                    self.started = true;
                    return Some(Ok(event));
                }
                Some(Err(error)) => {
                    let policy = &self.options.retry;
                    if self.started || self.retries >= policy.max_retries || !is_retryable(&error) {
                        if self.attempt.is_none() {
                            // The idle attempt was dropped and will not resume.
                            self.finished = true;
                        }
                        return Some(Err(error));
                    }
                    if require_timer().is_err() {
                        // Without a timer the call cannot wait, so it returns the error.
                        return Some(Err(error));
                    }
                    let wait = retry_delay(policy, self.retries, &error, SystemTime::now());
                    self.retries += 1;
                    self.attempt = None;
                    if !wait_for_retry(wait, self.deadline).await {
                        self.finished = true;
                        let timeout = self.options.timeout.unwrap_or_default();
                        return Some(Err(total_timeout_error(&self.model, timeout)));
                    }
                }
            }
        }
    }
}

/// The provider stream for one call, with its retries and timeouts.
pub(crate) fn call_stream(
    model: &Model,
    messages: Vec<Message>,
    options: StreamOptions,
) -> EventStream {
    call_stream_until(model, messages, options, None)
}

fn call_stream_until(
    model: &Model,
    messages: Vec<Message>,
    options: StreamOptions,
    deadline: Option<Instant>,
) -> EventStream {
    let call = StreamCall::new(model.clone(), messages, options, deadline);
    stream::unfold(call, |mut call| async move {
        let item = call.next().await?;
        Some((item, call))
    })
    .boxed()
}

/// One buffered generation, with its retries and timeouts. An unsupported
/// native generation falls back to draining the call's stream.
pub(crate) fn call_generate(
    model: &Model,
    messages: Vec<Message>,
    options: StreamOptions,
) -> BoxFuture<'static, Result<Message>> {
    let model = model.clone();
    Box::pin(async move {
        let deadline = match options.timeout {
            Some(timeout) => {
                require_timer()?;
                Some(Instant::now() + timeout)
            }
            None => None,
        };
        if options.idle_timeout.is_some() {
            require_timer()?;
        }
        let mut retries = 0;
        loop {
            let attempt = model
                .provider
                .generate(&model, messages.clone(), options.clone());
            let error = match bounded(attempt, options.idle_timeout, deadline).await {
                Wait::Ready(Ok(message)) => return Ok(message),
                Wait::Ready(Err(Error::UnsupportedOperation(_))) => {
                    let stream = call_stream_until(&model, messages, options, deadline);
                    let mut response = super::StreamHandle::new(
                        stream,
                        None,
                        Some(model.provider.name().to_owned()),
                    );
                    while response.next().await.transpose()?.is_some() {}
                    return Ok(response.message().clone());
                }
                Wait::Ready(Err(error)) => error,
                Wait::Idle => {
                    let idle = options.idle_timeout.unwrap_or_default();
                    timeout_error(
                        &model,
                        format!("the provider sent no response for {idle:?}"),
                    )
                }
                Wait::Deadline => {
                    let timeout = options.timeout.unwrap_or_default();
                    return Err(total_timeout_error(&model, timeout));
                }
            };
            if retries >= options.retry.max_retries
                || !is_retryable(&error)
                || require_timer().is_err()
            {
                return Err(error);
            }
            let wait = retry_delay(&options.retry, retries, &error, SystemTime::now());
            retries += 1;
            if !wait_for_retry(wait, deadline).await {
                let timeout = options.timeout.unwrap_or_default();
                return Err(total_timeout_error(&model, timeout));
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::HttpErrorContext;

    fn rate_limited(headers: &[(&str, &str)]) -> Error {
        ProviderError::from_status("rate limited", 429, None)
            .with_http_context(
                HttpErrorContext::new(429).with_response_headers(
                    headers
                        .iter()
                        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
                        .collect(),
                ),
            )
            .into()
    }

    #[test]
    fn backoff_doubles_from_the_initial_delay() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.backoff(0), Duration::from_secs(2));
        assert_eq!(policy.backoff(1), Duration::from_secs(4));
        // Large retry counts saturate instead of overflowing.
        assert_eq!(
            policy.backoff(40),
            Duration::from_secs(2).saturating_mul(u32::MAX)
        );
    }

    #[test]
    fn retry_after_headers_replace_the_backoff_within_limits() {
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000);
        let policy = RetryPolicy::default();
        let delay = |headers: &[(&str, &str)], retry| {
            retry_delay(&policy, retry, &rate_limited(headers), now)
        };
        assert_eq!(
            delay(&[("Retry-After-Ms", "1500")], 0),
            Duration::from_millis(1500)
        );
        assert_eq!(delay(&[("retry-after", "1")], 0), Duration::from_secs(1));
        assert_eq!(delay(&[("retry-after", "0")], 0), Duration::ZERO);
        // At or above 60 s, a header wait is used only when it is shorter than the backoff.
        assert_eq!(delay(&[("retry-after", "120")], 0), Duration::from_secs(2));
        let long = RetryPolicy {
            initial_delay: Duration::from_secs(100),
            ..RetryPolicy::default()
        };
        assert_eq!(
            retry_delay(&long, 0, &rate_limited(&[("retry-after", "70")]), now),
            Duration::from_secs(70)
        );
        // Negative and unreadable values fall back to the backoff.
        assert_eq!(delay(&[("retry-after", "-1")], 1), Duration::from_secs(4));
        assert_eq!(delay(&[("retry-after", "soon")], 0), Duration::from_secs(2));
        assert_eq!(
            delay(&[("retry-after-ms", "NaN"), ("retry-after", "3")], 0),
            Duration::from_secs(3)
        );
        // HTTP dates count from now.
        let date = httpdate::fmt_http_date(now + Duration::from_secs(5));
        assert_eq!(delay(&[("retry-after", &date)], 0), Duration::from_secs(5));
        let past = httpdate::fmt_http_date(now - Duration::from_secs(5));
        assert_eq!(delay(&[("retry-after", &past)], 0), Duration::from_secs(2));
        let obsolete = "Sunday, 09-Sep-01 01:46:45 GMT";
        assert!(header_delay(&rate_limited(&[("retry-after", obsolete)]), now).is_some());
    }

    #[test]
    fn only_retryable_errors_retry() {
        assert!(is_retryable(&rate_limited(&[])));
        assert!(!is_retryable(
            &ProviderError::from_status("bad", 400, None).into()
        ));
        assert!(is_retryable(&Error::Transport(TransportError::Timeout)));
        assert!(is_retryable(&Error::Transport(TransportError::Connection(
            "reset".into()
        ))));
        assert!(!is_retryable(&Error::Transport(TransportError::Body(
            "cut".into()
        ))));
        assert!(!is_retryable(&Error::Validation("bad".into())));
    }
}
