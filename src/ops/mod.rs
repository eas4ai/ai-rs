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

//! Model operations beyond chat: embeddings, images, video, speech,
//! transcription, reranking, file uploads, batches, streaming transcription
//! and speech translation, realtime sessions and evaluation.
//!
//! The front ends keep the names of the Python SDK 0.8.0 `ai.ops` module and
//! add the operations that the AI SDK for TypeScript 7.0.148 core runs. Each
//! calls the model's protocol, the model's override first and then the
//! provider's; the file and batch operations call the provider. An operation
//! the protocol does not implement fails with `Error::UnsupportedOperation`
//! before any request is sent.
//!
//! Every operation that returns one result retries a retryable error by the
//! `retry` policy of its options, 2 retries by default, as model calls do. A
//! streaming operation retries only an error that arrives before its first
//! part, and only while it has read no audio.
//!
//! ```no_run
//! use suprnova_ai::{Model, ops};
//! use suprnova_ai::providers::operations::EmbedParams;
//!
//! async fn vectors(model: &Model) -> Result<Vec<Vec<f64>>, suprnova_ai::Error> {
//!     let values = vec!["sunny day".to_owned(), "rainy night".to_owned()];
//!     let item = ops::embed(model, values, EmbedParams::default(), Default::default()).await?;
//!     Ok(item.value)
//! }
//! ```

use std::{
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll},
    time::Duration,
};

use futures::{
    Stream, StreamExt,
    future::{self, try_join_all},
    stream,
};
use serde_json::Value;
use tokio::time::Instant;

use crate::{
    Error, Model, Provider, ProviderProtocol, RetryPolicy,
    errors::{ProviderError, ProviderErrorKind},
    models::call::{is_retryable, require_timer, retry_delay, with_retries},
    providers::operations::*,
    types::messages::FilePart,
};

pub mod batch;
pub mod experimental;
pub mod realtime;

/// Options of an operation that returns one result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OperationOptions {
    /// Retries for a retryable error; 2 retries, waiting 2 s and then 4 s,
    /// by default.
    pub retry: RetryPolicy,
}

/// Options of a video generation that runs as a job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VideoOptions {
    pub retry: RetryPolicy,
    /// The wait between status checks; 5 s by default.
    pub poll_interval: Duration,
    /// How long the job may run before the call fails; 600 s by default.
    pub poll_timeout: Duration,
}

impl Default for VideoOptions {
    fn default() -> Self {
        Self {
            retry: RetryPolicy::default(),
            poll_interval: Duration::from_secs(5),
            poll_timeout: Duration::from_secs(600),
        }
    }
}

/// The protocol a model's operations use: its own override, then the
/// provider's.
pub(crate) fn protocol_of(model: &Model) -> Result<Arc<dyn ProviderProtocol>, Error> {
    model
        .protocol
        .clone()
        .or_else(|| model.provider.protocol())
        .ok_or_else(|| {
            Error::UnsupportedOperation(format!(
                "provider {:?} has no protocol",
                model.provider.name()
            ))
        })
}

/// Embed each value. Values are split into runs no longer than the
/// protocol's maximum per call, called concurrently when the protocol allows
/// it. An empty list returns an empty result without a call.
pub async fn embed(
    model: &Model,
    values: Vec<String>,
    params: EmbedParams,
    options: OperationOptions,
) -> Result<Item<Vec<Vec<f64>>>, Error> {
    if values.is_empty() {
        return Ok(Item::new(Vec::new()));
    }
    let limits = protocol_of(model)?.embed_limits(model);
    let runs = split(values, limits.max_per_call);
    let call = |run: Vec<String>| {
        let params = params.clone();
        let retry = options.retry.clone();
        async move {
            with_retries(&retry, || {
                model.provider.embed(model, run.clone(), params.clone())
            })
            .await
        }
    };
    let items = if limits.parallel {
        try_join_all(runs.into_iter().map(call)).await?
    } else {
        let mut items = Vec::new();
        for run in runs {
            items.push(call(run).await?);
        }
        items
    };
    Ok(merge(items))
}

/// Generate `params.n` images, in calls of at most the protocol's maximum
/// per call, run concurrently.
pub async fn generate_image(
    model: &Model,
    prompt: ImagePrompt,
    params: ImageParams,
    options: OperationOptions,
) -> Result<Item<Vec<FilePart>>, Error> {
    let limits = protocol_of(model)?.image_limits(model);
    let calls = batches(params.n, limits.max_per_call).into_iter().map(|n| {
        let prompt = prompt.clone();
        let params = ImageParams {
            n,
            ..params.clone()
        };
        let retry = options.retry.clone();
        async move {
            with_retries(&retry, || {
                model
                    .provider
                    .generate_image(model, prompt.clone(), params.clone())
            })
            .await
        }
    });
    Ok(merge(try_join_all(calls).await?))
}

/// Generate `params.n` videos. On a protocol with jobs, each batch of at
/// most the protocol's maximum per call starts a job and checks its status
/// every poll interval until it completes; videos returned as URLs are
/// fetched through the protocol. A protocol without jobs is called directly.
pub async fn generate_video(
    model: &Model,
    prompt: VideoPrompt,
    params: VideoParams,
    options: VideoOptions,
) -> Result<Item<Vec<FilePart>>, Error> {
    let protocol = protocol_of(model)?;
    let limits = protocol.video_limits(model);
    let calls = batches(params.n, limits.max_per_call).into_iter().map(|n| {
        let prompt = prompt.clone();
        let params = VideoParams {
            n,
            ..params.clone()
        };
        let protocol = protocol.clone();
        let options = options.clone();
        async move { video_batch(model, protocol, prompt, params, &options).await }
    });
    Ok(merge(try_join_all(calls).await?))
}

async fn video_batch(
    model: &Model,
    protocol: Arc<dyn ProviderProtocol>,
    prompt: VideoPrompt,
    params: VideoParams,
    options: &VideoOptions,
) -> Result<Item<Vec<FilePart>>, Error> {
    let provider = model.provider.as_ref();
    let started = with_retries(&options.retry, || {
        protocol.start_video(model, prompt.clone(), params.clone(), provider)
    })
    .await;
    let started = match started {
        Err(Error::UnsupportedOperation(_)) => {
            return with_retries(&options.retry, || {
                provider.generate_video(model, prompt.clone(), params.clone())
            })
            .await;
        }
        other => other?,
    };
    let mut item = poll_video(model, protocol.as_ref(), &started.value, options).await?;
    let mut videos = Vec::with_capacity(item.value.len());
    for video in std::mem::take(&mut item.value) {
        videos.push(
            with_retries(&options.retry, || {
                protocol.download_video(model, video.clone(), provider)
            })
            .await?,
        );
    }
    item.value = videos;
    let mut warnings = started.warnings;
    warnings.append(&mut item.warnings);
    item.warnings = warnings;
    item.provider_metadata = merge_metadata(started.provider_metadata, item.provider_metadata);
    Ok(item)
}

/// Check a job every poll interval until it completes, fails or runs past
/// the poll timeout.
async fn poll_video(
    model: &Model,
    protocol: &dyn ProviderProtocol,
    operation: &Value,
    options: &VideoOptions,
) -> Result<Item<Vec<FilePart>>, Error> {
    require_timer()?;
    let deadline = Instant::now() + options.poll_timeout;
    let timed_out = || {
        Error::from(
            ProviderError::new(
                ProviderErrorKind::Timeout,
                format!(
                    "video generation timed out after {:?}",
                    options.poll_timeout
                ),
            )
            .with_provider(model.provider.name()),
        )
    };
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(timed_out());
        }
        tokio::time::sleep_until((now + options.poll_interval).min(deadline)).await;
        if Instant::now() >= deadline {
            return Err(timed_out());
        }
        let status = tokio::time::timeout_at(
            deadline,
            with_retries(&options.retry, || {
                protocol.video_status(model, operation, model.provider.as_ref())
            }),
        )
        .await
        .map_err(|_| timed_out())??;
        match status {
            VideoStatus::Pending => {}
            VideoStatus::Completed(item) => return Ok(item),
            VideoStatus::Failed { message } => {
                return Err(ProviderError::new(ProviderErrorKind::Provider, message)
                    .with_provider(model.provider.name())
                    .into());
            }
        }
    }
}

/// Start a video job and return it without waiting for the video.
pub async fn start_video(
    model: &Model,
    prompt: VideoPrompt,
    params: VideoParams,
    options: OperationOptions,
) -> Result<Item<VideoJob>, Error> {
    let protocol = protocol_of(model)?;
    let started = with_retries(&options.retry, || {
        protocol.start_video(
            model,
            prompt.clone(),
            params.clone(),
            model.provider.as_ref(),
        )
    })
    .await?;
    Ok(Item {
        value: VideoJob {
            provider: model.provider.name().to_owned(),
            model_id: model.id.clone(),
            operation: started.value,
        },
        usage: started.usage,
        warnings: started.warnings,
        metadata: started.metadata,
        provider_metadata: started.provider_metadata,
    })
}

/// Check a video job once. The job must belong to this model.
pub async fn video_status(
    model: &Model,
    job: &VideoJob,
    options: OperationOptions,
) -> Result<VideoStatus, Error> {
    if job.provider != model.provider.name() || job.model_id != model.id {
        return Err(Error::Validation(format!(
            "video job of {}:{} checked with the model {}:{}",
            job.provider,
            job.model_id,
            model.provider.name(),
            model.id
        )));
    }
    let protocol = protocol_of(model)?;
    with_retries(&options.retry, || {
        protocol.video_status(model, &job.operation, model.provider.as_ref())
    })
    .await
}

/// Generate speech.
pub async fn generate_audio(
    model: &Model,
    prompt: AudioPrompt,
    params: AudioParams,
    options: OperationOptions,
) -> Result<Item<Vec<FilePart>>, Error> {
    with_retries(&options.retry, || {
        model
            .provider
            .generate_audio(model, prompt.clone(), params.clone())
    })
    .await
}

/// Transcribe a whole audio file.
pub async fn transcribe(
    model: &Model,
    audio: TranscribeInput,
    params: TranscribeParams,
    options: OperationOptions,
) -> Result<Item<Transcription>, Error> {
    with_retries(&options.retry, || {
        model
            .provider
            .transcribe(model, audio.clone(), params.clone())
    })
    .await
}

/// Rank documents by their relevance to a query.
pub async fn rerank(
    model: &Model,
    documents: Documents,
    query: String,
    params: RerankParams,
    options: OperationOptions,
) -> Result<Item<Vec<RankedDocument>>, Error> {
    with_retries(&options.retry, || {
        model
            .provider
            .rerank(model, documents.clone(), query.clone(), params.clone())
    })
    .await
}

/// Upload a file through the provider's files operation.
pub async fn upload_file(
    provider: &dyn Provider,
    file: UploadFile,
    params: UploadParams,
    options: OperationOptions,
) -> Result<Item<UploadedFile>, Error> {
    with_retries(&options.retry, || {
        provider.upload_file(file.clone(), params.clone())
    })
    .await
}

/// Transcribe a stream of raw audio chunks as they arrive. Dropping the
/// returned stream closes the provider connection and stops reading the
/// audio.
pub async fn stream_transcription(
    model: &Model,
    audio: AudioChunks,
    format: AudioFormat,
    params: StreamTranscribeParams,
    options: OperationOptions,
) -> Result<TranscriptionStream, Error> {
    let protocol = protocol_of(model)?;
    let audio = SharedAudio::new(audio);
    stream_with_retries(&options.retry, &audio, |chunks| {
        protocol.stream_transcription(
            model,
            chunks,
            format.clone(),
            params.clone(),
            model.provider.as_ref(),
        )
    })
    .await
}

/// Translate a stream of raw audio chunks as they arrive. Dropping the
/// returned stream closes the provider connection and stops reading the
/// audio.
pub async fn stream_translation(
    model: &Model,
    audio: AudioChunks,
    format: AudioFormat,
    params: StreamTranslateParams,
    options: OperationOptions,
) -> Result<TranslationStream, Error> {
    let protocol = protocol_of(model)?;
    let audio = SharedAudio::new(audio);
    stream_with_retries(&options.retry, &audio, |chunks| {
        protocol.stream_translation(
            model,
            chunks,
            format.clone(),
            params.clone(),
            model.provider.as_ref(),
        )
    })
    .await
}

/// Start a streaming operation, retrying an error that arrives before its
/// first part while no audio has been read, so that a retry sends the same
/// audio from its start.
async fn stream_with_retries<T, F, Fut>(
    policy: &RetryPolicy,
    audio: &SharedAudio,
    mut start: F,
) -> Result<futures::stream::BoxStream<'static, Result<T, Error>>, Error>
where
    T: Send + 'static,
    F: FnMut(AudioChunks) -> Fut,
    Fut: Future<Output = Result<futures::stream::BoxStream<'static, Result<T, Error>>, Error>>,
{
    let mut retries = 0;
    loop {
        let error = match start(audio.reader()).await {
            Ok(mut parts) => match parts.next().await {
                Some(Ok(first)) => {
                    return Ok(stream::once(future::ready(Ok(first))).chain(parts).boxed());
                }
                None => return Ok(stream::empty().boxed()),
                Some(Err(error)) => error,
            },
            Err(error) => error,
        };
        if audio.was_read()
            || retries >= policy.max_retries
            || !is_retryable(&error)
            || require_timer().is_err()
        {
            return Err(error);
        }
        let wait = retry_delay(policy, retries, &error, std::time::SystemTime::now());
        retries += 1;
        tokio::time::sleep(wait).await;
    }
}

/// An audio stream that several attempts can read in turn, which records
/// whether any chunk was read.
struct SharedAudio {
    inner: Arc<Mutex<AudioChunks>>,
    read: Arc<AtomicBool>,
}

impl SharedAudio {
    fn new(audio: AudioChunks) -> Self {
        Self {
            inner: Arc::new(Mutex::new(audio)),
            read: Arc::new(AtomicBool::new(false)),
        }
    }
    fn reader(&self) -> AudioChunks {
        AudioReader {
            inner: self.inner.clone(),
            read: self.read.clone(),
        }
        .boxed()
    }
    fn was_read(&self) -> bool {
        self.read.load(Ordering::SeqCst)
    }
}

struct AudioReader {
    inner: Arc<Mutex<AudioChunks>>,
    read: Arc<AtomicBool>,
}

impl Stream for AudioReader {
    type Item = Vec<u8>;
    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Vec<u8>>> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let polled = inner.poll_next_unpin(context);
        if matches!(polled, Poll::Ready(Some(_))) {
            self.read.store(true, Ordering::SeqCst);
        }
        polled
    }
}

/// Split values into runs of at most `max` (one run without a maximum).
fn split<T>(values: Vec<T>, max: Option<usize>) -> Vec<Vec<T>> {
    match max.filter(|max| *max > 0) {
        Some(max) => {
            let mut runs = Vec::new();
            let mut values = values.into_iter().peekable();
            while values.peek().is_some() {
                runs.push(values.by_ref().take(max).collect());
            }
            runs
        }
        None => vec![values],
    }
}

/// The counts of `n` results in calls of at most `max`, the last with the
/// remainder.
fn batches(n: u32, max: Option<usize>) -> Vec<u32> {
    let max = max
        .and_then(|max| u32::try_from(max).ok())
        .filter(|max| *max > 0);
    match max {
        Some(max) if n > max => {
            let mut counts = vec![max; (n / max) as usize];
            if n % max != 0 {
                counts.push(n % max);
            }
            counts
        }
        _ => vec![n],
    }
}

/// Join the results of several calls in call order: their values, the sum
/// of their usage, all their warnings and their metadata.
fn merge<T>(items: Vec<Item<Vec<T>>>) -> Item<Vec<T>> {
    let mut merged = Item::new(Vec::new());
    for item in items {
        merged.value.extend(item.value);
        merged.usage = match (merged.usage.take(), item.usage) {
            (Some(total), Some(usage)) => Some(total + usage),
            (total, usage) => total.or(usage),
        };
        merged.warnings.extend(item.warnings);
        merged.metadata = merge_metadata(merged.metadata.take(), item.metadata);
        merged.provider_metadata =
            merge_metadata(merged.provider_metadata.take(), item.provider_metadata);
    }
    merged
}

/// Merge metadata maps, the later entry's fields over the earlier one's for
/// each provider key.
fn merge_metadata(left: Option<Metadata>, right: Option<Metadata>) -> Option<Metadata> {
    match (left, right) {
        (Some(mut left), Some(right)) => {
            for (key, value) in right {
                match (left.get_mut(&key), value) {
                    (Some(Value::Object(existing)), Value::Object(fields)) => {
                        existing.extend(fields);
                    }
                    (_, value) => {
                        left.insert(key, value);
                    }
                }
            }
            Some(left)
        }
        (left, right) => left.or(right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_split_into_runs_of_the_maximum() {
        assert_eq!(
            split(vec![1, 2, 3, 4, 5], Some(2)),
            vec![vec![1, 2], vec![3, 4], vec![5]]
        );
        assert_eq!(split(vec![1, 2], None), vec![vec![1, 2]]);
    }

    #[test]
    fn counts_split_into_batches_with_the_remainder_last() {
        assert_eq!(batches(5, Some(4)), vec![4, 1]);
        assert_eq!(batches(3, Some(1)), vec![1, 1, 1]);
        assert_eq!(batches(4, Some(4)), vec![4]);
        assert_eq!(batches(2, None), vec![2]);
    }
}
