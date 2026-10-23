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

//! Gemini language model calls: `streamGenerateContent` and
//! `generateContent`, continued while a response finishes with
//! `CONTINUATION`.

use std::{collections::BTreeMap, sync::Arc};

use futures::{StreamExt, TryStreamExt, stream};
use serde_json::{Map, Value, json};

use super::{
    api::{self, Endpoint, MEDIA_RESPONSE_BYTES},
    messages::{self, Target},
    request::{self, Prepared},
    response::Reader,
    tool_results,
};
use crate::{
    Error, StreamOptions,
    models::{files::DownloadTransport, params::InferenceRequestParams},
    providers::{
        EventStream,
        http::{self, SseJson},
    },
    types::{Message, events::MessageHydrator},
};

/// What a Gemini call needs besides the request: the endpoint, the
/// download transport for file URLs the API does not fetch, and on Vertex
/// AI the size limit of downloaded tool result files.
#[derive(Clone)]
pub(crate) struct Call {
    pub endpoint: Endpoint,
    pub model_id: String,
    pub downloads: Option<Arc<dyn DownloadTransport>>,
    pub tool_result_limit: Option<usize>,
}

impl Call {
    pub(crate) fn target(&self) -> Target<'_> {
        Target {
            model_id: &self.model_id,
            flavor: self.endpoint.flavor,
            base_url: &self.endpoint.base,
            batch: false,
        }
    }

    /// Download the user and tool result file URLs the API does not fetch,
    /// after checking the request with each one standing in as empty bytes,
    /// so a call that fails validation downloads nothing.
    pub(crate) async fn prepare(
        &self,
        messages: Vec<Message>,
        options: &StreamOptions,
        streaming: bool,
    ) -> Result<(Prepared, Vec<Message>), Error> {
        let target = self.target();
        let downloads = self.downloads.as_deref();
        let user_files = messages::has_downloads(&messages, target)
            .then(|| messages::without_downloads(&messages, target));
        let stand_ins = match self.tool_result_limit {
            Some(_) => {
                let history = user_files.as_deref().unwrap_or(&messages);
                tool_results::without_downloads(history, target).or(user_files)
            }
            None => user_files,
        };
        if let Some(stand_ins) = stand_ins {
            request::build(target, &stand_ins, options, streaming)?;
        }
        let messages = messages::download_files(messages, target, downloads).await?;
        let messages = match self.tool_result_limit {
            Some(limit) => tool_results::download(messages, target, downloads, limit).await?,
            None => messages,
        };
        let prepared = request::build(target, &messages, options, streaming)?;
        Ok((prepared, messages))
    }

    fn query(params: Option<&InferenceRequestParams>, streaming: bool) -> BTreeMap<String, Value> {
        let mut query = params
            .and_then(|p| p.extra_query.clone())
            .unwrap_or_default();
        if streaming {
            query.insert("alt".into(), json!("sse"));
        }
        query
    }

    fn http_request(
        &self,
        prepared: &Prepared,
        body: Map<String, Value>,
        params: Option<&InferenceRequestParams>,
        streaming: bool,
    ) -> Result<crate::transport::HttpRequest, Error> {
        let method = if streaming {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        self.endpoint.request(
            "POST",
            &api::model_segments(&self.model_id, method),
            Some(Value::Object(body)),
            prepared.headers.clone(),
            Self::query(params, streaming),
        )
    }

    /// Stream a model call.
    pub(crate) fn stream(self, messages: Vec<Message>, options: StreamOptions) -> EventStream {
        stream::once(async move {
            let (prepared, _) = self.prepare(messages, &options, true).await?;
            let params = options.params.clone();
            let request =
                self.http_request(&prepared, prepared.body.clone(), params.as_ref(), true)?;
            let response = self.endpoint.send(request, Some(&self.model_id)).await?;
            let mut reader = Reader::new(
                self.endpoint.flavor,
                self.endpoint.provider.clone(),
                prepared.json_tool.clone(),
            );
            reader.headers(&response.headers);
            let source = sse(&self.endpoint, response.body);
            let state = StreamState {
                call: self,
                prepared,
                params,
                source: Some(source),
                reader,
                pending_token: None,
                done: false,
            };
            Ok::<_, Error>(
                stream::try_unfold(state, |mut state| async move {
                    let event = state.next().await?;
                    Ok(event.map(|event| (event, state)))
                })
                .boxed(),
            )
        })
        .try_flatten()
        .boxed()
    }

    /// A buffered model call: the message and its normalized finish reason.
    pub(crate) async fn generate(
        &self,
        messages: Vec<Message>,
        options: &StreamOptions,
    ) -> Result<(Message, Option<String>), Error> {
        let generated = self.generate_full(messages, options).await?;
        Ok((generated.message, generated.finish))
    }

    /// A buffered model call with the raw finish reason too.
    pub(crate) async fn generate_full(
        &self,
        messages: Vec<Message>,
        options: &StreamOptions,
    ) -> Result<Generated, Error> {
        let (prepared, _) = self.prepare(messages, options, false).await?;
        let params = options.params.as_ref();
        let mut reader = Reader::new(
            self.endpoint.flavor,
            self.endpoint.provider.clone(),
            prepared.json_tool.clone(),
        );
        let mut body = prepared.body.clone();
        loop {
            let request = self.http_request(&prepared, body.clone(), params, false)?;
            let (headers, value) = self
                .endpoint
                .json(request, Some(&self.model_id), MEDIA_RESPONSE_BYTES)
                .await?;
            reader.headers(&headers);
            match reader.chunk(&value)? {
                Some(token) => {
                    reader.start_continuation();
                    body.insert("continuationToken".into(), json!(token));
                }
                None => break,
            }
        }
        reader.end();
        let raw_finish = reader.raw_finish();
        let (message, finish) = hydrate(&self.endpoint.provider, reader.events.drain(..))?;
        Ok(Generated {
            message,
            finish,
            raw_finish,
        })
    }
}

/// A buffered call's message with its normalized and raw finish reasons.
pub(crate) struct Generated {
    pub message: Message,
    pub finish: Option<String>,
    pub raw_finish: Option<String>,
}

/// The server-sent events of a generate content stream, up to 256 MiB each.
fn sse(
    endpoint: &Endpoint,
    body: crate::transport::Body,
) -> futures::stream::BoxStream<'static, Result<SseJson, Error>> {
    http::sse_json_with_limit(
        *endpoint.flavor.api(),
        endpoint.provider.clone(),
        body,
        MEDIA_RESPONSE_BYTES,
    )
}

/// The message events describe, with its finish reason. A response without
/// a finish fails as incomplete, as its stream would.
pub(crate) fn hydrate(
    provider: &str,
    events: impl IntoIterator<Item = crate::types::Event>,
) -> Result<(Message, Option<String>), Error> {
    let mut hydrator = MessageHydrator::new(None);
    for event in events {
        hydrator
            .feed(event)
            .map_err(|error| Error::Validation(error.to_string()))?;
    }
    if !hydrator.ended {
        return Err(crate::errors::ProviderError::incomplete_response(
            "the response ended without a finish reason; it is incomplete",
            Some(provider.to_owned()),
        )
        .into());
    }
    let finish = hydrator.finish_reason.clone();
    Ok((hydrator.message, finish))
}

struct StreamState {
    call: Call,
    prepared: Prepared,
    params: Option<InferenceRequestParams>,
    source: Option<futures::stream::BoxStream<'static, Result<SseJson, Error>>>,
    reader: Reader,
    pending_token: Option<String>,
    done: bool,
}

impl StreamState {
    async fn next(&mut self) -> Result<Option<crate::types::Event>, Error> {
        loop {
            if let Some(event) = self.reader.events.pop_front() {
                return Ok(Some(event));
            }
            if self.done {
                return Ok(None);
            }
            let Some(source) = self.source.as_mut() else {
                return Ok(None);
            };
            match source.next().await {
                Some(Ok(SseJson::Object(value))) => {
                    if let Some(token) = self.reader.chunk(&value)? {
                        self.pending_token = Some(token);
                    }
                }
                Some(Ok(SseJson::Done)) => {}
                Some(Err(error)) => {
                    self.done = true;
                    return Err(error);
                }
                None => match self.pending_token.take() {
                    Some(token) => {
                        let mut body = self.prepared.body.clone();
                        body.insert("continuationToken".into(), json!(token));
                        let request = self.call.http_request(
                            &self.prepared,
                            body,
                            self.params.as_ref(),
                            true,
                        )?;
                        let response = self
                            .call
                            .endpoint
                            .send(request, Some(&self.call.model_id))
                            .await?;
                        self.reader.start_continuation();
                        self.source = Some(sse(&self.call.endpoint, response.body));
                    }
                    None => {
                        self.reader.end();
                        self.done = true;
                        self.source = None;
                    }
                },
            }
        }
    }
}
