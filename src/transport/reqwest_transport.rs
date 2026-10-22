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

use std::time::Duration;

use futures::{StreamExt, future::BoxFuture};

use super::{HttpRequest, HttpResponse, HttpTransport, TransportError};

/// Optional transport. Supply an existing client to control TLS, proxy, or timeouts.
#[derive(Clone, Debug)]
pub struct ReqwestTransport {
    client: reqwest::Client,
}

impl ReqwestTransport {
    /// Build a client that gives up opening a connection after 10 s and sets
    /// no other time limit, so a model may think for minutes before or while
    /// it streams. TCP keepalive ends a connection whose peer is gone. Bound a
    /// call with `StreamOptions::idle_timeout` or `StreamOptions::timeout`.
    /// The client follows no redirects.
    pub fn new() -> Result<Self, TransportError> {
        let client = reqwest::Client::builder()
            .no_proxy()
            .connect_timeout(Duration::from_secs(10))
            .tcp_keepalive(Duration::from_secs(30))
            .tcp_keepalive_interval(Duration::from_secs(10))
            .tcp_keepalive_retries(6)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(map_error)?;
        Ok(Self { client })
    }

    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }
}

impl HttpTransport for ReqwestTransport {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        Box::pin(async move {
            let method = reqwest::Method::from_bytes(request.method.as_bytes())
                .map_err(|_| TransportError::Request("invalid HTTP method".into()))?;
            let mut builder = self.client.request(method, &request.url);
            for (name, value) in request.headers {
                builder = builder.header(name, value);
            }
            let response = builder.body(request.body).send().await.map_err(map_error)?;
            let status = response.status().as_u16();
            let headers = response
                .headers()
                .iter()
                .map(|(name, value)| {
                    value
                        .to_str()
                        .map(|value| (name.to_string(), value.to_owned()))
                        .map_err(|_| {
                            TransportError::Protocol("non-text HTTP response header".into())
                        })
                })
                .collect::<Result<_, _>>()?;
            let body = response
                .bytes_stream()
                .map(|chunk| chunk.map_err(map_error))
                .boxed();
            Ok(HttpResponse {
                status,
                headers,
                body,
            })
        })
    }
}

fn map_error(error: reqwest::Error) -> TransportError {
    if error.is_timeout() {
        TransportError::Timeout
    } else if error.is_builder() {
        TransportError::Request(error.without_url().to_string())
    } else if error.is_body() || error.is_decode() {
        TransportError::Body(error.without_url().to_string())
    } else {
        TransportError::Connection(error.without_url().to_string())
    }
}
