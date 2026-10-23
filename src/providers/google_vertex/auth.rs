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

//! The Vertex auth headers (VTX-002): an Express Mode key goes out as
//! `x-goog-api-key` through the request builder; otherwise a transport
//! wrapper asks the credentials for their headers on every request and
//! WebSocket. The crate keeps no token: an access token source is asked
//! each time, and `google-cloud-auth` credentials cache and refresh their
//! own tokens.

use std::sync::Arc;

use futures::future::BoxFuture;

use crate::{
    Error, ProviderConfig,
    transport::{
        HttpRequest, HttpResponse, HttpTransport, TransportError, WebSocket, WebSocketRequest,
        WebSocketTransport,
    },
};

/// The env value of the service account's email.
pub(crate) const CLIENT_EMAIL_ENV: &str = "GOOGLE_CLIENT_EMAIL";
/// The env value of the service account's PKCS #8 PEM private key.
pub(crate) const PRIVATE_KEY_ENV: &str = "GOOGLE_PRIVATE_KEY";
/// The env value of the service account's private key id.
#[cfg(feature = "google-auth")]
pub(crate) const PRIVATE_KEY_ID_ENV: &str = "GOOGLE_PRIVATE_KEY_ID";
/// The OAuth scope of a Vertex AI service account.
#[cfg(feature = "google-auth")]
const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// A source of OAuth access tokens for Vertex AI, such as an application's
/// own Google sign-in. The crate asks it for a token on every request and
/// keeps none, so the source decides when to refresh. Its errors are not
/// shown: a failed token request ends the call with a provider error that
/// names only the source.
pub trait AccessTokenSource: Send + Sync {
    fn access_token(&self) -> BoxFuture<'_, Result<String, Error>>;
}

/// The credentials an application set on a provider.
#[derive(Clone)]
pub(crate) enum Credentials {
    Token(Arc<dyn AccessTokenSource>),
    #[cfg(feature = "google-auth")]
    Google(google_cloud_auth::credentials::Credentials),
}

/// Where a provider's Vertex auth headers come from: the credentials the
/// application set, or else, with the `google-auth` feature, a service
/// account built once from the configuration's `env` map.
#[derive(Clone, Default)]
pub(crate) struct Auth {
    credentials: Option<Credentials>,
    #[cfg(feature = "google-auth")]
    service_account: Arc<tokio::sync::OnceCell<google_cloud_auth::credentials::Credentials>>,
}

#[cfg(feature = "google-auth")]
fn env<'a>(config: &'a ProviderConfig, name: &str) -> Option<&'a str> {
    config
        .env
        .get(name)
        .map(String::as_str)
        .filter(|value| !value.is_empty())
}

/// Whether the configuration has an Express Mode API key.
pub(crate) fn express(config: &ProviderConfig) -> bool {
    config.api_key().is_some_and(|key| !key.is_empty())
}

/// Whether the configuration's `env` map holds a service account.
#[cfg(feature = "google-auth")]
fn has_service_account(config: &ProviderConfig) -> bool {
    env(config, CLIENT_EMAIL_ENV).is_some() && env(config, PRIVATE_KEY_ENV).is_some()
}

/// Whether the process has a rustls crypto provider for google-cloud-auth's
/// service account signer, which reads only the process default and panics
/// without one.
#[cfg(feature = "google-auth")]
fn has_crypto_provider() -> bool {
    rustls::crypto::CryptoProvider::get_default().is_some()
}

/// Whether the `env` map holds a service account the crate can sign with.
#[cfg(feature = "google-auth")]
fn service_account_ready(config: &ProviderConfig) -> bool {
    has_service_account(config) && has_crypto_provider()
}

#[cfg(not(feature = "google-auth"))]
fn service_account_ready(_config: &ProviderConfig) -> bool {
    false
}

/// Whether the `env` map holds a service account that only lacks the
/// process crypto provider.
#[cfg(feature = "google-auth")]
fn lacks_crypto_provider(config: &ProviderConfig) -> bool {
    has_service_account(config) && !has_crypto_provider()
}

#[cfg(not(feature = "google-auth"))]
fn lacks_crypto_provider(_config: &ProviderConfig) -> bool {
    false
}

/// The error of a call whose provider has no way to sign in, from its
/// configuration: no Express Mode key, no credentials set and no usable
/// service account.
pub(crate) fn unavailable(config: &ProviderConfig) -> Error {
    if lacks_crypto_provider(config) {
        return Error::Configuration(
            "the Google Vertex AI service account needs a rustls crypto provider: install one with rustls::crypto::CryptoProvider::install_default, such as rustls::crypto::ring::default_provider().install_default(), before the first call".into(),
        );
    }
    if cfg!(feature = "google-auth") {
        Error::Configuration(format!(
            "Google Vertex AI needs credentials: an Express Mode API key ({}), an access token source, google-cloud-auth Credentials, or a service account ({CLIENT_EMAIL_ENV} and {PRIVATE_KEY_ENV}) in the configuration's env",
            super::API_KEY_ENV
        ))
    } else {
        Error::Configuration(format!(
            "Google Vertex AI needs credentials: an Express Mode API key ({}) or an access token source; google-cloud-auth Credentials and service accounts need the google-auth feature",
            super::API_KEY_ENV
        ))
    }
}

impl Auth {
    pub(crate) fn with(credentials: Credentials) -> Self {
        Self {
            credentials: Some(credentials),
            #[cfg(feature = "google-auth")]
            service_account: Arc::default(),
        }
    }

    /// Whether a call can sign in without an Express Mode key.
    pub(crate) fn available(&self, config: &ProviderConfig) -> bool {
        self.credentials.is_some() || service_account_ready(config)
    }

    /// The auth headers of one request: none in Express Mode, else the
    /// credentials' headers.
    pub(crate) async fn headers(
        &self,
        config: &ProviderConfig,
    ) -> Result<Vec<(String, String)>, TransportError> {
        if express(config) {
            return Ok(Vec::new());
        }
        match &self.credentials {
            Some(Credentials::Token(source)) => {
                let token = source.access_token().await.map_err(|_| {
                    TransportError::Request(
                        "the Google Vertex AI access token source failed".into(),
                    )
                })?;
                Ok(vec![("authorization".into(), format!("Bearer {token}"))])
            }
            #[cfg(feature = "google-auth")]
            Some(Credentials::Google(credentials)) => google_headers(credentials).await,
            None => self.service_account_headers(config).await,
        }
    }

    #[cfg(feature = "google-auth")]
    async fn service_account_headers(
        &self,
        config: &ProviderConfig,
    ) -> Result<Vec<(String, String)>, TransportError> {
        let credentials = self
            .service_account
            .get_or_try_init(|| async { service_account(config) })
            .await?;
        google_headers(credentials).await
    }

    #[cfg(not(feature = "google-auth"))]
    async fn service_account_headers(
        &self,
        _config: &ProviderConfig,
    ) -> Result<Vec<(String, String)>, TransportError> {
        Err(TransportError::Request(
            "Google Vertex AI has no credentials".into(),
        ))
    }
}

/// The service account of the configuration's `env` map, with the
/// cloud-platform scope.
#[cfg(feature = "google-auth")]
fn service_account(
    config: &ProviderConfig,
) -> Result<google_cloud_auth::credentials::Credentials, TransportError> {
    use google_cloud_auth::credentials::service_account::{AccessSpecifier, Builder};
    let (Some(email), Some(key)) = (env(config, CLIENT_EMAIL_ENV), env(config, PRIVATE_KEY_ENV))
    else {
        return Err(TransportError::Request(
            "Google Vertex AI has no credentials".into(),
        ));
    };
    // The signer panics without a process crypto provider; `available`
    // keeps a call from getting here without one.
    if !has_crypto_provider() {
        return Err(TransportError::Request(
            "the Google Vertex AI service account needs a rustls crypto provider".into(),
        ));
    }
    let key = serde_json::json!({
        "type": "service_account",
        "client_email": email,
        "private_key": key,
        "private_key_id": env(config, PRIVATE_KEY_ID_ENV).unwrap_or_default(),
        "project_id": env(config, super::PROJECT_ENV).unwrap_or_default(),
    });
    // The builder's error can quote the key file, so it is not passed on.
    Builder::new(key)
        .with_access_specifier(AccessSpecifier::from_scopes([CLOUD_PLATFORM_SCOPE]))
        .build()
        .map_err(|_| {
            TransportError::Request(format!(
                "Google Vertex AI could not build the service account from {CLIENT_EMAIL_ENV} and {PRIVATE_KEY_ENV}; the key must be a PKCS #8 PEM private key"
            ))
        })
}

/// Every header of google-cloud-auth credentials.
#[cfg(feature = "google-auth")]
async fn google_headers(
    credentials: &google_cloud_auth::credentials::Credentials,
) -> Result<Vec<(String, String)>, TransportError> {
    use google_cloud_auth::credentials::CacheableResource;
    match credentials.headers(http::Extensions::new()).await {
        Ok(CacheableResource::New { data, .. }) => data
            .iter()
            .map(|(name, value)| {
                value
                    .to_str()
                    .map(|value| (name.as_str().to_owned(), value.to_owned()))
                    .map_err(|_| {
                        TransportError::Request(
                            "the Google Vertex AI credentials gave a header that is not text"
                                .into(),
                        )
                    })
            })
            .collect(),
        // Without an entity tag the credentials always give new headers.
        Ok(CacheableResource::NotModified) => Err(TransportError::Request(
            "the Google Vertex AI credentials gave no headers".into(),
        )),
        Err(error) if error.is_transient() => Err(TransportError::Connection(
            "the Google Vertex AI credentials are unavailable".into(),
        )),
        Err(_) => Err(TransportError::Request(
            "the Google Vertex AI credentials failed".into(),
        )),
    }
}

/// Merge auth headers into request headers: the configured headers keep
/// their values unless `auth_wins`, as MaaS and xAI on Vertex AI put the
/// auth headers last.
fn merge(headers: &mut Vec<(String, String)>, auth: Vec<(String, String)>, auth_wins: bool) {
    for (name, value) in auth {
        let present = headers.iter().any(|(n, _)| n.eq_ignore_ascii_case(&name));
        if auth_wins {
            headers.retain(|(n, _)| !n.eq_ignore_ascii_case(&name));
        } else if present {
            continue;
        }
        headers.push((name, value));
    }
}

/// Whether a request URL may carry the Vertex auth headers: a URL on the
/// base URL's host, or an `https` or `wss` URL on a `googleapis.com` host. A
/// token never goes to another host, such as a video's download URL
/// elsewhere.
fn signs(url: &str, base: Option<&str>) -> bool {
    let Ok(url) = url::Url::parse(url) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    let base_host = base
        .and_then(|base| url::Url::parse(base).ok())
        .and_then(|base| base.host_str().map(str::to_owned));
    if base_host.as_deref() == Some(host) {
        return true;
    }
    matches!(url.scheme(), "https" | "wss")
        && (host == "googleapis.com" || host.ends_with(".googleapis.com"))
}

/// An HTTP transport that adds the Vertex auth headers to each request to
/// the provider's hosts.
#[derive(Clone)]
pub(crate) struct AuthTransport {
    pub inner: Arc<dyn HttpTransport>,
    pub auth: Auth,
    pub config: ProviderConfig,
    pub auth_wins: bool,
    /// The provider's base URL, whose host is signed.
    pub base_url: fn(&ProviderConfig) -> Result<String, Error>,
}

impl HttpTransport for AuthTransport {
    fn send(
        &self,
        mut request: HttpRequest,
    ) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        Box::pin(async move {
            let base = (self.base_url)(&self.config).ok();
            if signs(&request.url, base.as_deref()) {
                let auth = self.auth.headers(&self.config).await?;
                merge(&mut request.headers, auth, self.auth_wins);
            }
            self.inner.send(request).await
        })
    }
}

/// A WebSocket transport that adds the Vertex auth headers to each
/// connection to the provider's hosts.
#[derive(Clone)]
pub(crate) struct AuthWebSocket {
    pub inner: Arc<dyn WebSocketTransport>,
    pub auth: Auth,
    pub config: ProviderConfig,
    pub base_url: fn(&ProviderConfig) -> Result<String, Error>,
}

impl WebSocketTransport for AuthWebSocket {
    fn connect(
        &self,
        mut request: WebSocketRequest,
    ) -> BoxFuture<'_, Result<WebSocket, TransportError>> {
        Box::pin(async move {
            let base = (self.base_url)(&self.config).ok();
            if signs(&request.url, base.as_deref()) {
                let auth = self.auth.headers(&self.config).await?;
                merge(&mut request.headers, auth, false);
            }
            self.inner.connect(request).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(headers: &[(&str, &str)]) -> Vec<(String, String)> {
        headers
            .iter()
            .map(|(n, v)| ((*n).to_owned(), (*v).to_owned()))
            .collect()
    }

    #[test]
    fn only_the_providers_hosts_are_signed() {
        let base = Some("https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p");
        for url in [
            "https://us-central1-aiplatform.googleapis.com/v1/x",
            "https://texttospeech.googleapis.com/v1/text:synthesize",
            "wss://us-central1-aiplatform.googleapis.com/ws/x",
        ] {
            assert!(signs(url, base), "{url}");
        }
        assert!(signs(
            "http://localhost:8080/v1",
            Some("http://localhost:8080")
        ));
        for url in [
            "https://cdn.example/v.mp4",
            "https://googleapis.com.evil.example/x",
            "http://speech.googleapis.com/v2",
            "gs://bucket/v.mp4",
            "not a url",
        ] {
            assert!(!signs(url, base), "{url}");
        }
    }

    #[test]
    fn configured_headers_win_unless_auth_does() {
        let mut headers = pairs(&[("Authorization", "Bearer mine"), ("accept", "*/*")]);
        merge(
            &mut headers,
            pairs(&[("authorization", "Bearer t1"), ("x-goog-user-project", "p")]),
            false,
        );
        assert_eq!(
            headers,
            pairs(&[
                ("Authorization", "Bearer mine"),
                ("accept", "*/*"),
                ("x-goog-user-project", "p")
            ])
        );
        merge(&mut headers, pairs(&[("authorization", "Bearer t2")]), true);
        assert_eq!(
            headers,
            pairs(&[
                ("accept", "*/*"),
                ("x-goog-user-project", "p"),
                ("authorization", "Bearer t2")
            ])
        );
    }
}
