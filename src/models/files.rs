//! Bounded public-network downloads for model input files.

use std::{
    collections::HashSet,
    error::Error as StdError,
    fmt,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    time::Duration,
};

use bytes::{Bytes, BytesMut};
use futures::{StreamExt, future::BoxFuture};
use url::{Host, Url};

use crate::transport::{HttpRequest, HttpResponse};

pub const DEFAULT_MAX_BYTES: usize = 100 * 1024 * 1024;
pub const MAX_REDIRECTS: usize = 10;
pub const DNS_TIMEOUT: Duration = Duration::from_secs(5);

/// Resolves and sends a request while retaining the original host for HTTP and TLS.
///
/// Implementations must connect only to one of `addresses`, must preserve the URL's
/// host as the Host header and TLS server name, and must not follow redirects.
/// Response bodies must yield decoded content so download limits apply after
/// decompression. A fresh connection per call prevents a connection pooled for
/// one hostname from being reused for a different redirect hostname that resolves to the same address.
pub trait DownloadTransport: Send + Sync {
    fn resolve<'a>(
        &'a self,
        host: &'a str,
        port: u16,
    ) -> BoxFuture<'a, Result<Vec<IpAddr>, DownloadError>>;

    fn send_pinned(
        &self,
        request: HttpRequest,
        addresses: Vec<IpAddr>,
    ) -> BoxFuture<'_, Result<HttpResponse, DownloadError>>;
}

#[derive(Debug)]
pub struct DownloadError {
    pub url: String,
    pub status_code: Option<u16>,
    detail: Option<String>,
    source: Option<Box<dyn StdError + Send + Sync>>,
}

impl DownloadError {
    #[must_use]
    pub fn new(url: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            status_code: None,
            detail: Some(detail.into()),
            source: None,
        }
    }

    #[must_use]
    pub fn status(url: impl Into<String>, status_code: u16, detail: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            status_code: Some(status_code),
            detail: Some(detail.into()),
            source: None,
        }
    }

    #[must_use]
    pub fn with_source(
        url: impl Into<String>,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        Self {
            url: url.into(),
            status_code: None,
            detail: None,
            source: Some(Box::new(source)),
        }
    }

    #[must_use]
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

impl fmt::Display for DownloadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "failed to download {:?}", self.url)?;
        if let Some(status_code) = self.status_code {
            write!(formatter, ": status={status_code}")?;
        }
        if let Some(detail) = &self.detail {
            write!(formatter, ": {detail}")?;
        }
        Ok(())
    }
}

impl StdError for DownloadError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_deref()
            .map(|source| source as &(dyn StdError + 'static))
    }
}

/// Download a public HTTP(S) URL with redirect and decoded-body size limits.
pub async fn download(
    transport: &dyn DownloadTransport,
    url: &str,
    max_bytes: usize,
) -> Result<(Bytes, Option<String>), DownloadError> {
    let original = url.to_owned();
    let mut cookies = cookie_store::CookieStore::default();
    let mut current = parse_and_validate_url(url, &original)?;

    for redirect_count in 0..=MAX_REDIRECTS {
        validate_url(&current, &original)?;
        let host = match current
            .host()
            .ok_or_else(|| DownloadError::new(&original, "disallowed download URL"))?
        {
            Host::Domain(host) => host.to_owned(),
            Host::Ipv4(address) => address.to_string(),
            Host::Ipv6(address) => address.to_string(),
        };
        let port = current
            .port_or_known_default()
            .ok_or_else(|| DownloadError::new(&original, "download URL has no port"))?;
        let addresses = tokio::time::timeout(DNS_TIMEOUT, transport.resolve(&host, port))
            .await
            .map_err(|error| DownloadError::with_source(&original, error))??;
        let addresses = validate_addresses(addresses, &original)?;
        let cookie_header = cookies
            .get_request_values(&current)
            .map(|(name, value)| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join("; ");
        let headers = if cookie_header.is_empty() {
            Vec::new()
        } else {
            vec![("Cookie".into(), cookie_header)]
        };
        let response = transport
            .send_pinned(
                HttpRequest {
                    method: "GET".to_owned(),
                    url: current.as_str().to_owned(),
                    headers,
                    body: Bytes::new(),
                },
                addresses,
            )
            .await?;

        for (_, value) in response
            .headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("set-cookie"))
        {
            // Invalid or out-of-scope cookies are ignored, as in a browser jar.
            // insert_raw avoids the convenience API's logging of cookie values.
            if let Ok(cookie) = cookie_store::RawCookie::parse(value.clone()) {
                let _ = cookies.insert_raw(&cookie, &current);
            }
        }

        if is_redirect(response.status) {
            if redirect_count == MAX_REDIRECTS {
                return Err(DownloadError::new(&original, "too many redirects"));
            }
            let location = header(&response, "location").ok_or_else(|| {
                DownloadError::status(&original, response.status, "redirect has no location")
            })?;
            current = current
                .join(location)
                .map_err(|error| DownloadError::with_source(&original, error))?;
            validate_url(&current, &original)?;
            continue;
        }

        if !(200..300).contains(&response.status) {
            return Err(DownloadError::status(
                &original,
                response.status,
                status_text(response.status),
            ));
        }

        if header(&response, "content-length")
            .and_then(|value| value.parse::<usize>().ok())
            .is_some_and(|length| length > max_bytes)
        {
            return Err(DownloadError::new(
                &original,
                "response exceeds maximum size",
            ));
        }

        let content_type = header(&response, "content-type")
            .and_then(|value| value.split(';').next())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        let mut data = BytesMut::new();
        let mut body = response.body;
        while let Some(chunk) = body.next().await {
            let chunk = chunk.map_err(|error| DownloadError::with_source(&original, error))?;
            if data.len().saturating_add(chunk.len()) > max_bytes {
                return Err(DownloadError::new(
                    &original,
                    "response exceeds maximum size",
                ));
            }
            data.extend_from_slice(&chunk);
        }
        return Ok((data.freeze(), content_type));
    }

    Err(DownloadError::new(&original, "too many redirects"))
}

fn parse_and_validate_url(url: &str, reported_url: &str) -> Result<Url, DownloadError> {
    let authority = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .and_then(|rest| rest.split(['/', '?', '#']).next());
    if authority.is_none_or(str::is_empty) {
        return Err(DownloadError::new(reported_url, "disallowed download URL"));
    }
    let parsed =
        Url::parse(url).map_err(|error| DownloadError::with_source(reported_url, error))?;
    validate_url(&parsed, reported_url)?;
    Ok(parsed)
}

fn validate_url(url: &Url, reported_url: &str) -> Result<(), DownloadError> {
    if !matches!(url.scheme(), "http" | "https") {
        return Err(DownloadError::new(
            reported_url,
            format!("unsupported URL scheme: {:?}", url.scheme()),
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(DownloadError::new(reported_url, "disallowed download URL"));
    }
    let host = url
        .host()
        .ok_or_else(|| DownloadError::new(reported_url, "disallowed download URL"))?;
    match host {
        Host::Domain(host) => {
            let host = host.trim_end_matches('.').to_ascii_lowercase();
            if host.is_empty()
                || host.contains('%')
                || host == "localhost"
                || host.ends_with(".localhost")
                || host.ends_with(".local")
            {
                return Err(DownloadError::new(reported_url, "disallowed download URL"));
            }
        }
        Host::Ipv4(address) => validate_address(IpAddr::V4(address), reported_url)?,
        Host::Ipv6(address) => validate_address(IpAddr::V6(address), reported_url)?,
    }
    Ok(())
}

fn validate_addresses(addresses: Vec<IpAddr>, url: &str) -> Result<Vec<IpAddr>, DownloadError> {
    if addresses.is_empty() {
        return Err(DownloadError::new(url, "hostname has no addresses"));
    }
    let mut seen = HashSet::new();
    let mut unique = Vec::with_capacity(addresses.len());
    for address in addresses {
        validate_address(address, url)?;
        if seen.insert(address) {
            unique.push(address);
        }
    }
    Ok(unique)
}

fn validate_address(address: IpAddr, url: &str) -> Result<(), DownloadError> {
    let public = match address {
        IpAddr::V4(address) => is_public_ipv4(address),
        IpAddr::V6(address) => is_public_ipv6(address),
    };
    if public {
        Ok(())
    } else {
        Err(DownloadError::new(url, "non-public IP address"))
    }
}

fn is_public_ipv4(address: Ipv4Addr) -> bool {
    let [a, b, c, _] = address.octets();
    if a == 192 && b == 0 && c == 0 && matches!(address.octets()[3], 9 | 10) {
        return true;
    }
    !(a == 0
        || a == 10
        || a == 127
        || a >= 224
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 168)
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113))
}

fn is_public_ipv6(address: Ipv6Addr) -> bool {
    if let Some(ipv4) = address.to_ipv4() {
        return is_public_ipv4(ipv4);
    }
    let value = u128::from(address);
    if has_prefix(
        value,
        u128::from(Ipv6Addr::new(0x0064, 0xff9b, 0, 0, 0, 0, 0, 0)),
        96,
    ) {
        return is_public_ipv4(Ipv4Addr::from(value as u32));
    }

    // Public unicast currently lives in 2000::/3. Match Python's conservative
    // IANA special-use table, including the globally reachable exceptions
    // inside 2001::/23.
    let special_2001 = has_prefix(
        value,
        u128::from(Ipv6Addr::new(0x2001, 0, 0, 0, 0, 0, 0, 0)),
        23,
    );
    let public_2001_exception = value == u128::from(Ipv6Addr::new(0x2001, 1, 0, 0, 0, 0, 0, 1))
        || value == u128::from(Ipv6Addr::new(0x2001, 1, 0, 0, 0, 0, 0, 2))
        || has_prefix(
            value,
            u128::from(Ipv6Addr::new(0x2001, 3, 0, 0, 0, 0, 0, 0)),
            32,
        )
        || has_prefix(
            value,
            u128::from(Ipv6Addr::new(0x2001, 4, 0x0112, 0, 0, 0, 0, 0)),
            48,
        )
        || has_prefix(
            value,
            u128::from(Ipv6Addr::new(0x2001, 0x0020, 0, 0, 0, 0, 0, 0)),
            28,
        )
        || has_prefix(
            value,
            u128::from(Ipv6Addr::new(0x2001, 0x0030, 0, 0, 0, 0, 0, 0)),
            28,
        );
    has_prefix(
        value,
        u128::from(Ipv6Addr::new(0x2000, 0, 0, 0, 0, 0, 0, 0)),
        3,
    ) && (!special_2001 || public_2001_exception)
        && !has_prefix(
            value,
            u128::from(Ipv6Addr::new(0x2001, 0x0db8, 0, 0, 0, 0, 0, 0)),
            32,
        )
        && !has_prefix(
            value,
            u128::from(Ipv6Addr::new(0x2002, 0, 0, 0, 0, 0, 0, 0)),
            16,
        )
        && !has_prefix(
            value,
            u128::from(Ipv6Addr::new(0x3fff, 0, 0, 0, 0, 0, 0, 0)),
            20,
        )
}

fn has_prefix(value: u128, network: u128, bits: u32) -> bool {
    let mask = if bits == 0 {
        0
    } else {
        u128::MAX << (128 - bits)
    };
    value & mask == network & mask
}

fn header<'a>(response: &'a HttpResponse, name: &str) -> Option<&'a str> {
    response
        .headers
        .iter()
        .find(|(header_name, _)| header_name.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

fn is_redirect(status: u16) -> bool {
    matches!(status, 301 | 302 | 303 | 307 | 308)
}

fn status_text(status: u16) -> &'static str {
    match status {
        400 => "bad request",
        401 => "unauthorized",
        403 => "forbidden",
        404 => "not found",
        408 => "request timeout",
        429 => "too many requests",
        500 => "internal server error",
        502 => "bad gateway",
        503 => "service unavailable",
        504 => "gateway timeout",
        _ => "HTTP request failed",
    }
}
