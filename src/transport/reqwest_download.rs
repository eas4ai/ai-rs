use super::ReqwestTransport;
use crate::{
    models::files::{DownloadError, DownloadTransport},
    transport::{HttpRequest, HttpResponse, HttpTransport},
};
use futures::future::BoxFuture;
use std::{
    net::{IpAddr, SocketAddr},
    time::Duration,
};

impl DownloadTransport for ReqwestTransport {
    fn resolve<'a>(
        &'a self,
        host: &'a str,
        port: u16,
    ) -> BoxFuture<'a, Result<Vec<IpAddr>, DownloadError>> {
        Box::pin(async move {
            tokio::net::lookup_host((host, port))
                .await
                .map(|addresses| addresses.map(|address| address.ip()).collect())
                .map_err(|error| DownloadError::with_source(host, error))
        })
    }

    fn send_pinned(
        &self,
        request: HttpRequest,
        addresses: Vec<IpAddr>,
    ) -> BoxFuture<'_, Result<HttpResponse, DownloadError>> {
        Box::pin(async move {
            let url = url::Url::parse(&request.url)
                .map_err(|error| DownloadError::with_source(&request.url, error))?;
            let host = match url.host() {
                Some(url::Host::Domain(host)) => host.to_owned(),
                Some(url::Host::Ipv4(address)) => address.to_string(),
                Some(url::Host::Ipv6(address)) => address.to_string(),
                None => return Err(DownloadError::new(&request.url, "missing host")),
            };
            let port = url
                .port_or_known_default()
                .ok_or_else(|| DownloadError::new(&request.url, "unsupported URL scheme"))?;
            if addresses.is_empty() {
                return Err(DownloadError::new(&request.url, "no pinned addresses"));
            }
            let sockets: Vec<_> = addresses
                .into_iter()
                .map(|ip| SocketAddr::new(ip, port))
                .collect();
            let client = reqwest::Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(10))
                .timeout(Duration::from_secs(30))
                .resolve_to_addrs(&host, &sockets)
                .pool_max_idle_per_host(0)
                .build()
                .map_err(|error| DownloadError::with_source(&request.url, error.without_url()))?;
            let response = ReqwestTransport::with_client(client)
                .send(request.clone())
                .await
                .map_err(|error| DownloadError::with_source(&request.url, error))?;
            Ok(response)
        })
    }
}
