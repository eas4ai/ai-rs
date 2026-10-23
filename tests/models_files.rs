use std::{
    collections::VecDeque,
    net::{IpAddr, Ipv4Addr},
    sync::Mutex,
};

use bytes::Bytes;
use futures::{FutureExt, StreamExt, future::BoxFuture, stream};
use suprnova_ai::{
    models::files::{DownloadError, DownloadTransport, MAX_REDIRECTS, download},
    transport::{HttpRequest, HttpResponse, TransportError},
};

const PUBLIC_IP: IpAddr = IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34));
const PUBLIC_URL: &str = "https://media.example/file?signature=a%2Fb";

#[derive(Debug)]
struct ResponseSpec {
    status: u16,
    headers: Vec<(String, String)>,
    chunks: Vec<Result<Bytes, TransportError>>,
}

impl ResponseSpec {
    fn ok(data: impl Into<Bytes>) -> Self {
        Self {
            status: 200,
            headers: Vec::new(),
            chunks: vec![Ok(data.into())],
        }
    }

    fn redirect(location: &str) -> Self {
        Self {
            status: 302,
            headers: vec![("location".to_owned(), location.to_owned())],
            chunks: Vec::new(),
        }
    }
}

#[derive(Default)]
struct FakeDownloadTransport {
    resolutions: Mutex<VecDeque<Vec<IpAddr>>>,
    responses: Mutex<VecDeque<ResponseSpec>>,
    resolved: Mutex<Vec<(String, u16)>>,
    sent: Mutex<Vec<(String, Vec<IpAddr>)>>,
    sent_headers: Mutex<Vec<suprnova_ai::transport::Headers>>,
}

impl FakeDownloadTransport {
    fn with_responses(responses: impl IntoIterator<Item = ResponseSpec>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
            ..Self::default()
        }
    }

    fn push_resolution(&self, addresses: Vec<IpAddr>) {
        self.resolutions.lock().unwrap().push_back(addresses);
    }
}

impl DownloadTransport for FakeDownloadTransport {
    fn resolve<'a>(
        &'a self,
        host: &'a str,
        port: u16,
    ) -> BoxFuture<'a, Result<Vec<IpAddr>, DownloadError>> {
        async move {
            self.resolved.lock().unwrap().push((host.to_owned(), port));
            Ok(self
                .resolutions
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| vec![PUBLIC_IP]))
        }
        .boxed()
    }

    fn send_pinned(
        &self,
        request: HttpRequest,
        addresses: Vec<IpAddr>,
    ) -> BoxFuture<'_, Result<HttpResponse, DownloadError>> {
        async move {
            self.sent_headers.lock().unwrap().push(request.headers);
            self.sent.lock().unwrap().push((request.url, addresses));
            let response = self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| DownloadError::new("fake", "no queued response"))?;
            Ok(HttpResponse {
                status: response.status,
                headers: response.headers,
                body: stream::iter(response.chunks).boxed(),
            })
        }
        .boxed()
    }
}

#[tokio::test]
async fn rejects_unsafe_urls_before_dns_or_http() {
    let transport = FakeDownloadTransport::default();
    for url in [
        "file:///etc/passwd",
        "ftp://example.com/file",
        "data:text/plain,hello",
        "http:///missing-host",
        "http://user:password@example.com/",
        "http://localhost/",
        "http://sub.localhost/",
        "http://service.local/",
        "http://127.0.0.1/",
        "http://127.255.255.255/",
        "http://2130706433/internal",
        "http://127.1/internal",
        "http://0177.0.0.1/internal",
        "http://0x7f000001/internal",
        "http://10.0.0.1/",
        "http://100.64.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://198.18.0.1/",
        "http://224.0.0.1/",
        "http://255.255.255.255/",
        "http://192.0.2.1/",
        "http://[::1]/",
        "http://[fc00::1]/",
        "http://[fe80::1]/",
        "http://[2001:db8::1]/",
        "http://[2001:10::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:0:127.0.0.1]/",
        "http://[::127.0.0.1]/",
        "http://[64:ff9b::169.254.169.254]/",
        "http://[64:ff9b:1::127.0.0.1]/",
        "http://[2002:7f00:1::]/",
    ] {
        assert!(
            download(&transport, url, 100).await.is_err(),
            "accepted {url}"
        );
    }
    assert!(transport.resolved.lock().unwrap().is_empty());
    assert!(transport.sent.lock().unwrap().is_empty());
}

#[tokio::test]
async fn dns_rebinding_on_a_redirect_is_blocked_before_the_second_send() {
    let transport = FakeDownloadTransport::with_responses([ResponseSpec::redirect("/internal")]);
    transport.push_resolution(vec![PUBLIC_IP]);
    transport.push_resolution(vec![IpAddr::V4(Ipv4Addr::LOCALHOST)]);

    let error = download(&transport, PUBLIC_URL, 100).await.unwrap_err();
    assert!(error.to_string().contains("non-public"));
    assert_eq!(transport.resolved.lock().unwrap().len(), 2);
    assert_eq!(transport.sent.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn permits_python_ipaddress_public_exceptions() {
    for url in [
        "http://192.0.0.9/file",
        "http://192.0.0.10/file",
        "http://192.88.99.1/file",
        "http://[2001:1::1]/file",
        "http://[2001:3::1]/file",
        "http://[2001:4:112::1]/file",
        "http://[2001:20::1]/file",
        "http://[2001:30::1]/file",
    ] {
        let transport = FakeDownloadTransport::with_responses([ResponseSpec::ok("ok")]);
        assert_eq!(
            download(&transport, url, 10).await.unwrap().0,
            Bytes::from_static(b"ok"),
            "rejected {url}"
        );
        if url.contains('[') {
            let resolved_host = &transport.resolved.lock().unwrap()[0].0;
            assert!(!resolved_host.contains(['[', ']']));
        }
    }
}

#[tokio::test]
async fn rejects_an_empty_or_partly_private_dns_answer() {
    for addresses in [Vec::new(), vec![PUBLIC_IP, IpAddr::V4(Ipv4Addr::LOCALHOST)]] {
        let transport = FakeDownloadTransport::default();
        transport.push_resolution(addresses);
        let error = download(&transport, PUBLIC_URL, 100).await.unwrap_err();
        assert!(error.to_string().contains("address"));
        assert!(transport.sent.lock().unwrap().is_empty());
    }
}

#[tokio::test]
async fn redirects_are_revalidated_reresolved_and_pinned() {
    let transport = FakeDownloadTransport::with_responses([
        ResponseSpec::redirect("https://cdn.example/final?signature=x%2Fy"),
        ResponseSpec {
            status: 200,
            headers: vec![("content-type".into(), "image/png; charset=utf-8".into())],
            chunks: vec![Ok(Bytes::from_static(b"image"))],
        },
    ]);
    let second_ip = IpAddr::V4(Ipv4Addr::new(1, 1, 1, 1));
    transport.push_resolution(vec![PUBLIC_IP]);
    transport.push_resolution(vec![second_ip]);

    let result = download(&transport, PUBLIC_URL, 100).await.unwrap();
    assert_eq!(
        result,
        (Bytes::from_static(b"image"), Some("image/png".into()))
    );
    assert_eq!(
        *transport.resolved.lock().unwrap(),
        vec![("media.example".into(), 443), ("cdn.example".into(), 443)]
    );
    assert_eq!(
        *transport.sent.lock().unwrap(),
        vec![
            (PUBLIC_URL.into(), vec![PUBLIC_IP]),
            (
                "https://cdn.example/final?signature=x%2Fy".into(),
                vec![second_ip]
            )
        ]
    );
}

#[tokio::test]
async fn a_private_redirect_is_never_sent() {
    let transport = FakeDownloadTransport::with_responses([ResponseSpec::redirect(
        "http://127.0.0.1/internal",
    )]);
    let error = download(&transport, PUBLIC_URL, 100).await.unwrap_err();
    assert!(error.to_string().contains("non-public"));
    assert_eq!(transport.sent.lock().unwrap().len(), 1);
    assert_eq!(transport.resolved.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn enforces_declared_and_streamed_size_limits() {
    let declared = FakeDownloadTransport::with_responses([ResponseSpec {
        status: 200,
        headers: vec![("content-length".into(), "100".into())],
        chunks: vec![Ok(Bytes::from_static(b"never read"))],
    }]);
    let error = download(&declared, PUBLIC_URL, 5).await.unwrap_err();
    assert!(error.to_string().contains("maximum size"));

    let streamed = FakeDownloadTransport::with_responses([ResponseSpec {
        status: 200,
        headers: Vec::new(),
        chunks: vec![
            Ok(Bytes::from_static(b"123")),
            Ok(Bytes::from_static(b"456")),
        ],
    }]);
    let error = download(&streamed, PUBLIC_URL, 5).await.unwrap_err();
    assert!(error.to_string().contains("maximum size"));
}

#[tokio::test]
async fn accepts_an_exact_size_and_reports_http_failures() {
    let exact =
        FakeDownloadTransport::with_responses([ResponseSpec::ok(Bytes::from_static(b"12345"))]);
    assert_eq!(
        download(&exact, PUBLIC_URL, 5).await.unwrap(),
        (Bytes::from_static(b"12345"), None)
    );

    let missing = FakeDownloadTransport::with_responses([ResponseSpec {
        status: 404,
        headers: Vec::new(),
        chunks: vec![Ok(Bytes::from_static(b"never read"))],
    }]);
    let error = download(&missing, PUBLIC_URL, 5).await.unwrap_err();
    assert_eq!(error.status_code, Some(404));
}

#[tokio::test]
async fn redirect_limit_matches_the_python_core() {
    let transport = FakeDownloadTransport::with_responses(
        (0..=MAX_REDIRECTS).map(|_| ResponseSpec::redirect("/again")),
    );
    let error = download(&transport, PUBLIC_URL, 100).await.unwrap_err();
    assert!(error.to_string().contains("too many redirects"));
    assert_eq!(transport.sent.lock().unwrap().len(), MAX_REDIRECTS + 1);
}

#[tokio::test]
async fn redirect_cookie_scope_matches_the_python_session() {
    for (target, expected) in [
        ("https://media.example.com/final", Some("access=ok")),
        ("https://cdn.example.com/final", Some("access=ok")),
        ("https://unrelated.test/final", None),
    ] {
        let mut redirect = ResponseSpec::redirect(target);
        redirect.headers.push((
            "Set-Cookie".into(),
            "access=ok; Domain=example.com; Path=/".into(),
        ));
        let transport = FakeDownloadTransport::with_responses([redirect, ResponseSpec::ok("ok")]);
        assert_eq!(
            download(&transport, "http://media.example.com/file", 100)
                .await
                .unwrap()
                .0,
            "ok"
        );
        let sent = transport.sent_headers.lock().unwrap();
        let cookie = sent[1]
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("cookie"))
            .map(|(_, v)| v.as_str());
        assert_eq!(cookie, expected, "{target}");
    }
}

#[tokio::test(start_paused = true)]
async fn dns_failure_and_timeout_do_not_send_http_requests() {
    struct FailingResolver(bool);
    impl DownloadTransport for FailingResolver {
        fn resolve<'a>(
            &'a self,
            host: &'a str,
            _port: u16,
        ) -> BoxFuture<'a, Result<Vec<IpAddr>, DownloadError>> {
            Box::pin(async move {
                if self.0 {
                    futures::future::pending().await
                } else {
                    Err(DownloadError::new(host, "DNS failed"))
                }
            })
        }
        fn send_pinned(
            &self,
            _request: HttpRequest,
            _addresses: Vec<IpAddr>,
        ) -> BoxFuture<'_, Result<HttpResponse, DownloadError>> {
            panic!("DNS failure must prevent HTTP requests")
        }
    }
    assert!(
        download(&FailingResolver(false), PUBLIC_URL, 100)
            .await
            .is_err()
    );
    assert!(
        download(&FailingResolver(true), PUBLIC_URL, 100)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn early_errors_and_cancellation_drop_the_response_body() {
    use futures::Stream;
    use std::{
        pin::Pin,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        task::{Context, Poll},
    };
    struct DropBody {
        flag: Arc<AtomicBool>,
        mode: u8,
        sent: bool,
    }
    impl Drop for DropBody {
        fn drop(&mut self) {
            self.flag.store(true, Ordering::SeqCst);
        }
    }
    impl Stream for DropBody {
        type Item = Result<Bytes, TransportError>;
        fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            if self.mode == 2 {
                return Poll::Pending;
            }
            if self.sent {
                return Poll::Ready(None);
            }
            self.sent = true;
            Poll::Ready(Some(if self.mode == 1 {
                Err(TransportError::Timeout)
            } else {
                Ok(Bytes::from_static(b"too-long"))
            }))
        }
    }
    struct CleanupTransport {
        status: u16,
        headers: Vec<(String, String)>,
        flag: Arc<AtomicBool>,
        mode: u8,
    }
    impl DownloadTransport for CleanupTransport {
        fn resolve<'a>(
            &'a self,
            _host: &'a str,
            _port: u16,
        ) -> BoxFuture<'a, Result<Vec<IpAddr>, DownloadError>> {
            Box::pin(async { Ok(vec![PUBLIC_IP]) })
        }
        fn send_pinned(
            &self,
            _request: HttpRequest,
            _addresses: Vec<IpAddr>,
        ) -> BoxFuture<'_, Result<HttpResponse, DownloadError>> {
            Box::pin(async move {
                Ok(HttpResponse {
                    status: self.status,
                    headers: self.headers.clone(),
                    body: DropBody {
                        flag: self.flag.clone(),
                        mode: self.mode,
                        sent: false,
                    }
                    .boxed(),
                })
            })
        }
    }
    for (status, headers, mode) in [
        (404, vec![], 0),
        (200, vec![("content-length".into(), "1000".into())], 0),
        (
            302,
            vec![("location".into(), "http://127.0.0.1/private".into())],
            0,
        ),
        (200, vec![], 0),
        (200, vec![], 1),
    ] {
        let flag = Arc::new(AtomicBool::new(false));
        let transport = CleanupTransport {
            status,
            headers,
            flag: flag.clone(),
            mode,
        };
        assert!(download(&transport, PUBLIC_URL, 3).await.is_err());
        assert!(flag.load(Ordering::SeqCst));
    }
    let flag = Arc::new(AtomicBool::new(false));
    let transport = CleanupTransport {
        status: 200,
        headers: vec![],
        flag: flag.clone(),
        mode: 2,
    };
    let mut future = Box::pin(download(&transport, PUBLIC_URL, 100));
    assert!(futures::poll!(&mut future).is_pending());
    assert!(!flag.load(Ordering::SeqCst));
    drop(future);
    assert!(flag.load(Ordering::SeqCst));
}
