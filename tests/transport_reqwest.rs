#![cfg(feature = "reqwest")]
use bytes::Bytes;
use futures::StreamExt;
use suprnova_ai::transport::{HttpRequest, HttpTransport, ReqwestTransport};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

#[tokio::test]
async fn reqwest_sends_request_and_streams_non_success_body() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        loop {
            let mut buf = [0; 1024];
            let n = socket.read(&mut buf).await.unwrap();
            assert_ne!(n, 0);
            request.extend_from_slice(&buf[..n]);
            if request.ends_with(b"hello") {
                break;
            }
        }
        let request = String::from_utf8(request).unwrap();
        assert!(request.starts_with("POST /test HTTP/1.1"));
        assert!(request.contains("x-test: yes"));
        socket.write_all(b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 4\r\nX-Test: response\r\nConnection: close\r\n\r\nslow").await.unwrap();
    });
    let response = ReqwestTransport::new()
        .unwrap()
        .send(HttpRequest {
            method: "POST".into(),
            url: format!("http://{addr}/test"),
            headers: vec![("X-Test".into(), "yes".into())],
            body: Bytes::from_static(b"hello"),
        })
        .await
        .unwrap();
    assert_eq!(response.status, 429);
    assert!(
        response
            .headers
            .contains(&("x-test".into(), "response".into()))
    );
    let chunks = response
        .body
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(chunks.concat(), b"slow");
    task.await.unwrap();
}

#[tokio::test]
async fn reqwest_does_not_follow_redirects() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0; 1024];
        assert_ne!(socket.read(&mut buf).await.unwrap(), 0);
        socket.write_all(b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/private\r\nContent-Length: 0\r\n\r\n").await.unwrap();
    });
    let response = ReqwestTransport::new()
        .unwrap()
        .send(HttpRequest {
            method: "GET".into(),
            url: format!("http://{addr}/"),
            headers: vec![],
            body: Bytes::new(),
        })
        .await
        .unwrap();
    assert_eq!(response.status, 302);
    task.await.unwrap();
}

#[tokio::test]
async fn reqwest_download_pins_destination_and_keeps_original_host() {
    use std::net::{IpAddr, Ipv4Addr};
    use suprnova_ai::models::files::DownloadTransport;
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0; 1024];
        let n = socket.read(&mut buf).await.unwrap();
        let request = String::from_utf8_lossy(&buf[..n]);
        assert!(request.contains(&format!("host: media.example.invalid:{port}")));
        socket
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            .await
            .unwrap();
    });
    let response = ReqwestTransport::new()
        .unwrap()
        .send_pinned(
            HttpRequest {
                method: "GET".into(),
                url: format!("http://media.example.invalid:{port}/image"),
                headers: vec![],
                body: Bytes::new(),
            },
            vec![IpAddr::V4(Ipv4Addr::LOCALHOST)],
        )
        .await
        .unwrap();
    assert_eq!(response.status, 200);
    let chunks = response
        .body
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(chunks.concat(), b"ok");
    task.await.unwrap();
}

#[tokio::test]
async fn reqwest_rejects_invalid_requests_without_exposing_url() {
    let error = ReqwestTransport::new()
        .unwrap()
        .send(HttpRequest {
            method: "GET".into(),
            url: "not a URL?key=secret".into(),
            headers: vec![],
            body: Bytes::new(),
        })
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        suprnova_ai::transport::TransportError::Request(_)
    ));
    assert!(!error.to_string().contains("secret"));
}

#[tokio::test]
async fn reqwest_streams_decoded_compressed_response_bytes() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buf = [0; 1024];
        assert_ne!(socket.read(&mut buf).await.unwrap(), 0);
        let compressed = include_bytes!("fixtures/body.gz");
        let headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            compressed.len()
        );
        socket.write_all(headers.as_bytes()).await.unwrap();
        socket.write_all(compressed).await.unwrap();
    });
    let response = ReqwestTransport::new()
        .unwrap()
        .send(HttpRequest {
            method: "GET".into(),
            url: format!("http://{addr}/"),
            headers: vec![],
            body: Bytes::new(),
        })
        .await
        .unwrap();
    let chunks = response
        .body
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(chunks.concat(), vec![b'x'; 256]);
    task.await.unwrap();
}
