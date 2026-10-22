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

use bytes::Bytes;
use futures::{StreamExt, stream};
use suprnova_ai::transport::{Body, HttpRequest, TransportError, sse};

fn body(chunks: Vec<&[u8]>) -> Body {
    stream::iter(
        chunks
            .into_iter()
            .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
            .collect::<Vec<_>>(),
    )
    .boxed()
}

#[tokio::test]
async fn sse_parses_every_chunk_boundary_and_mixed_line_endings() {
    let input =
        "\u{feff}: comment\r\nevent: text\rid: event-1\nretry: 123\ndata: hé\r\ndata: llo\n\n"
            .as_bytes();
    for split in 0..=input.len() {
        let events = sse::parse(body(vec![&input[..split], &input[split..]]))
            .collect::<Vec<_>>()
            .await;
        assert_eq!(events.len(), 1, "split {split}");
        let event = events[0].as_ref().unwrap();
        assert_eq!(event.data, "hé\nllo");
        assert_eq!(event.event, "text");
        assert_eq!(event.id.as_deref(), Some("event-1"));
        assert_eq!(event.retry, Some(123));
    }
}

#[tokio::test]
async fn sse_control_fields_persist_and_events_reset() {
    let input = b"id: a\nretry: 30\nevent: first\ndata:\n\nid: bad\0id\nretry: -1\ndata: next\n\nid:\ndata:last\n\n";
    let events = sse::parse(body(vec![input])).collect::<Vec<_>>().await;
    let events = events.into_iter().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].data, "");
    assert_eq!(events[1].id.as_deref(), Some("a"));
    assert_eq!(events[1].event, "message");
    assert_eq!(events[1].retry, Some(30));
    assert_eq!(events[2].id.as_deref(), Some(""));
}

#[tokio::test]
async fn sse_discards_unterminated_events_and_ignores_unknown_fields() {
    let events = sse::parse(body(vec![b"unknown:x\n:heartbeat\n\ndata:partial\n"]))
        .collect::<Vec<_>>()
        .await;
    assert!(events.is_empty());
}

#[tokio::test]
async fn sse_propagates_transport_utf8_and_limit_errors() {
    let events = sse::parse(stream::iter(vec![Err(TransportError::Timeout)]).boxed())
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events[0], Err(TransportError::Timeout)));
    let events = sse::parse(body(vec![b"data:\xff\n\n"]))
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events[0], Err(TransportError::Protocol(_))));
    let events = sse::parse_with_limit(body(vec![b"data:long\n\n"]), 4)
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events[0], Err(TransportError::Limit(4))));
    let events = sse::parse_with_limit(body(vec![b"data:123\ndata:456\ndata:789\n\n"]), 9)
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(events[0], Err(TransportError::Limit(9))));
}

#[test]
fn request_debug_redacts_sensitive_data() {
    let request = HttpRequest {
        method: "POST".into(),
        url: "https://secret.example/?key=secret".into(),
        headers: vec![("Authorization".into(), "Bearer secret".into())],
        body: Bytes::from_static(b"secret"),
    };
    assert!(!format!("{request:?}").contains("secret"));
}

#[tokio::test]
async fn sse_yields_prior_events_before_later_failure_in_same_chunk() {
    let events = sse::parse(body(vec![b"data:ok\n\ndata:\xff\n\n"]))
        .collect::<Vec<_>>()
        .await;
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].as_ref().unwrap().data, "ok");
    assert!(matches!(events[1], Err(TransportError::Protocol(_))));
}
