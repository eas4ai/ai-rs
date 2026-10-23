use std::sync::{Arc, Mutex};

use bytes::Bytes;
use futures::{StreamExt, TryStreamExt, future::BoxFuture, stream};
use serde_json::{Value, json};
use suprnova_ai::{
    Error, Model, Provider, ProviderConfig, ProviderProtocol, StreamOptions,
    providers::{ConfiguredProvider, EventStream},
    transport::{Body, HttpRequest, HttpResponse, HttpTransport, TransportError, sse},
    types::{Event, Message},
    ui::ai_sdk::{UI_MESSAGE_STREAM_HEADERS, outbound_stream::format_done_sse, to_sse},
};

struct ScriptedTransport {
    body: Mutex<Option<Body>>,
    requests: Mutex<Vec<HttpRequest>>,
}

impl HttpTransport for ScriptedTransport {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        Box::pin(async move {
            self.requests.lock().unwrap().push(request);
            Ok(HttpResponse {
                status: 200,
                headers: vec![("content-type".into(), "text/event-stream".into())],
                body: self
                    .body
                    .lock()
                    .unwrap()
                    .take()
                    .expect("one scripted response"),
            })
        })
    }
}

struct ScriptedProtocol(Arc<ScriptedTransport>);

#[async_trait::async_trait]
impl ProviderProtocol for ScriptedProtocol {
    fn protocol_class_id(&self) -> &str {
        "ui-script"
    }

    fn stream(
        &self,
        _model: &Model,
        messages: Vec<Message>,
        _options: StreamOptions,
        _provider: &dyn Provider,
    ) -> EventStream {
        let transport = self.0.clone();
        stream::once(async move {
            let response = transport
                .send(HttpRequest {
                    method: "POST".into(),
                    url: "https://script.invalid/chat".into(),
                    headers: vec![],
                    body: Bytes::from(serde_json::to_vec(&messages)?),
                })
                .await?;
            Ok::<_, Error>(
                sse::parse(response.body)
                    .map_err(Error::from)
                    .and_then(|event| async move {
                        serde_json::from_str::<Event>(&event.data).map_err(Error::from)
                    }),
            )
        })
        .try_flatten()
        .boxed()
    }
}

fn model(body: Body) -> (Model, Arc<ScriptedTransport>) {
    let transport = Arc::new(ScriptedTransport {
        body: Mutex::new(Some(body)),
        requests: Mutex::new(vec![]),
    });
    let protocol = Arc::new(ScriptedProtocol(transport.clone()));
    let provider = Arc::new(ConfiguredProvider::new(
        "ui-script",
        ProviderConfig::default(),
        protocol,
        Some(transport.clone()),
    ));
    (Model::new("script", provider), transport)
}

#[tokio::test]
async fn fake_transport_flows_through_core_hydration_and_ui_sse_lazily() {
    let frames = [
        json!({"kind":"stream_start","message":{"id":"server"}}),
        json!({"kind":"text_start","message":{"id":"server"},"block_id":"txt"}),
        json!({"kind":"text_delta","message":{"id":"server"},"block_id":"txt","chunk":"café 👋"}),
        json!({"kind":"stream_end","message":{"id":"server"},"finish_reason":"stop"}),
    ];
    let wire: String = frames.iter().map(|v| format!("data: {v}\n\n")).collect();
    // Split at every byte, including inside UTF-8, through the actual SSE parser.
    let body = stream::iter(
        wire.into_bytes()
            .into_iter()
            .map(|b| Ok(Bytes::from(vec![b]))),
    )
    .boxed();
    let (model, transport) = model(body);
    let input = vec![suprnova_ai::user_message("hello")];
    let mut ui = to_sse(suprnova_ai::stream(&model, input, StreamOptions::default()));
    assert!(transport.requests.lock().unwrap().is_empty());
    let first = ui.next().await.unwrap().unwrap();
    assert_eq!(transport.requests.lock().unwrap().len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(first.trim_start_matches("data: ")).unwrap()["type"],
        "start"
    );
    let rest: Vec<String> = ui.try_collect().await.unwrap();
    assert_eq!(rest.last().unwrap(), format_done_sse());
    assert!(
        rest.iter()
            .any(|frame| frame.contains("caf\\u00e9 \\ud83d\\udc4b"))
    );
    let finish: Value = serde_json::from_str(
        rest.iter()
            .find(|s| s.contains("\"type\": \"finish\""))
            .unwrap()
            .trim_start_matches("data: "),
    )
    .unwrap();
    assert_eq!(finish["finishReason"], "stop");
    assert_eq!(
        finish["messageMetadata"]["aiPython"]["sourceMessages"][0]["partIds"],
        json!(["txt"])
    );
    let requests = transport.requests.lock().unwrap();
    let sent: Vec<Message> = serde_json::from_slice(&requests[0].body).unwrap();
    assert_eq!(sent[0].text(), "hello");
    assert_eq!(
        UI_MESSAGE_STREAM_HEADERS,
        [
            ("Content-Type", "text/event-stream"),
            ("Cache-Control", "no-cache"),
            ("Connection", "keep-alive"),
            ("x-vercel-ai-ui-message-stream", "v1"),
            ("x-accel-buffering", "no"),
        ]
    );
}

#[tokio::test]
async fn failed_transport_never_emits_a_successful_finish_or_done() {
    let body = stream::iter([
        Ok(Bytes::from_static(
            b"data: {\"kind\":\"text_delta\",\"chunk\":\"partial\",\"block_id\":\"t\"}\n\n",
        )),
        Err(TransportError::Body("scripted failure".into())),
    ])
    .boxed();
    let (model, _) = model(body);
    let outputs: Vec<_> = to_sse(suprnova_ai::stream(
        &model,
        vec![],
        StreamOptions::default(),
    ))
    .collect()
    .await;
    assert!(
        matches!(outputs.last(), Some(Err(Error::Transport(TransportError::Body(message)))) if message == "scripted failure")
    );
    assert!(
        outputs
            .iter()
            .filter_map(|o| o.as_ref().ok())
            .all(|frame| !frame.contains("\"type\": \"finish\"") && frame != format_done_sse())
    );
}
