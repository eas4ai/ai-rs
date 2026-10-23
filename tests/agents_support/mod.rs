#![allow(dead_code)]
use bytes::Bytes;
use futures::{FutureExt, StreamExt, future::BoxFuture, stream};
use serde_json::Value;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};
use suprnova_ai::{
    Error, Model, ProviderConfig, ProviderProtocol, StreamOptions,
    providers::{ConfiguredProvider, EventStream},
    transport::{HttpRequest, HttpResponse, HttpTransport, TransportError},
    types::{Event, Message, MessageRole, Part, ToolCallPart, events::replay_message_events},
};

pub struct ScriptedTransport {
    scripts: Mutex<VecDeque<Result<Vec<Event>, TransportError>>>,
    pub requests: Mutex<Vec<Value>>,
}
impl HttpTransport for ScriptedTransport {
    fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<HttpResponse, TransportError>> {
        async move {
            assert_eq!(request.url, "https://script.invalid/infer");
            self.requests
                .lock()
                .unwrap()
                .push(serde_json::from_slice(&request.body).unwrap());
            let events = self
                .scripts
                .lock()
                .unwrap()
                .pop_front()
                .expect("unexpected model call")?;
            Ok(HttpResponse {
                status: 200,
                headers: vec![],
                body: stream::iter([Ok(Bytes::from(serde_json::to_vec(&events).unwrap()))]).boxed(),
            })
        }
        .boxed()
    }
}
struct ScriptedProtocol(Arc<ScriptedTransport>);
#[async_trait::async_trait]
impl ProviderProtocol for ScriptedProtocol {
    fn protocol_class_id(&self) -> &str {
        "agent-test"
    }
    fn stream(
        &self,
        _model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        _provider: &dyn suprnova_ai::Provider,
    ) -> EventStream {
        let transport = self.0.clone();
        stream::once(async move {
            let response=transport.send(HttpRequest {method:"POST".into(),url:"https://script.invalid/infer".into(),headers:vec![],body:Bytes::from(serde_json::to_vec(&serde_json::json!({"messages":messages,"tools":options.tools,"output_schema":options.output_schema,"params":options.params})).unwrap())}).await?;
            let bytes=response.body.try_collect::<Vec<_>>().await?.concat();
            let events:Vec<Event>=serde_json::from_slice(&bytes)?;
            Ok::<_,Error>(stream::iter(events.into_iter().map(Ok)).boxed())
        }).try_flatten().boxed()
    }
}
use futures::TryStreamExt;
pub fn scripted(scripts: Vec<Vec<Event>>) -> (Model, Arc<ScriptedTransport>) {
    scripted_results(scripts.into_iter().map(Ok).collect())
}
pub fn scripted_results(
    scripts: Vec<Result<Vec<Event>, TransportError>>,
) -> (Model, Arc<ScriptedTransport>) {
    let transport = Arc::new(ScriptedTransport {
        scripts: Mutex::new(scripts.into()),
        requests: Mutex::new(vec![]),
    });
    let provider = Arc::new(ConfiguredProvider::new(
        "test",
        ProviderConfig::default(),
        Arc::new(ScriptedProtocol(transport.clone())),
        Some(transport.clone()),
    ));
    (Model::new("test-model", provider), transport)
}
pub fn response(message: Message) -> Vec<Event> {
    replay_message_events(&message)
}
pub fn text(value: &str) -> Vec<Event> {
    response(suprnova_ai::assistant_message(value))
}
pub fn calls(calls: &[(&str, &str, &str)]) -> Vec<Event> {
    response(Message::new(
        MessageRole::Assistant,
        calls
            .iter()
            .map(|(id, name, args)| Part::ToolCall(ToolCallPart::new(*id, *name, *args)))
            .collect(),
    ))
}
