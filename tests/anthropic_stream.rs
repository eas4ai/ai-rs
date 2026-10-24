mod anthropic_support;
use anthropic_support::*;
use futures::StreamExt;
use serde_json::{Value, json};
use suprnova_ai::{
    Error, StreamOptions,
    errors::ProviderErrorKind,
    transport::{HttpResponse, TransportError},
    types::{Event, Part},
};

#[tokio::test]
async fn builtin_events_are_ordered_and_aggregate_call_parts_and_results() {
    let events = complete(
        vec![
            block_start(
                0,
                json!({"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}),
            ),
            delta(
                0,
                json!({"type":"input_json_delta","partial_json":"{\"query\""}),
            ),
            delta(
                0,
                json!({"type":"input_json_delta","partial_json":":\"weather\"}"}),
            ),
            stop(0),
            block_start(
                1,
                json!({"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"title":"Forecast","url":"https://example.com"}]}),
            ),
            stop(1),
        ],
        None,
    );
    let m = model(ScriptedTransport::events(events));
    let mut s = suprnova_ai::stream(
        &m,
        vec![suprnova_ai::user_message("Hi")],
        StreamOptions::default(),
    );
    let mut kinds = vec![];
    while let Some(e) = s.next().await {
        let e = e.unwrap();
        kinds.push(serde_json::to_value(&e).unwrap()["kind"].clone());
        if let Event::BuiltinToolEnd(end) = e {
            assert_eq!(end.tool_call.tool_args, "{\"query\":\"weather\"}");
        }
    }
    assert_eq!(
        kinds,
        json!([
            "stream_start",
            "builtin_tool_start",
            "builtin_tool_delta",
            "builtin_tool_delta",
            "builtin_tool_end",
            "builtin_tool_result",
            "stream_end"
        ])
        .as_array()
        .unwrap()
        .clone()
    );
    let call = s.message().builtin_tool_calls()[0];
    assert_eq!(call.tool_call_id, "srvtoolu_1");
    assert_eq!(call.tool_name, "web_search");
    assert!(call.provider_metadata.is_none());
    // A web search result in TypeScript's shape (ANT-008).
    let result = s.message().builtin_tool_returns()[0];
    assert_eq!(result.tool_name, "web_search");
    assert_eq!(
        result.result,
        json!([{"title":"Forecast","url":"https://example.com","pageAge":null}])
    );
    assert!(result.provider_metadata.is_none());
}
#[tokio::test]
async fn split_signatures_text_tools_and_usage_round_trip() {
    let mut events = complete(
        vec![
            block_start(0, json!({"type":"thinking","thinking":"","signature":""})),
            delta(0, json!({"type":"thinking_delta","thinking":"hidden"})),
            delta(0, json!({"type":"signature_delta","signature":"si"})),
            delta(0, json!({"type":"signature_delta","signature":"g"})),
            stop(0),
            block_start(1, json!({"type":"text","text":""})),
            delta(1, json!({"type":"text_delta","text":"héllo"})),
            stop(1),
            block_start(
                2,
                json!({"type":"tool_use","id":"tc1","name":"weather","input":{}}),
            ),
            delta(
                2,
                json!({"type":"input_json_delta","partial_json":"{\"city\":\"SF\"}"}),
            ),
            stop(2),
        ],
        Some("tool_use"),
    );
    events[0]["message"]["usage"] = json!({"input_tokens":10,"output_tokens":1,"cache_read_input_tokens":5,"cache_creation_input_tokens":3,"server_tool_use":{"web_search_requests":1}});
    let last = events.len() - 2;
    events[last]["usage"] = json!({"output_tokens":20});
    let m = model(ScriptedTransport::events(events));
    let mut s = suprnova_ai::stream(&m, vec![], StreamOptions::default());
    while let Some(e) = s.next().await {
        e.unwrap();
    }
    assert_eq!(s.text(), "héllo");
    assert_eq!(s.tool_calls()[0].tool_args, "{\"city\":\"SF\"}");
    let Part::Reasoning(p) = &s.message().parts[0] else {
        panic!()
    };
    assert_eq!(p.text, "hidden");
    assert_eq!(
        p.provider_metadata.as_ref().unwrap()["anthropic"],
        json!({"signature":"sig"})
    );
    let usage = s.usage().unwrap();
    // Input counts the cache writes and reads too (ANT-014).
    assert_eq!(usage.input_tokens, 18);
    assert_eq!(usage.output_tokens, 20);
    assert_eq!(usage.cache_read_tokens, Some(5));
    assert_eq!(usage.cache_write_tokens, Some(3));
    assert_eq!(
        usage.raw.as_ref().unwrap()["server_tool_use"],
        json!({"web_search_requests":1})
    );
    assert_eq!(s.response_id(), Some("msg_test"));
    assert_eq!(s.response_model(), Some("claude-test"));
    let (b, _) = capture(vec![s.message().clone()], StreamOptions::default()).await;
    assert_eq!(b["messages"][0]["content"][0]["signature"], "sig");
}
#[tokio::test]
async fn finish_reasons_response_identity_and_unknown_reason_metadata() {
    for (reason, expected) in [
        (Some("end_turn"), Some("stop")),
        (Some("stop_sequence"), Some("stop")),
        (Some("max_tokens"), Some("length")),
        (Some("model_context_window_exceeded"), Some("length")),
        (Some("tool_use"), Some("tool_call")),
        (Some("refusal"), Some("content_filter")),
        (Some("pause_turn"), Some("stop")),
        (None, None),
    ] {
        let m = model(ScriptedTransport::events(complete(vec![], reason)));
        let events = m
            .provider
            .stream(&m, vec![], StreamOptions::default())
            .collect::<Vec<_>>()
            .await;
        let Event::StreamEnd(end) = events.last().unwrap().as_ref().unwrap() else {
            panic!()
        };
        assert_eq!(end.finish_reason.as_deref(), expected);
        assert_eq!(end.response_id.as_deref(), Some("msg_test"));
        assert_eq!(end.response_model.as_deref(), Some("claude-test"));
        // The raw stop reason is in the final metadata (ANT-014).
        assert_eq!(
            end.event.provider_metadata.as_ref().unwrap()["anthropic"]["stopReason"],
            json!(reason)
        );
    }
}
#[tokio::test]
async fn all_server_result_types_and_unknown_blocks() {
    // Each error result in TypeScript's shape; an unknown result block is
    // not delivered (ANT-008), and redacted thinking keeps its data (ANT-013).
    for (ty, name, error_type) in [
        (
            "web_search_tool_result",
            "web_search",
            "web_search_tool_result_error",
        ),
        (
            "web_fetch_tool_result",
            "web_fetch",
            "web_fetch_tool_result_error",
        ),
        (
            "code_execution_tool_result",
            "code_execution",
            "code_execution_tool_result_error",
        ),
        ("bash_tool_result", "", ""),
        ("memory_tool_result", "", ""),
    ] {
        let m = model(ScriptedTransport::events(complete(
            vec![
                block_start(0, json!({"type":"redacted_thinking","data":"redacted"})),
                stop(0),
                delta(
                    10,
                    json!({"type":"input_json_delta","partial_json":"ignored"}),
                ),
                block_start(
                    1,
                    json!({"type":ty,"tool_use_id":"unknown","content":{"type":error_type,"error_code":"unavailable"}}),
                ),
                stop(1),
            ],
            Some("end_turn"),
        )));
        let mut s = suprnova_ai::stream(&m, vec![], StreamOptions::default());
        while let Some(e) = s.next().await {
            e.unwrap();
        }
        let Part::Reasoning(reasoning) = &s.message().parts[0] else {
            panic!("{:?}", s.message().parts)
        };
        assert_eq!(
            reasoning.provider_metadata.as_ref().unwrap()["anthropic"],
            json!({"redactedData": "redacted"})
        );
        let results = s.message().builtin_tool_returns();
        if name.is_empty() {
            assert!(results.is_empty(), "{ty}");
            continue;
        }
        assert_eq!(results[0].tool_name, name);
        assert!(results[0].is_error);
        assert_eq!(
            results[0].result,
            json!({"type": error_type, "errorCode": "unavailable"})
        );
    }
}
#[tokio::test]
async fn sse_parser_handles_every_byte_boundary() {
    let events = complete(
        vec![
            block_start(0, json!({"type":"text","text":""})),
            delta(0, json!({"type":"text_delta","text":"hé🙂"})),
            stop(0),
        ],
        Some("end_turn"),
    );
    let bytes = wire(&events);
    for split in 0..=bytes.len() {
        let fake = ScriptedTransport::new(vec![Ok(HttpResponse {
            status: 200,
            headers: vec![],
            body: body(vec![bytes[..split].to_vec(), bytes[split..].to_vec()]),
        })]);
        let m = model(fake);
        let mut s = suprnova_ai::stream(&m, vec![], StreamOptions::default());
        while let Some(e) = s.next().await {
            e.unwrap();
        }
        assert_eq!(s.text(), "hé🙂", "split {split}");
    }
}
#[tokio::test]
async fn status_transport_malformed_and_truncated_stream_errors() {
    let cases = vec![
        (
            ScriptedTransport::new(vec![Err(TransportError::Timeout)]),
            ProviderErrorKind::Timeout,
        ),
        (
            ScriptedTransport::new(vec![Err(TransportError::Connection("offline".into()))]),
            ProviderErrorKind::Connection,
        ),
        (
            ScriptedTransport::new(vec![Ok(response(
                529,
                json!({"error":{"message":"overloaded","type":"overloaded_error"}}),
            ))]),
            ProviderErrorKind::Overloaded,
        ),
        (
            ScriptedTransport::new(vec![Ok(response(
                404,
                json!({"error":{"message":"missing"}}),
            ))]),
            ProviderErrorKind::ModelNotFound,
        ),
        (
            ScriptedTransport::events(vec![start()]),
            ProviderErrorKind::IncompleteResponse,
        ),
        (
            ScriptedTransport::new(vec![Ok(HttpResponse {
                status: 200,
                headers: vec![],
                body: body(vec![b"data: {broken}\n\n".to_vec()]),
            })]),
            ProviderErrorKind::Response,
        ),
        (
            ScriptedTransport::events(vec![
                start(),
                json!({"type":"error","error":{"type":"api_error","message":"broken"}}),
            ]),
            // An api_error has status 500 (ANT-013).
            ProviderErrorKind::InternalServer,
        ),
    ];
    for (fake, kind) in cases {
        let m = model(fake);
        let values = m
            .provider
            .stream(&m, vec![], StreamOptions::default())
            .collect::<Vec<_>>()
            .await;
        let err = values
            .into_iter()
            .find_map(Result::err)
            .expect("expected error");
        let Error::Provider(err) = err else {
            panic!("wrong error {err:?}")
        };
        assert_eq!(err.kind, kind);
        assert_eq!(err.provider.as_deref(), Some("anthropic"));
    }
}
#[derive(serde::Deserialize, schemars::JsonSchema, Debug, PartialEq)]
struct Answer {
    answer: i32,
}
#[tokio::test]
async fn structured_output_uses_output_config_and_core_decoding() {
    let fake = ScriptedTransport::events(complete(
        vec![
            block_start(0, json!({"type":"text","text":""})),
            delta(0, json!({"type":"text_delta","text":"{\"answer\":42}"})),
            stop(0),
        ],
        Some("end_turn"),
    ));
    let m = model(fake.clone());
    let params = suprnova_ai::models::params::InferenceRequestParams::default()
        .with_reasoning_effort(suprnova_ai::models::params::ProviderValue::Value(Some(
            "high".into(),
        )));
    let mut s = suprnova_ai::stream(
        &m,
        vec![],
        StreamOptions {
            params: Some(params),
            ..Default::default()
        }
        .with_output::<Answer>(),
    );
    while let Some(e) = s.next().await {
        e.unwrap();
    }
    assert_eq!(s.output::<Answer>().unwrap(), Answer { answer: 42 });
    let b: Value = serde_json::from_slice(&fake.request().body).unwrap();
    assert_eq!(b["output_config"]["effort"], "high");
    assert_eq!(b["output_config"]["format"]["type"], "json_schema");
    assert_eq!(
        b["output_config"]["format"]["schema"]["additionalProperties"],
        false
    );
    assert_eq!(
        b["output_config"]["format"]["schema"]["properties"]["answer"]["type"],
        "integer"
    );
    assert!(b.get("output_format").is_none());
}

#[tokio::test]
async fn lazy_stream_cancels_its_body_on_drop() {
    use std::{
        pin::Pin,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        task::{Context, Poll},
    };
    struct PendingBody(Arc<AtomicBool>);
    impl futures::Stream for PendingBody {
        type Item = Result<bytes::Bytes, TransportError>;
        fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }
    impl Drop for PendingBody {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    let dropped = Arc::new(AtomicBool::new(false));
    let fake = ScriptedTransport::new(vec![Ok(HttpResponse {
        status: 200,
        headers: vec![],
        body: PendingBody(dropped.clone()).boxed(),
    })]);
    let m = model(fake.clone());
    let mut s = m.provider.stream(&m, vec![], StreamOptions::default());
    assert!(fake.requests.lock().unwrap().is_empty());
    assert!(matches!(
        s.next().await.unwrap().unwrap(),
        Event::StreamStart(_)
    ));
    assert!(!dropped.load(Ordering::SeqCst));
    drop(s);
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(fake.requests.lock().unwrap().len(), 1);
}
#[tokio::test]
async fn malformed_fields_usage_and_body_fail_without_a_success_end() {
    // A delta of a block that never started is passed over, as TypeScript
    // does, and the stream then ends without its stop.
    let m = model(ScriptedTransport::events(vec![
        start(),
        delta(0, json!({"type":"text_delta"})),
    ]));
    let values = m
        .provider
        .stream(&m, vec![], StreamOptions::default())
        .collect::<Vec<_>>()
        .await;
    let Error::Provider(err) = values.into_iter().find_map(Result::err).unwrap() else {
        panic!()
    };
    assert_eq!(err.kind, ProviderErrorKind::IncompleteResponse);
    for event in [
        block_start(0, json!({"type":"tool_use","name":"missing-id"})),
        json!({"type":"message_delta","usage":{"input_tokens":-1}}),
        json!({"type":"content_block_start","index":-1,"content_block":{"type":"text"}}),
    ] {
        let m = model(ScriptedTransport::events(vec![start(), event]));
        let values = m
            .provider
            .stream(&m, vec![], StreamOptions::default())
            .collect::<Vec<_>>()
            .await;
        assert!(values.iter().all(|e| !matches!(e, Ok(Event::StreamEnd(_)))));
        let Error::Provider(err) = values.into_iter().find_map(Result::err).unwrap() else {
            panic!()
        };
        assert_eq!(err.kind, ProviderErrorKind::Response);
    }
    let m = model(ScriptedTransport::new(vec![Ok(HttpResponse {
        status: 200,
        headers: vec![],
        body: futures::stream::iter([
            Ok(bytes::Bytes::from(wire(&[start()]))),
            Err(TransportError::Body("disconnected".into())),
        ])
        .boxed(),
    })]));
    let values = m
        .provider
        .stream(&m, vec![], StreamOptions::default())
        .collect::<Vec<_>>()
        .await;
    let Error::Provider(err) = values.into_iter().find_map(Result::err).unwrap() else {
        panic!()
    };
    assert_eq!(err.kind, ProviderErrorKind::Connection);
}
#[derive(serde::Deserialize, schemars::JsonSchema)]
#[allow(dead_code)]
struct NestedOutput {
    child: Option<Child>,
    values: Vec<Child>,
}
#[derive(serde::Deserialize, schemars::JsonSchema)]
#[allow(dead_code)]
struct Child {
    name: String,
}
#[tokio::test]
async fn structured_output_lowers_nested_definitions_nullable_types_and_constraints() {
    let (b, _) = capture(
        vec![],
        StreamOptions::default().with_output::<NestedOutput>(),
    )
    .await;
    let schema = &b["output_config"]["format"]["schema"];
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(schema["$defs"]["Child"]["additionalProperties"], false);
    assert!(schema["properties"]["child"]["anyOf"].is_array());
    assert_eq!(
        schema["properties"]["values"]["items"]["$ref"],
        "#/$defs/Child"
    );
    let schema=serde_json::from_value(json!({"type":"object","properties":{"value":{"type":["integer","null"],"minimum":0}},"required":["value"]})).unwrap();
    let (b, _) = capture(
        vec![],
        StreamOptions {
            output_schema: Some(schema),
            ..Default::default()
        },
    )
    .await;
    // A type list stays, and a constraint goes into the description
    // (ANT-005).
    assert_eq!(
        b["output_config"]["format"]["schema"]["properties"]["value"]["type"],
        json!(["integer", "null"])
    );
    assert_eq!(
        b["output_config"]["format"]["schema"]["properties"]["value"]["description"],
        "minimum: 0."
    );
}
#[tokio::test]
async fn a_done_marker_before_message_stop_is_invalid_json() {
    let fake = ScriptedTransport::new(vec![Ok(HttpResponse {
        status: 200,
        headers: vec![],
        body: body(vec![
            format!("event: message_start\ndata: {}\n\n", start()).into_bytes(),
            b"data: [DONE]\n\n".to_vec(),
        ]),
    })]);
    let m = model(fake);
    let values = m
        .provider
        .stream(&m, vec![], StreamOptions::default())
        .collect::<Vec<_>>()
        .await;
    let Error::Provider(err) = values.into_iter().find_map(Result::err).unwrap() else {
        panic!()
    };
    assert_eq!(err.kind, ProviderErrorKind::Response);
    assert!(!err.is_retryable);
}
