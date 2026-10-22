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

use futures::{StreamExt, stream};
use serde_json::{Value, json};
use suprnova_ai::Error;
use suprnova_ai::agents::{
    AgentTool, Aggregate, BoundToolCall, HookRegistry, Runtime, ToolContext,
};
use suprnova_ai::types::events::{Aggregator, AggregatorFactory, StreamEnd};
use suprnova_ai::types::messages::{
    Message, MessageRole, SpecialToolResult, TextPart, ToolCallPart, ToolResultValue,
};
use suprnova_ai::types::{Event, ModelEventMeta};

#[tokio::test]
async fn status_and_text_streams_forward_each_partial_before_finishing() {
    for (aggregate, expected) in [
        (Aggregate::last(), json!("answer")),
        (Aggregate::concat("|"), json!("working|answer")),
    ] {
        let tool = AgentTool::stream(
            "status",
            "Report progress.",
            |_: Value, _| stream::iter([Ok("working"), Ok("answer")]),
            aggregate,
        )
        .unwrap();
        let (runtime, mut events) = Runtime::new();
        let context = ToolContext::new(runtime.clone(), HookRegistry::default(), "tc", "status");
        let task = tokio::spawn(async move {
            BoundToolCall::new(ToolCallPart::new("tc", "status", "{}"), tool)
                .execute(context)
                .await
        });
        for expected in [json!("working"), json!("answer")] {
            let Event::PartialToolCallResult(partial) = events.next().await.unwrap() else {
                panic!("expected partial")
            };
            assert_eq!(partial.value, expected);
            assert_eq!(partial.tool_call_id.as_deref(), Some("tc"));
            assert!(partial.aggregator_factory.is_some());
        }
        let result = task.await.unwrap().unwrap();
        assert_eq!(result.results[0].result.as_json(), Some(&expected));
        assert_eq!(
            result.results[0].model_input.as_ref().unwrap().as_json(),
            Some(&expected)
        );
        runtime.signal_done().await;
        assert!(events.next().await.is_none());
    }
}

#[tokio::test]
async fn empty_stream_status_is_null_and_text_is_empty() {
    for (aggregate, expected) in [
        (Aggregate::last(), Value::Null),
        (Aggregate::concat(""), json!("")),
    ] {
        let tool = AgentTool::stream(
            "empty",
            "Empty stream.",
            |_: Value, _| stream::empty::<Result<String, Error>>(),
            aggregate,
        )
        .unwrap();
        let (runtime, _events) = Runtime::new();
        let result = BoundToolCall::new(ToolCallPart::new("tc", "empty", "{}"), tool)
            .execute(ToolContext::new(
                runtime,
                HookRegistry::default(),
                "tc",
                "empty",
            ))
            .await
            .unwrap();
        assert_eq!(result.results[0].result.as_json(), Some(&expected));
        assert_eq!(
            result.results[0].get_model_input().as_json(),
            Some(&expected)
        );
    }
}

#[tokio::test]
async fn sub_agent_events_form_rich_results_with_text_model_input() {
    let message = Message::with_id(
        "inner",
        MessageRole::Assistant,
        vec![TextPart::new("Mars has two moons.").into()],
    );
    let event = Event::StreamEnd(StreamEnd {
        event: ModelEventMeta::with_message(message),
        ..Default::default()
    });
    let tool = AgentTool::stream(
        "research",
        "Sub-agent.",
        move |_: Value, _| stream::iter([Ok(event.clone())]),
        Aggregate::messages(),
    )
    .unwrap();
    let (runtime, mut events) = Runtime::new();
    let context = ToolContext::new(runtime, HookRegistry::default(), "outer", "research");
    let result = BoundToolCall::new(ToolCallPart::new("outer", "research", "{}"), tool.clone())
        .execute(context)
        .await
        .unwrap();
    let ToolResultValue::Special(SpecialToolResult::Messages(bundle)) = &result.results[0].result
    else {
        panic!("expected rich message bundle")
    };
    assert_eq!(bundle.messages[0].text(), "Mars has two moons.");
    assert_eq!(
        result.results[0].get_model_input().as_json(),
        Some(&json!("Mars has two moons."))
    );
    let Event::PartialToolCallResult(partial) = events.next().await.unwrap() else {
        panic!("expected wrapped inner event")
    };
    assert_eq!(partial.value["kind"], "stream_end");
    assert_eq!(partial.tool_call_id.as_deref(), Some("outer"));
    let mut reducer = partial.aggregator_factory.unwrap().create();
    reducer.feed(partial.value);
    assert_eq!(reducer.get_model_input(), json!("Mars has two moons."));
    assert_eq!(
        tool.model_input(&result.results[0].result)
            .unwrap()
            .unwrap()
            .as_json(),
        Some(&json!("Mars has two moons."))
    );
}

#[tokio::test]
async fn custom_reducer_keeps_snapshot_distinct_from_model_input() {
    #[derive(Default)]
    struct Sum(i64);
    impl Aggregator<Value> for Sum {
        type Snapshot = Value;
        type ModelInput = Value;
        fn feed(&mut self, value: Value) {
            self.0 += value.as_i64().unwrap();
        }
        fn snapshot(&self) -> Value {
            json!({"sum":self.0})
        }
        fn to_model_input(snapshot: Value) -> Value {
            snapshot["sum"].clone()
        }
    }
    let aggregate =
        Aggregate::new(AggregatorFactory::new(Sum::default)).with_model_input(|value| {
            Ok(ToolResultValue::Json(
                value.as_json().unwrap()["sum"].clone(),
            ))
        });
    let tool = AgentTool::stream(
        "sum",
        "Sum.",
        |_: Value, _| stream::iter([Ok(1), Ok(2)]),
        aggregate,
    )
    .unwrap();
    let (runtime, _events) = Runtime::new();
    let result = BoundToolCall::new(ToolCallPart::new("tc", "sum", "{}"), tool)
        .execute(ToolContext::new(
            runtime,
            HookRegistry::default(),
            "tc",
            "sum",
        ))
        .await
        .unwrap();
    assert_eq!(result.results[0].result.as_json(), Some(&json!({"sum":3})));
    assert_eq!(
        result.results[0].get_model_input().as_json(),
        Some(&json!(3))
    );
}

#[tokio::test]
async fn stream_failure_retains_error_and_already_emitted_partials() {
    let tool = AgentTool::stream(
        "fail",
        "Fail midstream.",
        |_: Value, _| stream::iter([Ok("first"), Err(Error::Configuration("boom".into()))]),
        Aggregate::last(),
    )
    .unwrap();
    let (runtime, mut events) = Runtime::new();
    let result = BoundToolCall::new(ToolCallPart::new("tc", "fail", "{}"), tool)
        .execute(ToolContext::new(
            runtime,
            HookRegistry::default(),
            "tc",
            "fail",
        ))
        .await
        .unwrap();
    assert!(result.results[0].is_error());
    assert!(result.exception.unwrap().as_error().is::<Error>());
    let Event::PartialToolCallResult(partial) = events.next().await.unwrap() else {
        panic!("expected first partial")
    };
    assert_eq!(partial.value, json!("first"));
}

#[tokio::test]
async fn yield_from_routes_labeled_partials_and_returns_model_facing_value() {
    use suprnova_ai::agents::tools::yield_from;
    let (runtime, mut events) = Runtime::new();
    let result = yield_from(
        &runtime,
        stream::iter([Ok("a"), Ok("b")]),
        Aggregate::concat("|"),
        None,
        None,
        json!("researcher"),
    )
    .await
    .unwrap();
    assert_eq!(result.as_json(), Some(&json!("a|b")));
    for value in ["a", "b"] {
        let Event::PartialToolCallResult(partial) = events.next().await.unwrap() else {
            panic!("expected partial")
        };
        assert_eq!(partial.label, json!("researcher"));
        assert_eq!(partial.value, json!(value));
        assert!(partial.tool_call_id.is_none());
    }
}

#[tokio::test]
async fn nested_options_inherit_hook_registry_and_middleware() {
    use std::sync::Arc;
    use suprnova_ai::agents::Middleware;
    struct Marker;
    impl Middleware for Marker {}
    let (runtime, _events) = Runtime::new();
    let hooks = HookRegistry::default();
    hooks.resolve_hook("shared", json!(42)).unwrap();
    let mut context = ToolContext::new(runtime, hooks, "tc", "outer");
    context.middleware.push(Arc::new(Marker));
    let nested = context.nested_options();
    assert_eq!(nested.middleware.len(), 1);
    assert!(Arc::ptr_eq(&nested.middleware[0], &context.middleware[0]));
    let value: Value = nested
        .hook_registry
        .unwrap()
        .hook("shared", Default::default(), None, &context.runtime)
        .await
        .unwrap();
    assert_eq!(value, json!(42));
}
