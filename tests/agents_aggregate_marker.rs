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

use futures::stream;
use serde_json::{Value, json};
use suprnova_ai::agents::{AgentTool, Aggregate, ConcatAggregator, LastAggregator};
use suprnova_ai::types::events::Aggregator;

#[test]
fn marker_factories_capture_delimiters_and_create_independent_reducers() {
    let marker = Aggregate::concat("|");
    let mut first = marker.factory().create();
    first.feed(json!("a"));
    first.feed(json!("b"));
    assert_eq!(first.snapshot(), json!("a|b"));
    assert_eq!(marker.factory().create().snapshot(), json!(""));
    let mut status = Aggregate::last().factory().create();
    assert_eq!(status.snapshot(), Value::Null);
    status.feed(json!(1));
    status.feed(json!(2));
    assert_eq!(status.snapshot(), json!(2));
}

#[test]
fn typed_reducers_preserve_their_snapshot_types() {
    let mut concat = ConcatAggregator::new("\n");
    concat.feed("a".into());
    concat.feed("b".into());
    assert_eq!(concat.snapshot(), "a\nb");
    let mut last = LastAggregator::<u32>::default();
    assert_eq!(last.snapshot(), None);
    last.feed(7);
    assert_eq!(last.snapshot(), Some(7));
}

#[test]
fn streaming_tools_reject_a_second_model_conversion() {
    let tool = AgentTool::stream(
        "status",
        "Progress.",
        |_: Value, _| stream::iter([Ok("done")]),
        Aggregate::last(),
    )
    .unwrap();
    assert!(tool.with_model_input(|value| Ok(value.clone())).is_err());
}
