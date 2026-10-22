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

use serde_json::{Value, json};
use suprnova_ai::types::{
    events::{Aggregator, AggregatorFactory, ValueAggregator},
    messages::{MessageBundle, SpecialToolResult, ToolResultValue},
};

struct JsonSnapshot;
impl Aggregator<Value> for JsonSnapshot {
    type Snapshot = Value;
    type ModelInput = Value;
    fn feed(&mut self, _: Value) {}
    fn snapshot(&self) -> Value {
        json!({"type":"messages","messages":[]})
    }
    fn to_model_input(snapshot: Value) -> Value {
        snapshot
    }
}

struct TypedSnapshot;
impl ValueAggregator for TypedSnapshot {
    fn feed(&mut self, _: Value) {}
    fn snapshot(&self) -> Value {
        json!({"type":"messages","messages":[]})
    }
    fn snapshot_result(&self) -> ToolResultValue {
        MessageBundle::new(vec![]).into()
    }
    fn get_model_input(&self) -> Value {
        Value::Null
    }
}

#[test]
fn json_and_typed_snapshots_keep_distinct_runtime_identity() {
    let plain = AggregatorFactory::new(|| JsonSnapshot).create();
    assert_eq!(
        plain.snapshot_result(),
        ToolResultValue::Json(plain.snapshot())
    );
    let typed = AggregatorFactory::from_boxed(|| Box::new(TypedSnapshot)).create();
    assert_eq!(typed.snapshot(), plain.snapshot());
    assert!(
        matches!(typed.snapshot_result(), ToolResultValue::Special(SpecialToolResult::Messages(bundle)) if bundle.messages.is_empty())
    );
}
