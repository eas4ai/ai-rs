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

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use suprnova_ai::types::{Event, Message, tools::Tool, usage::Usage};

fn fixtures() -> Value {
    serde_json::from_str(include_str!("fixtures/python-0.8.0.json")).expect("valid Python fixture")
}

fn assert_roundtrip<T: DeserializeOwned + Serialize>(key: &str) {
    for (index, value) in fixtures()[key].as_array().unwrap().iter().enumerate() {
        let restored: T = serde_json::from_value(value.clone())
            .unwrap_or_else(|error| panic!("{key}[{index}]: {error}"));
        let encoded = serde_json::to_value(restored).unwrap();
        assert_eq!(
            &encoded, value,
            "{key}[{index}] must retain canonical Python JSON"
        );
    }
}

#[test]
fn every_event_variant_roundtrips_canonical_python_json() {
    assert_roundtrip::<Event>("events");
}
#[test]
fn all_message_roles_and_parts_roundtrip_canonical_python_json() {
    assert_roundtrip::<Message>("messages");
}
#[test]
fn function_and_provider_tools_roundtrip_canonical_python_json() {
    assert_roundtrip::<Tool>("tools");
}
#[test]
fn usage_roundtrips_canonical_python_json() {
    assert_roundtrip::<Usage>("usage");
}

#[test]
fn compact_serialization_keeps_full_non_model_event_messages() {
    use suprnova_ai::types::events::OmitEventMessages;
    for value in fixtures()["events"].as_array().unwrap() {
        let event: Event = serde_json::from_value(value.clone()).unwrap();
        if event.model_event().is_none() {
            assert_eq!(
                serde_json::to_value(OmitEventMessages(&event)).unwrap(),
                *value
            );
        }
    }
}

#[test]
fn replay_is_runtime_state_for_every_event_variant() {
    for original in fixtures()["events"].as_array().unwrap() {
        let mut value = original.clone();
        value["replay"] = Value::Bool(true);
        let event: Event = serde_json::from_value(value).unwrap();
        assert!(event.replay());
        assert_eq!(serde_json::to_value(event).unwrap(), *original);
    }
}

#[test]
fn agent_events_require_full_messages() {
    for original in fixtures()["events"].as_array().unwrap() {
        if ["tool_call_result", "hook"]
            .iter()
            .any(|kind| original["kind"] == *kind)
        {
            let mut value = original.clone();
            value["message"] = serde_json::json!({"id":"message"});
            assert!(serde_json::from_value::<Event>(value).is_err());
        }
    }
}
