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

use std::collections::BTreeSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use suprnova_ai::Error;
use suprnova_ai::types::tools::{
    Tool, ToolApproval, ToolCall, ToolChoice, ToolChoiceMode, ToolConfig, ToolDefinition, ToolKind,
    ToolRef, ToolResult, ToolSelection, ToolShapeError, ToolSpec, ToolValidationError,
    validate_schema_value,
};

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
struct WeatherArgs {
    city: String,
    units: Units,
    days: Vec<u8>,
    location: Location,
}

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Units {
    Celsius,
    Fahrenheit,
}

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
struct Location {
    latitude: f64,
    longitude: f64,
}

#[derive(Debug, JsonSchema)]
#[allow(dead_code)]
struct WeatherOutput {
    temperature: f64,
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition::from_types::<WeatherArgs, WeatherOutput>(
        "get_weather",
        Some("Get the weather."),
    )
    .unwrap()
}

fn valid_arguments() -> Value {
    json!({
        "city": "San Francisco",
        "units": "celsius",
        "days": [1, 2],
        "location": {"latitude": 37.77, "longitude": -122.42}
    })
}

fn object_schema(value: Value) -> schemars::Schema {
    schemars::Schema::try_from(value).unwrap()
}

fn roundtrip_tool(tool: &Tool) -> Tool {
    serde_json::from_str(&serde_json::to_string(tool).unwrap()).unwrap()
}

#[test]
fn python_function_tool_roundtrips_through_json() {
    let tool = Tool {
        kind: ToolKind::Function,
        name: "get_weather".into(),
        spec: Some(ToolSpec {
            description: Some("Get the weather.".into()),
            params: object_schema(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            })),
        }),
        tool_config: None,
        require_approval: true,
    };
    assert_eq!(roundtrip_tool(&tool), tool);
    assert_eq!(
        serde_json::to_value(&tool).unwrap(),
        json!({
            "kind": "function",
            "name": "get_weather",
            "spec": {
                "description": "Get the weather.",
                "params": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            },
            "tool_config": null,
            "require_approval": true
        })
    );
}

#[test]
fn python_provider_tool_roundtrips_through_json() {
    let args = serde_json::from_value(json!({
        "max_uses": 3,
        "allowed_domains": ["example.com"],
        "user_location": {"city": "SF", "country": "US"}
    }))
    .unwrap();
    let tool = Tool::provider(
        "web_search",
        ToolConfig {
            id: Some("anthropic.web_search_20260209".into()),
            args,
        },
    )
    .unwrap();
    assert_eq!(roundtrip_tool(&tool), tool);
    assert_eq!(
        serde_json::to_value(&tool).unwrap(),
        json!({
            "kind": "provider",
            "name": "web_search",
            "spec": null,
            "tool_config": {
                "id": "anthropic.web_search_20260209",
                "args": {
                    "max_uses": 3,
                    "allowed_domains": ["example.com"],
                    "user_location": {"city": "SF", "country": "US"}
                }
            },
            "require_approval": false
        })
    );
}

#[test]
fn python_provider_tool_with_opaque_payload_roundtrips() {
    let args = serde_json::from_value(json!({
        "server_label": "my-server",
        "headers": {"x_api_key": "secret"}
    }))
    .unwrap();
    let tool = Tool::provider(
        "mcp",
        ToolConfig {
            id: Some("openai.mcp".into()),
            args,
        },
    )
    .unwrap();
    assert_eq!(roundtrip_tool(&tool), tool);
}

#[test]
fn python_function_tool_requires_spec() {
    let error = serde_json::from_value::<Tool>(json!({
        "kind": "function",
        "name": "bad"
    }))
    .unwrap_err();
    assert!(error.to_string().contains("require spec"));
}

#[test]
fn python_provider_tool_rejects_spec() {
    let error = serde_json::from_value::<Tool>(json!({
        "kind": "provider",
        "name": "bad",
        "spec": {"params": {}},
        "tool_config": {"id": "x.y"}
    }))
    .unwrap_err();
    assert!(error.to_string().contains("cannot have spec"));
}

#[test]
fn python_provider_tool_requires_tool_config_id() {
    for value in [
        json!({"kind": "provider", "name": "bad"}),
        json!({
            "kind": "provider",
            "name": "bad",
            "tool_config": {}
        }),
    ] {
        let error = serde_json::from_value::<Tool>(value).unwrap_err();
        assert!(error.to_string().contains("tool_config"));
    }
}

#[test]
fn approval_payload_and_shape_error_are_typed() {
    let approval: ToolApproval = serde_json::from_value(json!({"granted": false})).unwrap();
    assert_eq!(approval.reason, None);
    assert_eq!(
        serde_json::to_value(&approval).unwrap(),
        json!({"granted": false, "reason": null})
    );
    assert_eq!(
        serde_json::to_value(ToolSpec {
            description: None,
            params: object_schema(json!({})),
        })
        .unwrap(),
        json!({"description": null, "params": {}})
    );
    assert_eq!(
        serde_json::to_value(ToolConfig::default()).unwrap(),
        json!({"id": null, "args": {}})
    );
    let invalid = Tool {
        kind: ToolKind::Function,
        name: "bad".into(),
        spec: None,
        tool_config: None,
        require_approval: false,
    };
    assert_eq!(
        invalid.validate_shape(),
        Err(ToolShapeError::FunctionRequiresSpec)
    );
}

#[test]
fn typed_definition_roundtrips_and_validates_refs_enums_and_arrays() {
    let definition = weather_tool();
    definition.validate_arguments(&valid_arguments()).unwrap();

    let serialized = serde_json::to_string(&definition).unwrap();
    let restored: ToolDefinition = serde_json::from_str(&serialized).unwrap();
    assert_eq!(restored, definition);
    assert_eq!(restored.name, "get_weather");
    assert!(restored.output_schema.is_some());

    let decoded: WeatherArgs = restored
        .decode_arguments(&valid_arguments().to_string())
        .unwrap();
    assert_eq!(decoded.units, Units::Celsius);
    assert_eq!(decoded.location.latitude, 37.77);
}

#[test]
fn validation_reports_required_type_enum_items_and_additional_properties() {
    let definition = weather_tool();

    for (arguments, expected) in [
        (
            json!({
                "units": "celsius",
                "days": [1],
                "location": {"latitude": 1.0, "longitude": 2.0}
            }),
            "required property \"city\" is missing",
        ),
        (
            json!({
                "city": 42,
                "units": "celsius",
                "days": [1],
                "location": {"latitude": 1.0, "longitude": 2.0}
            }),
            "expected type",
        ),
        (
            json!({
                "city": "SF",
                "units": "kelvin",
                "days": [1],
                "location": {"latitude": 1.0, "longitude": 2.0}
            }),
            "enum",
        ),
        (
            json!({
                "city": "SF",
                "units": "celsius",
                "days": ["tomorrow"],
                "location": {"latitude": 1.0, "longitude": 2.0}
            }),
            "expected type",
        ),
        (
            json!({
                "city": "SF",
                "units": "celsius",
                "days": [1],
                "location": {"latitude": 1.0, "longitude": 2.0},
                "secret": true
            }),
            "additional properties are not allowed",
        ),
    ] {
        let error = definition.validate_arguments(&arguments).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn schema_validation_rejects_unsupported_constraints() {
    let schema = schemars::Schema::try_from(json!({
        "type": "object",
        "properties": {"name": {"type": "string", "pattern": "^[a-z]+$"}}
    }))
    .unwrap();

    let error = ToolDefinition::try_new("named", None::<String>, schema, None).unwrap_err();
    assert!(matches!(
        error,
        ToolValidationError::UnsupportedSchema { ref keyword, .. } if keyword == "pattern"
    ));
}

#[test]
fn schema_inspection_bounds_programmatic_nesting() {
    let mut nested = json!({"type": "string"});
    for _ in 0..130 {
        nested = json!({"anyOf": [nested]});
    }
    let schema = schemars::Schema::try_from(nested).unwrap();
    let error = ToolDefinition::try_new("nested", None::<String>, schema, None).unwrap_err();
    assert!(matches!(error, ToolValidationError::InvalidSchema { .. }));
    assert!(error.to_string().contains("nesting exceeds"));
}

#[test]
fn recursive_schema_errors_are_not_swallowed_by_combinators() {
    for schema in [
        json!({"$ref": "#"}),
        json!({"anyOf": [{"$ref": "#"}, {"type": "string"}]}),
        json!({"oneOf": [{"$ref": "#"}, {"type": "string"}]}),
        json!({"not": {"$ref": "#"}}),
    ] {
        let schema = schemars::Schema::try_from(schema).unwrap();
        let definition =
            ToolDefinition::try_new("recursive", None::<String>, schema, None).unwrap();
        let error = definition.validate_arguments(&json!("value")).unwrap_err();
        assert!(matches!(error, ToolValidationError::InvalidSchema { .. }));
    }
}

#[test]
fn boolean_schemas_and_local_refs_are_enforced() {
    let schema = schemars::Schema::try_from(json!({
        "$defs": {"name": {"type": "string"}},
        "type": "object",
        "properties": {"name": {"$ref": "#/$defs/name"}},
        "required": ["name"],
        "additionalProperties": false
    }))
    .unwrap();
    let definition = ToolDefinition::try_new("named", None::<String>, schema, None).unwrap();
    definition
        .validate_arguments(&json!({"name": "Ada"}))
        .unwrap();
    assert!(
        definition
            .validate_arguments(&json!({"name": false}))
            .is_err()
    );
}

#[test]
fn additional_properties_can_have_their_own_schema() {
    let schema = schemars::Schema::try_from(json!({
        "type": "object",
        "additionalProperties": {"type": "integer"}
    }))
    .unwrap();
    let definition = ToolDefinition::try_new("scores", None::<String>, schema, None).unwrap();
    definition
        .validate_arguments(&json!({"alice": 10, "bob": 12}))
        .unwrap();
    assert!(
        definition
            .validate_arguments(&json!({"alice": "ten"}))
            .is_err()
    );
}

#[test]
fn tool_choice_uses_python_wire_shapes() {
    for (choice, expected) in [
        (ToolChoice::Auto, json!("auto")),
        (ToolChoice::None, json!("none")),
        (ToolChoice::Required, json!("required")),
        (
            ToolChoice::Tool(ToolRef::new("get_weather")),
            json!("get_weather"),
        ),
        (
            ToolChoice::Selection(ToolSelection::new(
                ["search", "get_weather"],
                ToolChoiceMode::Required,
            )),
            json!({"tools": ["get_weather", "search"], "mode": "required"}),
        ),
    ] {
        let serialized = serde_json::to_value(&choice).unwrap();
        assert_eq!(serialized, expected);
        assert_eq!(
            serde_json::from_value::<ToolChoice>(serialized).unwrap(),
            choice
        );
    }

    assert_eq!(
        ToolSelection::new(["b", "a", "a"], ToolChoiceMode::Auto).tools,
        BTreeSet::from([ToolRef::new("a"), ToolRef::new("b")])
    );
}

#[test]
fn call_and_result_types_roundtrip_and_validate() {
    let definition = weather_tool();
    let call = ToolCall::new("call-1", "get_weather", valid_arguments());
    call.validate(&definition).unwrap();
    assert_eq!(
        serde_json::from_value::<ToolCall>(serde_json::to_value(&call).unwrap()).unwrap(),
        call
    );

    let result = ToolResult::success("call-1", json!({"temperature": 20.5}));
    definition.validate_output(&result.result).unwrap();
    assert!(!result.is_error);
    assert!(ToolResult::error("call-1", json!("unavailable")).is_error);
}

#[test]
fn decode_arguments_distinguishes_json_and_schema_failures() {
    let definition = weather_tool();
    assert!(matches!(
        definition.decode_arguments::<WeatherArgs>("{"),
        Err(Error::Json(_))
    ));
    assert!(matches!(
        definition.decode_arguments::<WeatherArgs>(r#"{"city":"SF"}"#),
        Err(Error::Validation(_))
    ));
}

#[test]
fn output_without_a_schema_accepts_any_json_value() {
    let definition = ToolDefinition::from_input::<WeatherArgs>("weather", None::<String>).unwrap();
    definition.validate_output(&json!(["anything"])).unwrap();
}

#[test]
fn tool_types_derive_json_schema() {
    let schema = schemars::schema_for!(ToolDefinition);
    assert!(schema.get("type").is_some() || schema.get("$ref").is_some());
    let schema = schemars::schema_for!(ToolChoice);
    for value in [
        json!("auto"),
        json!("get_weather"),
        json!({"tools": ["get_weather"], "mode": "required"}),
    ] {
        validate_schema_value(&schema, &value).unwrap();
    }
}

#[allow(dead_code)]
fn assert_traits<T: Clone + std::fmt::Debug + Serialize>() {}

#[test]
fn stream_facing_tool_types_are_clone_debug_and_serializable() {
    assert_traits::<Tool>();
    assert_traits::<ToolDefinition>();
    assert_traits::<ToolChoice>();
}

#[test]
fn numeric_schema_equality_is_mathematical_and_recursive() {
    for (schema, value) in [
        (json!({"enum":[1]}), json!(1.0)),
        (json!({"const":1}), json!(1.0)),
        (json!({"const":{"values":[1]}}), json!({"values":[1.0]})),
    ] {
        assert!(validate_schema_value(&schema.try_into().unwrap(), &value).is_ok());
    }
    let unique: schemars::Schema = json!({"uniqueItems":true}).try_into().unwrap();
    assert!(validate_schema_value(&unique, &json!([1, 1.0])).is_err());
    assert!(validate_schema_value(&unique, &json!([{"a":1},{"a":1.0}])).is_err());
    let distinct: schemars::Schema = json!({"const":9007199254740992u64}).try_into().unwrap();
    assert!(validate_schema_value(&distinct, &json!(9007199254740993u64)).is_err());
}

#[test]
fn numeric_bounds_and_multiple_of_keep_integer_and_decimal_precision() {
    let invalid: schemars::Schema = json!({"multipleOf":0}).try_into().unwrap();
    assert!(validate_schema_value(&invalid, &Value::Null).is_err());
    for (schema, value) in [
        (
            json!({"minimum":9007199254740993u64}),
            json!(9007199254740992u64),
        ),
        (
            json!({"maximum":9007199254740992u64}),
            json!(9007199254740993u64),
        ),
        (
            json!({"exclusiveMinimum":9007199254740992u64}),
            json!(9007199254740992u64),
        ),
        (json!({"multipleOf":2}), json!(u64::MAX)),
    ] {
        assert!(validate_schema_value(&schema.try_into().unwrap(), &value).is_err());
    }
    let decimal: schemars::Schema = json!({"multipleOf":0.1}).try_into().unwrap();
    assert!(validate_schema_value(&decimal, &json!(0.3)).is_ok());
    assert!(validate_schema_value(&decimal, &json!(0.31)).is_err());
}

#[test]
fn construction_rejects_invalid_numeric_constraints_and_reference_targets() {
    for value in [
        json!({"multipleOf":0}),
        json!({"multipleOf":-1}),
        json!({"examples":[7],"$ref":"#/examples/0"}),
        json!({"examples":[{"type":"invalid"}],"$ref":"#/examples/0"}),
    ] {
        let schema: schemars::Schema = value.try_into().unwrap();
        assert!(ToolDefinition::try_new("invalid", None::<String>, schema, None).is_err());
    }
}
