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

//! Model-facing tool declarations, calls, results, and JSON Schema validation.

use std::collections::{BTreeSet, HashSet};
use std::fmt;

use bigdecimal::{BigDecimal, Zero};
use schemars::{JsonSchema, Schema};
use serde::de::{self, DeserializeOwned};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::errors::{Error, Result};

/// Payload used by external tool-approval hooks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ToolApproval {
    /// Whether execution was approved.
    pub granted: bool,
    /// Optional human-readable approval or denial reason.
    #[serde(default)]
    pub reason: Option<String>,
}

/// Provider-facing declaration of a host-executed function tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolSpec {
    /// Human-readable guidance for when and how to use the tool.
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for the arguments supplied by the model.
    pub params: Schema,
}

/// Execution configuration for a provider-executed tool.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolConfig {
    /// Canonical provider tool id.
    #[serde(default)]
    pub id: Option<String>,
    /// Opaque provider wire arguments in snake_case form.
    #[serde(default)]
    pub args: Map<String, Value>,
}

/// Where a tool executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ToolKind {
    /// The host executes a model-requested function.
    Function,
    /// The model provider executes the tool.
    Provider,
}

/// Model-facing declaration of a function or provider-executed tool.
#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
pub struct Tool {
    /// Tool execution kind.
    pub kind: ToolKind,
    /// Name exposed to the model.
    pub name: String,
    /// Function tool schema.
    #[serde(default)]
    pub spec: Option<ToolSpec>,
    /// Provider-executed tool configuration.
    #[serde(default)]
    pub tool_config: Option<ToolConfig>,
    /// Whether host execution requires external approval.
    #[serde(default)]
    pub require_approval: bool,
}

impl Tool {
    /// Creates a host-executed function declaration.
    pub fn function(name: impl Into<String>, spec: ToolSpec) -> Self {
        Self {
            kind: ToolKind::Function,
            name: name.into(),
            spec: Some(spec),
            tool_config: None,
            require_approval: false,
        }
    }

    /// Creates a provider-executed declaration after checking its id.
    pub fn provider(
        name: impl Into<String>,
        tool_config: ToolConfig,
    ) -> std::result::Result<Self, ToolShapeError> {
        let tool = Self {
            kind: ToolKind::Provider,
            name: name.into(),
            spec: None,
            tool_config: Some(tool_config),
            require_approval: false,
        };
        tool.validate_shape()?;
        Ok(tool)
    }

    /// Checks the relationship between kind, schema, and provider config.
    pub fn validate_shape(&self) -> std::result::Result<(), ToolShapeError> {
        match self.kind {
            ToolKind::Function if self.spec.is_none() => Err(ToolShapeError::FunctionRequiresSpec),
            ToolKind::Provider if self.spec.is_some() => Err(ToolShapeError::ProviderRejectsSpec),
            ToolKind::Provider
                if self
                    .tool_config
                    .as_ref()
                    .and_then(|config| config.id.as_ref())
                    .is_none() =>
            {
                Err(ToolShapeError::ProviderRequiresConfigId)
            }
            _ => Ok(()),
        }
    }
}

impl<'de> Deserialize<'de> for Tool {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct WireTool {
            kind: ToolKind,
            name: String,
            #[serde(default)]
            spec: Option<ToolSpec>,
            #[serde(default)]
            tool_config: Option<ToolConfig>,
            #[serde(default)]
            require_approval: bool,
        }

        let wire = WireTool::deserialize(deserializer)?;
        let tool = Self {
            kind: wire.kind,
            name: wire.name,
            spec: wire.spec,
            tool_config: wire.tool_config,
            require_approval: wire.require_approval,
        };
        tool.validate_shape().map_err(de::Error::custom)?;
        Ok(tool)
    }
}

/// Invalid relationship between a tool kind and its fields.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum ToolShapeError {
    /// Function tools need a model-facing schema.
    #[error("function tools require spec=ToolSpec(...)")]
    FunctionRequiresSpec,
    /// Provider tools cannot carry a function schema.
    #[error("provider tools cannot have spec")]
    ProviderRejectsSpec,
    /// Provider tools need a canonical provider id.
    #[error("provider tools require tool_config with an id")]
    ProviderRequiresConfigId,
}

/// A model-facing function tool declaration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolDefinition {
    /// Name exposed to the model.
    pub name: String,
    /// Human-readable guidance for when and how to use the tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the arguments supplied by the model.
    pub input_schema: Schema,
    /// JSON Schema for the value returned by the host, when declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Schema>,
}

impl ToolDefinition {
    /// Builds a declaration from already-generated schemas.
    ///
    /// The schemas are checked up front so unsupported validation keywords do
    /// not turn into silently ignored constraints at call time.
    pub fn try_new(
        name: impl Into<String>,
        description: Option<impl Into<String>>,
        input_schema: Schema,
        output_schema: Option<Schema>,
    ) -> std::result::Result<Self, ToolValidationError> {
        inspect_schema(&input_schema)?;
        if let Some(schema) = &output_schema {
            inspect_schema(schema)?;
        }

        Ok(Self {
            name: name.into(),
            description: description.map(Into::into),
            input_schema,
            output_schema,
        })
    }

    /// Generates a declaration for a typed input and untyped output.
    pub fn from_input<Input: JsonSchema>(
        name: impl Into<String>,
        description: Option<impl Into<String>>,
    ) -> std::result::Result<Self, ToolValidationError> {
        Self::try_new(name, description, schemars::schema_for!(Input), None)
    }

    /// Generates a declaration for typed input and output values.
    pub fn from_types<Input: JsonSchema, Output: JsonSchema>(
        name: impl Into<String>,
        description: Option<impl Into<String>>,
    ) -> std::result::Result<Self, ToolValidationError> {
        Self::try_new(
            name,
            description,
            schemars::schema_for!(Input),
            Some(schemars::schema_for!(Output)),
        )
    }

    /// Validates model-supplied arguments against this tool's input schema.
    pub fn validate_arguments(
        &self,
        arguments: &Value,
    ) -> std::result::Result<(), ToolValidationError> {
        validate_schema_value(&self.input_schema, arguments)
    }

    /// Parses JSON arguments, validates them, and deserializes the typed input.
    pub fn decode_arguments<Input: DeserializeOwned>(&self, arguments: &str) -> Result<Input> {
        let value: Value = serde_json::from_str(arguments)?;
        self.validate_arguments(&value)
            .map_err(|error| Error::Validation(error.to_string()))?;
        serde_json::from_value(value).map_err(Error::from)
    }

    /// Validates a host result against the declared output schema.
    ///
    /// Tools without an output schema accept any JSON value.
    pub fn validate_output(&self, output: &Value) -> std::result::Result<(), ToolValidationError> {
        match &self.output_schema {
            Some(schema) => validate_schema_value(schema, output),
            None => Ok(()),
        }
    }
}

impl From<ToolDefinition> for Tool {
    fn from(definition: ToolDefinition) -> Self {
        Self::function(
            definition.name,
            ToolSpec {
                description: definition.description,
                params: definition.input_schema,
            },
        )
    }
}

impl From<&ToolDefinition> for Tool {
    fn from(definition: &ToolDefinition) -> Self {
        definition.clone().into()
    }
}

/// A function call emitted by a model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolCall {
    /// Provider-assigned call identifier.
    pub id: String,
    /// Tool name requested by the model.
    pub name: String,
    /// Decoded JSON arguments.
    pub arguments: Value,
}

impl ToolCall {
    /// Creates a tool call.
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    /// Validates this call against a declaration, including its tool name.
    pub fn validate(
        &self,
        definition: &ToolDefinition,
    ) -> std::result::Result<(), ToolValidationError> {
        if self.name != definition.name {
            return Err(ToolValidationError::InvalidValue {
                instance_path: "$.name".to_owned(),
                message: format!(
                    "expected tool name {:?}, got {:?}",
                    definition.name, self.name
                ),
            });
        }
        definition.validate_arguments(&self.arguments)
    }
}

/// The host result for one model tool call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolResult {
    /// Identifier of the call this result answers.
    pub tool_call_id: String,
    /// JSON value returned to the model.
    #[serde(default)]
    pub result: Value,
    /// Whether the result represents a tool execution error.
    #[serde(default)]
    pub is_error: bool,
}

impl ToolResult {
    /// Creates a successful result.
    pub fn success(tool_call_id: impl Into<String>, result: Value) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            result,
            is_error: false,
        }
    }

    /// Creates a result that reports a tool execution error to the model.
    pub fn error(tool_call_id: impl Into<String>, result: Value) -> Self {
        Self {
            tool_call_id: tool_call_id.into(),
            result,
            is_error: true,
        }
    }
}

/// Built-in policies for model tool selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceMode {
    /// The model decides whether to call a tool.
    #[default]
    Auto,
    /// The model must not call a tool.
    None,
    /// The model must call at least one tool.
    Required,
}

/// A reference to a specific tool by name.
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct ToolRef(pub String);

impl ToolRef {
    /// Creates a reference from a tool name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl fmt::Display for ToolRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl From<String> for ToolRef {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ToolRef {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// A subset of tools paired with a choice policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ToolSelection {
    /// Tools available for this selection.
    pub tools: BTreeSet<ToolRef>,
    /// Policy applied to the selected tools.
    pub mode: ToolChoiceMode,
}

impl ToolSelection {
    /// Creates a deterministically ordered selection.
    pub fn new(tools: impl IntoIterator<Item = impl Into<ToolRef>>, mode: ToolChoiceMode) -> Self {
        Self {
            tools: tools.into_iter().map(Into::into).collect(),
            mode,
        }
    }
}

/// Tool choice policy, serialized in the same shapes as the Python core.
///
/// Modes are lowercase strings, a forced tool is its name string, and a tool
/// subset is an object containing `tools` and `mode`.
#[derive(Clone, Debug, Default, PartialEq, Eq, JsonSchema)]
#[schemars(untagged)]
pub enum ToolChoice {
    /// The model decides whether to call a tool.
    #[default]
    Auto,
    /// The model must not call a tool.
    None,
    /// The model must call at least one tool.
    Required,
    /// The model must call the named tool.
    Tool(ToolRef),
    /// Only a subset of tools participates in selection.
    Selection(ToolSelection),
}

impl From<ToolChoiceMode> for ToolChoice {
    fn from(value: ToolChoiceMode) -> Self {
        match value {
            ToolChoiceMode::Auto => Self::Auto,
            ToolChoiceMode::None => Self::None,
            ToolChoiceMode::Required => Self::Required,
        }
    }
}

impl From<ToolRef> for ToolChoice {
    fn from(value: ToolRef) -> Self {
        Self::Tool(value)
    }
}

impl From<ToolSelection> for ToolChoice {
    fn from(value: ToolSelection) -> Self {
        Self::Selection(value)
    }
}

impl Serialize for ToolChoice {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Auto => serializer.serialize_str("auto"),
            Self::None => serializer.serialize_str("none"),
            Self::Required => serializer.serialize_str("required"),
            Self::Tool(tool) => tool.serialize(serializer),
            Self::Selection(selection) => selection.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ToolChoice {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum WireChoice {
            Name(String),
            Selection(ToolSelection),
        }

        match WireChoice::deserialize(deserializer)? {
            WireChoice::Name(name) => match name.as_str() {
                "auto" => Ok(Self::Auto),
                "none" => Ok(Self::None),
                "required" => Ok(Self::Required),
                _ => Ok(Self::Tool(ToolRef(name))),
            },
            WireChoice::Selection(selection) => Ok(Self::Selection(selection)),
        }
    }
}

/// A precise schema or instance validation failure.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum ToolValidationError {
    /// A schema is malformed.
    #[error("invalid JSON Schema at {schema_path}: {message}")]
    InvalidSchema {
        /// Location within the schema.
        schema_path: String,
        /// Description of the malformed value.
        message: String,
    },
    /// A valid JSON Schema feature is outside the supported validation subset.
    #[error("unsupported JSON Schema keyword {keyword:?} at {schema_path}")]
    UnsupportedSchema {
        /// Location within the schema.
        schema_path: String,
        /// Unsupported keyword.
        keyword: String,
    },
    /// A value does not satisfy the schema.
    #[error("value at {instance_path} is invalid: {message}")]
    InvalidValue {
        /// Location within the validated JSON value.
        instance_path: String,
        /// Description of the failed constraint.
        message: String,
    },
}

/// Validates a JSON value against the supported JSON Schema subset.
pub fn validate_schema_value(
    schema: &Schema,
    value: &Value,
) -> std::result::Result<(), ToolValidationError> {
    inspect_schema(schema)?;
    validate_value(schema.as_value(), schema.as_value(), value, "$", 0)
}

fn inspect_schema(schema: &Schema) -> std::result::Result<(), ToolValidationError> {
    inspect_schema_value(
        schema.as_value(),
        schema.as_value(),
        "#",
        0,
        &mut HashSet::new(),
    )
}

fn inspect_schema_value(
    root: &Value,
    schema: &Value,
    path: &str,
    depth: usize,
    seen: &mut HashSet<*const Value>,
) -> std::result::Result<(), ToolValidationError> {
    if depth > 128 {
        return Err(invalid_schema(path, "schema nesting exceeds 128 levels"));
    }

    if !seen.insert(std::ptr::from_ref(schema)) {
        return Ok(());
    }
    let Value::Object(object) = schema else {
        if schema.is_boolean() {
            return Ok(());
        }
        return Err(invalid_schema(path, "schema must be an object or boolean"));
    };

    for (keyword, constraint) in object {
        let keyword_path = join_pointer(path, keyword);
        match keyword.as_str() {
            "$schema" | "$id" | "$anchor" | "title" | "description" | "default" | "examples"
            | "deprecated" | "readOnly" | "writeOnly" | "format" | "contentEncoding"
            | "contentMediaType" | "discriminator" => {}
            "$ref" => {
                let reference = constraint
                    .as_str()
                    .ok_or_else(|| invalid_schema(&keyword_path, "$ref must be a string"))?;
                if !reference.starts_with('#') {
                    return Err(ToolValidationError::UnsupportedSchema {
                        schema_path: keyword_path,
                        keyword: format!("remote $ref {reference:?}"),
                    });
                }
                let target = resolve_local_ref(root, reference).map_err(|_| {
                    invalid_schema(
                        &keyword_path,
                        format!("local reference {reference:?} does not resolve"),
                    )
                })?;
                inspect_schema_value(root, target, reference, depth + 1, seen)?;
            }
            "$defs" | "definitions" | "properties" => {
                let children = constraint.as_object().ok_or_else(|| {
                    invalid_schema(&keyword_path, "keyword value must be an object")
                })?;
                for (name, child) in children {
                    inspect_schema_value(
                        root,
                        child,
                        &join_pointer(&keyword_path, name),
                        depth + 1,
                        seen,
                    )?;
                }
            }
            "additionalProperties" | "items" | "not" => {
                inspect_schema_value(root, constraint, &keyword_path, depth + 1, seen)?;
            }
            "prefixItems" | "anyOf" | "oneOf" | "allOf" => {
                let children = constraint.as_array().ok_or_else(|| {
                    invalid_schema(&keyword_path, "keyword value must be an array")
                })?;
                if children.is_empty() && matches!(keyword.as_str(), "anyOf" | "oneOf" | "allOf") {
                    return Err(invalid_schema(
                        &keyword_path,
                        "schema list must not be empty",
                    ));
                }
                for (index, child) in children.iter().enumerate() {
                    inspect_schema_value(
                        root,
                        child,
                        &format!("{keyword_path}/{index}"),
                        depth + 1,
                        seen,
                    )?;
                }
            }
            "type" => inspect_type_constraint(constraint, &keyword_path)?,
            "required" => inspect_string_array(constraint, &keyword_path)?,
            "enum" => {
                let values = constraint
                    .as_array()
                    .ok_or_else(|| invalid_schema(&keyword_path, "enum must be an array"))?;
                if values.is_empty() {
                    return Err(invalid_schema(&keyword_path, "enum must not be empty"));
                }
            }
            "const" => {}
            "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" | "multipleOf" => {
                let number = constraint.as_number().ok_or_else(|| {
                    invalid_schema(&keyword_path, "numeric constraint must be a number")
                })?;
                let number = decimal(number, &keyword_path)
                    .map_err(|error| invalid_schema(&keyword_path, error.to_string()))?;
                if keyword == "multipleOf" && number <= 0 {
                    return Err(invalid_schema(&keyword_path, "multipleOf must be positive"));
                }
            }
            "minLength" | "maxLength" | "minItems" | "maxItems" | "minProperties"
            | "maxProperties" => inspect_nonnegative_integer(constraint, &keyword_path)?,
            "uniqueItems" => {
                if !constraint.is_boolean() {
                    return Err(invalid_schema(
                        &keyword_path,
                        "uniqueItems must be a boolean",
                    ));
                }
            }
            key if key.starts_with("x-") => {}
            _ => {
                return Err(ToolValidationError::UnsupportedSchema {
                    schema_path: keyword_path,
                    keyword: keyword.clone(),
                });
            }
        }
    }

    Ok(())
}

fn inspect_type_constraint(
    constraint: &Value,
    path: &str,
) -> std::result::Result<(), ToolValidationError> {
    match constraint {
        Value::String(kind) => inspect_type_name(kind, path),
        Value::Array(kinds) if !kinds.is_empty() => {
            for kind in kinds {
                let kind = kind
                    .as_str()
                    .ok_or_else(|| invalid_schema(path, "type array must contain strings"))?;
                inspect_type_name(kind, path)?;
            }
            Ok(())
        }
        _ => Err(invalid_schema(
            path,
            "type must be a string or non-empty string array",
        )),
    }
}

fn inspect_type_name(kind: &str, path: &str) -> std::result::Result<(), ToolValidationError> {
    match kind {
        "null" | "boolean" | "object" | "array" | "number" | "integer" | "string" => Ok(()),
        _ => Err(invalid_schema(path, format!("unknown JSON type {kind:?}"))),
    }
}

fn inspect_string_array(
    constraint: &Value,
    path: &str,
) -> std::result::Result<(), ToolValidationError> {
    let values = constraint
        .as_array()
        .ok_or_else(|| invalid_schema(path, "required must be an array"))?;
    if values.iter().all(Value::is_string) {
        Ok(())
    } else {
        Err(invalid_schema(path, "required must contain only strings"))
    }
}

fn inspect_nonnegative_integer(
    constraint: &Value,
    path: &str,
) -> std::result::Result<(), ToolValidationError> {
    if constraint.as_u64().is_some() {
        Ok(())
    } else {
        Err(invalid_schema(
            path,
            "constraint must be a non-negative integer",
        ))
    }
}

fn validate_value(
    root: &Value,
    schema: &Value,
    value: &Value,
    instance_path: &str,
    depth: usize,
) -> std::result::Result<(), ToolValidationError> {
    if depth > 128 {
        return Err(ToolValidationError::InvalidSchema {
            schema_path: "#".to_owned(),
            message: "schema reference nesting exceeds 128 levels".to_owned(),
        });
    }

    match schema {
        Value::Bool(true) => return Ok(()),
        Value::Bool(false) => {
            return Err(invalid_value(instance_path, "boolean schema is false"));
        }
        Value::Object(_) => {}
        _ => return Err(invalid_schema("#", "schema must be an object or boolean")),
    }
    let object = schema.as_object().expect("object checked above");

    if let Some(reference) = object.get("$ref") {
        let reference = reference
            .as_str()
            .expect("schema inspected before validation");
        let target = resolve_local_ref(root, reference)?;
        validate_value(root, target, value, instance_path, depth + 1)?;
    }

    if let Some(constraint) = object.get("type") {
        validate_type(constraint, value, instance_path)?;
    }
    if let Some(values) = object.get("enum").and_then(Value::as_array)
        && !contains_equivalent(values, value, instance_path)?
    {
        return Err(invalid_value(instance_path, "value is not in enum"));
    }
    if let Some(expected) = object.get("const")
        && !json_equal(value, expected, instance_path, 0)?
    {
        return Err(invalid_value(instance_path, "value does not match const"));
    }

    validate_combinators(root, object, value, instance_path, depth)?;
    validate_number(object, value, instance_path)?;
    validate_string(object, value, instance_path)?;
    validate_array(root, object, value, instance_path, depth)?;
    validate_object(root, object, value, instance_path, depth)?;

    if let Some(negated) = object.get("not")
        && schema_matches(root, negated, value, instance_path, depth + 1)?
    {
        return Err(invalid_value(
            instance_path,
            "value matches disallowed schema",
        ));
    }

    Ok(())
}

fn validate_type(
    constraint: &Value,
    value: &Value,
    path: &str,
) -> std::result::Result<(), ToolValidationError> {
    let matches = match constraint {
        Value::String(kind) => value_matches_type(value, kind),
        Value::Array(kinds) => kinds
            .iter()
            .filter_map(Value::as_str)
            .any(|kind| value_matches_type(value, kind)),
        _ => false,
    };
    if matches {
        Ok(())
    } else {
        Err(invalid_value(
            path,
            format!("expected type {}, got {}", constraint, value_type(value)),
        ))
    }
}

fn value_matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "number" => value.is_number(),
        "integer" => value
            .as_f64()
            .is_some_and(|number| number.is_finite() && number.fract() == 0.0),
        "string" => value.is_string(),
        _ => false,
    }
}

fn validate_combinators(
    root: &Value,
    object: &Map<String, Value>,
    value: &Value,
    path: &str,
    depth: usize,
) -> std::result::Result<(), ToolValidationError> {
    if let Some(schemas) = object.get("allOf").and_then(Value::as_array) {
        for schema in schemas {
            validate_value(root, schema, value, path, depth + 1)?;
        }
    }

    if let Some(schemas) = object.get("anyOf").and_then(Value::as_array) {
        let mut matched = false;
        for schema in schemas {
            matched |= schema_matches(root, schema, value, path, depth + 1)?;
        }
        if !matched {
            return Err(invalid_value(path, "value does not match anyOf"));
        }
    }

    if let Some(schemas) = object.get("oneOf").and_then(Value::as_array) {
        let mut matches = 0;
        for schema in schemas {
            if schema_matches(root, schema, value, path, depth + 1)? {
                matches += 1;
            }
        }
        if matches != 1 {
            return Err(invalid_value(
                path,
                format!("value must match exactly one oneOf schema; matched {matches}"),
            ));
        }
    }
    Ok(())
}

fn schema_matches(
    root: &Value,
    schema: &Value,
    value: &Value,
    path: &str,
    depth: usize,
) -> std::result::Result<bool, ToolValidationError> {
    match validate_value(root, schema, value, path, depth) {
        Ok(()) => Ok(true),
        Err(ToolValidationError::InvalidValue { .. }) => Ok(false),
        Err(error) => Err(error),
    }
}

fn validate_number(
    object: &Map<String, Value>,
    value: &Value,
    path: &str,
) -> std::result::Result<(), ToolValidationError> {
    let Value::Number(number) = value else {
        return Ok(());
    };
    let number = decimal(number, path)?;
    for keyword in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"] {
        if let Some(Value::Number(limit)) = object.get(keyword) {
            let limit = decimal(limit, path)?;
            let valid = match keyword {
                "minimum" => number >= limit,
                "maximum" => number <= limit,
                "exclusiveMinimum" => number > limit,
                "exclusiveMaximum" => number < limit,
                _ => unreachable!(),
            };
            if !valid {
                return Err(invalid_value(
                    path,
                    format!("number violates {keyword} {limit}"),
                ));
            }
        }
    }
    if let Some(Value::Number(divisor)) = object.get("multipleOf") {
        let divisor = decimal(divisor, path)?;
        if divisor <= 0 {
            return Err(invalid_schema(
                "#/multipleOf",
                "multipleOf must be positive",
            ));
        }
        if !(&number % &divisor).is_zero() {
            return Err(invalid_value(
                path,
                format!("number must be a multiple of {divisor}"),
            ));
        }
    }
    Ok(())
}

fn decimal(
    number: &serde_json::Number,
    path: &str,
) -> std::result::Result<BigDecimal, ToolValidationError> {
    let text = number.to_string();
    if text.len() > 8192 {
        return Err(invalid_value(
            path,
            "decimal representation exceeds 8192 bytes",
        ));
    }
    let decimal: BigDecimal = text
        .parse()
        .map_err(|_| invalid_value(path, "number is outside the supported decimal range"))?;
    if !(-4096..=4096).contains(&decimal.as_bigint_and_exponent().1) {
        return Err(invalid_value(path, "decimal exponent exceeds 4096 places"));
    }
    Ok(decimal)
}

fn contains_equivalent(
    values: &[Value],
    value: &Value,
    path: &str,
) -> std::result::Result<bool, ToolValidationError> {
    for candidate in values {
        if json_equal(candidate, value, path, 0)? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn json_equal(
    left: &Value,
    right: &Value,
    path: &str,
    depth: usize,
) -> std::result::Result<bool, ToolValidationError> {
    if depth > 128 {
        return Err(invalid_value(path, "value nesting exceeds 128 levels"));
    }
    match (left, right) {
        (Value::Number(left), Value::Number(right)) => {
            Ok(decimal(left, path)? == decimal(right, path)?)
        }
        (Value::Array(left), Value::Array(right)) if left.len() == right.len() => {
            for (left, right) in left.iter().zip(right) {
                if !json_equal(left, right, path, depth + 1)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        (Value::Object(left), Value::Object(right)) if left.len() == right.len() => {
            for (key, left) in left {
                let Some(right) = right.get(key) else {
                    return Ok(false);
                };
                if !json_equal(left, right, path, depth + 1)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(left == right),
    }
}

fn validate_string(
    object: &Map<String, Value>,
    value: &Value,
    path: &str,
) -> std::result::Result<(), ToolValidationError> {
    let Some(value) = value.as_str() else {
        return Ok(());
    };
    let length = value.chars().count() as u64;
    if let Some(minimum) = object.get("minLength").and_then(Value::as_u64)
        && length < minimum
    {
        return Err(invalid_value(
            path,
            format!("string length must be at least {minimum}"),
        ));
    }
    if let Some(maximum) = object.get("maxLength").and_then(Value::as_u64)
        && length > maximum
    {
        return Err(invalid_value(
            path,
            format!("string length must be at most {maximum}"),
        ));
    }
    Ok(())
}

fn validate_array(
    root: &Value,
    object: &Map<String, Value>,
    value: &Value,
    path: &str,
    depth: usize,
) -> std::result::Result<(), ToolValidationError> {
    let Some(values) = value.as_array() else {
        return Ok(());
    };
    let length = values.len() as u64;
    validate_length(object, "minItems", "maxItems", length, "array", path)?;

    if object
        .get("uniqueItems")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        for (index, item) in values.iter().enumerate() {
            if contains_equivalent(&values[..index], item, path)? {
                return Err(invalid_value(path, "array items must be unique"));
            }
        }
    }

    let prefix_count = if let Some(prefixes) = object.get("prefixItems").and_then(Value::as_array) {
        for (index, (item, schema)) in values.iter().zip(prefixes).enumerate() {
            validate_value(
                root,
                schema,
                item,
                &join_instance(path, &index.to_string()),
                depth + 1,
            )?;
        }
        prefixes.len()
    } else {
        0
    };

    if let Some(schema) = object.get("items") {
        for (index, item) in values.iter().enumerate().skip(prefix_count) {
            validate_value(
                root,
                schema,
                item,
                &join_instance(path, &index.to_string()),
                depth + 1,
            )?;
        }
    }
    Ok(())
}

fn validate_object(
    root: &Value,
    object: &Map<String, Value>,
    value: &Value,
    path: &str,
    depth: usize,
) -> std::result::Result<(), ToolValidationError> {
    let Some(value) = value.as_object() else {
        return Ok(());
    };
    let length = value.len() as u64;
    validate_length(
        object,
        "minProperties",
        "maxProperties",
        length,
        "object",
        path,
    )?;

    if let Some(required) = object.get("required").and_then(Value::as_array) {
        for name in required.iter().filter_map(Value::as_str) {
            if !value.contains_key(name) {
                return Err(invalid_value(
                    path,
                    format!("required property {name:?} is missing"),
                ));
            }
        }
    }

    let properties = object
        .get("properties")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    for (name, schema) in &properties {
        if let Some(property) = value.get(name) {
            validate_value(
                root,
                schema,
                property,
                &join_instance(path, name),
                depth + 1,
            )?;
        }
    }

    if let Some(additional) = object.get("additionalProperties") {
        for (name, property) in value {
            if properties.contains_key(name) {
                continue;
            }
            match additional {
                Value::Bool(true) => {}
                Value::Bool(false) => {
                    return Err(invalid_value(
                        &join_instance(path, name),
                        "additional properties are not allowed",
                    ));
                }
                schema => validate_value(
                    root,
                    schema,
                    property,
                    &join_instance(path, name),
                    depth + 1,
                )?,
            }
        }
    }
    Ok(())
}

fn validate_length(
    object: &Map<String, Value>,
    minimum_keyword: &str,
    maximum_keyword: &str,
    length: u64,
    subject: &str,
    path: &str,
) -> std::result::Result<(), ToolValidationError> {
    if let Some(minimum) = object.get(minimum_keyword).and_then(Value::as_u64)
        && length < minimum
    {
        return Err(invalid_value(
            path,
            format!("{subject} length must be at least {minimum}"),
        ));
    }
    if let Some(maximum) = object.get(maximum_keyword).and_then(Value::as_u64)
        && length > maximum
    {
        return Err(invalid_value(
            path,
            format!("{subject} length must be at most {maximum}"),
        ));
    }
    Ok(())
}

fn resolve_local_ref<'a>(
    root: &'a Value,
    reference: &str,
) -> std::result::Result<&'a Value, ToolValidationError> {
    if reference == "#" {
        return Ok(root);
    }
    let pointer =
        reference
            .strip_prefix('#')
            .ok_or_else(|| ToolValidationError::UnsupportedSchema {
                schema_path: "#/$ref".to_owned(),
                keyword: format!("remote $ref {reference:?}"),
            })?;
    root.pointer(pointer).ok_or_else(|| {
        invalid_schema(
            "#/$ref",
            format!("local reference {reference:?} does not resolve"),
        )
    })
}

fn invalid_schema(path: &str, message: impl Into<String>) -> ToolValidationError {
    ToolValidationError::InvalidSchema {
        schema_path: path.to_owned(),
        message: message.into(),
    }
}

fn invalid_value(path: &str, message: impl Into<String>) -> ToolValidationError {
    ToolValidationError::InvalidValue {
        instance_path: path.to_owned(),
        message: message.into(),
    }
}

fn join_pointer(path: &str, segment: &str) -> String {
    format!("{path}/{}", segment.replace('~', "~0").replace('/', "~1"))
}

fn join_instance(path: &str, segment: &str) -> String {
    format!("{path}/{}", segment.replace('~', "~0").replace('/', "~1"))
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
