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

//! Provider-independent inference request parameters.

use std::collections::{BTreeMap, BTreeSet};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::types::tools::ToolChoice;

/// A parameter left to the model or provider, or explicitly supplied by the caller.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProviderValue<T> {
    #[default]
    Default,
    Value(T),
}

impl<T> ProviderValue<T> {
    pub fn as_ref(&self) -> ProviderValue<&T> {
        match self {
            Self::Default => ProviderValue::Default,
            Self::Value(value) => ProviderValue::Value(value),
        }
    }
}

impl<T> From<T> for ProviderValue<T> {
    fn from(value: T) -> Self {
        Self::Value(value)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TemperatureSamplerParams {
    pub temperature: ProviderValue<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TopKSamplerParams {
    pub top_k: ProviderValue<Option<u32>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TopPSamplerParams {
    pub top_p: ProviderValue<Option<f64>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct MinPSamplerParams {
    pub min_p: ProviderValue<Option<f64>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RepetitionPenaltyParams {
    pub repetition_penalty: ProviderValue<Option<f64>>,
    pub frequency_penalty: ProviderValue<Option<f64>>,
    pub presence_penalty: ProviderValue<Option<f64>>,
    pub consideration_window: ProviderValue<Option<u32>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SeedSamplerParams {
    pub seed: ProviderValue<Option<Seed>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Seed {
    Value(i64),
    Random,
}

/// Sampling settings keyed by their stable public names instead of Rust type ids.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SamplerParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<TemperatureSamplerParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<TopKSamplerParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<TopPSamplerParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<MinPSamplerParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<RepetitionPenaltyParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<SeedSamplerParams>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolCallingParams {
    #[serde(default)]
    pub max_tool_calls: ProviderValue<Option<u32>>,
    #[serde(default)]
    pub parallel_tool_calls: ProviderValue<bool>,
    pub tool_choice: ToolChoice,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ReasoningParams {
    /// `Value(None)` disables reasoning; `Default` leaves the choice to the provider.
    pub effort: ProviderValue<Option<String>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ProviderServiceParams {
    pub service_tier: ProviderValue<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(transparent)]
pub struct TokenThreshold(pub u64);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ContextManagementParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction: Option<TokenThreshold>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OutputParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<BTreeSet<String>>,
    #[serde(default)]
    pub text_verbosity: ProviderValue<Option<String>>,
    #[serde(default)]
    pub reasoning_summary: ProviderValue<Option<String>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CacheParams {
    #[serde(default)]
    pub mode: ProviderValue<String>,
    #[serde(default)]
    pub retention: ProviderValue<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ExtraHeaderValue {
    Value(String),
    Unset,
}

/// Complete provider-independent options for one inference request.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct InferenceRequestParams {
    #[serde(default)]
    pub sampling: ProviderValue<SamplerParams>,
    #[serde(default)]
    pub reasoning: ProviderValue<ReasoningParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calling: Option<ToolCallingParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_service: Option<ProviderServiceParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<ContextManagementParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_params: Option<BTreeMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_headers: Option<BTreeMap<String, ExtraHeaderValue>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_query: Option<BTreeMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<BTreeMap<String, Value>>,
}

impl InferenceRequestParams {
    #[must_use]
    pub fn with_temperature(mut self, temperature: impl Into<ProviderValue<f64>>) -> Self {
        let sampling = match &mut self.sampling {
            ProviderValue::Default => {
                self.sampling = ProviderValue::Value(SamplerParams::default());
                match &mut self.sampling {
                    ProviderValue::Value(sampling) => sampling,
                    ProviderValue::Default => unreachable!("sampling was just initialized"),
                }
            }
            ProviderValue::Value(sampling) => sampling,
        };
        sampling.temperature = Some(TemperatureSamplerParams {
            temperature: temperature.into(),
        });
        self
    }

    #[must_use]
    pub fn with_reasoning_effort(mut self, effort: ProviderValue<Option<String>>) -> Self {
        self.reasoning = ProviderValue::Value(ReasoningParams { effort });
        self
    }

    #[must_use]
    pub fn with_provider_param(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.provider_params
            .get_or_insert_with(BTreeMap::new)
            .insert(name.into(), value.into());
        self
    }
}
