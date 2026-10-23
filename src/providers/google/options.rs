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

//! Provider options, with the TypeScript option names
//! (`google-language-model-options.ts`). An unknown option name or a value
//! outside an option's set fails with `Error::Validation`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

use super::api::Flavor;
use crate::Error;

/// Parse the flavor's options object as `T`. No options give `T::default()`.
pub(crate) fn parse<T: DeserializeOwned + Default>(
    flavor: Flavor,
    params: Option<&BTreeMap<String, Value>>,
) -> Result<T, Error> {
    match flavor.options(params) {
        Some((key, value)) => from_value(key, value.clone()),
        None => Ok(T::default()),
    }
}

/// Parse an options object of the provider key `key` as `T`.
pub(crate) fn from_value<T: DeserializeOwned>(key: &str, value: Value) -> Result<T, Error> {
    serde_json::from_value(value)
        .map_err(|error| Error::Validation(format!("invalid {key} provider options: {error}")))
}

/// The flavor's options object as a map, empty without one.
pub(crate) fn object(
    flavor: Flavor,
    params: Option<&BTreeMap<String, Value>>,
) -> Result<(&'static str, Map<String, Value>), Error> {
    match flavor.options(params) {
        Some((key, Value::Object(map))) => Ok((key, map.clone())),
        Some((key, _)) => Err(Error::Validation(format!(
            "{key} provider options must be an object"
        ))),
        None => Ok((flavor.option_keys()[0], Map::new())),
    }
}

/// Fail unless `value` is one of `allowed`.
pub(crate) fn check_one_of(name: &str, value: &str, allowed: &[&str]) -> Result<(), Error> {
    if allowed.contains(&value) {
        Ok(())
    } else {
        Err(Error::Validation(format!(
            "google option {name} must be one of {}, not {value:?}",
            allowed.join(", ")
        )))
    }
}

const THRESHOLDS: &[&str] = &[
    "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
    "BLOCK_LOW_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_NONE",
    "OFF",
];

const CATEGORIES: &[&str] = &[
    "HARM_CATEGORY_UNSPECIFIED",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_CIVIC_INTEGRITY",
];

/// The language model options (`googleLanguageModelOptions`).
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct LanguageOptions {
    pub response_modalities: Option<Vec<String>>,
    pub thinking_config: Option<ThinkingConfig>,
    pub cached_content: Option<String>,
    pub structured_outputs: Option<bool>,
    pub safety_settings: Option<Vec<SafetySetting>>,
    pub threshold: Option<String>,
    pub audio_timestamp: Option<bool>,
    pub labels: Option<BTreeMap<String, String>>,
    pub media_resolution: Option<String>,
    pub image_config: Option<ImageConfig>,
    pub retrieval_config: Option<RetrievalConfig>,
    pub stream_function_call_arguments: Option<bool>,
    pub service_tier: Option<String>,
    pub continuation_token: Option<String>,
    pub shared_request_type: Option<String>,
    pub request_type: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<serde_json::Number>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_thoughts: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct SafetySetting {
    pub category: String,
    pub threshold: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ImageConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub person_generation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prominent_people: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_output_options: Option<ImageOutputOptions>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ImageOutputOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression_quality: Option<serde_json::Number>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct RetrievalConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lat_lng: Option<LatLng>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct LatLng {
    pub latitude: f64,
    pub longitude: f64,
}

pub(crate) const ASPECT_RATIOS: &[&str] = &[
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "1:8", "8:1", "1:4",
    "4:1",
];

impl LanguageOptions {
    /// Check each value against its set, and the options that only one API
    /// takes against the flavor.
    pub(crate) fn validate(&self, flavor: Flavor) -> Result<(), Error> {
        if let Some(modalities) = &self.response_modalities {
            for modality in modalities {
                check_one_of("responseModalities", modality, &["TEXT", "IMAGE", "AUDIO"])?;
            }
        }
        if let Some(level) = self
            .thinking_config
            .as_ref()
            .and_then(|config| config.thinking_level.as_deref())
        {
            check_one_of(
                "thinkingConfig.thinkingLevel",
                level,
                &["minimal", "low", "medium", "high"],
            )?;
        }
        for setting in self.safety_settings.iter().flatten() {
            check_one_of("safetySettings.category", &setting.category, CATEGORIES)?;
            check_one_of("safetySettings.threshold", &setting.threshold, THRESHOLDS)?;
        }
        if let Some(threshold) = &self.threshold {
            check_one_of("threshold", threshold, THRESHOLDS)?;
        }
        if let Some(resolution) = &self.media_resolution {
            check_one_of(
                "mediaResolution",
                resolution,
                &[
                    "MEDIA_RESOLUTION_UNSPECIFIED",
                    "MEDIA_RESOLUTION_LOW",
                    "MEDIA_RESOLUTION_MEDIUM",
                    "MEDIA_RESOLUTION_HIGH",
                ],
            )?;
        }
        if let Some(config) = &self.image_config {
            if let Some(ratio) = &config.aspect_ratio {
                check_one_of("imageConfig.aspectRatio", ratio, ASPECT_RATIOS)?;
            }
            if let Some(size) = &config.image_size {
                check_one_of("imageConfig.imageSize", size, &["1K", "2K", "4K", "512"])?;
            }
            if let Some(value) = &config.person_generation {
                check_one_of(
                    "imageConfig.personGeneration",
                    value,
                    &[
                        "PERSON_GENERATION_UNSPECIFIED",
                        "ALLOW_ALL",
                        "ALLOW_ADULT",
                        "ALLOW_NONE",
                    ],
                )?;
            }
            if let Some(value) = &config.prominent_people {
                check_one_of(
                    "imageConfig.prominentPeople",
                    value,
                    &[
                        "PROMINENT_PEOPLE_UNSPECIFIED",
                        "ALLOW_PROMINENT_PEOPLE",
                        "BLOCK_PROMINENT_PEOPLE",
                    ],
                )?;
            }
            if let Some(mime) = config
                .image_output_options
                .as_ref()
                .and_then(|options| options.mime_type.as_deref())
            {
                check_one_of(
                    "imageConfig.imageOutputOptions.mimeType",
                    mime,
                    &["image/jpeg", "image/png"],
                )?;
            }
        }
        if let Some(tier) = &self.service_tier {
            check_one_of("serviceTier", tier, &["standard", "flex", "priority"])?;
        }
        if let Some(value) = &self.shared_request_type {
            check_one_of(
                "sharedRequestType",
                value,
                &["priority", "flex", "standard"],
            )?;
        }
        if let Some(value) = &self.request_type {
            check_one_of("requestType", value, &["shared"])?;
        }
        match flavor {
            Flavor::Gemini => {
                let mut vertex_only = Vec::new();
                if self.stream_function_call_arguments.is_some() {
                    vertex_only.push("streamFunctionCallArguments");
                }
                if self.shared_request_type.is_some() {
                    vertex_only.push("sharedRequestType");
                }
                if self.request_type.is_some() {
                    vertex_only.push("requestType");
                }
                if let Some(config) = &self.image_config {
                    if config.person_generation.is_some() {
                        vertex_only.push("imageConfig.personGeneration");
                    }
                    if config.prominent_people.is_some() {
                        vertex_only.push("imageConfig.prominentPeople");
                    }
                    if config.image_output_options.is_some() {
                        vertex_only.push("imageConfig.imageOutputOptions");
                    }
                }
                if !vertex_only.is_empty() {
                    return Err(Error::Validation(format!(
                        "{} {} Vertex AI only; the Gemini API does not take {}",
                        vertex_only.join(", "),
                        if vertex_only.len() == 1 { "is" } else { "are" },
                        if vertex_only.len() == 1 { "it" } else { "them" },
                    )));
                }
            }
            Flavor::Vertex => {
                if self.service_tier.is_some() {
                    return Err(Error::Validation(
                        "serviceTier is a Gemini API option that Vertex AI does not take; use sharedRequestType and requestType".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// The safety settings: as given, or the threshold for each of the four
    /// configurable categories.
    pub(crate) fn safety_settings(&self) -> Option<Value> {
        if let Some(settings) = &self.safety_settings {
            return serde_json::to_value(settings).ok();
        }
        self.threshold.as_ref().map(|threshold| {
            Value::Array(
                [
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                ]
                .iter()
                .map(|category| serde_json::json!({"category": category, "threshold": threshold}))
                .collect(),
            )
        })
    }
}
