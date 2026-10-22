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

//! Provider operation payloads. These types keep the base contract independent of adapters.
use crate::types::{messages::FilePart, usage::Usage};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub type Metadata = BTreeMap<String, Value>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Item<T> {
    pub value: T,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub warnings: Vec<Warning>,
    #[serde(default)]
    pub metadata: Option<Metadata>,
    #[serde(default)]
    pub provider_metadata: Option<Metadata>,
}
impl<T> Item<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            usage: None,
            warnings: Vec::new(),
            metadata: None,
            provider_metadata: None,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Warning {
    pub kind: String,
    pub message: Option<String>,
    pub feature: Option<String>,
    pub setting: Option<String>,
    pub details: Option<String>,
}

impl Default for Warning {
    fn default() -> Self {
        Self {
            kind: "other".into(),
            message: None,
            feature: None,
            setting: None,
            details: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MediaInput {
    File(FilePart),
    Bytes(Vec<u8>),
    Url(String),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct ImagePrompt {
    pub text: Option<String>,
    #[serde(default)]
    pub images: Vec<MediaInput>,
    pub mask: Option<MediaInput>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FrameImage {
    pub image: MediaInput,
    pub frame_type: FrameType,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FrameType {
    FirstFrame,
    LastFrame,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct VideoPrompt {
    pub text: Option<String>,
    pub image: Option<MediaInput>,
    #[serde(default)]
    pub frame_images: Vec<FrameImage>,
    #[serde(default)]
    pub references: Vec<MediaInput>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AudioPrompt {
    pub text: String,
    pub instructions: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum TranscribeInput {
    File(FilePart),
    Bytes(Vec<u8>),
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionSegment {
    pub text: String,
    pub start_second: f64,
    pub end_second: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Transcription {
    pub text: String,
    #[serde(default)]
    pub segments: Vec<TranscriptionSegment>,
    pub language: Option<String>,
    pub duration_seconds: Option<f64>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RankedDocument {
    pub index: usize,
    pub score: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Documents {
    Text(Vec<String>),
    Json(Vec<Metadata>),
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EvaluationInput {
    Text(String),
    Object(Metadata),
    Array(Vec<Value>),
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
#[serde(try_from = "QuestionWire")]
pub enum Question {
    Choice {
        instructions: EvaluationInput,
        criteria: BTreeMap<String, Option<EvaluationInput>>,
    },
    Score {
        instructions: EvaluationInput,
        criteria: Vec<Option<EvaluationInput>>,
    },
    Noul {
        instructions: EvaluationInput,
        criteria: Option<NoulCriteria>,
    },
}
impl Question {
    pub fn validate(&self) -> Result<(), crate::errors::Error> {
        let message = match self {
            Self::Choice { criteria, .. } if criteria.is_empty() => {
                Some("choice criteria must not be empty")
            }
            Self::Score { criteria, .. } if criteria.len() < 2 => {
                Some("score criteria must contain at least two levels")
            }
            _ => None,
        };
        match message {
            Some(message) => Err(crate::errors::Error::Validation(message.into())),
            None => Ok(()),
        }
    }
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum QuestionWire {
    Choice {
        instructions: EvaluationInput,
        criteria: BTreeMap<String, Option<EvaluationInput>>,
    },
    Score {
        instructions: EvaluationInput,
        criteria: Vec<Option<EvaluationInput>>,
    },
    Noul {
        instructions: EvaluationInput,
        criteria: Option<NoulCriteria>,
    },
}
impl TryFrom<QuestionWire> for Question {
    type Error = crate::errors::Error;
    fn try_from(value: QuestionWire) -> Result<Self, Self::Error> {
        let question = match value {
            QuestionWire::Choice {
                instructions,
                criteria,
            } => Self::Choice {
                instructions,
                criteria,
            },
            QuestionWire::Score {
                instructions,
                criteria,
            } => Self::Score {
                instructions,
                criteria,
            },
            QuestionWire::Noul {
                instructions,
                criteria,
            } => Self::Noul {
                instructions,
                criteria,
            },
        };
        question.validate()?;
        Ok(question)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct NoulCriteria {
    pub r#true: Option<EvaluationInput>,
    pub r#false: Option<EvaluationInput>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ImageParams {
    pub n: u32,
    pub size: Option<String>,
    pub aspect_ratio: Option<String>,
    pub seed: Option<i64>,
    #[serde(default)]
    pub provider_options: Metadata,
}
impl Default for ImageParams {
    fn default() -> Self {
        Self {
            n: 1,
            size: None,
            aspect_ratio: None,
            seed: None,
            provider_options: Metadata::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct VideoParams {
    pub n: u32,
    pub aspect_ratio: Option<String>,
    pub resolution: Option<String>,
    pub duration: Option<u32>,
    pub fps: Option<u32>,
    pub seed: Option<i64>,
    pub generate_audio: Option<bool>,
    #[serde(default)]
    pub provider_options: Metadata,
}
impl Default for VideoParams {
    fn default() -> Self {
        Self {
            n: 1,
            aspect_ratio: None,
            resolution: None,
            duration: None,
            fps: None,
            seed: None,
            generate_audio: None,
            provider_options: Metadata::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct AudioParams {
    pub voice: Option<String>,
    pub output_format: Option<String>,
    pub speed: Option<f64>,
    pub language: Option<String>,
    #[serde(default)]
    pub provider_options: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct EmbedParams {
    #[serde(default)]
    pub provider_options: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct TranscribeParams {
    #[serde(default)]
    pub provider_options: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct RerankParams {
    pub top_n: Option<usize>,
    #[serde(default)]
    pub provider_options: Metadata,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct EvaluationParams {
    #[serde(default)]
    pub provider_options: Metadata,
}
