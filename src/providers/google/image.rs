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

//! Gemini image generation, one image per call, on the buffered language
//! model call (`google-image-model.ts` and `google-vertex-image-model.ts`).

use serde_json::{Map, Value, json};

use super::{
    api::Flavor,
    language::Call,
    options::{self, LanguageOptions},
    tools,
};
use crate::{
    Error, StreamOptions,
    models::params::{
        InferenceRequestParams, ProviderValue, SamplerParams, Seed, SeedSamplerParams,
    },
    providers::operations::*,
    types::{
        Message, MessageRole, Part, media,
        messages::{FileData, FilePart, TextPart},
    },
};

/// A request of an image generation: the history, the call options and the
/// warnings, for the model call and for a batch request.
pub(crate) struct ImageRequest {
    pub messages: Vec<Message>,
    pub options: StreamOptions,
    pub warnings: Vec<Warning>,
}

fn file_part(input: &MediaInput) -> Result<FilePart, Error> {
    Ok(match input {
        MediaInput::File(file) => file.clone(),
        MediaInput::Url(url) => FilePart {
            id: crate::types::messages::generate_id("part"),
            data: FileData::Text(url.clone()),
            media_type: "image/*".into(),
            filename: None,
            provider_metadata: None,
        },
        MediaInput::Bytes(bytes) => {
            let media_type = media::detect_image_media_type(bytes.as_slice()).unwrap_or("image/*");
            FilePart {
                id: crate::types::messages::generate_id("part"),
                data: FileData::Bytes(bytes.clone()),
                media_type: media_type.into(),
                filename: None,
                provider_metadata: None,
            }
        }
    })
}

/// Build the language model call of an image generation (GGL-014,
/// VTX-007).
pub(crate) fn request(
    flavor: Flavor,
    model_id: &str,
    prompt: &ImagePrompt,
    params: &ImageParams,
) -> Result<ImageRequest, Error> {
    if !model_id.starts_with("gemini-") {
        return Err(Error::Validation(
            "Google image models other than Gemini are no longer supported. Use a model ID that starts with `gemini-`.".into(),
        ));
    }
    if prompt.mask.is_some() {
        return Err(Error::Validation(
            "Gemini image models do not support mask-based image editing.".into(),
        ));
    }
    if params.n != 1 {
        return Err(Error::Validation(format!(
            "Gemini image models return one image per call, not {}; ops::generate_image makes one call per image",
            params.n
        )));
    }
    let mut warnings = Vec::new();
    if params.size.is_some() {
        warnings.push(Warning {
            kind: "unsupported".into(),
            feature: Some("size".into()),
            details: Some(
                "This model does not support the `size` option. Use `aspectRatio` instead.".into(),
            ),
            ..Default::default()
        });
    }
    let mut parts = Vec::new();
    if let Some(text) = &prompt.text {
        parts.push(Part::Text(TextPart::new(text.clone())));
    }
    for image in &prompt.images {
        parts.push(Part::File(file_part(image)?));
    }
    let messages = vec![Message::new(MessageRole::User, parts)];

    let (key, mut google) = options::object(flavor, Some(&params.provider_options))?;
    let search = match flavor {
        Flavor::Gemini => google.remove("googleSearch"),
        Flavor::Vertex => None,
    };
    if google.contains_key("responseModalities") {
        return Err(Error::Validation(
            "image generation sets responseModalities itself; leave it out of the options".into(),
        ));
    }
    let image_config = google.remove("imageConfig");
    // Check the rest against the language model options.
    let _: LanguageOptions = options::from_value(key, Value::Object(google.clone()))?;
    google.insert("responseModalities".into(), json!(["IMAGE"]));
    if image_config.is_some() || params.aspect_ratio.is_some() {
        let mut config = match image_config {
            Some(Value::Object(config)) => config,
            Some(Value::Null) | None => Map::new(),
            Some(_) => {
                return Err(Error::Validation(format!(
                    "{key} option imageConfig must be an object"
                )));
            }
        };
        if let Some(ratio) = &params.aspect_ratio {
            config.insert("aspectRatio".into(), json!(ratio));
        }
        google.insert("imageConfig".into(), Value::Object(config));
    }
    let mut request = InferenceRequestParams {
        provider_params: Some([(flavor.option_keys()[0].to_owned(), Value::Object(google))].into()),
        ..Default::default()
    };
    if let Some(seed) = params.seed {
        request.sampling = ProviderValue::Value(SamplerParams {
            seed: Some(SeedSamplerParams {
                seed: ProviderValue::Value(Some(Seed::Value(seed))),
            }),
            ..Default::default()
        });
    }
    let mut tools_list = Vec::new();
    if let Some(args) = search.filter(|args| !args.is_null()) {
        tools_list.push(tools::google_search(args)?);
    }
    Ok(ImageRequest {
        messages,
        options: StreamOptions {
            tools: tools_list,
            params: Some(request),
            ..Default::default()
        },
        warnings,
    })
}

/// The images of a generate content message, with the image metadata.
pub(crate) fn result(
    flavor: Flavor,
    message: &Message,
    raw_finish: Option<String>,
    warnings: Vec<Warning>,
) -> Item<Vec<FilePart>> {
    let images: Vec<FilePart> = message
        .files()
        .into_iter()
        .filter(|file| file.media_type.starts_with("image/"))
        .map(|file| FilePart {
            provider_metadata: None,
            ..file.clone()
        })
        .collect();
    let entries: Vec<Value> = images.iter().map(|_| json!({})).collect();
    let metadata = match flavor {
        Flavor::Gemini => {
            let mut google = message
                .provider_metadata
                .as_ref()
                .and_then(|m| m.get("google"))
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            google.insert("finishReason".into(), json!(raw_finish));
            google.insert("images".into(), json!(entries));
            [("google".to_owned(), Value::Object(google))].into()
        }
        Flavor::Vertex => flavor.metadata(json!({"images": entries})),
    };
    let mut item = Item::new(images);
    item.usage = message
        .usage
        .clone()
        .map(|usage| crate::types::usage::Usage { raw: None, ..usage });
    item.warnings = warnings;
    item.provider_metadata = Some(metadata);
    item
}

/// Generate one image.
pub(crate) async fn generate(
    call: &Call,
    prompt: &ImagePrompt,
    params: &ImageParams,
) -> Result<Item<Vec<FilePart>>, Error> {
    let request = request(call.endpoint.flavor, &call.model_id, prompt, params)?;
    let generated = call
        .generate_full(request.messages, &request.options)
        .await?;
    Ok(result(
        call.endpoint.flavor,
        &generated.message,
        generated.raw_finish,
        request.warnings,
    ))
}
