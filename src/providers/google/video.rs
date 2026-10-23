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

//! Veo video jobs (`google-video-model.ts` and `google-vertex-video-model.ts`).
//! A completed Gemini API job returns download URLs, which the protocol
//! fetches with the key in a header, and only from the base URL's origin.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use bytes::Bytes;
use serde_json::{Map, Value, json};

use super::{
    api::{self, Endpoint, Flavor, MEDIA_RESPONSE_BYTES},
    options,
};
use crate::{
    Error,
    errors::{ProviderError, ProviderErrorKind},
    providers::{http, operations::*},
    transport::HttpRequest,
    types::{
        media,
        messages::{FileData, FilePart, generate_id},
    },
};

/// The most videos one job returns.
pub(crate) const MAX_VIDEOS: usize = 4;

fn url_warning(flavor: Flavor) -> Warning {
    Warning {
        kind: "unsupported".into(),
        feature: Some("URL-based image input".into()),
        details: Some(match flavor {
            Flavor::Gemini => "Google Generative AI video models require base64-encoded images or GCS URIs. URL will be ignored.".into(),
            Flavor::Vertex => "Vertex AI video models require base64-encoded images or GCS URIs. URL will be ignored.".into(),
        }),
        ..Default::default()
    }
}

/// An image input as a Veo image: base64 bytes or a `gs://` URI.
fn image(input: &MediaInput, flavor: Flavor, warnings: &mut Vec<Warning>) -> Option<Value> {
    let url = |url: &str, warnings: &mut Vec<Warning>| {
        if url.starts_with("gs://") {
            Some(json!({"gcsUri": url, "mimeType": "image/png"}))
        } else {
            warnings.push(url_warning(flavor));
            None
        }
    };
    match input {
        MediaInput::Url(text) => url(text, warnings),
        MediaInput::Bytes(bytes) => Some(json!({
            "bytesBase64Encoded": STANDARD.encode(bytes),
            "mimeType": media::detect_image_media_type(bytes.as_slice()).unwrap_or("image/png"),
        })),
        MediaInput::File(file) => {
            let media_type = if file.media_type.is_empty() || file.media_type.ends_with("/*") {
                "image/png".to_owned()
            } else {
                file.media_type.clone()
            };
            match &file.data {
                FileData::Bytes(bytes) => Some(
                    json!({"bytesBase64Encoded": STANDARD.encode(bytes), "mimeType": media_type}),
                ),
                FileData::Text(text)
                    if text.starts_with("gs://") || media::is_downloadable_url(text) =>
                {
                    url(text, warnings)
                }
                FileData::Text(text) => Some(
                    json!({"bytesBase64Encoded": media::data_to_base64(text), "mimeType": media_type}),
                ),
            }
        }
    }
}

fn warning(feature: &str, details: &str) -> Warning {
    Warning {
        kind: "unsupported".into(),
        feature: Some(feature.into()),
        details: Some(details.into()),
        ..Default::default()
    }
}

/// The `predictLongRunning` body and the warnings.
pub(crate) fn request(
    flavor: Flavor,
    prompt: &VideoPrompt,
    params: &VideoParams,
) -> Result<(Value, Vec<Warning>), Error> {
    if params.n == 0 || params.n as usize > MAX_VIDEOS {
        return Err(Error::Validation(format!(
            "Veo returns 1 to {MAX_VIDEOS} videos per job, not {}",
            params.n
        )));
    }
    let (_, mut google) = options::object(flavor, Some(&params.provider_options))?;
    let mut warnings = Vec::new();
    let mut instance = Map::new();
    if let Some(text) = &prompt.text {
        instance.insert("prompt".into(), json!(text));
    }
    let frame = |kind: FrameType| {
        prompt
            .frame_images
            .iter()
            .find(|f| f.frame_type == kind)
            .map(|f| &f.image)
    };
    if let Some(start) = frame(FrameType::FirstFrame).or(prompt.image.as_ref())
        && let Some(image) = image(start, flavor, &mut warnings)
    {
        instance.insert("image".into(), image);
    }
    if let Some(last) = frame(FrameType::LastFrame)
        && let Some(image) = image(last, flavor, &mut warnings)
    {
        instance.insert("lastFrame".into(), image);
    }
    let option_references = google.remove("referenceImages").filter(|v| !v.is_null());
    if prompt.frame_images.is_empty() && !prompt.references.is_empty() {
        let references: Vec<Value> = prompt
            .references
            .iter()
            .filter_map(|reference| image(reference, flavor, &mut warnings))
            .map(|image| json!({"image": image, "referenceType": "asset"}))
            .collect();
        instance.insert("referenceImages".into(), json!(references));
    } else if let Some(references) = option_references {
        let Value::Array(references) = references else {
            return Err(Error::Validation(
                "google option referenceImages must be a list".into(),
            ));
        };
        let references: Vec<Value> = match flavor {
            Flavor::Vertex => references,
            Flavor::Gemini => references
                .into_iter()
                .map(|reference| {
                    if let Some(data) = reference.get("bytesBase64Encoded").and_then(Value::as_str).filter(|d| !d.is_empty()) {
                        json!({"image": {"bytesBase64Encoded": data, "mimeType": "image/png"}, "referenceType": "asset"})
                    } else if let Some(uri) = reference.get("gcsUri").and_then(Value::as_str).filter(|u| !u.is_empty()) {
                        json!({"image": {"gcsUri": uri, "mimeType": "image/png"}, "referenceType": "asset"})
                    } else {
                        reference
                    }
                })
                .collect(),
        };
        instance.insert("referenceImages".into(), json!(references));
    }
    let mut parameters = Map::new();
    parameters.insert("sampleCount".into(), json!(params.n));
    if let Some(ratio) = &params.aspect_ratio {
        parameters.insert("aspectRatio".into(), json!(ratio));
    }
    if let Some(resolution) = &params.resolution {
        let wire = match resolution.as_str() {
            "1280x720" => "720p",
            "1920x1080" => "1080p",
            "3840x2160" => "4k",
            other => other,
        };
        parameters.insert("resolution".into(), json!(wire));
    }
    if let Some(duration) = params.duration.filter(|d| *d != 0) {
        parameters.insert("durationSeconds".into(), json!(duration));
    }
    if let Some(seed) = params.seed.filter(|s| *s != 0) {
        parameters.insert("seed".into(), json!(seed));
    }
    if params.fps.is_some() {
        warnings.push(warning(
            "fps",
            "Veo does not take a frame rate. It was ignored.",
        ));
    }
    for key in ["pollIntervalMs", "pollTimeoutMs"] {
        if let Some(value) = google.remove(key).filter(|v| !v.is_null())
            && !value.as_f64().is_some_and(|v| v > 0.0)
        {
            return Err(Error::Validation(format!(
                "google option {key} must be a positive number"
            )));
        }
    }
    if let Some(person) = google.get("personGeneration").and_then(Value::as_str) {
        options::check_one_of(
            "personGeneration",
            person,
            &["dont_allow", "allow_adult", "allow_all"],
        )?;
    }
    if google
        .get("negativePrompt")
        .is_some_and(|v| !v.is_null() && !v.is_string())
    {
        return Err(Error::Validation(
            "google option negativePrompt must be text".into(),
        ));
    }
    match flavor {
        Flavor::Vertex => {
            let audio = params
                .generate_audio
                .map(Value::Bool)
                .or_else(|| google.remove("generateAudio"));
            if let Some(audio) = audio.filter(|v| !v.is_null()) {
                parameters.insert("generateAudio".into(), audio);
            }
            if let Some(directory) = google.remove("gcsOutputDirectory").filter(|v| !v.is_null()) {
                parameters.insert("gcsOutputDirectory".into(), directory);
            }
        }
        Flavor::Gemini => {
            if params.generate_audio.is_some() {
                warnings.push(warning(
                    "generateAudio",
                    "The Gemini API decides whether a Veo video has audio. It was ignored.",
                ));
            }
        }
    }
    for (key, value) in google {
        if !value.is_null() {
            parameters.insert(key, value);
        }
    }
    Ok((
        json!({"instances": [instance], "parameters": parameters}),
        warnings,
    ))
}

fn video_error(provider: &str, message: String) -> Error {
    ProviderError::new(ProviderErrorKind::Response, message)
        .with_provider(provider)
        .into()
}

/// Start a job and return its operation value.
pub(crate) async fn start(
    endpoint: &Endpoint,
    model_id: &str,
    prompt: &VideoPrompt,
    params: &VideoParams,
) -> Result<Item<Value>, Error> {
    let (body, warnings) = request(endpoint.flavor, prompt, params)?;
    let http = endpoint.request(
        "POST",
        &["models".into(), format!("{model_id}:predictLongRunning")],
        Some(body),
        Default::default(),
        Default::default(),
    )?;
    let (_, operation) = endpoint
        .json(http, Some(model_id), http::MAX_RESPONSE_BYTES)
        .await?;
    let name = operation
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .ok_or_else(|| {
            video_error(
                &endpoint.provider,
                "No operation name returned from API".into(),
            )
        })?;
    let mut item = Item::new(json!({"operationName": name}));
    item.warnings = warnings;
    Ok(item)
}

/// Check a job once.
pub(crate) async fn status(
    endpoint: &Endpoint,
    model_id: &str,
    operation: &Value,
) -> Result<VideoStatus, Error> {
    let name = operation
        .get("operationName")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .ok_or_else(|| Error::Validation("a Veo job needs its operationName".into()))?;
    let http = match endpoint.flavor {
        Flavor::Gemini => endpoint.request(
            "GET",
            &name.split('/').map(str::to_owned).collect::<Vec<_>>(),
            None,
            Default::default(),
            Default::default(),
        )?,
        Flavor::Vertex => endpoint.request(
            "POST",
            &["models".into(), format!("{model_id}:fetchPredictOperation")],
            Some(json!({"operationName": name})),
            Default::default(),
            Default::default(),
        )?,
    };
    let (_, operation) = endpoint
        .json(http, Some(model_id), MEDIA_RESPONSE_BYTES)
        .await?;
    if operation.get("done") != Some(&Value::Bool(true)) {
        return Ok(VideoStatus::Pending);
    }
    if let Some(error) = operation.get("error").filter(|e| !e.is_null()) {
        let message = error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("unknown error");
        return Ok(VideoStatus::Failed {
            message: format!("Video generation failed: {message}"),
        });
    }
    let provider = &endpoint.provider;
    match endpoint.flavor {
        Flavor::Gemini => {
            let samples = operation["response"]["generateVideoResponse"]["generatedSamples"]
                .as_array()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    video_error(
                        provider,
                        format!("No videos in response. Response: {operation}"),
                    )
                })?;
            let mut videos = Vec::new();
            let mut metadata = Vec::new();
            for sample in samples {
                if let Some(uri) = sample["video"]["uri"].as_str().filter(|u| !u.is_empty()) {
                    videos.push(FilePart {
                        id: generate_id("part"),
                        data: FileData::Text(uri.to_owned()),
                        media_type: "video/mp4".into(),
                        filename: None,
                        provider_metadata: None,
                    });
                    metadata.push(json!({"uri": uri}));
                }
            }
            if videos.is_empty() {
                return Err(video_error(provider, "No valid videos in response".into()));
            }
            let mut item = Item::new(videos);
            item.provider_metadata =
                Some([("google".to_owned(), json!({"videos": metadata}))].into());
            Ok(VideoStatus::Completed(item))
        }
        Flavor::Vertex => {
            let entries = operation["response"]["videos"]
                .as_array()
                .filter(|v| !v.is_empty())
                .ok_or_else(|| {
                    video_error(
                        provider,
                        format!("No videos in response. Response: {operation}"),
                    )
                })?;
            let mut videos = Vec::new();
            let mut metadata = Vec::new();
            for entry in entries {
                let mime = entry
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .unwrap_or("video/mp4");
                let data =
                    if let Some(data) = entry.get("bytesBase64Encoded").and_then(Value::as_str) {
                        FileData::Bytes(STANDARD.decode(data).map_err(|_| {
                            video_error(provider, "a Vertex AI video is not valid base64".into())
                        })?)
                    } else if let Some(uri) = entry.get("gcsUri").and_then(Value::as_str) {
                        FileData::Text(uri.to_owned())
                    } else {
                        continue;
                    };
                let mut meta = json!({"mimeType": mime});
                if let Some(uri) = entry.get("gcsUri").and_then(Value::as_str) {
                    meta["gcsUri"] = json!(uri);
                }
                metadata.push(meta);
                videos.push(FilePart {
                    id: generate_id("part"),
                    data,
                    media_type: mime.to_owned(),
                    filename: None,
                    provider_metadata: None,
                });
            }
            if videos.is_empty() {
                return Err(video_error(provider, "No valid videos in response".into()));
            }
            let payload = json!({"videos": metadata});
            let mut item = Item::new(videos);
            item.provider_metadata = Some(
                ["googleVertex", "google-vertex", "vertex"]
                    .iter()
                    .map(|key| ((*key).to_owned(), payload.clone()))
                    .collect(),
            );
            Ok(VideoStatus::Completed(item))
        }
    }
}

/// Fetch a video a job returned as an `http` or `https` URL: with the key
/// header from the base URL's origin, without credentials from any other.
/// Bytes, and `gs://` URIs, which only Google Cloud reads, come back as they
/// are.
pub(crate) async fn download(endpoint: &Endpoint, video: FilePart) -> Result<FilePart, Error> {
    let FileData::Text(url) = &video.data else {
        return Ok(video);
    };
    if !media::is_downloadable_url(url) {
        return Ok(video);
    }
    let same_origin = api::origin(url).is_some();
    let mut headers = Vec::new();
    if same_origin {
        if let Some(key) = endpoint.config.api_key().filter(|key| !key.is_empty()) {
            headers.push(("x-goog-api-key".to_owned(), key.to_owned()));
        }
        for (name, value) in &endpoint.config.headers {
            headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
            headers.push((name.to_ascii_lowercase(), value.clone()));
        }
    }
    let request = HttpRequest {
        method: "GET".into(),
        url: url.clone(),
        headers,
        body: Bytes::new(),
    };
    let response = endpoint.send(request, None).await?;
    let (headers, bytes) = http::read_body_with_limit(
        endpoint.flavor.api(),
        &endpoint.provider,
        response,
        MEDIA_RESPONSE_BYTES,
    )
    .await?;
    let media_type = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .and_then(|(_, value)| value.split(';').next())
        .map(str::trim)
        .filter(|value| value.starts_with("video/"))
        .map_or_else(|| video.media_type.clone(), str::to_owned);
    Ok(FilePart {
        data: FileData::Bytes(bytes.to_vec()),
        media_type,
        ..video
    })
}
