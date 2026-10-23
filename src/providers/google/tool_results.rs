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

//! Tool result file URLs that Vertex AI cannot reference, downloaded before
//! the request is built (`download-tool-result-files.ts`, VTX-005).

use super::messages::{self, Target};
use crate::{
    Error,
    errors::{ProviderError, ProviderErrorKind},
    models::files::{self, DownloadTransport},
    types::{
        media,
        messages::{
            ContentPart, FileData, Message, MessageRole, Part, SpecialToolResult, ToolResultValue,
        },
    },
};

/// Download each tool result file URL that the model cannot take as
/// `fileData`, up to `limit` bytes, and send it as bytes.
pub(crate) async fn download(
    mut history: Vec<Message>,
    target: Target<'_>,
    downloads: Option<&dyn DownloadTransport>,
    limit: usize,
) -> Result<Vec<Message>, Error> {
    for message in history.iter_mut().filter(|m| m.role == MessageRole::Tool) {
        for part in &mut message.parts {
            let Part::ToolResult(result) = part else {
                continue;
            };
            let value = match &mut result.model_input {
                Some(value) => value,
                None => &mut result.result,
            };
            let ToolResultValue::Special(SpecialToolResult::Content(content)) = value else {
                continue;
            };
            for item in &mut content.value {
                let ContentPart::File(file) = item else {
                    continue;
                };
                let Some(url) = messages::file_url(file).map(str::to_owned) else {
                    continue;
                };
                if messages::function_response_fetches(target, &url, &file.media_type) {
                    continue;
                }
                if !media::is_downloadable_url(&url) {
                    return Err(Error::Validation(format!(
                        "{} cannot take the tool result file URL of type {} for {}; send the file's bytes",
                        target.flavor.api().label,
                        file.media_type,
                        target.model_id
                    )));
                }
                let Some(transport) = downloads else {
                    return Err(Error::Validation(format!(
                        "{} needs a download transport to send tool result file URLs",
                        target.flavor.api().label
                    )));
                };
                let (bytes, content_type) =
                    files::download(transport, &url, limit)
                        .await
                        .map_err(|error| {
                            Error::from(
                                ProviderError::new(
                                    ProviderErrorKind::Response,
                                    format!(
                                        "{} tool result file download failed",
                                        target.flavor.api().label
                                    ),
                                )
                                .with_source(error),
                            )
                        })?;
                let detected = media::detect_image_media_type(bytes.as_ref()).map(str::to_owned);
                let full =
                    |media_type: &str| !media_type.ends_with("/*") && media_type.contains('/');
                file.media_type = detected
                    .or_else(|| content_type.filter(|_| !full(&file.media_type)))
                    .unwrap_or_else(|| file.media_type.clone());
                file.data = FileData::Bytes(bytes.to_vec());
            }
        }
    }
    Ok(history)
}
