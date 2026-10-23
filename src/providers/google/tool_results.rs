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
            ContentPart, FileData, FilePart, Message, MessageRole, Part, SpecialToolResult,
            ToolResultValue,
        },
    },
};

/// The files of the tool results in a history.
fn result_files(history: &mut [Message]) -> Vec<&mut FilePart> {
    let mut files = Vec::new();
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
                if let ContentPart::File(file) = item {
                    files.push(file);
                }
            }
        }
    }
    files
}

/// The URL of a tool result file the model cannot take as `fileData`.
fn pending(file: &FilePart, target: Target<'_>) -> Option<String> {
    messages::file_url(file)
        .filter(|url| !messages::function_response_fetches(target, url, &file.media_type))
        .map(str::to_owned)
}

/// The history with each tool result file URL to download standing in as
/// empty bytes, to check a request before downloading; `None` when there
/// is none.
pub(crate) fn without_downloads(history: &[Message], target: Target<'_>) -> Option<Vec<Message>> {
    let mut history = history.to_vec();
    let mut found = false;
    for file in result_files(&mut history) {
        if pending(file, target).is_some() {
            file.data = FileData::Bytes(Vec::new());
            found = true;
        }
    }
    found.then_some(history)
}

/// Download each tool result file URL that the model cannot take as
/// `fileData`, up to `limit` bytes, and send it as bytes. Every URL is
/// checked before the first download.
pub(crate) async fn download(
    mut history: Vec<Message>,
    target: Target<'_>,
    downloads: Option<&dyn DownloadTransport>,
    limit: usize,
) -> Result<Vec<Message>, Error> {
    {
        let mut files = Vec::new();
        for file in result_files(&mut history) {
            let Some(url) = pending(file, target) else {
                continue;
            };
            if !media::is_downloadable_url(&url) {
                return Err(Error::Validation(format!(
                    "{} cannot take the tool result file URL of type {} for {}; send the file's bytes",
                    target.flavor.api().label,
                    file.media_type,
                    target.model_id
                )));
            }
            files.push((file, url));
        }
        if files.is_empty() {
            return Ok(history);
        }
        let Some(transport) = downloads else {
            return Err(Error::Validation(format!(
                "{} needs a download transport to send tool result file URLs",
                target.flavor.api().label
            )));
        };
        for (file, url) in files {
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
            let full = |media_type: &str| !media_type.ends_with("/*") && media_type.contains('/');
            file.media_type = detected
                .or_else(|| content_type.filter(|_| !full(&file.media_type)))
                .unwrap_or_else(|| file.media_type.clone());
            file.data = FileData::Bytes(bytes.to_vec());
        }
    }
    Ok(history)
}
