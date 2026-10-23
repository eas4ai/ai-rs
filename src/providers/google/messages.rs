//! History as Gemini `contents` and `systemInstruction`
//! (`convert-to-google-messages.ts`).

use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};

use super::{api::Flavor, capabilities};
use crate::{
    Error,
    errors::{ProviderError, ProviderErrorKind},
    models::files::{DEFAULT_MAX_BYTES, DownloadTransport, download},
    types::{
        media,
        messages::{
            ContentPart, FileData, FilePart, Message, MessageRole, Part, SpecialToolResult,
            ToolResultPart, ToolResultValue,
        },
    },
};

/// The signature Google documents for replaying a function call whose own
/// thought signature is not available.
pub(crate) const SKIP_THOUGHT_SIGNATURE: &str = "skip_thought_signature_validator";

/// The media types of `https` URLs that Gemini fetches itself for a model
/// that fetches external URLs (`google-provider.ts`).
const EXTERNAL_URL_MEDIA_TYPES: &[&str] = &[
    "text/html",
    "text/css",
    "text/plain",
    "text/xml",
    "text/csv",
    "text/rtf",
    "text/javascript",
    "application/json",
    "application/pdf",
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/webp",
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/avi",
    "video/x-flv",
    "video/mpg",
    "video/webm",
    "video/wmv",
    "video/3gpp",
];

/// What converts a history for one model.
#[derive(Clone, Copy)]
pub(crate) struct Target<'a> {
    pub model_id: &'a str,
    pub flavor: Flavor,
    /// The base URL, whose `/files/` URLs Gemini fetches.
    pub base_url: &'a str,
    /// A batch request: the request is built before its model is known to
    /// fetch external URLs, so only the URLs every model fetches go out.
    pub batch: bool,
}

/// A URL kept in a file part's data: `http`, `https` or `gs`.
pub(crate) fn file_url(part: &FilePart) -> Option<&str> {
    match &part.data {
        FileData::Text(text)
            if text.starts_with("http://")
                || text.starts_with("https://")
                || text.starts_with("gs://") =>
        {
            Some(text)
        }
        _ => None,
    }
}

/// A Gemini API files URL: Gemini's own, or under the base URL.
pub(crate) fn is_files_url(url: &str, base_url: &str) -> bool {
    url.starts_with("https://generativelanguage.googleapis.com/v1beta/files/")
        || url.starts_with(&format!("{}/files/", base_url.trim_end_matches('/')))
}

/// A YouTube video URL Gemini fetches: `youtube.com/watch?v=` or `youtu.be/`.
pub(crate) fn is_youtube_url(url: &str) -> bool {
    let word = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    let query = |c: char| word(c) || matches!(c, '=' | '&' | '.');
    if let Some(rest) = url
        .strip_prefix("https://www.youtube.com/watch?v=")
        .or_else(|| url.strip_prefix("https://youtube.com/watch?v="))
    {
        let (id, tail) = rest.split_at(rest.find(|c: char| !word(c)).unwrap_or(rest.len()));
        return !id.is_empty()
            && (tail.is_empty() || (tail.starts_with('&') && tail[1..].chars().all(query)));
    }
    if let Some(rest) = url.strip_prefix("https://youtu.be/") {
        let (id, tail) = rest.split_at(rest.find(|c: char| !word(c)).unwrap_or(rest.len()));
        return !id.is_empty()
            && (tail.is_empty() || (tail.starts_with('?') && tail[1..].chars().all(query)));
    }
    false
}

/// Whether the API fetches a user file's URL itself, so that it goes out as
/// `fileData`. Vertex AI fetches every `http`, `https` and `gs` URL.
pub(crate) fn fetches(target: Target<'_>, url: &str, media_type: &str) -> bool {
    match target.flavor {
        Flavor::Vertex => true,
        Flavor::Gemini => {
            is_files_url(url, target.base_url)
                || is_youtube_url(url)
                || (url.starts_with("https://")
                    && !target.batch
                    && capabilities::fetches_external_urls(target.model_id)
                    && EXTERNAL_URL_MEDIA_TYPES.contains(&media_type))
        }
    }
}

/// The user file URLs the API does not fetch, which the call downloads.
fn needs_download(target: Target<'_>, part: &FilePart) -> Option<String> {
    let url = file_url(part)?;
    (!fetches(target, url, &part.media_type)).then(|| url.to_owned())
}

/// Download the user file URLs the API does not fetch through the download
/// transport, or fail with `Error::Validation` when there is none.
pub(crate) async fn download_files(
    mut messages: Vec<Message>,
    target: Target<'_>,
    downloads: Option<&dyn DownloadTransport>,
) -> Result<Vec<Message>, Error> {
    for message in messages.iter_mut().filter(|m| m.role == MessageRole::User) {
        for part in &mut message.parts {
            let Part::File(file) = part else { continue };
            let Some(url) = needs_download(target, file) else {
                continue;
            };
            let Some(transport) = downloads else {
                return Err(Error::Validation(format!(
                    "{} does not fetch the file URL of type {}; build the provider with a download transport to send it",
                    target.flavor.api().label,
                    file.media_type
                )));
            };
            let (bytes, _) =
                download(transport, &url, DEFAULT_MAX_BYTES)
                    .await
                    .map_err(|error| {
                        Error::from(
                            ProviderError::new(
                                ProviderErrorKind::Response,
                                format!("{} input download failed", target.flavor.api().label),
                            )
                            .with_source(error),
                        )
                    })?;
            file.data = FileData::Bytes(bytes.to_vec());
        }
    }
    Ok(messages)
}

/// Whether a history has a user file URL to download.
pub(crate) fn has_downloads(messages: &[Message], target: Target<'_>) -> bool {
    messages
        .iter()
        .filter(|m| m.role == MessageRole::User)
        .flat_map(|m| &m.parts)
        .any(|part| matches!(part, Part::File(file) if needs_download(target, file).is_some()))
}

/// The history with each user file URL to download standing in as empty
/// bytes, to check a request before downloading.
pub(crate) fn without_downloads(messages: &[Message], target: Target<'_>) -> Vec<Message> {
    let mut messages = messages.to_vec();
    for message in messages.iter_mut().filter(|m| m.role == MessageRole::User) {
        for part in &mut message.parts {
            if let Part::File(file) = part
                && needs_download(target, file).is_some()
            {
                file.data = FileData::Bytes(Vec::new());
            }
        }
    }
    messages
}

/// A file's media type and base64 data, from bytes, base64 text or a data
/// URL.
pub(crate) fn inline_data(part: &FilePart) -> (String, String) {
    match &part.data {
        FileData::Bytes(bytes) => (full_media_type(part, Some(bytes)), STANDARD.encode(bytes)),
        FileData::Text(text) => {
            if let (Some(media_type), Some(data)) = media::split_data_url(text) {
                (media_type.to_owned(), data.to_owned())
            } else {
                (full_media_type(part, None), text.clone())
            }
        }
    }
}

/// A full media type for a wildcard one such as `image/*`, detected from
/// the bytes when it can be.
fn full_media_type(part: &FilePart, bytes: Option<&[u8]>) -> String {
    if !part.media_type.ends_with("/*") {
        return part.media_type.clone();
    }
    let decoded;
    let bytes = match bytes {
        Some(bytes) => Some(bytes),
        None => match &part.data {
            FileData::Text(text) => {
                decoded = STANDARD.decode(text).ok();
                decoded.as_deref()
            }
            FileData::Bytes(bytes) => Some(bytes.as_slice()),
        },
    };
    let detected = bytes.and_then(|bytes| match part.media_type.as_str() {
        "image/*" => media::detect_image_media_type(bytes),
        "audio/*" => media::detect_audio_media_type(bytes),
        "video/*" => media::detect_video_media_type(bytes),
        _ => None,
    });
    detected.map_or_else(|| part.media_type.clone(), str::to_owned)
}

/// A part's metadata value under the flavor's keys.
fn meta<'a>(
    flavor: Flavor,
    metadata: Option<&'a crate::types::messages::Metadata>,
    key: &str,
) -> Option<&'a Value> {
    flavor
        .part_metadata(metadata)?
        .get(key)
        .filter(|v| !v.is_null())
}

fn meta_text(
    flavor: Flavor,
    metadata: Option<&crate::types::messages::Metadata>,
    key: &str,
) -> Option<String> {
    meta(flavor, metadata, key).map(|value| match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    })
}

fn with_signature(mut part: Value, signature: Option<String>) -> Value {
    if let Some(signature) = signature {
        part["thoughtSignature"] = json!(signature);
    }
    part
}

fn parse_args(text: &str, what: &str) -> Result<Value, Error> {
    if text.trim().is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(text)
        .map_err(|_| Error::Validation(format!("{what} arguments are not valid JSON")))
}

/// Convert a history. Returns the system instruction and the contents.
pub(crate) fn convert(
    messages: &[Message],
    target: Target<'_>,
) -> Result<(Option<Value>, Vec<Value>), Error> {
    let flavor = target.flavor;
    let gemini_3 = capabilities::uses_gemini_3_features(target.model_id);
    let gemma = capabilities::is_gemma(target.model_id);
    let ids = !flavor.is_vertex();
    let mut system: Vec<String> = Vec::new();
    let mut contents: Vec<Value> = Vec::new();
    let mut system_allowed = true;
    for message in messages {
        match message.role {
            MessageRole::Internal => continue,
            MessageRole::System => {
                if !system_allowed {
                    return Err(Error::Validation(
                        "system messages are only supported at the beginning of the conversation"
                            .into(),
                    ));
                }
                system.push(message.text());
            }
            MessageRole::User => {
                system_allowed = false;
                let mut parts = Vec::new();
                for part in &message.parts {
                    match part {
                        Part::Text(text) => parts.push(json!({"text": text.text})),
                        Part::File(file) => parts.push(user_file(file, target)?),
                        _ => {}
                    }
                }
                if !parts.is_empty() {
                    contents.push(json!({"role": "user", "parts": parts}));
                }
            }
            MessageRole::Assistant => {
                system_allowed = false;
                let parts = assistant_parts(message, target, gemini_3, ids)?;
                if !parts.is_empty() {
                    contents.push(json!({"role": "model", "parts": parts}));
                }
            }
            MessageRole::Tool => {
                system_allowed = false;
                let mut parts = Vec::new();
                for part in &message.parts {
                    let Part::ToolResult(result) = part else {
                        continue;
                    };
                    let metadata = result.provider_metadata.as_ref();
                    if let (Some(id), Some(tool_type)) = (
                        meta_text(flavor, metadata, "serverToolCallId"),
                        meta_text(flavor, metadata, "serverToolType"),
                    ) && let Some(last) = contents.last_mut()
                        && last["role"] == "model"
                    {
                        let response = match result.get_model_input() {
                            ToolResultValue::Json(value) if value.is_object() => value.clone(),
                            _ => json!({}),
                        };
                        let part = with_signature(
                            json!({"toolResponse": {"toolType": tool_type, "response": response, "id": id}}),
                            meta_text(flavor, metadata, "thoughtSignature"),
                        );
                        if let Some(parts) = last["parts"].as_array_mut() {
                            parts.push(part);
                        }
                        continue;
                    }
                    tool_result(&mut parts, result, target, gemini_3, ids)?;
                }
                if !parts.is_empty() {
                    contents.push(json!({"role": "user", "parts": parts}));
                }
            }
        }
    }
    if gemma && !system.is_empty() {
        if let Some(first) = contents.first_mut()
            && first["role"] == "user"
            && let Some(parts) = first["parts"].as_array_mut()
        {
            parts.insert(0, json!({"text": format!("{}\n\n", system.join("\n\n"))}));
        }
        return Ok((None, contents));
    }
    let instruction = (!system.is_empty()).then(
        || json!({"parts": system.iter().map(|text| json!({"text": text})).collect::<Vec<_>>()}),
    );
    Ok((instruction, contents))
}

fn user_file(file: &FilePart, target: Target<'_>) -> Result<Value, Error> {
    if let Some(url) = file_url(file) {
        if fetches(target, url, &file.media_type) {
            return Ok(json!({"fileData": {"mimeType": file.media_type, "fileUri": url}}));
        }
        return Err(Error::Validation(format!(
            "{} does not fetch the file URL of type {}; build the provider with a download transport to send it",
            target.flavor.api().label,
            file.media_type
        )));
    }
    let (mime_type, data) = inline_data(file);
    Ok(json!({"inlineData": {"mimeType": mime_type, "data": data}}))
}

fn assistant_parts(
    message: &Message,
    target: Target<'_>,
    gemini_3: bool,
    ids: bool,
) -> Result<Vec<Value>, Error> {
    let flavor = target.flavor;
    let mut parts = Vec::new();
    let mut signed_call = false;
    for part in &message.parts {
        match part {
            Part::Text(text) if !text.text.is_empty() => {
                let signature =
                    meta_text(flavor, text.provider_metadata.as_ref(), "thoughtSignature");
                parts.push(with_signature(json!({"text": text.text}), signature));
            }
            Part::Reasoning(reasoning) if !reasoning.text.is_empty() => {
                let signature = meta_text(
                    flavor,
                    reasoning.provider_metadata.as_ref(),
                    "thoughtSignature",
                );
                parts.push(with_signature(
                    json!({"text": reasoning.text, "thought": true}),
                    signature,
                ));
            }
            Part::File(file) => {
                let metadata = file.provider_metadata.as_ref();
                let signature = meta_text(flavor, metadata, "thoughtSignature");
                let thought = meta(flavor, metadata, "thought") == Some(&Value::Bool(true));
                let mut wire = if let Some(url) = file_url(file) {
                    if flavor.is_vertex() || !is_files_url(url, target.base_url) {
                        return Err(Error::Validation(
                            "file URLs in assistant messages are not supported; send the file's bytes"
                                .into(),
                        ));
                    }
                    json!({"fileData": {"mimeType": file.media_type, "fileUri": url}})
                } else {
                    let (mime_type, data) = inline_data(file);
                    json!({"inlineData": {"mimeType": mime_type, "data": data}})
                };
                if thought {
                    wire["thought"] = json!(true);
                }
                parts.push(with_signature(wire, signature));
            }
            Part::ToolCall(call) => {
                let signature =
                    meta_text(flavor, call.provider_metadata.as_ref(), "thoughtSignature");
                // Gemini 3 returns one signature for parallel calls, on the
                // first; later calls of the same response have none.
                let skip_sentinel = signature.is_none() && signed_call;
                let effective = signature.clone().or_else(|| {
                    (gemini_3 && !skip_sentinel).then(|| SKIP_THOUGHT_SIGNATURE.to_owned())
                });
                if signature.is_some() {
                    signed_call = true;
                }
                let mut function = json!({
                    "name": call.tool_name,
                    "args": parse_args(&call.tool_args, &format!("tool call {}", call.tool_name))?,
                });
                if ids {
                    function["id"] = json!(call.tool_call_id);
                }
                parts.push(with_signature(json!({"functionCall": function}), effective));
            }
            Part::BuiltinToolCall(call) => {
                let metadata = call.provider_metadata.as_ref();
                if call.tool_name == "code_execution" {
                    let code = parse_args(&call.tool_args, "code execution")?;
                    let (Some(language), Some(source)) = (
                        code.get("language").and_then(Value::as_str),
                        code.get("code").and_then(Value::as_str),
                    ) else {
                        return Err(Error::Validation(
                            "a code_execution call needs a language and code".into(),
                        ));
                    };
                    parts.push(json!({"executableCode": {"language": language, "code": source}}));
                    continue;
                }
                let (Some(id), Some(tool_type)) = (
                    meta_text(flavor, metadata, "serverToolCallId"),
                    meta_text(flavor, metadata, "serverToolType"),
                ) else {
                    continue;
                };
                let signature = meta_text(flavor, metadata, "thoughtSignature")
                    .or_else(|| gemini_3.then(|| SKIP_THOUGHT_SIGNATURE.to_owned()));
                parts.push(with_signature(
                    json!({"toolCall": {"toolType": tool_type, "args": parse_args(&call.tool_args, &call.tool_name)?, "id": id}}),
                    signature,
                ));
            }
            Part::BuiltinToolReturn(result) => {
                let metadata = result.provider_metadata.as_ref();
                if result.tool_name == "code_execution" {
                    let outcome = result.result.get("outcome").and_then(Value::as_str);
                    let output = result.result.get("output").and_then(Value::as_str);
                    let (Some(outcome), Some(output)) = (outcome, output) else {
                        return Err(Error::Validation(
                            "a code_execution result needs an outcome and output".into(),
                        ));
                    };
                    parts.push(
                        json!({"codeExecutionResult": {"outcome": outcome, "output": output}}),
                    );
                    continue;
                }
                let (Some(id), Some(tool_type)) = (
                    meta_text(flavor, metadata, "serverToolCallId"),
                    meta_text(flavor, metadata, "serverToolType"),
                ) else {
                    continue;
                };
                let response = if result.result.is_object() {
                    result.result.clone()
                } else {
                    json!({})
                };
                parts.push(with_signature(
                    json!({"toolResponse": {"toolType": tool_type, "response": response, "id": id}}),
                    meta_text(flavor, metadata, "thoughtSignature"),
                ));
            }
            _ => {}
        }
    }
    Ok(parts)
}

/// Google reserves `{"$ref": <name>}` in function responses; a result with a
/// `$ref` key anywhere goes out as its JSON text.
fn response_value(value: &Value) -> Value {
    fn has_ref(value: &Value) -> bool {
        match value {
            Value::Array(items) => items.iter().any(has_ref),
            Value::Object(map) => map
                .iter()
                .any(|(key, value)| key == "$ref" || has_ref(value)),
            _ => false,
        }
    }
    if has_ref(value) {
        Value::String(value.to_string())
    } else {
        value.clone()
    }
}

fn function_response(
    id: Option<&str>,
    name: &str,
    response: Map<String, Value>,
    parts: Option<Vec<Value>>,
) -> Value {
    let mut function = json!({"name": name, "response": response});
    if let Some(id) = id {
        function["id"] = json!(id);
    }
    if let Some(parts) = parts.filter(|parts| !parts.is_empty()) {
        function["parts"] = json!(parts);
    }
    json!({"functionResponse": function})
}

fn tool_result(
    parts: &mut Vec<Value>,
    result: &ToolResultPart,
    target: Target<'_>,
    gemini_3: bool,
    ids: bool,
) -> Result<(), Error> {
    let id = ids.then_some(result.tool_call_id.as_str());
    let name = result.tool_name.as_str();
    let input = result.get_model_input();
    if let ToolResultValue::Special(SpecialToolResult::Content(content)) = input
        && !result.is_error()
    {
        if gemini_3 {
            let mut texts = Vec::new();
            let mut files = Vec::new();
            for item in &content.value {
                match item {
                    ContentPart::Text(text) => texts.push(text.text.clone()),
                    ContentPart::File(file) => match file_url(file) {
                        None => {
                            let (mime_type, data) = inline_data(file);
                            files
                                .push(json!({"inlineData": {"mimeType": mime_type, "data": data}}));
                        }
                        Some(url) if function_response_fetches(target, url, &file.media_type) => {
                            files.push(
                                json!({"fileData": {"mimeType": file.media_type, "fileUri": url}}),
                            );
                        }
                        Some(_) => {
                            texts.push(serde_json::to_string(&ContentPart::File(file.clone()))?)
                        }
                    },
                }
            }
            let text = if texts.is_empty() {
                "Tool executed successfully.".to_owned()
            } else {
                texts.join("\n")
            };
            let mut response = Map::new();
            response.insert("name".into(), json!(name));
            response.insert("content".into(), json!(text));
            parts.push(function_response(id, name, response, Some(files)));
        } else {
            for item in &content.value {
                match item {
                    ContentPart::Text(text) => {
                        let mut response = Map::new();
                        response.insert("name".into(), json!(name));
                        response.insert("content".into(), json!(text.text));
                        parts.push(function_response(id, name, response, None));
                    }
                    ContentPart::File(file) if file_url(file).is_none() => {
                        let (mime_type, data) = inline_data(file);
                        let kind = if file.media_type.starts_with("image/") {
                            "image"
                        } else {
                            "file"
                        };
                        parts.push(json!({"inlineData": {"mimeType": mime_type, "data": data}}));
                        parts.push(json!({"text": format!("Tool executed successfully and returned this {kind} as a response")}));
                    }
                    ContentPart::File(file) => {
                        parts.push(json!({"text": serde_json::to_string(&ContentPart::File(file.clone()))?}));
                    }
                }
            }
        }
        return Ok(());
    }
    let value = match input {
        ToolResultValue::Json(value) => response_value(value),
        special => response_value(&serde_json::to_value(special)?),
    };
    let mut response = Map::new();
    response.insert("name".into(), json!(name));
    response.insert(
        if result.is_error() {
            "error"
        } else {
            "content"
        }
        .into(),
        value,
    );
    parts.push(function_response(id, name, response, None));
    Ok(())
}

/// Whether a tool result's file URL goes out as `fileData`: on Vertex AI,
/// a `gs://` URL of an image, PDF or text file for a model with Gemini 3
/// features (VTX-005); never on the Gemini API.
pub(crate) fn function_response_fetches(target: Target<'_>, url: &str, media_type: &str) -> bool {
    target.flavor.is_vertex()
        && capabilities::uses_gemini_3_features(target.model_id)
        && url.starts_with("gs://")
        && matches!(
            media_type,
            "image/png" | "image/jpeg" | "image/webp" | "application/pdf" | "text/plain"
        )
}
