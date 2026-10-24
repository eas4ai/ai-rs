//! The files API of the hosts whose TypeScript package ships one: DeepSeek's
//! (`deepseek-files.ts`, HST-037), which uploads only. File info, download
//! and delete fail with `Error::UnsupportedOperation`, as `Provider` does
//! by default.
use std::time::{Duration, UNIX_EPOCH};

use serde_json::{Map, Value};

use crate::{
    Error, Provider,
    providers::{
        google::realtime::iso_time,
        http::{self, ApiProfile},
        multipart::Multipart,
        operations::{Item, UploadFile, UploadParams, UploadedFile},
    },
    types::media,
};

fn invalid(message: impl Into<String>) -> Error {
    Error::Validation(message.into())
}

const MAX_FILE_BYTES: usize = 64 * 1024 * 1024;
const MAX_FILENAME_CHARS: usize = 512;
const SUPPORTED_TYPES: [&str; 5] = [
    "image/gif",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
];
/// Declared types that say nothing about the file, which its bytes or its
/// filename then must.
const GENERIC_TYPES: [&str; 6] = [
    "",
    "application/binary",
    "application/octet-stream",
    "binary/octet-stream",
    "image",
    "image/*",
];
const EXTENSIONS: [&str; 5] = ["gif", "jpeg", "jpg", "png", "webp"];
const SUPPORTED: &str = "DeepSeek file uploads support JPEG, PNG, GIF and WebP images";

/// `POST <base>/files` as `deepseek-files.ts` sends it: a form with the
/// file, `purpose: user_data` and the `expiresAfter` option, checked before
/// any request.
pub(super) async fn deepseek_upload(
    api: &'static ApiProfile,
    provider: &dyn Provider,
    file: UploadFile,
    params: UploadParams,
) -> Result<Item<UploadedFile>, Error> {
    let expires_after = expires_after(&params)?;
    check_deepseek(&file)?;
    let mut form = Multipart::new();
    // A file without a name goes as a browser sends an unnamed blob.
    form.file(
        "file",
        file.filename.as_deref().unwrap_or("blob"),
        &file.media_type,
        &file.data,
    );
    form.text("purpose", "user_data");
    if let Some(seconds) = expires_after {
        form.text("expires_after[anchor]", "created_at");
        form.text("expires_after[seconds]", &seconds.to_string());
    }
    let transport = provider.transport().ok_or_else(|| {
        Error::Configuration(format!(
            "provider {:?} has no transport for files",
            provider.name()
        ))
    })?;
    let request = http::request(
        api,
        provider.config(),
        "POST",
        &["files"],
        None,
        Default::default(),
        Default::default(),
    )?;
    let response = http::json(
        api,
        transport,
        form.into_request(request),
        provider.name(),
        None,
    )
    .await?;
    let id = response["id"]
        .as_str()
        .ok_or_else(|| http::response_error(provider.name(), "DeepSeek file response has no id"))?;
    // A time the system clock cannot hold is left out rather than trusted.
    let time = |field: &str| {
        response[field]
            .as_u64()
            .and_then(|seconds| UNIX_EPOCH.checked_add(Duration::from_secs(seconds)))
            .map(iso_time)
    };
    let mut metadata = Map::new();
    for (from, to) in [
        ("object", "object"),
        ("filename", "filename"),
        ("purpose", "purpose"),
        ("bytes", "bytes"),
        ("created_at", "createdAt"),
        ("expires_at", "expiresAt"),
    ] {
        if !response[from].is_null() {
            metadata.insert(to.into(), response[from].clone());
        }
    }
    let mut item = Item::new(UploadedFile {
        provider_reference: [("deepseek".to_owned(), id.to_owned())].into(),
        filename: response["filename"]
            .as_str()
            .map(str::to_owned)
            .or(file.filename),
        media_type: Some(file.media_type),
        size_bytes: response["bytes"].as_u64(),
        created_at: time("created_at"),
        expires_at: time("expires_at"),
    });
    item.provider_metadata = Some([("deepseek".to_owned(), Value::Object(metadata))].into());
    Ok(item)
}

/// The `expiresAfter` of the `deepseek` file options
/// (`deepSeekFilesOptionsSchema`): a whole number of seconds from an hour
/// to 30 days.
fn expires_after(params: &UploadParams) -> Result<Option<u64>, Error> {
    let Some(entry) = params.provider_options.get("deepseek") else {
        return Ok(None);
    };
    let Value::Object(entry) = entry else {
        return Err(invalid("deepseek file options must be an object"));
    };
    let mut expires_after = None;
    for (option, value) in entry {
        if option != "expiresAfter" {
            return Err(invalid(format!(
                "deepseek file option {option} is not known"
            )));
        }
        let seconds = value
            .as_u64()
            .filter(|seconds| (3600..=2_592_000).contains(seconds))
            .ok_or_else(|| {
                invalid(format!(
                    "deepseek file option expiresAfter must be a whole number of seconds from 3600 to 2592000, not {value}"
                ))
            })?;
        expires_after = Some(seconds);
    }
    Ok(expires_after)
}

/// What `validateFileUpload` in `deepseek-files.ts` refuses: a file over
/// 64 MiB, a filename over 512 characters, bytes that are another kind of
/// file, a declared type that is neither an image DeepSeek takes nor
/// generic, and a generic type with neither detectable bytes nor an image
/// filename extension.
fn check_deepseek(file: &UploadFile) -> Result<(), Error> {
    if file.data.len() > MAX_FILE_BYTES {
        return Err(invalid(format!(
            "DeepSeek file uploads must not exceed 64 MiB ({MAX_FILE_BYTES} bytes); the file is {} bytes",
            file.data.len()
        )));
    }
    if let Some(name) = &file.filename
        && name.chars().count() > MAX_FILENAME_CHARS
    {
        return Err(invalid(format!(
            "DeepSeek filenames must not exceed {MAX_FILENAME_CHARS} characters; the filename is {} characters",
            name.chars().count()
        )));
    }
    let declared = file
        .media_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    let detected = detect(&file.data);
    if let Some(detected) = detected
        && !SUPPORTED_TYPES.contains(&detected)
    {
        return Err(invalid(format!(
            "{SUPPORTED}; the file's bytes are {detected}"
        )));
    }
    if SUPPORTED_TYPES.contains(&declared.as_str()) {
        return Ok(());
    }
    if !GENERIC_TYPES.contains(&declared.as_str()) {
        return Err(invalid(format!(
            "{SUPPORTED}; the media type {:?} is not one of them",
            file.media_type
        )));
    }
    if detected.is_some() || has_image_extension(file.filename.as_deref()) {
        return Ok(());
    }
    Err(invalid(format!(
        "{SUPPORTED}; give a supported media type or a filename ending in .jpg, .jpeg, .png, .gif or .webp, not the media type {:?}",
        file.media_type
    )))
}

/// The media type the bytes show, as TypeScript's `detectMediaType` finds it
/// without a top-level type: an image, a PDF, audio or video.
fn detect(data: &[u8]) -> Option<&'static str> {
    media::detect_image_media_type(data)
        .or_else(|| data.starts_with(b"%PDF").then_some("application/pdf"))
        .or_else(|| media::detect_audio_media_type(data))
        .or_else(|| media::detect_video_media_type(data))
}

fn has_image_extension(filename: Option<&str>) -> bool {
    filename
        .and_then(|name| name.rsplit_once('.'))
        .is_some_and(|(_, extension)| EXTENSIONS.contains(&extension.to_ascii_lowercase().as_str()))
}
