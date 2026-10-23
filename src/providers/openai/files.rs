//! OpenAI's files (`files/openai-files.ts`): upload, metadata, download and
//! delete, OAI-021.

use std::{
    collections::BTreeMap,
    time::{Duration, UNIX_EPOCH},
};

use serde::Deserialize;
use serde_json::{Map, Value};

use super::{errors, params::invalid, protocol::http_request};
use crate::{
    Error, ProviderConfig,
    providers::{
        google::realtime::iso_time,
        http,
        multipart::Multipart,
        operations::{DownloadedFile, Item, UploadFile, UploadParams, UploadedFile},
    },
    transport::HttpTransport,
};

/// Where a files call goes: the provider's configuration and transport.
pub(super) struct Files<'a> {
    pub config: &'a ProviderConfig,
    pub transport: &'a dyn HttpTransport,
    pub provider: &'a str,
}

/// The upload options (`openaiFilesOptionsSchema`).
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FileOptions {
    purpose: Option<String>,
    expires_after: Option<serde_json::Number>,
}

/// The OpenAI file id of a provider reference.
pub(super) fn file_id(reference: &BTreeMap<String, String>) -> Result<&str, Error> {
    reference
        .get("openai")
        .map(String::as_str)
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| invalid("the file reference has no openai file id"))
}

fn time(seconds: &Value) -> Option<String> {
    seconds
        .as_u64()
        .map(|seconds| iso_time(UNIX_EPOCH + Duration::from_secs(seconds)))
}

/// A file object as an `UploadedFile`, with the fields OpenAI reports and
/// its metadata under `openai` (`toFileMetadata`).
pub(super) fn uploaded(response: &Value, provider: &str) -> Result<Item<UploadedFile>, Error> {
    let id = response["id"]
        .as_str()
        .ok_or_else(|| http::response_error(provider, "OpenAI file response has no id"))?;
    let mut item = Item::new(UploadedFile {
        provider_reference: [("openai".to_owned(), id.to_owned())].into(),
        media_type: None,
        filename: response["filename"].as_str().map(str::to_owned),
        size_bytes: response["bytes"].as_u64(),
        created_at: time(&response["created_at"]),
        expires_at: time(&response["expires_at"]),
    });
    let mut metadata = Map::new();
    for (src, dst) in [
        ("filename", "filename"),
        ("purpose", "purpose"),
        ("bytes", "bytes"),
        ("created_at", "createdAt"),
        ("status", "status"),
        ("expires_at", "expiresAt"),
    ] {
        if !response[src].is_null() {
            metadata.insert(dst.into(), response[src].clone());
        }
    }
    item.provider_metadata = Some([("openai".to_owned(), Value::Object(metadata))].into());
    Ok(item)
}

impl Files<'_> {
    pub(super) async fn upload(
        &self,
        file: UploadFile,
        params: UploadParams,
    ) -> Result<Item<UploadedFile>, Error> {
        let options: FileOptions = match params.provider_options.get("openai") {
            None => FileOptions::default(),
            Some(value) => serde_json::from_value(value.clone())
                .map_err(|e| invalid(format!("invalid openai file options: {e}")))?,
        };
        let mut form = Multipart::new();
        form.file(
            "file",
            file.filename.as_deref().unwrap_or("file"),
            &file.media_type,
            &file.data,
        );
        form.text(
            "purpose",
            options.purpose.as_deref().unwrap_or("assistants"),
        );
        if let Some(seconds) = &options.expires_after {
            form.text("expires_after[anchor]", "created_at");
            form.text("expires_after[seconds]", &seconds.to_string());
        }
        let request = http_request(&errors::API, "POST", self.config, &["files"], None, None)?;
        let response = http::json(
            &errors::API,
            self.transport,
            form.into_request(request),
            self.provider,
            None,
        )
        .await?;
        uploaded(&response, self.provider)
    }

    pub(super) async fn info(
        &self,
        reference: &BTreeMap<String, String>,
    ) -> Result<Item<UploadedFile>, Error> {
        let id = file_id(reference)?;
        let request = http_request(&errors::API, "GET", self.config, &["files", id], None, None)?;
        let response =
            http::json(&errors::API, self.transport, request, self.provider, None).await?;
        uploaded(&response, self.provider)
    }

    pub(super) async fn download(
        &self,
        reference: &BTreeMap<String, String>,
        max_bytes: usize,
    ) -> Result<Item<DownloadedFile>, Error> {
        let id = file_id(reference)?;
        let mut request = http_request(
            &errors::API,
            "GET",
            self.config,
            &["files", id, "content"],
            None,
            None,
        )?;
        request
            .headers
            .retain(|(name, _)| !name.eq_ignore_ascii_case("accept"));
        request.headers.push(("accept".into(), "*/*".into()));
        let response =
            http::send(&errors::API, self.transport, request, self.provider, None).await?;
        let (headers, data) =
            http::read_body_with_limit(&errors::API, self.provider, response, max_bytes).await?;
        let media_type = headers
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case("content-type"))
            .and_then(|(_, value)| value.split(';').next())
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
        Ok(Item::new(DownloadedFile { data, media_type }))
    }

    pub(super) async fn delete(
        &self,
        reference: &BTreeMap<String, String>,
    ) -> Result<Item<bool>, Error> {
        let id = file_id(reference)?;
        let request = http_request(
            &errors::API,
            "DELETE",
            self.config,
            &["files", id],
            None,
            None,
        )?;
        let response =
            http::json(&errors::API, self.transport, request, self.provider, None).await?;
        let deleted = response["deleted"].as_bool().ok_or_else(|| {
            http::response_error(
                self.provider,
                "OpenAI file delete response has no deleted flag",
            )
        })?;
        Ok(Item::new(deleted))
    }
}
