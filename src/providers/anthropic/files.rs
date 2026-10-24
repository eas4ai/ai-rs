//! Anthropic's files and skills APIs: file upload (`anthropic-files.ts`,
//! ANT-017) and skill upload (`skills/anthropic-skills.ts`, ANT-018).
use super::{
    errors::API,
    protocol::{ANTHROPIC_OWN, authorize, with_betas},
};
use crate::{
    Error, ProviderConfig,
    providers::{
        http,
        multipart::Multipart,
        operations::{Item, SkillFile, SkillParams, UploadFile, UploadedFile, UploadedSkill},
    },
    transport::{HttpRequest, HttpTransport},
};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

/// Where a files or skills call goes: the provider's configuration and
/// transport.
pub(super) struct Files<'a> {
    pub config: &'a ProviderConfig,
    pub transport: &'a dyn HttpTransport,
    pub provider: &'a str,
}

/// An id that goes into a URL path as one segment: one that names no
/// segment, or holds a control character the URL parser drops, would send
/// the call elsewhere.
pub(super) fn path_id<'a>(what: &str, id: &'a str) -> Result<&'a str, Error> {
    if id.trim().is_empty() || id == "." || id == ".." || id.chars().any(char::is_control) {
        return Err(Error::Validation(format!(
            "{id:?} is not an Anthropic {what} id"
        )));
    }
    Ok(id)
}

impl Files<'_> {
    /// A request on Anthropic's own paths with these betas.
    pub(super) fn request(
        &self,
        method: &str,
        path: &[&str],
        body: Option<Value>,
        betas: &[&str],
        query: BTreeMap<String, Value>,
    ) -> Result<HttpRequest, Error> {
        let betas: Vec<String> = betas.iter().map(|b| (*b).to_owned()).collect();
        let headers = with_betas(self.config, &BTreeMap::new(), &betas);
        let mut request = http::request(&API, self.config, method, path, body, headers, query)?;
        authorize(&ANTHROPIC_OWN, self.config, &mut request)?;
        Ok(request)
    }

    pub(super) async fn json(&self, request: HttpRequest) -> Result<Value, Error> {
        http::json(&API, self.transport, request, self.provider, None).await
    }

    fn string<'v>(&self, value: &'v Value, key: &str, what: &str) -> Result<&'v str, Error> {
        value.get(key).and_then(Value::as_str).ok_or_else(|| {
            http::response_error(
                self.provider,
                format!("Anthropic {what} response has no {key}"),
            )
        })
    }

    pub(super) async fn upload(&self, file: UploadFile) -> Result<Item<UploadedFile>, Error> {
        let mut form = Multipart::new();
        // A file without a name goes as a browser sends an unnamed blob.
        let filename = file.filename.clone().unwrap_or_else(|| "blob".into());
        form.file("file", &filename, &file.media_type, &file.data);
        let request = self.request(
            "POST",
            &["v1", "files"],
            None,
            &["files-api-2025-04-14"],
            BTreeMap::new(),
        )?;
        let response = self.json(form.into_request(request)).await?;
        let id = self.string(&response, "id", "file upload")?;
        let mut md = Map::new();
        for (from, to) in [
            ("filename", "filename"),
            ("mime_type", "mimeType"),
            ("size_bytes", "sizeBytes"),
            ("created_at", "createdAt"),
            ("downloadable", "downloadable"),
        ] {
            if let Some(value) = response.get(from).filter(|v| !v.is_null()) {
                md.insert(to.into(), value.clone());
            }
        }
        let text = |key: &str| response.get(key).and_then(Value::as_str).map(str::to_owned);
        let mut item = Item::new(UploadedFile {
            provider_reference: [("anthropic".to_owned(), id.to_owned())].into(),
            media_type: text("mime_type").or(Some(file.media_type)),
            filename: text("filename").or(file.filename),
            size_bytes: response.get("size_bytes").and_then(Value::as_u64),
            created_at: text("created_at"),
            expires_at: None,
        });
        item.provider_metadata = Some([("anthropic".to_owned(), Value::Object(md))].into());
        Ok(item)
    }

    pub(super) async fn upload_skill(
        &self,
        files: Vec<SkillFile>,
        params: SkillParams,
    ) -> Result<Item<UploadedSkill>, Error> {
        const BETA: &str = "skills-2025-10-02";
        let mut form = Multipart::new();
        if let Some(title) = &params.display_title {
            form.text("display_title", title);
        }
        for file in &files {
            form.file(
                "files[]",
                &file.path,
                "application/octet-stream",
                &file.data,
            );
        }
        let request = self.request("POST", &["v1", "skills"], None, &[BETA], BTreeMap::new())?;
        let skill = self.json(form.into_request(request)).await?;
        let id = self.string(&skill, "id", "skill upload")?.to_owned();
        let text =
            |value: &Value, key: &str| value.get(key).and_then(Value::as_str).map(str::to_owned);
        let latest = text(&skill, "latest_version");
        let version = match &latest {
            Some(latest) => {
                let request = self.request(
                    "GET",
                    &[
                        "v1",
                        "skills",
                        path_id("skill", &id)?,
                        "versions",
                        path_id("skill version", latest)?,
                    ],
                    None,
                    &[BETA],
                    BTreeMap::new(),
                )?;
                self.json(request).await?
            }
            None => json!({}),
        };
        let mut md = Map::new();
        for (from, to) in [
            ("source", "source"),
            ("created_at", "createdAt"),
            ("updated_at", "updatedAt"),
        ] {
            if let Some(value) = skill.get(from).filter(|v| !v.is_null()) {
                md.insert(to.into(), value.clone());
            }
        }
        let mut item = Item::new(UploadedSkill {
            provider_reference: [("anthropic".to_owned(), id)].into(),
            display_title: text(&skill, "display_title"),
            name: text(&version, "name").or_else(|| text(&skill, "name")),
            description: text(&version, "description").or_else(|| text(&skill, "description")),
            latest_version: latest,
        });
        item.provider_metadata = Some([("anthropic".to_owned(), Value::Object(md))].into());
        Ok(item)
    }
}
