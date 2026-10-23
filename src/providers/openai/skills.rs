//! OpenAI's skills (`skills/openai-skills.ts`), OAI-022.

use serde_json::{Map, Value};

use super::{errors, files::Files, params::invalid, protocol::http_request};
use crate::{
    Error,
    providers::{
        http,
        multipart::Multipart,
        operations::{Item, SkillFile, SkillParams, UploadedSkill},
    },
};

impl Files<'_> {
    /// Upload a skill with POST <base>/skills, each file as `files[]` named
    /// by its path.
    pub(super) async fn upload_skill(
        &self,
        files: Vec<SkillFile>,
        params: SkillParams,
    ) -> Result<Item<UploadedSkill>, Error> {
        if params.display_title.is_some() {
            return Err(invalid("OpenAI skills take no display title"));
        }
        if let Some(options) = params.provider_options.get("openai")
            && !options.as_object().is_some_and(Map::is_empty)
        {
            return Err(invalid("OpenAI skills take no openai options"));
        }
        let mut form = Multipart::new();
        for file in &files {
            form.file(
                "files[]",
                &file.path,
                "application/octet-stream",
                &file.data,
            );
        }
        let request = http_request(&errors::API, "POST", self.config, &["skills"], None, None)?;
        let response = http::json(
            &errors::API,
            self.transport,
            form.into_request(request),
            self.provider,
            None,
        )
        .await?;
        let id = response["id"].as_str().ok_or_else(|| {
            http::response_error(self.provider, "OpenAI skill response has no id")
        })?;
        let text = |key: &str| response[key].as_str().map(str::to_owned);
        let mut item = Item::new(UploadedSkill {
            provider_reference: [("openai".to_owned(), id.to_owned())].into(),
            display_title: None,
            name: text("name"),
            description: text("description"),
            latest_version: text("latest_version"),
        });
        let mut metadata = Map::new();
        for (src, dst) in [
            ("default_version", "defaultVersion"),
            ("created_at", "createdAt"),
            ("updated_at", "updatedAt"),
        ] {
            if !response[src].is_null() {
                metadata.insert(dst.into(), response[src].clone());
            }
        }
        item.provider_metadata = Some([("openai".to_owned(), Value::Object(metadata))].into());
        Ok(item)
    }
}
