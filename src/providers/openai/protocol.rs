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

use super::{errors, messages, params, streaming};
use crate::{
    Error, Model, Provider, ProviderConfig, ProviderProtocol, StreamOptions,
    models::params::{ExtraHeaderValue, InferenceRequestParams},
    providers::EventStream,
    transport::HttpRequest,
    types::Message,
};
use futures::{StreamExt, stream};
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAIChatCompletionsProtocol;
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAIResponsesProtocol;
pub fn default_protocol(provider: &str) -> Arc<dyn ProviderProtocol> {
    if provider == "openai" {
        Arc::new(OpenAIResponsesProtocol)
    } else {
        Arc::new(OpenAIChatCompletionsProtocol)
    }
}
impl ProviderProtocol for OpenAIChatCompletionsProtocol {
    fn protocol_class_id(&self) -> &str {
        "openai_chat_completions"
    }
    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        start(model, messages, options, provider, false, None)
    }
}
impl ProviderProtocol for OpenAIResponsesProtocol {
    fn protocol_class_id(&self) -> &str {
        "openai_responses"
    }
    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        start(model, messages, options, provider, true, None)
    }
}
/// Chat Completions with explicit public-network media downloads.
/// Download transports must honor the core DNS-pinning contract. Runtime transports
/// are not serialized; configure downloads again after restoring a snapshot.
#[derive(Clone)]
pub struct OpenAIChatCompletionsWithDownloads {
    downloads: Arc<dyn crate::models::files::DownloadTransport>,
}
impl OpenAIChatCompletionsProtocol {
    pub fn with_download_transport(
        self,
        downloads: Arc<dyn crate::models::files::DownloadTransport>,
    ) -> OpenAIChatCompletionsWithDownloads {
        OpenAIChatCompletionsWithDownloads { downloads }
    }
}
impl ProviderProtocol for OpenAIChatCompletionsWithDownloads {
    fn protocol_class_id(&self) -> &str {
        "openai_chat_completions"
    }
    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        start(
            model,
            messages,
            options,
            provider,
            false,
            Some(self.downloads.clone()),
        )
    }
}
fn start(
    model: &Model,
    messages: Vec<Message>,
    options: StreamOptions,
    provider: &dyn Provider,
    responses: bool,
    downloads: Option<Arc<dyn crate::models::files::DownloadTransport>>,
) -> EventStream {
    let config = provider.config().clone();
    let provider_name = provider.name().to_owned();
    let model_id = model.id.clone();
    let transport = provider.shared_transport();
    stream::once(async move {
        let messages = messages::download_files(messages, downloads.as_deref()).await?;
        let request = build_request(&model_id, &config, messages, &options, responses)?;
        let transport = transport.ok_or_else(|| {
            Error::Configuration("OpenAI protocols require Provider::shared_transport".into())
        })?;
        let response = transport
            .send(request.clone())
            .await
            .map_err(|e| errors::map_transport(e, &provider_name))?;
        let response =
            errors::check_status(response, &request, &provider_name, Some(&model_id)).await?;
        let image_type = options
            .tools
            .iter()
            .find_map(|t| {
                t.tool_config
                    .as_ref()
                    .filter(|c| c.id.as_deref() == Some("openai.image_generation"))
                    .map(|c| {
                        c.args
                            .get("output_format")
                            .and_then(Value::as_str)
                            .unwrap_or("png")
                    })
            })
            .unwrap_or("png");
        Ok::<_, Error>(streaming::parse(
            response.body,
            provider_name,
            responses,
            format!("image/{image_type}"),
        ))
    })
    .try_flatten()
    .boxed()
}
use futures::TryStreamExt;
/// Build a complete HTTP request without sending it.
pub fn build_request(
    model: &str,
    config: &ProviderConfig,
    messages: Vec<Message>,
    options: &StreamOptions,
    responses: bool,
) -> Result<HttpRequest, Error> {
    let p = options.params.clone().unwrap_or_default();
    let mut body = params::build(&p, responses)?;
    let use_references = body.get("store") != Some(&Value::Bool(false));
    body.insert("model".into(), json!(model));
    body.insert("stream".into(), json!(true));
    body.insert(
        if responses { "input" } else { "messages" }.into(),
        messages::convert(messages, responses, use_references)?.into(),
    );
    if !responses {
        body.insert("stream_options".into(), json!({"include_usage":true}));
    }
    if !options.tools.is_empty() {
        body.insert(
            "tools".into(),
            super::tools::convert(&options.tools, responses)?.into(),
        );
    }
    if let Some(choice) = &options.tool_choice {
        body.insert(
            "tool_choice".into(),
            params::tool_choice(choice, responses)?,
        );
    }
    if let Some(schema) = &options.output_schema {
        let mut schema = serde_json::to_value(schema)?;
        strict_schema(&mut schema, 0)?;
        let name = options.output_type_name.as_deref().unwrap_or("output");
        if responses {
            let text = body.entry("text").or_insert_with(|| json!({}));
            text["format"] =
                json!({"type":"json_schema","name":name,"schema":schema,"strict":true});
        } else {
            body.insert("response_format".into(),json!({"type":"json_schema","json_schema":{"name":name,"schema":schema,"strict":true}}));
        }
    }
    // Match SDK extra_body merging: explicit escape-hatch keys win at the wire boundary.
    if let Some(extra) = &p.extra_body {
        body.extend(extra.clone());
    }
    http_request(
        "POST",
        config,
        if responses {
            "responses"
        } else {
            "chat/completions"
        },
        Some(Value::Object(body)),
        Some(&p),
    )
}
pub(super) fn http_request(
    method: &str,
    config: &ProviderConfig,
    suffix: &str,
    body: Option<Value>,
    params: Option<&InferenceRequestParams>,
) -> Result<HttpRequest, Error> {
    let base = config.base_url();
    let mut url = url::Url::parse(&base)
        .map_err(|_| Error::Configuration("invalid OpenAI base URL".into()))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(Error::Configuration(
            "OpenAI base URL must use HTTP or HTTPS".into(),
        ));
    }
    let path = format!(
        "{}/{}",
        url.path().trim_end_matches('/'),
        suffix.trim_start_matches('/')
    );
    url.set_path(&path);
    if let Some(query) = params.and_then(|p| p.extra_query.as_ref()) {
        let mut pairs = url.query_pairs_mut();
        for (key, v) in query {
            pairs.append_pair(
                key,
                &v.as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| v.to_string()),
            );
        }
    }
    let mut headers = std::collections::BTreeMap::from([
        ("accept".into(), "application/json".into()),
        ("content-type".into(), "application/json".into()),
    ]);
    if let Some(key) = config.api_key().filter(|k| !k.is_empty()) {
        headers.insert("authorization".into(), format!("Bearer {key}"));
    }
    for (name, value) in &config.headers {
        headers.insert(name.to_ascii_lowercase(), value.clone());
    }
    if let Some(extra) = params.and_then(|p| p.extra_headers.as_ref()) {
        for (name, value) in extra {
            if let ExtraHeaderValue::Value(value) = value {
                headers.insert(name.to_ascii_lowercase(), value.clone());
            }
        }
    }
    Ok(HttpRequest {
        method: method.into(),
        url: url.into(),
        headers: headers.into_iter().collect(),
        body: match body {
            Some(body) => serde_json::to_vec(&body)?.into(),
            None => bytes::Bytes::new(),
        },
    })
}
fn strict_schema(schema: &mut Value, _depth: usize) -> Result<(), Error> {
    let root = schema.clone();
    strict_node(schema, &root, 0)
}
fn strict_node(schema: &mut Value, root: &Value, depth: usize) -> Result<(), Error> {
    if depth > 128 {
        return Err(params::invalid("OpenAI output schema exceeds 128 levels"));
    }
    let Some(object) = schema.as_object_mut() else {
        return Ok(());
    };
    if object.len() > 1
        && let Some(reference) = object.get("$ref").and_then(Value::as_str)
    {
        let resolved = reference
            .strip_prefix('#')
            .and_then(|p| root.pointer(p))
            .and_then(Value::as_object)
            .ok_or_else(|| params::invalid("OpenAI schema references must resolve locally"))?;
        let mut expanded = resolved.clone();
        expanded.extend(object.clone());
        expanded.remove("$ref");
        *object = expanded;
    }
    if object.get("default") == Some(&Value::Null) {
        object.remove("default");
    }
    if object.get("type").and_then(Value::as_str) == Some("object")
        && !object.contains_key("additionalProperties")
    {
        object.insert("additionalProperties".into(), json!(false));
    }
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        object.insert(
            "required".into(),
            json!(properties.keys().collect::<Vec<_>>()),
        );
    }
    for key in ["properties", "$defs", "definitions"] {
        if let Some(map) = object.get_mut(key).and_then(Value::as_object_mut) {
            for value in map.values_mut() {
                strict_node(value, root, depth + 1)?;
            }
        }
    }
    for key in ["items", "additionalProperties"] {
        if let Some(value) = object.get_mut(key) {
            strict_node(value, root, depth + 1)?;
        }
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(values) = object.get_mut(key).and_then(Value::as_array_mut) {
            for value in values {
                strict_node(value, root, depth + 1)?;
            }
        }
    }
    if let Some(all) = object
        .get("allOf")
        .and_then(Value::as_array)
        .filter(|a| a.len() == 1)
        && let Some(only) = all[0].as_object()
    {
        let mut flattened = object.clone();
        flattened.remove("allOf");
        flattened.extend(only.clone());
        *object = flattened;
        return strict_node(schema, root, depth + 1);
    }
    Ok(())
}
