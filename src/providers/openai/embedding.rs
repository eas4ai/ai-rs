//! Embeddings: OpenAI's (`embedding/openai-embedding-model.ts`) and an
//! OpenAI-compatible server's (`openai-compatible-embedding-model.ts`),
//! OAI-016.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{errors, params::invalid, protocol::http_request};
use crate::{
    Error, Model, Provider, ProviderProtocol,
    providers::{
        http,
        operations::{CallLimits, EmbedParams, Item},
    },
    types::usage::Usage,
};

/// OpenAI's embedding model limit of values per call.
const MAX_VALUES: usize = 2048;

/// Embeds with `POST <base>/embeddings`. [`OpenAIEmbeddingProtocol::new`]
/// speaks OpenAI's API, which OpenAI's language model protocols also use
/// for `embed`; [`OpenAIEmbeddingProtocol::compatible`] an OpenAI-compatible
/// server's.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAIEmbeddingProtocol {
    compatible: Option<Compatible>,
}

#[derive(Clone, Copy, Debug)]
struct Compatible {
    max_per_call: usize,
    parallel: bool,
}

impl OpenAIEmbeddingProtocol {
    /// OpenAI's embedding API: 2048 values per call, calls in parallel.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The embedding API of an OpenAI-compatible server, which takes at most
    /// `max_per_call` values per call, and calls in parallel when `parallel`.
    /// Its options are the entry under the provider's name, that name in
    /// camelCase, or `openaiCompatible`, and the `providerMetadata` of its
    /// response becomes the result's provider metadata.
    #[must_use]
    pub fn compatible(max_per_call: usize, parallel: bool) -> Self {
        Self {
            compatible: Some(Compatible {
                max_per_call: max_per_call.max(1),
                parallel,
            }),
        }
    }

    fn limits(self) -> CallLimits {
        match self.compatible {
            Some(c) => CallLimits {
                max_per_call: Some(c.max_per_call),
                parallel: c.parallel,
            },
            None => CallLimits {
                max_per_call: Some(MAX_VALUES),
                parallel: true,
            },
        }
    }
}

/// The embedding options (`openaiEmbeddingModelOptions`).
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EmbeddingOptions {
    dimensions: Option<serde_json::Number>,
    user: Option<String>,
}

/// `name` in camelCase: `my-server` becomes `myServer`.
fn camel_case(name: &str) -> String {
    let mut out = String::new();
    let mut upper = false;
    for c in name.chars() {
        if c == '-' || c == '_' {
            upper = true;
        } else if upper {
            out.extend(c.to_uppercase());
            upper = false;
        } else {
            out.push(c);
        }
    }
    out
}

fn options(
    protocol: OpenAIEmbeddingProtocol,
    provider: &dyn Provider,
    params: &EmbedParams,
) -> Result<EmbeddingOptions, Error> {
    let name = provider.name();
    let mut keys = match protocol.compatible {
        Some(_) => vec![
            "openaiCompatible".to_owned(),
            name.to_owned(),
            camel_case(name),
        ],
        None => vec!["openai".to_owned()],
    };
    // A host's own entry, whatever the provider is named (HST-028).
    if let Some(key) = protocol
        .compatible
        .and_then(|_| crate::providers::hosts::entry_key(provider))
        .filter(|key| !keys.iter().any(|k| k == key))
    {
        keys.push(key.to_owned());
    }
    let mut merged = Map::new();
    for key in keys {
        match params.provider_options.get(&key) {
            None => {}
            Some(Value::Object(entry)) => merged.extend(entry.clone()),
            Some(_) => {
                return Err(invalid(format!(
                    "{key} embedding options must be an object"
                )));
            }
        }
    }
    serde_json::from_value(Value::Object(merged))
        .map_err(|e| invalid(format!("invalid embedding options: {e}")))
}

pub(super) async fn embed(
    protocol: OpenAIEmbeddingProtocol,
    model: &Model,
    values: Vec<String>,
    params: EmbedParams,
    provider: &dyn Provider,
) -> Result<Item<Vec<Vec<f64>>>, Error> {
    let max = protocol.limits().max_per_call.unwrap_or(MAX_VALUES);
    if values.len() > max {
        return Err(invalid(format!(
            "{} takes at most {max} values per call, not {}",
            model.id,
            values.len()
        )));
    }
    let options = options(protocol, provider, &params)?;
    let mut body = json!({"model": model.id, "input": values, "encoding_format": "float"});
    if let Some(dimensions) = options.dimensions {
        body["dimensions"] = Value::Number(dimensions);
    }
    if let Some(user) = options.user {
        body["user"] = json!(user);
    }
    // A host's errors read as its chat errors do (HST-028); another
    // OpenAI-compatible server gets no OpenAI organization or project.
    let api = match protocol.compatible {
        Some(_) => crate::providers::hosts::chat_api(provider).unwrap_or(&errors::COMPATIBLE_API),
        None => &errors::API,
    };
    let request = http_request(
        api,
        "POST",
        provider.config(),
        &["embeddings"],
        Some(body),
        None,
    )?;
    let transport = provider.transport().ok_or_else(|| {
        Error::Configuration(format!(
            "provider {:?} has no transport for embeddings",
            provider.name()
        ))
    })?;
    let response = http::json(api, transport, request, provider.name(), Some(&model.id)).await?;
    let bad = || {
        http::response_error(
            provider.name(),
            "OpenAI embedding response has no embeddings",
        )
    };
    let embeddings = response["data"]
        .as_array()
        .ok_or_else(bad)?
        .iter()
        .map(|item| {
            item["embedding"]
                .as_array()
                .and_then(|values| {
                    values
                        .iter()
                        .map(Value::as_f64)
                        .collect::<Option<Vec<f64>>>()
                })
                .ok_or_else(bad)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut item = Item::new(embeddings);
    if let Some(tokens) = response["usage"]["prompt_tokens"].as_u64() {
        item.usage = Some(Usage {
            input_tokens: tokens,
            raw: Some(response["usage"].clone()),
            ..Default::default()
        });
    }
    if protocol.compatible.is_some()
        && let Some(Value::Object(metadata)) = response.get("providerMetadata")
    {
        item.provider_metadata = Some(metadata.clone().into_iter().collect());
    }
    Ok(item)
}

/// The limits OpenAI's language model protocols give `embed`.
pub(super) fn openai_limits() -> CallLimits {
    OpenAIEmbeddingProtocol::new().limits()
}

/// Restore the protocol from its snapshot value.
pub(super) fn restore(value: &Value) -> OpenAIEmbeddingProtocol {
    match value.get("protocol_class_id").and_then(Value::as_str) {
        Some("openai_compatible_embedding") => OpenAIEmbeddingProtocol::compatible(
            value
                .get("max_per_call")
                .and_then(Value::as_u64)
                .map_or(MAX_VALUES, |n| usize::try_from(n).unwrap_or(MAX_VALUES)),
            value
                .get("parallel")
                .and_then(Value::as_bool)
                .unwrap_or(true),
        ),
        _ => OpenAIEmbeddingProtocol::new(),
    }
}

#[async_trait]
impl ProviderProtocol for OpenAIEmbeddingProtocol {
    fn protocol_class_id(&self) -> &str {
        match self.compatible {
            Some(_) => "openai_compatible_embedding",
            None => "openai_embedding",
        }
    }

    fn to_value(&self) -> Result<Value, Error> {
        let mut value = json!({"protocol_class_id": self.protocol_class_id()});
        if let Some(c) = self.compatible {
            value["max_per_call"] = json!(c.max_per_call);
            value["parallel"] = json!(c.parallel);
        }
        Ok(value)
    }

    async fn embed(
        &self,
        model: &Model,
        values: Vec<String>,
        params: EmbedParams,
        provider: &dyn Provider,
    ) -> Result<Item<Vec<Vec<f64>>>, Error> {
        embed(*self, model, values, params, provider).await
    }

    fn embed_limits(&self, _model: &Model) -> CallLimits {
        self.limits()
    }
}
