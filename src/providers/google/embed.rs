//! Gemini embeddings: `embedContent` for one value and
//! `batchEmbedContents` for several (`google-embedding-model.ts`).

use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    api::{Endpoint, Flavor},
    options,
};
use crate::{
    Error,
    providers::{http, operations::*},
};

/// The most values one Gemini embedding call takes.
pub(crate) const MAX_VALUES: usize = 100;

const TASK_TYPES: &[&str] = &[
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
];

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EmbedOptions {
    output_dimensionality: Option<serde_json::Number>,
    task_type: Option<String>,
    /// One entry per value: `null`, or parts appended after the value's text.
    content: Option<Vec<Option<Vec<Value>>>>,
}

fn parse(params: &EmbedParams) -> Result<EmbedOptions, Error> {
    let provider_params = Some(&params.provider_options);
    let options: EmbedOptions = options::parse(Flavor::Gemini, provider_params)?;
    if let Some(task) = &options.task_type {
        options::check_one_of("taskType", task, TASK_TYPES)?;
    }
    for parts in options.content.iter().flatten().flatten() {
        if parts.is_empty() {
            return Err(Error::Validation(
                "a google content entry must be null or hold at least one part".into(),
            ));
        }
        for part in parts {
            let valid = part.get("text").is_some_and(Value::is_string)
                || part.get("inlineData").is_some_and(|d| {
                    d.get("mimeType").is_some_and(Value::is_string)
                        && d.get("data").is_some_and(Value::is_string)
                })
                || part.get("fileData").is_some_and(|d| {
                    d.get("fileUri").is_some_and(Value::is_string)
                        && d.get("mimeType").is_some_and(Value::is_string)
                });
            if !valid {
                return Err(Error::Validation(
                    "a google content part must be text, inlineData or fileData".into(),
                ));
            }
        }
    }
    Ok(options)
}

fn check_content(content: Option<&Vec<Option<Vec<Value>>>>, values: usize) -> Result<(), Error> {
    if let Some(content) = content
        && content.len() != values
    {
        return Err(Error::Validation(format!(
            "The number of multimodal content entries ({}) must match the number of values ({values}).",
            content.len()
        )));
    }
    Ok(())
}

/// The embed parameters for the values `start..end` of a split call: the
/// `content` option sliced to those values.
pub(crate) fn params_for(
    params: &EmbedParams,
    values: usize,
    range: std::ops::Range<usize>,
) -> Result<EmbedParams, Error> {
    let Some(Value::Object(google)) = params.provider_options.get("google") else {
        return Ok(params.clone());
    };
    let Some(Value::Array(content)) = google.get("content") else {
        return Ok(params.clone());
    };
    if content.len() != values {
        return Err(Error::Validation(format!(
            "The number of multimodal content entries ({}) must match the number of values ({values}).",
            content.len()
        )));
    }
    let mut google = google.clone();
    google.insert("content".into(), Value::Array(content[range].to_vec()));
    let mut params = params.clone();
    params
        .provider_options
        .insert("google".into(), Value::Object(google));
    Ok(params)
}

fn parts(value: &str, extra: Option<&Vec<Value>>) -> Vec<Value> {
    match extra {
        Some(extra) => {
            let mut parts = if value.is_empty() {
                Vec::new()
            } else {
                vec![json!({"text": value})]
            };
            parts.extend(extra.iter().cloned());
            parts
        }
        None => vec![json!({"text": value})],
    }
}

fn vector(value: &Value) -> Result<Vec<f64>, Error> {
    value
        .get("values")
        .and_then(Value::as_array)
        .and_then(|values| values.iter().map(Value::as_f64).collect::<Option<Vec<_>>>())
        .ok_or_else(|| Error::Validation("embedding values are missing".into()))
}

/// Embed up to 100 values.
pub(crate) async fn embed(
    endpoint: &Endpoint,
    model_id: &str,
    values: Vec<String>,
    params: &EmbedParams,
) -> Result<Item<Vec<Vec<f64>>>, Error> {
    if values.len() > MAX_VALUES {
        return Err(Error::Validation(format!(
            "{model_id} takes at most {MAX_VALUES} values per call, not {}",
            values.len()
        )));
    }
    let options = parse(params)?;
    check_content(options.content.as_ref(), values.len())?;
    let extra = |index: usize| {
        options
            .content
            .as_ref()
            .and_then(|content| content[index].as_ref())
    };
    let name = format!("models/{model_id}");
    let request = |content: Value| {
        let mut request = json!({"model": name, "content": content});
        if let Some(dimensions) = &options.output_dimensionality {
            request["outputDimensionality"] = json!(dimensions);
        }
        if let Some(task) = &options.task_type {
            request["taskType"] = json!(task);
        }
        request
    };
    let invalid = |what: &str| {
        http::response_error(
            &endpoint.provider,
            format!("Google embedding response {what}"),
        )
    };
    if values.len() == 1 {
        let body = request(json!({"parts": parts(&values[0], extra(0))}));
        let http = endpoint.request(
            "POST",
            &["models".into(), format!("{model_id}:embedContent")],
            Some(body),
            Default::default(),
            Default::default(),
        )?;
        let (_, response) = endpoint
            .json(http, Some(model_id), http::MAX_RESPONSE_BYTES)
            .await?;
        let embedding = response
            .get("embedding")
            .ok_or_else(|| invalid("has no embedding"))?;
        return Ok(Item::new(vec![
            vector(embedding).map_err(|_| invalid("has no values"))?,
        ]));
    }
    let requests: Vec<Value> = values
        .iter()
        .enumerate()
        .map(|(index, value)| request(json!({"role": "user", "parts": parts(value, extra(index))})))
        .collect();
    let http = endpoint.request(
        "POST",
        &["models".into(), format!("{model_id}:batchEmbedContents")],
        Some(json!({"requests": requests})),
        Default::default(),
        Default::default(),
    )?;
    let (_, response) = endpoint
        .json(http, Some(model_id), http::MAX_RESPONSE_BYTES)
        .await?;
    let embeddings = response
        .get("embeddings")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("has no embeddings"))?
        .iter()
        .map(vector)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| invalid("has an embedding without values"))?;
    if embeddings.len() != values.len() {
        return Err(invalid("has another number of embeddings than values"));
    }
    Ok(Item::new(embeddings))
}

/// Whether a Vertex AI embedding model uses `embedContent`, one value per
/// call; the others use `predict` (`google-vertex-embedding-model.ts`).
fn uses_embed_content(model_id: &str) -> bool {
    matches!(
        model_id,
        "gemini-embedding-2" | "gemini-embedding-2-preview"
    )
}

/// The most values one Vertex AI embedding call takes for a model (VTX-006).
pub(crate) fn vertex_max_values(model_id: &str) -> usize {
    if uses_embed_content(model_id) { 1 } else { 250 }
}

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct VertexEmbedOptions {
    output_dimensionality: Option<serde_json::Number>,
    task_type: Option<String>,
    title: Option<String>,
    auto_truncate: Option<bool>,
}

/// A token count, which Vertex AI can write as a whole float.
fn token_count(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| {
        value
            .as_f64()
            .filter(|n| n.is_finite() && *n >= 0.0 && n.fract() == 0.0)
            .map(|n| n as u64)
    })
}

/// Embed values on Vertex AI: one value with `embedContent`, or up to 250
/// with `predict` (VTX-006).
pub(crate) async fn embed_vertex(
    endpoint: &Endpoint,
    model_id: &str,
    values: Vec<String>,
    params: &EmbedParams,
) -> Result<Item<Vec<Vec<f64>>>, Error> {
    let max = vertex_max_values(model_id);
    if values.len() > max {
        return Err(Error::Validation(format!(
            "{model_id} takes at most {max} values per call, not {}",
            values.len()
        )));
    }
    let options: VertexEmbedOptions =
        options::parse(Flavor::Vertex, Some(&params.provider_options))?;
    if let Some(task) = &options.task_type {
        options::check_one_of("taskType", task, TASK_TYPES)?;
    }
    let invalid = |what: &str| {
        http::response_error(
            &endpoint.provider,
            format!("Google Vertex AI embedding response {what}"),
        )
    };
    let mut parameters = serde_json::Map::new();
    if let Some(dimensions) = &options.output_dimensionality {
        parameters.insert("outputDimensionality".into(), json!(dimensions));
    }
    if uses_embed_content(model_id) {
        let mut config = parameters;
        if let Some(task) = &options.task_type {
            config.insert("taskType".into(), json!(task));
        }
        if let Some(title) = &options.title {
            config.insert("title".into(), json!(title));
        }
        if let Some(truncate) = options.auto_truncate {
            config.insert("autoTruncate".into(), json!(truncate));
        }
        let text = values.first().cloned().unwrap_or_default();
        let body = json!({"content": {"parts": [{"text": text}]}, "embedContentConfig": config});
        let request = endpoint.request(
            "POST",
            &["models".into(), format!("{model_id}:embedContent")],
            Some(body),
            Default::default(),
            Default::default(),
        )?;
        let (_, response) = endpoint
            .json(request, Some(model_id), http::MAX_RESPONSE_BYTES)
            .await?;
        let embedding = response
            .get("embedding")
            .ok_or_else(|| invalid("has no embedding"))?;
        let mut item = Item::new(vec![
            vector(embedding).map_err(|_| invalid("has no values"))?,
        ]);
        item.usage = response
            .get("usageMetadata")
            .and_then(|usage| usage.get("promptTokenCount"))
            .and_then(Value::as_u64)
            .map(|tokens| crate::types::usage::Usage {
                input_tokens: tokens,
                ..Default::default()
            });
        return Ok(item);
    }
    if let Some(truncate) = options.auto_truncate {
        parameters.insert("autoTruncate".into(), json!(truncate));
    }
    let instances: Vec<Value> = values
        .iter()
        .map(|value| {
            let mut instance = json!({"content": value});
            if let Some(task) = &options.task_type {
                instance["task_type"] = json!(task);
            }
            if let Some(title) = &options.title {
                instance["title"] = json!(title);
            }
            instance
        })
        .collect();
    let request = endpoint.request(
        "POST",
        &["models".into(), format!("{model_id}:predict")],
        Some(json!({"instances": instances, "parameters": parameters})),
        Default::default(),
        Default::default(),
    )?;
    let (_, response) = endpoint
        .json(request, Some(model_id), http::MAX_RESPONSE_BYTES)
        .await?;
    let predictions = response
        .get("predictions")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("has no predictions"))?;
    let mut embeddings = Vec::with_capacity(predictions.len());
    let mut tokens = 0u64;
    for prediction in predictions {
        let embedding = prediction
            .get("embeddings")
            .ok_or_else(|| invalid("has a prediction without embeddings"))?;
        embeddings.push(vector(embedding).map_err(|_| invalid("has an embedding without values"))?);
        tokens = tokens.saturating_add(
            embedding
                .get("statistics")
                .and_then(|s| s.get("token_count"))
                .and_then(token_count)
                .ok_or_else(|| invalid("has an embedding without a token count"))?,
        );
    }
    if embeddings.len() != values.len() {
        return Err(invalid("has another number of embeddings than values"));
    }
    let mut item = Item::new(embeddings);
    item.usage = Some(crate::types::usage::Usage {
        input_tokens: tokens,
        ..Default::default()
    });
    Ok(item)
}
