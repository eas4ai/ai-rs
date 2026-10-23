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
