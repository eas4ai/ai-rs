//! Together AI's reranking (`reranking/togetherai-reranking-model.ts`,
//! HST-032).
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    Error, Model, Provider,
    providers::{
        http::{self, ApiProfile},
        operations::{Documents, Item, RankedDocument, RerankParams},
    },
    types::messages::Metadata,
};

/// The `togetherai` reranking options.
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Options {
    rank_fields: Option<Vec<String>>,
}

/// `POST <base>/rerank`, without the documents in the response.
pub(super) async fn together(
    api: &'static ApiProfile,
    model: &Model,
    documents: Documents,
    query: String,
    params: RerankParams,
    provider: &dyn Provider,
) -> Result<Item<Vec<RankedDocument>>, Error> {
    let options: Options = match params.provider_options.get("togetherai") {
        None => Options::default(),
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|e| Error::Validation(format!("invalid togetherai reranking options: {e}")))?,
    };
    let documents = match documents {
        Documents::Text(texts) => json!(texts),
        Documents::Json(objects) => json!(objects),
    };
    let mut body = json!({"model": model.id, "query": query, "documents": documents});
    if let Some(top_n) = params.top_n {
        body["top_n"] = json!(top_n);
    }
    if let Some(fields) = options.rank_fields {
        body["rank_fields"] = json!(fields);
    }
    // The response need not repeat the documents.
    body["return_documents"] = json!(false);
    let request = http::request(
        api,
        provider.config(),
        "POST",
        &["rerank"],
        Some(body),
        Default::default(),
        Default::default(),
    )?;
    let transport = provider.transport().ok_or_else(|| {
        Error::Configuration(format!(
            "provider {:?} has no transport for reranking",
            provider.name()
        ))
    })?;
    let response = http::json(api, transport, request, provider.name(), Some(&model.id)).await?;
    let bad = || {
        http::response_error(
            provider.name(),
            "Together AI rerank response has no results",
        )
    };
    let ranking = response["results"]
        .as_array()
        .ok_or_else(bad)?
        .iter()
        .map(|result| {
            let index = result["index"]
                .as_u64()
                .and_then(|i| usize::try_from(i).ok());
            let score = result["relevance_score"].as_f64();
            match (index, score) {
                (Some(index), Some(score)) => Ok(RankedDocument { index, score }),
                _ => Err(bad()),
            }
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let mut metadata = Metadata::new();
    for key in ["id", "model"] {
        if let Some(value) = response.get(key).filter(|v| !v.is_null()) {
            metadata.insert(key.into(), Value::clone(value));
        }
    }
    let mut item = Item::new(ranking);
    item.metadata = (!metadata.is_empty()).then_some(metadata);
    Ok(item)
}
