//! OpenAI's native evaluation with `POST <base>/decisions`
//! (`openai-decision-model.ts`), OAI-025.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use super::{errors, params::invalid, protocol::http_request};
use crate::{
    Error, Model, Provider, ProviderProtocol,
    errors::{ProviderError, ProviderErrorKind},
    providers::{
        http,
        operations::{EvaluationInput, EvaluationParams, Item, Question},
    },
    types::{messages::Metadata, usage::Usage},
};

/// Evaluates with OpenAI's Decisions API, as `gpt-6-luna` takes it: build a
/// model with `Model::with_protocol(Arc::new(OpenAIDecisionsProtocol))`.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAIDecisionsProtocol;

/// The decision options (`openaiDecisionModelOptions`).
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DecisionOptions {
    safety_identifier: Option<String>,
}

fn to_text(input: &EvaluationInput) -> Result<String, Error> {
    Ok(match input {
        EvaluationInput::Text(text) => text.clone(),
        EvaluationInput::Object(value) => serde_json::to_string(value)?,
        EvaluationInput::Array(value) => serde_json::to_string(value)?,
    })
}

fn question(name: &str, question: &Question) -> Result<Value, Error> {
    Ok(match question {
        Question::Choice {
            instructions,
            criteria,
        } => {
            let choices = criteria
                .iter()
                .map(|(value, description)| {
                    let mut choice = json!({"value": value});
                    if let Some(description) = description {
                        choice["description"] = json!(to_text(description)?);
                    }
                    Ok(choice)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            json!({"type": "choice", "name": name, "instructions": to_text(instructions)?, "choices": choices})
        }
        Question::Score {
            instructions,
            criteria,
        } => {
            // Score levels have no labels of their own; their indices name them.
            let levels = criteria
                .iter()
                .enumerate()
                .map(|(index, description)| {
                    let mut level = json!({"label": index.to_string()});
                    if let Some(description) = description {
                        level["description"] = json!(to_text(description)?);
                    }
                    Ok(level)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            json!({"type": "score", "name": name, "instructions": to_text(instructions)?, "levels": levels})
        }
        Question::Noul {
            instructions,
            criteria,
        } => {
            let mut parts = vec![to_text(instructions)?];
            if let Some(criteria) = criteria {
                if let Some(truth) = &criteria.r#true {
                    parts.push(format!("Criteria for true:\n{}", to_text(truth)?));
                }
                if let Some(falsity) = &criteria.r#false {
                    parts.push(format!("Criteria for false:\n{}", to_text(falsity)?));
                }
            }
            json!({"type": "predicate", "name": name, "instructions": parts.join("\n\n")})
        }
    })
}

fn bad(provider: &str, message: &str) -> Error {
    ProviderError::new(ProviderErrorKind::Response, message)
        .with_provider(provider)
        .into()
}

fn probability(value: &Value) -> Option<f64> {
    value.as_f64().filter(|p| (0.0..=1.0).contains(p))
}

pub(super) async fn evaluate(
    model: &Model,
    state: EvaluationInput,
    questions: BTreeMap<String, Question>,
    params: EvaluationParams,
    provider: &dyn Provider,
) -> Result<Item<Metadata>, Error> {
    if questions.is_empty() {
        return Err(invalid("Decision requires at least one question."));
    }
    for question in questions.values() {
        question.validate()?;
    }
    let options: DecisionOptions = match params.provider_options.get("openai") {
        None => DecisionOptions::default(),
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|e| invalid(format!("invalid openai decision options: {e}")))?,
    };
    if options
        .safety_identifier
        .as_ref()
        .is_some_and(|id| id.chars().count() > 128)
    {
        return Err(invalid(
            "openai decision option safetyIdentifier must be at most 128 characters",
        ));
    }
    let mut body = json!({
        "model": model.id,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": to_text(&state)?}]}],
        "questions": questions
            .iter()
            .map(|(name, q)| question(name, q))
            .collect::<Result<Vec<_>, Error>>()?,
    });
    if let Some(id) = options.safety_identifier {
        body["safety_identifier"] = json!(id);
    }
    let request = http_request(
        &errors::API,
        "POST",
        provider.config(),
        &["decisions"],
        Some(body),
        None,
    )?;
    let transport = provider.transport().ok_or_else(|| {
        Error::Configuration(format!(
            "provider {:?} has no transport for decisions",
            provider.name()
        ))
    })?;
    let response = http::json(
        &errors::API,
        transport,
        request,
        provider.name(),
        Some(&model.id),
    )
    .await?;
    let name = provider.name();
    let answers_list = response["answers"]
        .as_array()
        .ok_or_else(|| bad(name, "OpenAI Decisions response has no answers"))?;
    let mut answers = Metadata::new();
    let mut confidence = Map::new();
    let mut probabilities = Map::new();
    let mut seen = BTreeSet::new();
    for answer in answers_list {
        let id = answer["name"]
            .as_str()
            .ok_or_else(|| bad(name, "OpenAI Decisions answered a question without a name"))?;
        let Some(question) = questions.get(id) else {
            return Err(bad(
                name,
                "Decisions must return exactly one answer for every question.",
            ));
        };
        if !seen.insert(id.to_owned()) {
            return Err(bad(
                name,
                "Decisions must return exactly one answer for every question.",
            ));
        }
        let value = match (answer["type"].as_str().unwrap_or_default(), question) {
            ("refusal", _) => json!({"type": "refusal"}),
            ("predicate", Question::Noul { .. }) => {
                let p = probability(&answer["probability"]).ok_or_else(|| {
                    bad(
                        name,
                        "OpenAI Decisions returned a probability outside 0 to 1",
                    )
                })?;
                json!({"type": "noul", "noul": p})
            }
            ("choice", Question::Choice { criteria, .. }) => {
                let choice = answer["choice"]
                    .as_str()
                    .filter(|choice| criteria.contains_key(*choice))
                    .ok_or_else(|| bad(name, "OpenAI Decisions returned an unknown choice"))?;
                json!({"type": "choice", "choice": choice})
            }
            ("score", Question::Score { criteria, .. }) => {
                let score = answer["score"]
                    .as_f64()
                    .filter(|s| (0.0..=(criteria.len() - 1) as f64).contains(s))
                    .ok_or_else(|| {
                        bad(name, "OpenAI Decisions returned a score outside its levels")
                    })?;
                json!({"type": "score", "score": score})
            }
            _ => {
                return Err(bad(
                    name,
                    "OpenAI Decisions answered a question with another type",
                ));
            }
        };
        if let Some(c) = answer.get("confidence").filter(|c| !c.is_null()) {
            confidence.insert(id.to_owned(), c.clone());
        }
        if let Some(list) = answer["probabilities"].as_array() {
            let mut by_value = Map::new();
            for entry in list {
                let key = match &entry["value"] {
                    Value::String(text) => text.clone(),
                    other => other.to_string(),
                };
                let p = probability(&entry["probability"]).ok_or_else(|| {
                    bad(
                        name,
                        "OpenAI Decisions returned a probability outside 0 to 1",
                    )
                })?;
                if by_value.insert(key, json!(p)).is_some() {
                    return Err(bad(
                        name,
                        "Decisions returned duplicate probability values.",
                    ));
                }
            }
            probabilities.insert(id.to_owned(), Value::Object(by_value));
        }
        answers.insert(id.to_owned(), value);
    }
    if seen.len() != questions.len() {
        return Err(bad(
            name,
            "Decisions must return exactly one answer for every question.",
        ));
    }
    let mut item = Item::new(answers);
    if response["usage"].is_object() {
        item.usage = Some(Usage {
            input_tokens: response["usage"]["input_tokens"].as_u64().unwrap_or(0),
            output_tokens: response["usage"]["output_tokens"].as_u64().unwrap_or(0),
            raw: Some(response["usage"].clone()),
            ..Default::default()
        });
    }
    let mut openai = json!({"confidence": confidence, "probabilities": probabilities});
    if response["usage"].is_object() {
        openai["usage"] = response["usage"].clone();
    }
    item.provider_metadata = Some([("openai".to_owned(), openai)].into());
    Ok(item)
}

#[async_trait]
impl ProviderProtocol for OpenAIDecisionsProtocol {
    fn protocol_class_id(&self) -> &str {
        "openai_decisions"
    }

    async fn evaluate(
        &self,
        model: &Model,
        state: EvaluationInput,
        questions: BTreeMap<String, Question>,
        params: EvaluationParams,
        provider: &dyn Provider,
    ) -> Result<Item<Metadata>, Error> {
        evaluate(model, state, questions, params, provider).await
    }
}
