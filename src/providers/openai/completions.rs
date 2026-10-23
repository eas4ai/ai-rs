//! The Completions API: OpenAI's (`completion/openai-completion-language-model.ts`)
//! and an OpenAI-compatible server's
//! (`openai-compatible-completion-language-model.ts`), OAI-015.

use std::collections::VecDeque;

use futures::{StreamExt, TryStreamExt, stream};
use serde_json::{Map, Value, json};

use super::{dialect, errors, params::invalid, protocol::http_request};
use crate::{
    Error, Model, Provider, ProviderProtocol, StreamOptions,
    models::params::{InferenceRequestParams, ProviderValue, Seed},
    providers::{EventStream, http},
    transport::{Body, HttpRequest},
    types::{
        Message, MessageRole, Part,
        events::{Event, ModelEventMeta, StreamEnd, StreamStart, TextDelta, TextEnd, TextStart},
        usage::Usage,
    },
};

/// Streams `POST <base>/completions` with a prompt made from the history.
/// [`OpenAICompletionsProtocol::new`] speaks OpenAI's API;
/// [`OpenAICompletionsProtocol::compatible`] an OpenAI-compatible server's,
/// which takes the call's other provider options in the body.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAICompletionsProtocol {
    compatible: Option<Compatible>,
}

#[derive(Clone, Copy, Debug)]
struct Compatible {
    include_usage: bool,
}

impl OpenAICompletionsProtocol {
    /// OpenAI's Completions API.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The Completions API of an OpenAI-compatible server. Its provider
    /// options are the entry under the provider's name, or that name in
    /// camelCase, and the options other than `echo`, `logitBias`, `suffix`
    /// and `user` go into the body as given. `include_usage` asks for usage
    /// in the stream with `stream_options`, which some servers reject.
    #[must_use]
    pub fn compatible(include_usage: bool) -> Self {
        Self {
            compatible: Some(Compatible { include_usage }),
        }
    }
}

impl ProviderProtocol for OpenAICompletionsProtocol {
    fn protocol_class_id(&self) -> &str {
        match self.compatible {
            Some(_) => "openai_compatible_completions",
            None => "openai_completions",
        }
    }

    fn to_value(&self) -> Result<Value, Error> {
        let mut value = json!({"protocol_class_id": self.protocol_class_id()});
        if let Some(compatible) = self.compatible {
            value["include_usage"] = json!(compatible.include_usage);
        }
        Ok(value)
    }

    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        let request = request(*self, &model.id, provider, messages, &options);
        let provider_name = provider.name().to_owned();
        let model_id = model.id.clone();
        let transport = provider.shared_transport();
        stream::once(async move {
            let request = request?;
            let transport = transport.ok_or_else(|| {
                Error::Configuration(
                    "OpenAI completions protocols require Provider::shared_transport".into(),
                )
            })?;
            let response = http::send(
                &errors::API,
                transport.as_ref(),
                request,
                &provider_name,
                Some(&model_id),
            )
            .await?;
            Ok::<_, Error>(read(response.body, provider_name))
        })
        .try_flatten()
        .boxed()
    }
}

/// Restore the protocol from its snapshot value.
pub(super) fn restore(value: &Value) -> OpenAICompletionsProtocol {
    match value.get("protocol_class_id").and_then(Value::as_str) {
        Some("openai_compatible_completions") => OpenAICompletionsProtocol::compatible(
            value
                .get("include_usage")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        ),
        _ => OpenAICompletionsProtocol::new(),
    }
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

/// The provider options of the call: the `openai` entry for OpenAI, the
/// entries under the provider's name and its camelCase form for another
/// server.
fn provider_options(
    protocol: OpenAICompletionsProtocol,
    provider: &str,
    p: &InferenceRequestParams,
) -> Result<Map<String, Value>, Error> {
    let keys = match protocol.compatible {
        Some(_) => vec![provider.to_owned(), camel_case(provider)],
        None => vec!["openai".to_owned()],
    };
    let mut options = Map::new();
    for key in keys {
        match p.provider_params.as_ref().and_then(|params| params.get(&key)) {
            None => {}
            Some(Value::Object(entry)) => options.extend(entry.clone()),
            Some(_) => {
                return Err(invalid(format!(
                    "{key} provider options must be an object"
                )));
            }
        }
    }
    Ok(options)
}

/// Fail on the core settings the Completions API never takes.
fn check_params(p: &InferenceRequestParams) -> Result<(), Error> {
    let unsupported = |name: &str| {
        Err(invalid(format!(
            "OpenAI completions does not support {name}"
        )))
    };
    if let ProviderValue::Value(sampling) = &p.sampling {
        if sampling
            .top_k
            .as_ref()
            .is_some_and(|t| matches!(t.top_k, ProviderValue::Value(Some(_))))
        {
            return unsupported("top_k");
        }
        if sampling
            .min_p
            .as_ref()
            .is_some_and(|t| matches!(t.min_p, ProviderValue::Value(Some(_))))
        {
            return unsupported("min_p");
        }
        if let Some(r) = &sampling.repetition_penalty
            && (matches!(r.repetition_penalty, ProviderValue::Value(Some(_)))
                || matches!(r.consideration_window, ProviderValue::Value(Some(_))))
        {
            return unsupported("repetition_penalty");
        }
    }
    if matches!(p.reasoning, ProviderValue::Value(_)) {
        return unsupported("reasoning");
    }
    if p.tool_calling.is_some() {
        return unsupported("tool calling");
    }
    if p.provider_service.is_some() {
        return unsupported("service tiers");
    }
    if p.safety_identifier.is_some() {
        return unsupported("safety identifiers");
    }
    if p.metadata.is_some() {
        return unsupported("metadata");
    }
    if p.cache.is_some() {
        return unsupported("prompt caching");
    }
    if p.context_management.is_some() {
        return unsupported("context management");
    }
    if let Some(output) = &p.output
        && (output.include.is_some()
            || matches!(output.text_verbosity, ProviderValue::Value(_))
            || matches!(output.reasoning_summary, ProviderValue::Value(_)))
    {
        return unsupported("output include, verbosity or reasoning summaries");
    }
    Ok(())
}

/// The text of a message's text parts.
fn text_of(message: &Message) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            Part::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// The prompt of a history: the first system message and a blank line, then
/// `user:` and `assistant:` turns, ending with `assistant:`
/// (`convert-to-openai-completion-prompt.ts`).
fn prompt(messages: &[Message]) -> Result<String, Error> {
    let mut messages: Vec<&Message> = messages
        .iter()
        .filter(|m| m.role != MessageRole::Internal)
        .collect();
    let mut text = String::new();
    if messages.first().is_some_and(|m| m.role == MessageRole::System) {
        text.push_str(&text_of(messages.remove(0)));
        text.push_str("\n\n");
    }
    for message in messages {
        match message.role {
            MessageRole::System => {
                return Err(invalid(
                    "OpenAI completions takes a system message only as the first message",
                ));
            }
            MessageRole::User => {
                text.push_str(&format!("user:\n{}\n\n", text_of(message)));
            }
            MessageRole::Assistant => {
                if message
                    .parts
                    .iter()
                    .any(|p| matches!(p, Part::ToolCall(_) | Part::BuiltinToolCall(_)))
                {
                    return Err(invalid("OpenAI completions does not take tool calls"));
                }
                text.push_str(&format!("assistant:\n{}\n\n", text_of(message)));
            }
            MessageRole::Tool => {
                return Err(invalid("OpenAI completions does not take tool results"));
            }
            MessageRole::Internal => {}
        }
    }
    text.push_str("assistant:\n");
    Ok(text)
}

fn request(
    protocol: OpenAICompletionsProtocol,
    model: &str,
    provider: &dyn Provider,
    messages: Vec<Message>,
    options: &StreamOptions,
) -> Result<HttpRequest, Error> {
    if !options.tools.is_empty() {
        return Err(invalid("OpenAI completions does not take tools"));
    }
    if options.tool_choice.is_some() {
        return Err(invalid("OpenAI completions does not take a tool choice"));
    }
    if options.output_schema.is_some() {
        return Err(invalid("OpenAI completions does not take an output schema"));
    }
    let p = options.params.clone().unwrap_or_default();
    check_params(&p)?;
    let mut provider_options = provider_options(protocol, provider.name(), &p)?;
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("prompt".into(), json!(prompt(&messages)?));
    body.insert("stream".into(), json!(true));
    if protocol.compatible.is_none_or(|c| c.include_usage) {
        body.insert("stream_options".into(), json!({"include_usage": true}));
    }
    body.insert("stop".into(), json!(["\nuser:"]));
    if let Some(max) = p.output.as_ref().and_then(|o| o.max_tokens) {
        body.insert("max_tokens".into(), json!(max));
    }
    if let ProviderValue::Value(sampling) = &p.sampling {
        if let Some(ProviderValue::Value(t)) = sampling.temperature.as_ref().map(|t| &t.temperature) {
            body.insert("temperature".into(), json!(t));
        }
        if let Some(ProviderValue::Value(Some(t))) = sampling.top_p.as_ref().map(|t| &t.top_p) {
            body.insert("top_p".into(), json!(t));
        }
        if let Some(r) = &sampling.repetition_penalty {
            if let ProviderValue::Value(Some(v)) = r.frequency_penalty {
                body.insert("frequency_penalty".into(), json!(v));
            }
            if let ProviderValue::Value(Some(v)) = r.presence_penalty {
                body.insert("presence_penalty".into(), json!(v));
            }
        }
        if let Some(seed) = &sampling.seed
            && let ProviderValue::Value(Some(Seed::Value(seed))) = &seed.seed
            && *seed != -1
        {
            body.insert("seed".into(), json!(seed));
        }
    }
    let known = ["echo", "logitBias", "suffix", "user", "logprobs"];
    let take = |options: &mut Map<String, Value>, name: &str| options.remove(name);
    if let Some(echo) = take(&mut provider_options, "echo") {
        let echo = echo
            .as_bool()
            .ok_or_else(|| invalid("completions option echo must be a boolean"))?;
        body.insert("echo".into(), json!(echo));
    }
    if let Some(bias) = take(&mut provider_options, "logitBias") {
        let valid = bias
            .as_object()
            .is_some_and(|bias| bias.values().all(Value::is_number));
        if !valid {
            return Err(invalid(
                "completions option logitBias must map tokens to numbers",
            ));
        }
        body.insert("logit_bias".into(), bias);
    }
    for name in ["suffix", "user"] {
        if let Some(value) = take(&mut provider_options, name) {
            if !value.is_string() {
                return Err(invalid(format!(
                    "completions option {name} must be a string"
                )));
            }
            body.insert(name.into(), value);
        }
    }
    match protocol.compatible {
        None => {
            if let Some(logprobs) = take(&mut provider_options, "logprobs") {
                match logprobs {
                    Value::Bool(true) => {
                        body.insert("logprobs".into(), json!(0));
                    }
                    Value::Bool(false) => {}
                    Value::Number(n) => {
                        body.insert("logprobs".into(), Value::Number(n));
                    }
                    other => {
                        return Err(invalid(format!(
                            "completions option logprobs must be a boolean or a number, not {other}"
                        )));
                    }
                }
            }
            if let Some(name) = provider_options.keys().next() {
                return Err(invalid(format!(
                    "unknown openai completions option {name:?}; it takes {}",
                    known.join(", ")
                )));
            }
        }
        // Another server takes its other options as given.
        Some(_) => body.extend(provider_options),
    }
    if let Some(extra) = &p.extra_body {
        body.extend(extra.clone());
    }
    http_request(
        &errors::API,
        "POST",
        provider.config(),
        &["completions"],
        Some(Value::Object(body)),
        Some(&p),
    )
}

/// The reader of a Completions stream: `choices[0].text` as text, the
/// finish reasons of Chat Completions, the logprobs into the final
/// `openai.logprobs`, and `prompt_tokens` and `completion_tokens` as usage.
fn read(body: Body, provider: String) -> EventStream {
    struct Reader {
        input: futures::stream::BoxStream<'static, Result<http::SseJson, Error>>,
        provider: String,
        queue: VecDeque<Event>,
        started: bool,
        text: bool,
        ended: bool,
        usage: Option<Usage>,
        finish: Option<String>,
        logprobs: Vec<Value>,
        response_id: Option<String>,
        response_model: Option<String>,
    }
    impl Reader {
        fn start(&mut self) {
            if !self.started {
                self.started = true;
                self.queue.push_back(Event::StreamStart(StreamStart::default()));
            }
        }
        fn chunk(&mut self, data: &Value) {
            self.start();
            if let Some(id) = data["id"].as_str() {
                self.response_id = Some(id.to_owned());
            }
            if let Some(model) = data["model"].as_str() {
                self.response_model = Some(model.to_owned());
            }
            if data["usage"].is_object() {
                self.usage = Some(Usage {
                    input_tokens: data["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
                    output_tokens: data["usage"]["completion_tokens"].as_u64().unwrap_or(0),
                    raw: Some(data["usage"].clone()),
                    ..Default::default()
                });
            }
            let choice = &data["choices"][0];
            if let Some(reason) = choice["finish_reason"].as_str() {
                self.finish = Some(
                    match reason {
                        "stop" => "stop",
                        "length" => "length",
                        "content_filter" => "content_filter",
                        "function_call" | "tool_calls" => "tool_call",
                        _ => "other",
                    }
                    .to_owned(),
                );
            }
            if !choice["logprobs"].is_null() {
                self.logprobs.push(choice["logprobs"].clone());
            }
            if let Some(text) = choice["text"].as_str().filter(|t| !t.is_empty()) {
                if !self.text {
                    self.text = true;
                    self.queue.push_back(Event::TextStart(TextStart {
                        event: Default::default(),
                        block_id: "text".into(),
                    }));
                }
                self.queue.push_back(Event::TextDelta(TextDelta {
                    event: Default::default(),
                    block_id: "text".into(),
                    chunk: text.to_owned(),
                }));
            }
        }
        fn complete(&mut self) {
            self.start();
            if self.text {
                self.queue.push_back(Event::TextEnd(TextEnd {
                    event: Default::default(),
                    block_id: "text".into(),
                }));
            }
            let metadata = (!self.logprobs.is_empty()).then(|| {
                [(
                    "openai".to_owned(),
                    json!({"logprobs": std::mem::take(&mut self.logprobs)}),
                )]
                .into()
            });
            self.queue.push_back(Event::StreamEnd(StreamEnd {
                event: ModelEventMeta {
                    usage: self.usage.take(),
                    provider_metadata: metadata,
                    ..Default::default()
                },
                finish_reason: self.finish.take(),
                response_id: self.response_id.take(),
                response_model: self.response_model.take(),
            }));
            self.ended = true;
        }
    }
    let reader = Reader {
        input: http::sse_json(errors::API, provider.clone(), body),
        provider,
        queue: VecDeque::new(),
        started: false,
        text: false,
        ended: false,
        usage: None,
        finish: None,
        logprobs: Vec::new(),
        response_id: None,
        response_model: None,
    };
    stream::try_unfold(reader, |mut r| async move {
        loop {
            if let Some(event) = r.queue.pop_front() {
                return Ok(Some((event, r)));
            }
            if r.ended {
                return Ok(None);
            }
            match r.input.next().await {
                Some(Ok(http::SseJson::Done)) | None => r.complete(),
                Some(Ok(http::SseJson::Object(data))) => {
                    if data.get("error").is_some_and(|e| !e.is_null()) {
                        return Err(dialect::stream_error(true, &errors::API, &r.provider, &data));
                    }
                    r.chunk(&data);
                }
                Some(Err(error)) => return Err(error),
            }
        }
    })
    .boxed()
}
