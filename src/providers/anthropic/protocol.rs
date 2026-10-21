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

//! Request conversion and the Anthropic Messages wire protocol.
pub use super::messages::{file_to_anthropic, messages_to_anthropic, tool_result_to_anthropic};
use super::{http, params, tools};
use crate::{
    Error, Model, Provider, ProviderProtocol, StreamOptions, providers::EventStream,
    transport::HttpRequest, types::Message,
};
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, Default)]
pub struct AnthropicMessagesProtocol;
fn build_request(
    model: &Model,
    messages: &[Message],
    options: &StreamOptions,
    provider: &dyn Provider,
) -> Result<HttpRequest, Error> {
    let mut params = params::translate(options.params.as_ref())?;
    if let Some(choice) = &options.tool_choice {
        params
            .body
            .insert("tool_choice".into(), params::tool_choice(choice)?);
    }
    let (system, messages) = messages_to_anthropic(messages)?;
    let (wire_tools, betas) = tools::to_wire(&options.tools)?;
    params.body.entry("max_tokens").or_insert(json!(8192));
    params.body.insert("model".into(), json!(model.id));
    params.body.insert("messages".into(), json!(messages));
    params.body.insert("stream".into(), json!(true));
    if let Some(system) = system {
        params.body.insert("system".into(), json!(system));
    }
    if !wire_tools.is_empty() {
        params.body.insert("tools".into(), json!(wire_tools));
    }
    params::add_betas(&mut params.headers, &betas);
    if let Some(schema) = &options.output_schema {
        let format =
            json!({"type":"json_schema","schema":super::schema::transform(schema.as_value(),0)?});
        let output = params.body.entry("output_config").or_insert(json!({}));
        output["format"] = format;
    }
    // The Python SDK merges extra_body after all named request parameters.
    params.body.extend(params.extra_body);
    http::request(
        provider.config(),
        "POST",
        &["v1", "messages"],
        Some(Value::Object(params.body)),
        params.headers,
        params.query,
    )
}
#[async_trait]
impl ProviderProtocol for AnthropicMessagesProtocol {
    fn protocol_class_id(&self) -> &str {
        "anthropic_messages"
    }
    fn stream(
        &self,
        model: &Model,
        messages: Vec<Message>,
        options: StreamOptions,
        provider: &dyn Provider,
    ) -> EventStream {
        let request = build_request(model, &messages, &options, provider);
        let transport = provider.transport_arc();
        let name = provider.name().to_owned();
        let model_id = model.id.clone();
        stream::once(async move {
            let request=request?;
            let transport=transport.ok_or_else(||Error::Configuration("Anthropic streams require an owned HttpTransport handle (Provider::transport_arc)".into()))?;
            let response=http::send(transport.as_ref(),request,&name,Some(&model_id)).await?;
            Ok::<_,Error>(super::stream::parse(response.body,name))
        }).try_flatten().boxed()
    }
}
