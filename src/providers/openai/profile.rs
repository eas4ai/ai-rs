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

//! What differs between servers that speak the Chat Completions protocol.
use super::errors;
use crate::providers::http::ApiProfile;

/// How a server's Chat Completions API differs from OpenAI's. The host
/// profiles follow the AI SDK for TypeScript 7.0.148 host packages.
#[derive(Debug)]
pub(crate) struct ChatProfile {
    /// The protocol's class id in snapshots.
    pub protocol_class_id: &'static str,
    pub api: ApiProfile,
    /// Ask for usage in the stream with `stream_options.include_usage`,
    /// which some servers reject.
    pub include_usage: bool,
    /// The request field of the output token limit.
    pub max_tokens_field: &'static str,
    /// The assistant message field that carries reasoning history.
    pub reasoning_field: &'static str,
    /// Reasoning efforts the server takes under another name.
    pub effort_aliases: &'static [(&'static str, &'static str)],
    /// The models whose output schema goes out as a JSON schema; the other
    /// models get JSON mode, which does not enforce the schema.
    pub json_schema: JsonSchemaModels,
    /// Read the `openai` provider parameters (`OpenAIParams`).
    pub openai_params: bool,
    /// Take only automatic tool choice; tool choice none sends no tools.
    pub auto_tool_choice_only: bool,
    /// Reject frequency and presence penalties and seeds.
    pub rejects_penalties_and_seed: bool,
    /// Finish reasons the server adds, with their normalized names.
    pub finish_reasons: &'static [(&'static str, &'static str)],
    /// The server may leave reasoning tokens out of `completion_tokens`.
    pub reasoning_outside_completion: bool,
    /// Under an output schema, drop tool calls that follow text.
    pub json_drops_tool_calls_after_text: bool,
}

#[derive(Debug)]
pub(crate) enum JsonSchemaModels {
    All,
    Only(&'static [&'static str]),
    None,
}

impl ChatProfile {
    pub(crate) fn sends_json_schema(&self, model: &str) -> bool {
        match self.json_schema {
            JsonSchemaModels::All => true,
            JsonSchemaModels::Only(models) => models.contains(&model),
            JsonSchemaModels::None => false,
        }
    }

    pub(crate) fn effort<'a>(&self, effort: &'a str) -> &'a str {
        self.effort_aliases
            .iter()
            .find(|(from, _)| *from == effort)
            .map_or(effort, |(_, to)| to)
    }

    /// The normalized name of a finish reason this server adds.
    pub(crate) fn finish_reason(&self, raw: &str) -> Option<&'static str> {
        self.finish_reasons
            .iter()
            .find(|(from, _)| *from == raw)
            .map(|(_, to)| *to)
    }
}

/// OpenAI's Chat Completions, also used by OpenAI-compatible servers without
/// a host profile.
pub(crate) static OPENAI: ChatProfile = ChatProfile {
    protocol_class_id: "openai_chat_completions",
    api: errors::API,
    include_usage: true,
    max_tokens_field: "max_completion_tokens",
    reasoning_field: "reasoning",
    effort_aliases: &[],
    json_schema: JsonSchemaModels::All,
    openai_params: true,
    auto_tool_choice_only: false,
    rejects_penalties_and_seed: false,
    finish_reasons: &[],
    reasoning_outside_completion: false,
    json_drops_tool_calls_after_text: false,
};
