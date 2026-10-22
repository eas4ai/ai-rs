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

use std::ops::Add;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Normalized token usage from a single model call.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Usage {
    /// Input tokens, including cached and uncached input.
    #[serde(default)]
    pub input_tokens: u64,
    /// Generated output tokens.
    #[serde(default)]
    pub output_tokens: u64,
    /// Optional provider-reported reasoning token count.
    #[serde(default)]
    pub reasoning_tokens: Option<u64>,
    /// Optional provider-reported cache read token count.
    #[serde(default)]
    pub cache_read_tokens: Option<u64>,
    /// Optional provider-reported cache write token count.
    #[serde(default)]
    pub cache_write_tokens: Option<u64>,
    /// Unmodified provider-specific usage payload.
    #[serde(default)]
    pub raw: Option<Value>,
}

impl Usage {
    #[must_use]
    pub const fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

impl Add for Usage {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
            reasoning_tokens: add_optional(self.reasoning_tokens, other.reasoning_tokens),
            cache_read_tokens: add_optional(self.cache_read_tokens, other.cache_read_tokens),
            cache_write_tokens: add_optional(self.cache_write_tokens, other.cache_write_tokens),
            // Raw payloads describe one provider call and cannot be merged safely.
            raw: None,
        }
    }
}

impl Add<&Usage> for Usage {
    type Output = Self;

    fn add(self, other: &Usage) -> Self::Output {
        self + other.clone()
    }
}

fn add_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (None, None) => None,
        (left, right) => Some(left.unwrap_or(0) + right.unwrap_or(0)),
    }
}
