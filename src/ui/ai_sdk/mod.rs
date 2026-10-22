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

//! AI SDK v6 UI messages and server-sent events.
//!
//! This adapter preserves the Python 0.8.0 wire protocol, including the
//! `aiPython` identity metadata used when persisting collapsed assistant turns.

pub mod approvals;
pub mod id_utils;
pub mod inbound_messages;
pub mod outbound_messages;
pub mod outbound_stream;
pub mod tool_utils;
pub mod ui_events;
pub mod ui_messages;

pub use approvals::{ApprovalResolver, ApprovalResponse, apply_approvals, extract_approvals};
pub use inbound_messages::{ToolResultValidator, to_messages};
pub use outbound_messages::to_ui_messages;
pub use outbound_stream::{to_sse, to_stream};
pub use ui_events::UI_MESSAGE_STREAM_HEADERS;
pub use ui_messages::UIMessage;
