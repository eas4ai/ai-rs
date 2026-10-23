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
