//! Tool approvals extracted from AI SDK UI messages.

use super::ui_messages::UIMessage;
use crate::{
    Error,
    types::messages::{HookPart, HookStatus, Message, MessageRole, Metadata, Part, generate_id},
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const TOOL_APPROVAL_HOOK_TYPE: &str = "ToolApproval";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovalResponse {
    pub hook_id: String,
    pub granted: bool,
    pub reason: Option<String>,
    pub tool_call_id: String,
}

#[must_use]
pub fn tool_call_id_for(part: &HookPart) -> Option<&str> {
    (part.hook_type == TOOL_APPROVAL_HOOK_TYPE)
        .then_some(part.tool_call_id.as_deref())
        .flatten()
}

#[must_use]
pub fn hook_part_from_tool_part(part: &Value) -> Option<HookPart> {
    let approval = part.get("approval")?.as_object()?;
    let state = part.get("state")?.as_str()?;
    let (status, resolution) = match state {
        "approval-requested" => (HookStatus::Pending, None),
        "approval-responded" => (
            HookStatus::Resolved,
            Some(
                json!({"granted":approval.get("approved")?.as_bool()?,"reason":approval.get("reason").unwrap_or(&Value::Null)}),
            ),
        ),
        "output-denied" => (
            HookStatus::Resolved,
            Some(json!({"granted":false,"reason":approval.get("reason").unwrap_or(&Value::Null)})),
        ),
        _ => return None,
    };
    let mut metadata = Metadata::new();
    for (source, key) in [
        (approval.get("isAutomatic"), "isAutomatic"),
        (part.get("providerExecuted"), "providerExecuted"),
        (part.get("callProviderMetadata"), "callProviderMetadata"),
    ] {
        if let Some(value) = source.filter(|v| !v.is_null()) {
            metadata.insert(key.to_owned(), value.clone());
        }
    }
    Some(HookPart {
        id: generate_id("part"),
        hook_id: approval.get("id")?.as_str()?.to_owned(),
        hook_type: TOOL_APPROVAL_HOOK_TYPE.to_owned(),
        status,
        metadata,
        resolution,
        tool_call_id: Some(part.get("toolCallId")?.as_str()?.to_owned()),
    })
}

#[must_use]
pub fn extract_approvals(messages: &[UIMessage]) -> Vec<ApprovalResponse> {
    messages
        .iter()
        .flat_map(|message| &message.parts)
        .filter_map(|part| {
            let kind = part.get("type")?.as_str()?;
            if !(kind == "dynamic-tool" || kind.starts_with("tool-") && kind != "tool-invocation")
                || part.get("state")?.as_str()? != "approval-responded"
            {
                return None;
            }
            let approval = part.get("approval")?;
            Some(ApprovalResponse {
                hook_id: approval.get("id")?.as_str()?.to_owned(),
                granted: approval.get("approved")?.as_bool()?,
                reason: approval
                    .get("reason")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                tool_call_id: part.get("toolCallId")?.as_str()?.to_owned(),
            })
        })
        .collect()
}

/// An explicit run-scoped registry, supplied by the application or agent runtime.
pub trait ApprovalResolver {
    fn resolve_hook(&mut self, hook_id: &str, resolution: Value) -> Result<(), Error>;
}

pub fn apply_approvals(
    approvals: &[ApprovalResponse],
    registry: &mut dyn ApprovalResolver,
) -> Result<(), Error> {
    for approval in approvals {
        registry.resolve_hook(
            &approval.hook_id,
            json!({"granted":approval.granted,"reason":approval.reason}),
        )?;
    }
    Ok(())
}

#[must_use]
pub fn is_resolved_approval_message(message: &Message) -> bool {
    message.role == MessageRole::Internal
        && matches!(message.parts.as_slice(),[Part::Hook(part)] if part.hook_type==TOOL_APPROVAL_HOOK_TYPE && part.status==HookStatus::Resolved)
}
