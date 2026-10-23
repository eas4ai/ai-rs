//! Events and headers understood by Vercel AI SDK UI clients.

use crate::types::messages::Metadata;
use serde::{Serialize, Serializer, ser::SerializeMap};
use serde_json::Value;

pub const UI_MESSAGE_STREAM_HEADERS: [(&str, &str); 5] = [
    ("Content-Type", "text/event-stream"),
    ("Cache-Control", "no-cache"),
    ("Connection", "keep-alive"),
    ("x-vercel-ai-ui-message-stream", "v1"),
    ("x-accel-buffering", "no"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Error,
    Other,
}

// Dataclasses put required fields before `type`, then optional fields. Keep
// that order, including omitted top-level nulls, in the Python wire format.
macro_rules! required_field {
    ($map:ident, $self:ident, $field:ident, $key:literal, Value) => {
        if !$self.$field.is_null() {
            $map.serialize_entry($key, &$self.$field)?;
        }
    };
    ($map:ident, $self:ident, $field:ident, $key:literal, $ty:ident) => {
        $map.serialize_entry($key, &$self.$field)?;
    };
}
macro_rules! optional_field {
    ($map:ident, $self:ident, $field:ident, $key:literal, Value) => {
        if let Some(value) = &$self.$field
            && !value.is_null()
        {
            $map.serialize_entry($key, value)?;
        }
    };
    ($map:ident, $self:ident, $field:ident, $key:literal, $ty:ident) => {
        if let Some(value) = &$self.$field {
            $map.serialize_entry($key, value)?;
        }
    };
}
macro_rules! ui_event {
    ($name:ident, $kind:literal, {$($field:ident:$ty:ident => $key:literal),* $(,)?}, {$($optional:ident:$opt_ty:ident => $opt_key:literal),* $(,)?}) => {
        #[derive(Clone, Debug, Default, PartialEq)]
        pub struct $name { $(pub $field: $ty,)* $(pub $optional: Option<$opt_ty>,)* }
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut map = serializer.serialize_map(None)?;
                $(required_field!(map, self, $field, $key, $ty);)*
                map.serialize_entry("type", $kind)?;
                $(optional_field!(map, self, $optional, $opt_key, $opt_ty);)*
                map.end()
            }
        }
    };
}

ui_event!(UIStartEvent, "start", {}, {message_id:String=>"messageId", message_metadata:Value=>"messageMetadata"});
ui_event!(UITextStartEvent, "text-start", {id:String=>"id"}, {provider_metadata:Metadata=>"providerMetadata"});
ui_event!(UITextDeltaEvent, "text-delta", {id:String=>"id", delta:String=>"delta"}, {provider_metadata:Metadata=>"providerMetadata"});
ui_event!(UITextEndEvent, "text-end", {id:String=>"id"}, {provider_metadata:Metadata=>"providerMetadata"});
ui_event!(UIReasoningStartEvent, "reasoning-start", {id:String=>"id"}, {provider_metadata:Metadata=>"providerMetadata"});
ui_event!(UIReasoningDeltaEvent, "reasoning-delta", {id:String=>"id", delta:String=>"delta"}, {provider_metadata:Metadata=>"providerMetadata"});
ui_event!(UIReasoningEndEvent, "reasoning-end", {id:String=>"id"}, {provider_metadata:Metadata=>"providerMetadata"});
ui_event!(UICustomEvent, "custom", {kind:String=>"kind"}, {provider_metadata:Metadata=>"providerMetadata"});
ui_event!(UISourceUrlEvent, "source-url", {source_id:String=>"sourceId", url:String=>"url"}, {title:String=>"title",provider_metadata:Metadata=>"providerMetadata"});
ui_event!(UISourceDocumentEvent, "source-document", {source_id:String=>"sourceId",media_type:String=>"mediaType",title:String=>"title"}, {filename:String=>"filename",provider_metadata:Metadata=>"providerMetadata"});
ui_event!(UIFileEvent, "file", {url:String=>"url",media_type:String=>"mediaType"}, {provider_metadata:Metadata=>"providerMetadata"});
ui_event!(UIReasoningFileEvent, "reasoning-file", {url:String=>"url",media_type:String=>"mediaType"}, {provider_metadata:Metadata=>"providerMetadata"});
ui_event!(UIToolInputStartEvent,"tool-input-start",{tool_call_id:String=>"toolCallId",tool_name:String=>"toolName"},{provider_executed:bool=>"providerExecuted",provider_metadata:Metadata=>"providerMetadata",tool_metadata:Metadata=>"toolMetadata",dynamic:bool=>"dynamic",title:String=>"title"});
ui_event!(UIToolInputDeltaEvent,"tool-input-delta",{tool_call_id:String=>"toolCallId",input_text_delta:String=>"inputTextDelta"},{});
ui_event!(UIToolInputAvailableEvent,"tool-input-available",{tool_call_id:String=>"toolCallId",tool_name:String=>"toolName",input:Value=>"input"},{provider_executed:bool=>"providerExecuted",provider_metadata:Metadata=>"providerMetadata",tool_metadata:Metadata=>"toolMetadata",dynamic:bool=>"dynamic",title:String=>"title"});
ui_event!(UIToolInputErrorEvent,"tool-input-error",{tool_call_id:String=>"toolCallId",tool_name:String=>"toolName",input:Value=>"input",error_text:String=>"errorText"},{provider_executed:bool=>"providerExecuted",provider_metadata:Metadata=>"providerMetadata",tool_metadata:Metadata=>"toolMetadata",dynamic:bool=>"dynamic",title:String=>"title"});
ui_event!(UIToolOutputAvailableEvent,"tool-output-available",{tool_call_id:String=>"toolCallId",output:Value=>"output"},{provider_executed:bool=>"providerExecuted",provider_metadata:Metadata=>"providerMetadata",tool_metadata:Metadata=>"toolMetadata",dynamic:bool=>"dynamic",preliminary:bool=>"preliminary"});
ui_event!(UIToolOutputErrorEvent,"tool-output-error",{tool_call_id:String=>"toolCallId",error_text:String=>"errorText"},{provider_executed:bool=>"providerExecuted",provider_metadata:Metadata=>"providerMetadata",tool_metadata:Metadata=>"toolMetadata",dynamic:bool=>"dynamic"});
ui_event!(UIToolOutputDeniedEvent,"tool-output-denied",{tool_call_id:String=>"toolCallId"},{});
ui_event!(UIToolApprovalRequestEvent,"tool-approval-request",{approval_id:String=>"approvalId",tool_call_id:String=>"toolCallId"},{is_automatic:bool=>"isAutomatic"});
ui_event!(UIToolApprovalResponseEvent,"tool-approval-response",{approval_id:String=>"approvalId",approved:bool=>"approved"},{reason:String=>"reason",provider_executed:bool=>"providerExecuted",provider_metadata:Metadata=>"providerMetadata"});
ui_event!(UIStartStepEvent, "start-step", {}, {});
ui_event!(UIFinishStepEvent, "finish-step", {}, {});
ui_event!(UIResetStepEvent, "reset-step", {}, {});
ui_event!(UIFinishEvent,"finish",{},{finish_reason:FinishReason=>"finishReason",message_metadata:Value=>"messageMetadata"});
ui_event!(UIAbortEvent,"abort",{},{reason:String=>"reason"});
ui_event!(UIMessageMetadataEvent,"message-metadata",{message_metadata:Value=>"messageMetadata"},{});
ui_event!(UIErrorEvent,"error",{error_text:String=>"errorText"},{});

#[derive(Clone, Debug, Default, PartialEq)]
pub struct UIDataEvent {
    pub data_type: String,
    pub data: Value,
    pub id: Option<String>,
    pub transient: Option<bool>,
}
impl Serialize for UIDataEvent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        if !self.data.is_null() {
            map.serialize_entry("data", &self.data)?;
        }
        if let Some(id) = &self.id {
            map.serialize_entry("id", id)?;
        }
        if let Some(transient) = self.transient {
            map.serialize_entry("transient", &transient)?;
        }
        map.serialize_entry("type", &format!("data-{}", self.data_type))?;
        map.end()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum UIMessageStreamEvent {
    Start(UIStartEvent),
    TextStart(UITextStartEvent),
    TextDelta(UITextDeltaEvent),
    TextEnd(UITextEndEvent),
    ReasoningStart(UIReasoningStartEvent),
    ReasoningDelta(UIReasoningDeltaEvent),
    ReasoningEnd(UIReasoningEndEvent),
    Custom(UICustomEvent),
    SourceUrl(UISourceUrlEvent),
    SourceDocument(UISourceDocumentEvent),
    File(UIFileEvent),
    ReasoningFile(UIReasoningFileEvent),
    Data(UIDataEvent),
    ToolInputStart(UIToolInputStartEvent),
    ToolInputDelta(UIToolInputDeltaEvent),
    ToolInputAvailable(UIToolInputAvailableEvent),
    ToolInputError(UIToolInputErrorEvent),
    ToolOutputAvailable(UIToolOutputAvailableEvent),
    ToolOutputError(UIToolOutputErrorEvent),
    ToolOutputDenied(UIToolOutputDeniedEvent),
    ToolApprovalRequest(UIToolApprovalRequestEvent),
    ToolApprovalResponse(UIToolApprovalResponseEvent),
    StartStep(UIStartStepEvent),
    FinishStep(UIFinishStepEvent),
    ResetStep(UIResetStepEvent),
    Finish(UIFinishEvent),
    Abort(UIAbortEvent),
    MessageMetadata(UIMessageMetadataEvent),
    Error(UIErrorEvent),
}
