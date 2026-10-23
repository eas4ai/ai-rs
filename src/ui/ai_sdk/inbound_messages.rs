//! Convert AI SDK v6 UI messages into core history and approval responses.

use super::{
    approvals::{self, ApprovalResponse},
    id_utils,
    tool_utils::normalize_tool_args,
    ui_messages::UIMessage,
};
use crate::{
    Error,
    types::messages::{
        BuiltinToolCallPart, BuiltinToolReturnPart, ContentOutput, FileData, FilePart, HookPart,
        Message, MessageBundle, MessageRole, Metadata, Part, ReasoningPart, TextPart, ToolCallPart,
        ToolResultPart, ToolResultValue, generate_id,
    },
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Validate and reconstruct a named tool's output using an explicit tool registry.
/// Return a core special result when the tool's declared output requires one.
pub trait ToolResultValidator {
    fn validate_result(&self, tool_name: &str, value: Value) -> Result<ToolResultValue, Error>;
}

/// Heal stale persisted tool states without mutating the caller's messages.
#[must_use]
pub fn normalize_ui_messages(messages: &[UIMessage]) -> Vec<UIMessage> {
    let mut normalized = messages.to_vec();
    for part in normalized.iter_mut().flat_map(|message| &mut message.parts) {
        if let Some(state) = normalized_tool_state(part) {
            part["state"] = Value::String(state.to_owned());
        }
    }
    normalized
}

fn normalized_tool_state(part: &Value) -> Option<&'static str> {
    let kind = part.get("type")?.as_str()?;
    if !(kind.starts_with("tool-") || kind == "dynamic-tool") {
        return None;
    }
    let state = part.get("state").and_then(Value::as_str).unwrap_or("");
    if part.get("output").is_none_or(Value::is_null) {
        return (state == "call").then_some("input-available");
    }
    if state == "output-error" || part.get("errorText").is_some_and(|value| !value.is_null()) {
        return Some("output-error");
    }
    let denied = part
        .get("approval")
        .and_then(|approval| approval.get("approved"))
        .and_then(Value::as_bool)
        == Some(false);
    Some(if state == "output-denied" || denied {
        "output-denied"
    } else {
        "output-available"
    })
}

/// Parse UI history and approvals. Tool result validation is optional and scoped.
/// Both deserialized messages and directly constructed messages are validated here.
pub fn to_messages(
    ui_messages: &[UIMessage],
    tools: Option<&dyn ToolResultValidator>,
) -> Result<(Vec<Message>, Vec<ApprovalResponse>), Error> {
    let validated = ui_messages
        .iter()
        .map(|message| serde_json::from_value(serde_json::to_value(message)?).map_err(Error::from))
        .collect::<Result<Vec<UIMessage>, _>>()?;
    let normalized = normalize_ui_messages(&validated);
    let responses = approvals::extract_approvals(&normalized);
    let mut messages = parse(&normalized, tools)?;
    messages.retain(|message| !approvals::is_resolved_approval_message(message));
    patch_deferred_hook_results(&mut messages, &responses);
    Ok((messages, responses))
}

fn string<'a>(part: &'a Value, key: &str) -> Result<&'a str, Error> {
    part.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Validation(format!("UI part requires string {key}")))
}

fn metadata(part: &Value, key: &str) -> Result<Option<Metadata>, Error> {
    match part.get(key).filter(|value| !value.is_null()) {
        Some(value) => Ok(Some(serde_json::from_value(value.clone())?)),
        None => Ok(None),
    }
}

fn normalize_tool_result(output: Value) -> Value {
    if output.is_null() || output.is_object() {
        output
    } else {
        json!({"value":output})
    }
}

fn tool_result_output(part: &Value) -> Value {
    let output = part.get("output").cloned().unwrap_or(Value::Null);
    match part.get("state").and_then(Value::as_str) {
        Some("output-error") => {
            let mut normalized = normalize_tool_result(output);
            if let Some(error) = part
                .get("errorText")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
            {
                if normalized.is_null() {
                    normalized = json!({"error":error});
                } else if let Some(object) = normalized.as_object_mut() {
                    object
                        .entry("error")
                        .or_insert_with(|| Value::String(error.to_owned()));
                }
            }
            normalized
        }
        Some("output-denied") => {
            let reason = part
                .get("approval")
                .and_then(|approval| approval.get("reason"))
                .and_then(Value::as_str)
                .filter(|reason| !reason.is_empty())
                .unwrap_or("Tool call execution denied.");
            json!({"type": "error-text", "value": reason})
        }
        _ => output,
    }
}

fn build_result(
    tool_call_id: &str,
    tool_name: &str,
    output: Value,
    is_error: bool,
    kind_hint: Option<&str>,
    tools: Option<&dyn ToolResultValidator>,
) -> Result<ToolResultPart, Error> {
    if is_error {
        return Ok(ToolResultPart::error(tool_call_id, tool_name, output));
    }
    let result: ToolResultValue = match kind_hint {
        Some("content") => ContentOutput::new(serde_json::from_value(output)?).into(),
        Some("messages") => {
            let raw = match output {
                Value::Array(values) => values,
                value => vec![value],
            };
            let messages = raw
                .into_iter()
                .map(serde_json::from_value)
                .collect::<Result<Vec<UIMessage>, _>>()?;
            MessageBundle::new(parse(&messages, tools)?).into()
        }
        _ => {
            let normalized = normalize_tool_result(output);
            match tools {
                Some(tools) => tools.validate_result(tool_name, normalized)?,
                None => normalized.into(),
            }
        }
    };
    Ok(ToolResultPart::new(tool_call_id, tool_name, result))
}

struct ParseContext<'a> {
    result_kinds: BTreeMap<String, String>,
    tools: Option<&'a dyn ToolResultValidator>,
}

#[derive(Default)]
struct ParsedParts {
    content: Vec<Part>,
    results: Vec<ToolResultPart>,
    hooks: Vec<HookPart>,
}

impl ParsedParts {
    fn append(&mut self, part: &Value, context: &ParseContext<'_>) -> Result<(), Error> {
        match string(part, "type")? {
            "text" | "reasoning" => self.append_text(part)?,
            "file" => self.content.push(Part::File(FilePart {
                id: generate_id("part"),
                data: FileData::Text(string(part, "url")?.to_owned()),
                media_type: string(part, "mediaType")?.to_owned(),
                filename: part
                    .get("filename")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                provider_metadata: metadata(part, "providerMetadata")?,
            })),
            kind if kind == "dynamic-tool" || kind.starts_with("tool-") => {
                self.append_tool(part, context)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn append_text(&mut self, part: &Value) -> Result<(), Error> {
        let text = string(part, "text")?;
        if text.is_empty() {
            return Ok(());
        }
        let provider_metadata = metadata(part, "providerMetadata")?;
        let content = if part["type"] == "text" {
            Part::Text(TextPart {
                provider_metadata,
                ..TextPart::new(text)
            })
        } else {
            Part::Reasoning(ReasoningPart {
                provider_metadata,
                ..ReasoningPart::new(text)
            })
        };
        self.content.push(content);
        Ok(())
    }

    fn append_tool(&mut self, part: &Value, context: &ParseContext<'_>) -> Result<(), Error> {
        let tool = ParsedTool::from_ui(part)?;
        let call = &tool.call;
        if tool.provider_executed {
            self.content
                .push(Part::BuiltinToolCall(BuiltinToolCallPart {
                    id: call.id.clone(),
                    tool_call_id: call.tool_call_id.clone(),
                    tool_name: call.tool_name.clone(),
                    tool_args: call.tool_args.clone(),
                    provider_metadata: call.provider_metadata.clone(),
                }));
        } else {
            self.content.push(Part::ToolCall(call.clone()));
        }
        if let Some(hook) = tool.hook {
            self.hooks.push(hook);
        }
        if !tool.completed {
            return Ok(());
        }
        if tool.provider_executed {
            self.content
                .push(Part::BuiltinToolReturn(BuiltinToolReturnPart {
                    id: generate_id("part"),
                    tool_call_id: call.tool_call_id.clone(),
                    tool_name: call.tool_name.clone(),
                    result: tool.output,
                    is_error: tool.is_error,
                    provider_metadata: tool
                        .result_metadata
                        .filter(|value| !value.is_empty())
                        .or_else(|| call.provider_metadata.clone()),
                }));
        } else {
            let kind = context
                .result_kinds
                .get(&call.tool_call_id)
                .map(String::as_str);
            let mut result = build_result(
                &call.tool_call_id,
                &call.tool_name,
                tool.output,
                tool.is_error,
                kind,
                context.tools,
            )?;
            result.provider_metadata = tool.result_metadata;
            self.results.push(result);
        }
        Ok(())
    }

    fn into_messages(self, ui: &UIMessage) -> Result<Vec<Message>, Error> {
        match ui.role.as_str() {
            "assistant" => {
                let mut parsed = split_assistant_parts(self.content, self.results, &ui.id);
                for hook in self.hooks {
                    parsed.push(turn_message(
                        MessageRole::Internal,
                        vec![Part::Hook(hook)],
                        &ui.id,
                    ));
                }
                Ok(parsed)
            }
            "user" | "system" => {
                if self.content.is_empty() {
                    return Err(Error::Validation(format!(
                        "Message {:?} has role {:?} but no content. User and system messages require non-empty content.",
                        ui.id, ui.role
                    )));
                }
                let role = if ui.role == "user" {
                    MessageRole::User
                } else {
                    MessageRole::System
                };
                Ok(vec![Message::with_id(&ui.id, role, self.content)])
            }
            _ => Err(Error::Validation(
                "UI role must be user, assistant, or system".into(),
            )),
        }
    }
}

struct ParsedTool {
    call: ToolCallPart,
    provider_executed: bool,
    completed: bool,
    is_error: bool,
    output: Value,
    result_metadata: Option<Metadata>,
    hook: Option<HookPart>,
}

impl ParsedTool {
    fn from_ui(part: &Value) -> Result<Self, Error> {
        let kind = string(part, "type")?;
        let legacy = kind == "tool-invocation";
        let id = string(
            part,
            if legacy {
                "toolInvocationId"
            } else {
                "toolCallId"
            },
        )?;
        let name = if legacy || kind == "dynamic-tool" {
            string(part, "toolName")?
        } else {
            kind.strip_prefix("tool-").unwrap_or(kind)
        };
        let state = string(part, "state")?;
        let input = if legacy {
            &part["args"]
        } else if state == "output-error" && part["input"].is_null() {
            &part["rawInput"]
        } else {
            &part["input"]
        };
        let mut call = ToolCallPart::new(id, name, normalize_tool_args(input));
        if !legacy {
            call.provider_metadata = metadata(part, "callProviderMetadata")?;
        }
        Ok(Self {
            call,
            provider_executed: part.get("providerExecuted").and_then(Value::as_bool) == Some(true),
            completed: matches!(state, "output-available" | "output-error" | "output-denied"),
            is_error: matches!(state, "output-error" | "output-denied"),
            output: if legacy {
                part["result"].clone()
            } else {
                tool_result_output(part)
            },
            result_metadata: if legacy {
                None
            } else {
                metadata(part, "resultProviderMetadata")?
            },
            hook: if legacy {
                None
            } else {
                approvals::hook_part_from_tool_part(part)
            },
        })
    }
}

fn parse(
    ui_messages: &[UIMessage],
    tools: Option<&dyn ToolResultValidator>,
) -> Result<Vec<Message>, Error> {
    let mut result = Vec::new();
    for ui in ui_messages {
        let sources = id_utils::source_messages_from(&ui.metadata);
        let context = ParseContext {
            result_kinds: id_utils::tool_result_kinds_from(&ui.metadata),
            tools,
        };
        let mut parts = ParsedParts::default();
        for part in &ui.parts {
            parts.append(part, &context)?;
        }
        result.extend(id_utils::restore_source_ids(
            parts.into_messages(ui)?,
            &sources,
        ));
    }
    Ok(result)
}

fn turn_message(role: MessageRole, parts: Vec<Part>, turn_id: &str) -> Message {
    Message {
        turn_id: Some(turn_id.to_owned()),
        ..Message::new(role, parts)
    }
}

fn split_assistant_parts(
    parts: Vec<Part>,
    results: Vec<ToolResultPart>,
    turn_id: &str,
) -> Vec<Message> {
    let results: BTreeMap<_, _> = results
        .into_iter()
        .map(|result| (result.tool_call_id.clone(), result))
        .collect();
    let mut messages = Vec::new();
    let mut current = Vec::new();
    let mut current_results = Vec::new();
    let mut seen_tool = false;
    for part in parts {
        if seen_tool && !current_results.is_empty() && !matches!(part, Part::ToolCall(_)) {
            messages.push(turn_message(
                MessageRole::Assistant,
                std::mem::take(&mut current),
                turn_id,
            ));
            messages.push(turn_message(
                MessageRole::Tool,
                std::mem::take(&mut current_results),
                turn_id,
            ));
            seen_tool = false;
        }
        if let Part::ToolCall(call) = &part {
            seen_tool = true;
            if let Some(result) = results.get(&call.tool_call_id) {
                current_results.push(Part::ToolResult(result.clone()));
            }
        }
        current.push(part);
    }
    if !current.is_empty() {
        messages.push(turn_message(MessageRole::Assistant, current, turn_id));
    }
    if !current_results.is_empty() {
        messages.push(turn_message(MessageRole::Tool, current_results, turn_id));
    }
    messages
}

fn patch_deferred_hook_results(messages: &mut [Message], approvals: &[ApprovalResponse]) {
    let Some((tool, previous)) = messages.split_last_mut() else {
        return;
    };
    let Some(assistant) = previous.last() else {
        return;
    };
    if tool.role != MessageRole::Tool || assistant.role != MessageRole::Assistant {
        return;
    }
    let hooks: BTreeMap<_, _> = approvals
        .iter()
        .map(|approval| (&approval.tool_call_id, approval))
        .collect();
    for call in assistant.tool_calls() {
        if tool
            .tool_results()
            .iter()
            .any(|result| result.tool_call_id == call.tool_call_id)
        {
            continue;
        }
        if let Some(hook) = hooks.get(&call.tool_call_id) {
            let mut result = ToolResultPart::error(
                &call.tool_call_id,
                &call.tool_name,
                json!(format!("Deferred on hook '{}'", hook.hook_id)),
            );
            result.is_hook_deferred = true;
            tool.parts.push(Part::ToolResult(result));
        }
    }
}
