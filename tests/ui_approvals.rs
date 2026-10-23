use serde_json::{Value, json};
use suprnova_ai::types::messages::{HookPart, HookStatus};
use suprnova_ai::types::{Message, MessageRole, Part};
use suprnova_ai::ui::ai_sdk::{UIMessage, approvals::*};

fn hook(kind: &str) -> HookPart {
    serde_json::from_value(
        json!({"hook_id":"custom_gate","hook_type":kind,"status":"pending","tool_call_id":"tc_42"}),
    )
    .unwrap()
}
#[test]
fn only_tool_approval_hooks_expose_tool_call_id() {
    assert_eq!(tool_call_id_for(&hook("ToolApproval")), Some("tc_42"));
    for kind in ["Confirmation", "SomethingElse"] {
        assert_eq!(tool_call_id_for(&hook(kind)), None);
    }
}
#[test]
fn extracts_static_and_dynamic_approval_responses() {
    for (kind, granted, reason) in [("tool-x", false, "nope"), ("dynamic-tool", true, "ok")] {
        let message:UIMessage=serde_json::from_value(json!({"role":"assistant","parts":[{"type":kind,"toolName":"web_search","toolCallId":"tc1","state":"approval-responded","approval":{"id":"approve_tc1","approved":granted,"reason":reason,"isAutomatic":true},"providerExecuted":true}]})).unwrap();
        let responses = extract_approvals(&[message]);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].hook_id, "approve_tc1");
        assert_eq!(responses[0].granted, granted);
        assert_eq!(responses[0].reason.as_deref(), Some(reason));
        assert_eq!(responses[0].tool_call_id, "tc1");
    }
}
#[test]
fn reconstructs_hook_states_and_metadata() {
    for (state, status, approved) in [
        ("approval-requested", HookStatus::Pending, Value::Null),
        ("approval-responded", HookStatus::Resolved, json!(true)),
        ("output-denied", HookStatus::Resolved, json!(false)),
    ] {
        let part = json!({"type":"tool-x","toolCallId":"tc1","state":state,"approval":{"id":"gate","approved":approved,"reason":"ok","isAutomatic":false},"providerExecuted":true,"callProviderMetadata":{"p":1}});
        let hook = hook_part_from_tool_part(&part).unwrap();
        assert_eq!(hook.status, status);
        assert_eq!(hook.tool_call_id.as_deref(), Some("tc1"));
        assert_eq!(hook.metadata["isAutomatic"], false);
        assert_eq!(hook.metadata["providerExecuted"], true);
        assert_eq!(hook.metadata["callProviderMetadata"], json!({"p":1}));
        let message = Message::new(MessageRole::Internal, vec![Part::Hook(hook)]);
        assert_eq!(
            is_resolved_approval_message(&message),
            status == HookStatus::Resolved
        );
    }
}
#[derive(Default)]
struct Registry(Vec<(String, Value)>);
impl ApprovalResolver for Registry {
    fn resolve_hook(&mut self, id: &str, resolution: Value) -> Result<(), suprnova_ai::Error> {
        self.0.push((id.into(), resolution));
        Ok(())
    }
}
#[test]
fn applies_responses_to_explicit_registry() {
    let responses = vec![ApprovalResponse {
        hook_id: "gate".into(),
        granted: false,
        reason: Some("nope".into()),
        tool_call_id: "tc1".into(),
    }];
    let mut registry = Registry::default();
    apply_approvals(&responses, &mut registry).unwrap();
    assert_eq!(
        registry.0,
        vec![("gate".into(), json!({"granted":false,"reason":"nope"}))]
    );
}
