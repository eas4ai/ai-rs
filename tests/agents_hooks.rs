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

use ai::{
    agents::{
        hooks::{HookError, HookRegistry},
        runtime::Runtime,
    },
    types::{Event, HookPart, HookStatus, messages::Metadata},
};
use futures::{StreamExt, poll};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct Confirmation {
    approved: bool,
    #[serde(default)]
    reason: String,
}

fn hook(event: Event) -> HookPart {
    match event {
        Event::Hook(event) => {
            assert_eq!(event.message.role, ai::types::MessageRole::Internal);
            assert_eq!(event.message.parts, vec![event.hook.clone().into()]);
            event.hook
        }
        _ => panic!("expected hook event"),
    }
}

#[tokio::test]
async fn live_resolution_emits_pending_and_resolved_with_metadata_and_attribution() {
    let registry = HookRegistry::new();
    let (runtime, mut events) = Runtime::new();
    let metadata = Metadata::from([("tool".to_owned(), json!("delete"))]);
    let mut future = Box::pin(registry.hook::<Confirmation>(
        "confirm",
        metadata.clone(),
        Some("tc1".into()),
        &runtime,
    ));
    assert!(poll!(&mut future).is_pending());
    let pending = hook(events.next().await.unwrap());
    assert_eq!(pending.status, HookStatus::Pending);
    assert_eq!(pending.hook_type, "Confirmation");
    assert_eq!(pending.metadata, metadata);
    assert_eq!(pending.tool_call_id.as_deref(), Some("tc1"));
    registry
        .resolve_hook("confirm", json!({"approved":true,"reason":"okay"}))
        .unwrap();
    assert_eq!(
        future.await.unwrap(),
        Confirmation {
            approved: true,
            reason: "okay".into()
        }
    );
    let resolved = hook(events.next().await.unwrap());
    assert_eq!(resolved.status, HookStatus::Resolved);
    assert_eq!(
        resolved.resolution,
        Some(json!({"approved":true,"reason":"okay"}))
    );
    assert_eq!(resolved.metadata, metadata);
    assert_eq!(resolved.tool_call_id.as_deref(), Some("tc1"));
}

#[tokio::test]
async fn pre_registration_is_shared_consumed_once_and_does_not_emit() {
    let registry = HookRegistry::default();
    let (runtime, mut events) = Runtime::new();
    registry
        .clone()
        .resolve_hook("confirm", json!({"approved":true}))
        .unwrap();
    assert!(
        registry
            .hook::<Confirmation>("confirm", Metadata::new(), None, &runtime)
            .await
            .unwrap()
            .approved
    );
    assert!(poll!(events.next()).is_pending());
    let mut second =
        Box::pin(registry.hook::<Confirmation>("confirm", Metadata::new(), None, &runtime));
    assert!(poll!(&mut second).is_pending());
    assert_eq!(
        hook(events.next().await.unwrap()).status,
        HookStatus::Pending
    );
}

#[tokio::test]
async fn live_cancel_emits_cancelled_and_preserves_reason() {
    let registry = HookRegistry::default();
    let (runtime, mut events) = Runtime::new();
    let mut future = Box::pin(registry.hook::<Confirmation>(
        "cancel",
        Metadata::new(),
        Some("tc1".into()),
        &runtime,
    ));
    assert!(poll!(&mut future).is_pending());
    events.next().await.unwrap();
    registry
        .cancel_hook("cancel", Some("denied".into()))
        .await
        .unwrap();
    assert!(
        matches!(future.await, Err(HookError::Cancelled { reason: Some(reason) }) if reason == "denied")
    );
    let cancelled = hook(events.next().await.unwrap());
    assert_eq!(cancelled.status, HookStatus::Cancelled);
    assert_eq!(cancelled.tool_call_id.as_deref(), Some("tc1"));
    assert!(matches!(
        registry.cancel_hook("cancel", None).await,
        Err(HookError::Missing { .. })
    ));
}

#[tokio::test]
async fn explicit_registries_isolate_same_labels() {
    let registry = HookRegistry::default();
    let other = HookRegistry::default();
    let (runtime, mut events) = Runtime::new();
    let mut future =
        Box::pin(registry.hook::<Confirmation>("same", Metadata::new(), None, &runtime));
    assert!(poll!(&mut future).is_pending());
    events.next().await.unwrap();
    assert!(matches!(
        other.cancel_hook("same", None).await,
        Err(HookError::Missing { .. })
    ));
    other
        .resolve_hook("same", json!({"approved":false}))
        .unwrap();
    assert!(poll!(&mut future).is_pending());
    registry
        .resolve_hook("same", json!({"approved":true}))
        .unwrap();
    assert!(future.await.unwrap().approved);
    assert!(
        !other
            .hook::<Confirmation>("same", Metadata::new(), None, &runtime)
            .await
            .unwrap()
            .approved
    );
}

#[tokio::test]
async fn resolution_is_validated_when_consumed_for_live_and_replay() {
    let registry = HookRegistry::default();
    let (runtime, mut events) = Runtime::new();
    registry
        .resolve_hook("invalid", json!({"approved":"not_bool"}))
        .unwrap();
    assert!(matches!(
        registry
            .hook::<Confirmation>("invalid", Metadata::new(), None, &runtime)
            .await,
        Err(HookError::Validation(_))
    ));
    let mut future =
        Box::pin(registry.hook::<Confirmation>("invalid", Metadata::new(), None, &runtime));
    assert!(poll!(&mut future).is_pending());
    events.next().await.unwrap();
    registry
        .resolve_hook("invalid", json!({"approved":"not_bool"}))
        .unwrap();
    assert!(matches!(future.await, Err(HookError::Validation(_))));
    assert_eq!(
        hook(events.next().await.unwrap()).status,
        HookStatus::Resolved
    );
}

#[tokio::test]
async fn deferral_reaches_live_and_pre_registered_awaiter() {
    let registry = HookRegistry::default();
    let (runtime, mut events) = Runtime::new();
    let mut future =
        Box::pin(registry.hook::<Confirmation>("defer", Metadata::new(), None, &runtime));
    assert!(poll!(&mut future).is_pending());
    let part = hook(events.next().await.unwrap());
    registry.defer_hook("defer", part.clone()).unwrap();
    assert!(matches!(future.await, Err(HookError::Deferred(found)) if *found == part));
    registry.defer_hook("defer", part.clone()).unwrap();
    assert!(
        matches!(registry.hook::<Confirmation>("defer", Metadata::new(), None, &runtime).await, Err(HookError::Deferred(found)) if *found == part)
    );
    assert!(poll!(events.next()).is_pending());
}

#[tokio::test]
async fn dropping_suspended_future_cleans_live_registry() {
    let registry = HookRegistry::default();
    let (runtime, mut events) = Runtime::new();
    let mut future =
        Box::pin(registry.hook::<Confirmation>("drop", Metadata::new(), None, &runtime));
    assert!(poll!(&mut future).is_pending());
    events.next().await.unwrap();
    drop(future);
    assert!(matches!(
        registry.cancel_hook("drop", None).await,
        Err(HookError::Missing { .. })
    ));
    registry
        .resolve_hook("drop", json!({"approved":true}))
        .unwrap();
    assert!(
        registry
            .hook::<Confirmation>("drop", Metadata::new(), None, &runtime)
            .await
            .unwrap()
            .approved
    );
}

#[tokio::test]
async fn duplicate_live_labels_fail_without_replacing_original() {
    let registry = HookRegistry::default();
    let (runtime, _events) = Runtime::new();
    let mut future =
        Box::pin(registry.hook::<Confirmation>("same", Metadata::new(), None, &runtime));
    assert!(poll!(&mut future).is_pending());
    assert!(matches!(
        registry
            .hook::<Confirmation>("same", Metadata::new(), None, &runtime)
            .await,
        Err(HookError::Duplicate { .. })
    ));
    registry
        .resolve_hook("same", json!({"approved":true}))
        .unwrap();
    assert!(future.await.unwrap().approved);
}

#[tokio::test]
async fn closed_runtime_removes_failed_suspension() {
    let registry = HookRegistry::default();
    let (runtime, events) = Runtime::new();
    drop(events);
    assert!(matches!(
        registry
            .hook::<Confirmation>("closed", Metadata::new(), None, &runtime)
            .await,
        Err(HookError::Runtime(_))
    ));
    assert!(matches!(
        registry.cancel_hook("closed", None).await,
        Err(HookError::Missing { .. })
    ));
}

#[tokio::test]
async fn scoped_middleware_wraps_hooks_and_can_short_circuit_without_suspension() {
    use ai::agents::middleware::{HookContext, HookNext, Middleware};
    use std::sync::{Arc, Mutex};
    struct Wrapper {
        name: &'static str,
        calls: Arc<Mutex<Vec<String>>>,
        short_circuit: bool,
    }
    #[async_trait::async_trait]
    impl Middleware for Wrapper {
        async fn wrap_hook(
            &self,
            mut call: HookContext,
            next: HookNext,
        ) -> Result<serde_json::Value, HookError> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("{} before", self.name));
            assert_eq!(call.payload_name, "Confirmation");
            assert!(call.payload.as_value().is_object());
            call.metadata.insert(self.name.into(), json!(true));
            let result = if self.short_circuit {
                Ok(json!({"approved":true}))
            } else {
                next(call).await
            };
            self.calls
                .lock()
                .unwrap()
                .push(format!("{} after", self.name));
            result
        }
    }
    let registry = HookRegistry::default();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let scoped = registry.with_middleware(vec![
        Arc::new(Wrapper {
            name: "outer",
            calls: calls.clone(),
            short_circuit: false,
        }),
        Arc::new(Wrapper {
            name: "inner",
            calls: calls.clone(),
            short_circuit: true,
        }),
    ]);
    let (runtime, mut events) = Runtime::new();
    assert!(
        scoped
            .hook::<Confirmation>("scoped", Metadata::new(), None, &runtime)
            .await
            .unwrap()
            .approved
    );
    assert_eq!(
        *calls.lock().unwrap(),
        ["outer before", "inner before", "inner after", "outer after"]
    );
    assert!(poll!(events.next()).is_pending());
    registry
        .resolve_hook("scoped", json!({"approved":false}))
        .unwrap();
    assert!(
        !registry
            .hook::<Confirmation>("scoped", Metadata::new(), None, &runtime)
            .await
            .unwrap()
            .approved
    );
}

#[tokio::test]
async fn cancel_returns_even_with_many_queued_side_events() {
    let registry = HookRegistry::default();
    let (runtime, mut events) = Runtime::new();
    let mut future =
        Box::pin(registry.hook::<Confirmation>("burst", Metadata::new(), None, &runtime));
    assert!(poll!(&mut future).is_pending());
    events.next().await.unwrap();
    for _ in 0..1000 {
        runtime
            .put_event(Event::text_delta("partial"))
            .await
            .unwrap();
    }
    registry.cancel_hook("burst", None).await.unwrap();
    assert!(matches!(
        future.await,
        Err(HookError::Cancelled { reason: None })
    ));
    for _ in 0..1000 {
        assert!(matches!(events.next().await, Some(Event::TextDelta(_))));
    }
    assert_eq!(
        hook(events.next().await.unwrap()).status,
        HookStatus::Cancelled
    );
}

#[tokio::test]
async fn middleware_rewrites_hook_label_and_metadata_before_suspending() {
    use ai::agents::middleware::{HookContext, HookNext, Middleware};
    use std::sync::Arc;
    struct Rewrite;
    #[async_trait::async_trait]
    impl Middleware for Rewrite {
        async fn wrap_hook(
            &self,
            mut call: HookContext,
            next: HookNext,
        ) -> Result<serde_json::Value, HookError> {
            call.label = "rewritten".into();
            call.payload_name = "Prompt".into();
            call.metadata.insert("source".into(), json!("wrapper"));
            next(call).await
        }
    }
    let registry = HookRegistry::default();
    let scoped = registry.with_middleware(vec![Arc::new(Rewrite)]);
    let (runtime, mut events) = Runtime::new();
    let mut future =
        Box::pin(scoped.hook::<Confirmation>("original", Metadata::new(), None, &runtime));
    assert!(poll!(&mut future).is_pending());
    let pending = hook(events.next().await.unwrap());
    assert_eq!(pending.hook_id, "rewritten");
    assert_eq!(pending.hook_type, "Prompt");
    assert_eq!(pending.metadata["source"], json!("wrapper"));
    registry
        .resolve_hook("rewritten", json!({"approved":true}))
        .unwrap();
    assert!(future.await.unwrap().approved);
}

#[tokio::test]
async fn cancelled_awaiter_cleanup_does_not_remove_new_same_label_hook() {
    let registry = HookRegistry::default();
    let (runtime, mut events) = Runtime::new();
    let mut old = Box::pin(registry.hook::<Confirmation>("reuse", Metadata::new(), None, &runtime));
    assert!(poll!(&mut old).is_pending());
    events.next().await.unwrap();
    registry.cancel_hook("reuse", None).await.unwrap();
    events.next().await.unwrap();
    let mut new = Box::pin(registry.hook::<Confirmation>("reuse", Metadata::new(), None, &runtime));
    assert!(poll!(&mut new).is_pending());
    events.next().await.unwrap();
    drop(old);
    registry
        .resolve_hook("reuse", json!({"approved":true}))
        .unwrap();
    assert!(new.await.unwrap().approved);
}

#[tokio::test]
async fn cancellation_event_is_queued_before_waiter_wakes() {
    use ai::agents::runtime::RuntimeEvents;
    use futures::{
        Future, Stream,
        task::{ArcWake, noop_waker_ref, waker_ref},
    };
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };
    use std::task::Context;

    struct ObserveWake {
        events: Mutex<RuntimeEvents>,
        cancelled: AtomicBool,
    }
    impl ArcWake for ObserveWake {
        fn wake_by_ref(this: &Arc<Self>) {
            let mut context = Context::from_waker(noop_waker_ref());
            let event =
                std::pin::Pin::new(&mut *this.events.lock().unwrap()).poll_next(&mut context);
            this.cancelled.store(matches!(event, std::task::Poll::Ready(Some(Event::Hook(event))) if event.hook.status == HookStatus::Cancelled), Ordering::SeqCst);
        }
    }
    let registry = HookRegistry::default();
    let (runtime, events) = Runtime::new();
    let observer = Arc::new(ObserveWake {
        events: Mutex::new(events),
        cancelled: AtomicBool::new(false),
    });
    let waker = waker_ref(&observer);
    let mut context = Context::from_waker(&waker);
    let mut future =
        Box::pin(registry.hook::<Confirmation>("ordered", Metadata::new(), None, &runtime));
    assert!(future.as_mut().poll(&mut context).is_pending());
    let pending = std::pin::Pin::new(&mut *observer.events.lock().unwrap())
        .poll_next(&mut Context::from_waker(noop_waker_ref()));
    assert!(
        matches!(pending, std::task::Poll::Ready(Some(Event::Hook(event))) if event.hook.status == HookStatus::Pending)
    );
    registry.cancel_hook("ordered", None).await.unwrap();
    assert!(
        observer.cancelled.load(Ordering::SeqCst),
        "waiter woke before its cancellation event was enqueued"
    );
    assert!(matches!(
        future.await,
        Err(HookError::Cancelled { reason: None })
    ));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_cancel_signal_precedes_execution_completion() {
    for _ in 0..1000 {
        let registry = HookRegistry::default();
        let producer = registry.clone();
        let (runtime, mut events) = Runtime::new();
        let completion = tokio::spawn(async move {
            assert!(
                matches!(producer.hook::<Confirmation>("concurrent", Metadata::new(), None, &runtime).await,
                Err(HookError::Cancelled { reason: Some(reason) }) if reason == "stop")
            );
            runtime
                .put_event(Event::text_delta("completed"))
                .await
                .unwrap();
            runtime.signal_done().await;
        });
        assert_eq!(
            hook(events.next().await.unwrap()).status,
            HookStatus::Pending
        );
        registry
            .cancel_hook("concurrent", Some("stop".into()))
            .await
            .unwrap();
        assert_eq!(
            hook(events.next().await.unwrap()).status,
            HookStatus::Cancelled
        );
        assert!(
            matches!(events.next().await, Some(Event::TextDelta(event)) if event.chunk == "completed")
        );
        assert!(events.next().await.is_none());
        completion.await.unwrap();
    }
}

#[tokio::test]
async fn failed_cancellation_event_still_unblocks_waiter() {
    let registry = HookRegistry::default();
    let (runtime, mut events) = Runtime::new();
    let mut future =
        Box::pin(registry.hook::<Confirmation>("closed_cancel", Metadata::new(), None, &runtime));
    assert!(poll!(&mut future).is_pending());
    events.next().await.unwrap();
    drop(events);
    assert!(matches!(
        registry
            .cancel_hook("closed_cancel", Some("stop".into()))
            .await,
        Err(HookError::Runtime(_))
    ));
    assert!(
        matches!(future.await, Err(HookError::Cancelled { reason: Some(reason) }) if reason == "stop")
    );
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ConstrainedResolution {
    #[schemars(range(min = 1))]
    count: i32,
    #[schemars(length(min = 3, max = 8))]
    name: String,
}

#[tokio::test]
async fn resolution_schema_constraints_apply_to_live_and_pre_registered_input() {
    let registry = HookRegistry::default();
    let (runtime, mut events) = Runtime::new();
    for invalid in [
        json!({"count":0,"name":"valid"}),
        json!({"count":1,"name":"x"}),
        json!({"count":1,"name":"too-long-name"}),
    ] {
        registry
            .resolve_hook("constrained", invalid.clone())
            .unwrap();
        assert!(
            matches!(
                registry
                    .hook::<ConstrainedResolution>("constrained", Metadata::new(), None, &runtime)
                    .await,
                Err(HookError::Schema(_))
            ),
            "pre-registered invalid value accepted: {invalid}"
        );
        let mut future = Box::pin(registry.hook::<ConstrainedResolution>(
            "constrained",
            Metadata::new(),
            None,
            &runtime,
        ));
        assert!(poll!(&mut future).is_pending());
        events.next().await.unwrap();
        registry
            .resolve_hook("constrained", invalid.clone())
            .unwrap();
        assert!(
            matches!(future.await, Err(HookError::Schema(_))),
            "live invalid value accepted: {invalid}"
        );
        assert_eq!(
            hook(events.next().await.unwrap()).status,
            HookStatus::Resolved
        );
    }
    registry
        .resolve_hook("valid", json!({"count":1,"name":"valid"}))
        .unwrap();
    let accepted = registry
        .hook::<ConstrainedResolution>("valid", Metadata::new(), None, &runtime)
        .await
        .unwrap();
    assert_eq!(accepted.count, 1);
    assert_eq!(accepted.name, "valid");
}

#[tokio::test]
async fn middleware_short_circuit_must_satisfy_original_payload_schema() {
    use ai::agents::middleware::{HookContext, HookNext, Middleware};
    use std::sync::Arc;
    struct InvalidResolution;
    #[async_trait::async_trait]
    impl Middleware for InvalidResolution {
        async fn wrap_hook(
            &self,
            mut call: HookContext,
            _: HookNext,
        ) -> Result<serde_json::Value, HookError> {
            call.payload = schemars::Schema::from(true);
            Ok(json!({"count":0,"name":"valid"}))
        }
    }
    let registry = HookRegistry::default().with_middleware(vec![Arc::new(InvalidResolution)]);
    let (runtime, mut events) = Runtime::new();
    assert!(matches!(
        registry
            .hook::<ConstrainedResolution>("wrapped", Metadata::new(), None, &runtime)
            .await,
        Err(HookError::Schema(_))
    ));
    assert!(poll!(events.next()).is_pending());
}

#[tokio::test]
async fn disappearing_waiter_reports_cancellation_delivery_failure() {
    use futures::{
        Future, Stream,
        task::{ArcWake, noop_waker_ref, waker_ref},
    };
    use std::{
        pin::Pin,
        sync::{Arc, Mutex},
        task::Context,
    };
    type Waiter =
        Arc<Mutex<Option<Pin<Box<dyn Future<Output = Result<Confirmation, HookError>> + Send>>>>>;
    struct DropWaiter(Waiter);
    impl ArcWake for DropWaiter {
        fn wake_by_ref(this: &Arc<Self>) {
            this.0.lock().unwrap().take();
        }
    }
    let registry = HookRegistry::default();
    let owner = registry.clone();
    let (runtime, mut events) = Runtime::new();
    let waiter: Waiter = Arc::new(Mutex::new(Some(Box::pin(async move {
        owner
            .hook::<Confirmation>("disappeared", Metadata::new(), None, &runtime)
            .await
    }))));
    assert!(
        waiter
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .as_mut()
            .poll(&mut Context::from_waker(noop_waker_ref()))
            .is_pending()
    );
    events.next().await.unwrap();
    let drop_waiter = Arc::new(DropWaiter(waiter.clone()));
    let waker = waker_ref(&drop_waiter);
    assert!(
        Pin::new(&mut events)
            .poll_next(&mut Context::from_waker(&waker))
            .is_pending()
    );
    assert!(
        matches!(registry.cancel_hook("disappeared", None).await, Err(HookError::Missing { label }) if label == "disappeared")
    );
    assert!(waiter.lock().unwrap().is_none());
    assert_eq!(
        hook(events.next().await.unwrap()).status,
        HookStatus::Cancelled
    );
}
