//! Explicit hook state shared by a run and its consumer.

use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tokio::sync::oneshot;

use super::{
    middleware::{HookContext, Middleware, hook_chain},
    runtime::Runtime,
};
use crate::types::{
    HookPart, HookStatus,
    messages::{Metadata, generate_id},
    tools::{ToolValidationError, validate_schema_value},
};

/// A hook's external outcome or failure to create a suspension.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HookError {
    #[error("hook deferred")]
    Deferred(Box<HookPart>),
    #[error("hook cancelled")]
    Cancelled { reason: Option<String> },
    #[error("hook resolution validation failed")]
    Validation(#[source] serde_json::Error),
    #[error("hook resolution does not satisfy its schema")]
    Schema(#[source] ToolValidationError),
    #[error("hook event emission failed")]
    Runtime(#[from] crate::Error),
    #[error("no deferred hook with label {label:?}")]
    Missing { label: String },
    #[error("hook label is already live: {label:?}")]
    Duplicate { label: String },
    #[error("hook cancellation event and waiter delivery failed for label {label:?}")]
    CancellationDelivery {
        label: String,
        #[source]
        emission: Box<crate::Error>,
    },
}

type Outcome = Result<Value, HookError>;

struct LiveHook {
    token: Arc<()>,
    sender: Option<oneshot::Sender<Outcome>>,
    part: HookPart,
    runtime: Runtime,
}

#[derive(Default)]
struct State {
    live: BTreeMap<String, LiveHook>,
    pending: BTreeMap<String, Outcome>,
}

/// Caller-owned registry. Clones share live and pre-registered resolutions.
#[derive(Clone, Default)]
pub struct HookRegistry {
    state: Arc<Mutex<State>>,
    middleware: Vec<Arc<dyn Middleware>>,
}

impl HookRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Suspend until external input arrives, or consume one pre-registered outcome.
    pub async fn hook<T: DeserializeOwned + JsonSchema>(
        &self,
        label: impl Into<String>,
        metadata: Metadata,
        tool_call_id: Option<String>,
        runtime: &Runtime,
    ) -> Result<T, HookError> {
        let schema = schemars::schema_for!(T);
        let call = HookContext {
            label: label.into(),
            payload: schema.clone(),
            payload_name: T::schema_name().into_owned(),
            metadata,
            tool_call_id,
            registry: self.clone(),
            runtime: runtime.clone(),
        };
        let chain = hook_chain(
            &self.middleware,
            Arc::new(|call| Box::pin(wait_for_hook(call))),
        );
        let resolution = chain(call).await?;
        let payload = T::deserialize(&resolution).map_err(HookError::Validation)?;
        validate_schema_value(&schema, &resolution).map_err(HookError::Schema)?;
        Ok(payload)
    }

    /// Scope wrappers to this handle while retaining shared hook state.
    #[must_use]
    pub fn with_middleware(&self, middleware: Vec<Arc<dyn Middleware>>) -> Self {
        Self {
            state: self.state.clone(),
            middleware,
        }
    }

    /// Resolve a live hook, or save input for one future hook with this label.
    pub fn resolve_hook(&self, label: impl Into<String>, data: Value) -> Result<(), HookError> {
        self.resolve(label.into(), Ok(data))
    }

    /// Propagate a structured deferral to a live hook or the next replay.
    pub fn defer_hook(&self, label: impl Into<String>, part: HookPart) -> Result<(), HookError> {
        self.resolve(label.into(), Err(HookError::Deferred(Box::new(part))))
    }

    fn resolve(&self, label: String, outcome: Outcome) -> Result<(), HookError> {
        let mut state = self.state.lock().expect("hook registry mutex poisoned");
        if let Some(live) = state.live.get_mut(&label) {
            let sender = live.sender.take().ok_or_else(|| HookError::Duplicate {
                label: label.clone(),
            })?;
            sender
                .send(outcome)
                .map_err(|_| HookError::Missing { label })
        } else {
            state.pending.insert(label, outcome);
            Ok(())
        }
    }

    /// Cancel an existing suspension and emit its cancelled signal.
    pub async fn cancel_hook(
        &self,
        label: impl Into<String>,
        reason: Option<String>,
    ) -> Result<(), HookError> {
        let label = label.into();
        let live = self
            .state
            .lock()
            .expect("hook registry mutex poisoned")
            .live
            .remove(&label)
            .ok_or_else(|| HookError::Missing {
                label: label.clone(),
            })?;
        // Make the signal visible before the waiter can finish its run and close
        // the sink. Wake it even if emission fails, so cancellation never strands it.
        let emission = live
            .runtime
            .put_hook(HookPart {
                status: HookStatus::Cancelled,
                ..live.part
            })
            .await;
        let delivery = live.sender.map_or(Ok(()), |sender| {
            sender
                .send(Err(HookError::Cancelled { reason }))
                .map_err(|_| ())
        });
        match (emission, delivery) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) => Err(HookError::Runtime(error)),
            (Ok(()), Err(())) => Err(HookError::Missing { label }),
            (Err(error), Err(())) => Err(HookError::CancellationDelivery {
                label,
                emission: Box::new(error),
            }),
        }
    }
}

async fn wait_for_hook(call: HookContext) -> Outcome {
    let HookContext {
        label,
        payload_name,
        metadata,
        tool_call_id,
        registry,
        runtime,
        ..
    } = call;
    let part = HookPart {
        id: generate_id("part"),
        hook_id: label.clone(),
        hook_type: payload_name,
        status: HookStatus::Pending,
        metadata,
        resolution: None,
        tool_call_id,
    };
    let (receiver, guard) = {
        let mut state = registry.state.lock().expect("hook registry mutex poisoned");
        if let Some(outcome) = state.pending.remove(&label) {
            return outcome;
        }
        if state.live.contains_key(&label) {
            return Err(HookError::Duplicate { label });
        }
        let (sender, receiver) = oneshot::channel();
        let token = Arc::new(());
        state.live.insert(
            label.clone(),
            LiveHook {
                token: token.clone(),
                sender: Some(sender),
                part: part.clone(),
                runtime: runtime.clone(),
            },
        );
        (
            receiver,
            LiveGuard {
                registry: registry.clone(),
                label,
                token,
            },
        )
    };
    runtime.put_hook(part.clone()).await?;
    let resolution = receiver
        .await
        .unwrap_or(Err(HookError::Cancelled { reason: None }))?;
    drop(guard);
    runtime
        .put_hook(HookPart {
            status: HookStatus::Resolved,
            resolution: Some(resolution.clone()),
            ..part
        })
        .await?;
    Ok(resolution)
}

struct LiveGuard {
    registry: HookRegistry,
    label: String,
    token: Arc<()>,
}

impl Drop for LiveGuard {
    fn drop(&mut self) {
        let mut state = self
            .registry
            .state
            .lock()
            .expect("hook registry mutex poisoned");
        if state
            .live
            .get(&self.label)
            .is_some_and(|live| Arc::ptr_eq(&live.token, &self.token))
        {
            state.live.remove(&self.label);
        }
    }
}
