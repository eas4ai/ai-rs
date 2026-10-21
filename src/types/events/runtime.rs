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

//! Cloneable runtime handles excluded from event serialization.

use super::Aggregator;
use serde_json::Value;
use std::{error::Error, fmt, sync::Arc};

/// Retain the original tool error and its source chain across event clones.
#[derive(Clone)]
pub struct RuntimeError(Arc<dyn Error + Send + Sync>);

impl RuntimeError {
    pub fn new(error: impl Error + Send + Sync + 'static) -> Self {
        Self(Arc::new(error))
    }
    pub fn from_arc(error: Arc<dyn Error + Send + Sync>) -> Self {
        Self(error)
    }
    pub fn as_error(&self) -> &(dyn Error + Send + Sync + 'static) {
        self.0.as_ref()
    }
}
impl fmt::Debug for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RuntimeError(..)")
    }
}
impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.0.as_ref(), f)
    }
}
impl Error for RuntimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.0.as_ref())
    }
}
impl PartialEq for RuntimeError {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for RuntimeError {}

/// Object-safe reducer interface for JSON values carried by partial-result events.
pub trait ValueAggregator: Send {
    fn feed(&mut self, value: Value);
    fn snapshot(&self) -> Value;
    /// Preserve the runtime type of special tool snapshots.
    ///
    /// JSON reducers keep their existing behavior. Reducers that produce a
    /// message bundle can override this method so adapters do not infer special
    /// results from an ordinary JSON object's shape.
    fn snapshot_result(&self) -> crate::types::messages::ToolResultValue {
        self.snapshot().into()
    }
    fn get_model_input(&self) -> Value;
}
impl<T: Aggregator<Value, Snapshot = Value, ModelInput = Value> + Send> ValueAggregator for T {
    fn feed(&mut self, value: Value) {
        Aggregator::feed(self, value);
    }
    fn snapshot(&self) -> Value {
        Aggregator::snapshot(self)
    }
    fn get_model_input(&self) -> Value {
        Aggregator::get_model_input(self)
    }
}

type Factory = dyn Fn() -> Box<dyn ValueAggregator> + Send + Sync;

/// A reusable factory for independent streaming-tool reducer instances.
#[derive(Clone)]
pub struct AggregatorFactory(Arc<Factory>);
impl AggregatorFactory {
    pub fn new<F, A>(factory: F) -> Self
    where
        F: Fn() -> A + Send + Sync + 'static,
        A: ValueAggregator + 'static,
    {
        Self::from_boxed(move || Box::new(factory()))
    }
    pub fn from_boxed(
        factory: impl Fn() -> Box<dyn ValueAggregator> + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(factory))
    }
    pub fn create(&self) -> Box<dyn ValueAggregator> {
        (self.0)()
    }
}
impl fmt::Debug for AggregatorFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AggregatorFactory(..)")
    }
}
impl PartialEq for AggregatorFactory {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for AggregatorFactory {}
