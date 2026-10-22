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

//! Stream composition uses the standard futures primitives. Rust streams own
//! cancellation and resource cleanup through Drop.

/// Merge streams as their next elements become ready, with consumer backpressure.
/// Streams are polled in place. This does not spawn background tasks or restart
/// exhausted streams. Use an explicit stream factory when restart is needed.
pub use futures::stream::{SelectAll, select_all as merge};
