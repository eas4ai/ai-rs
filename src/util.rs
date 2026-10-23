//! Stream composition uses the standard futures primitives. Rust streams own
//! cancellation and resource cleanup through Drop.

/// Merge streams as their next elements become ready, with consumer backpressure.
/// Streams are polled in place. This does not spawn background tasks or restart
/// exhausted streams. Use an explicit stream factory when restart is needed.
pub use futures::stream::{SelectAll, select_all as merge};
