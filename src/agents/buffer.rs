//! Consumer buffering with owned task cancellation and failure propagation.
use crate::{Error, providers::EventStream, types::Event};
use futures::{Future, Stream, StreamExt};
use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context as TaskContext, Poll},
};
enum BufferReceiver {
    Unbounded(tokio::sync::mpsc::UnboundedReceiver<Result<Event, Error>>),
    Bounded(tokio::sync::mpsc::Receiver<Result<Event, Error>>),
}
/// The loop task of a buffered run, once it has started.
type TaskSlot = Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>;
pub(super) struct BufferedStream {
    source: Option<EventStream>,
    capacity: Option<usize>,
    receiver: Option<BufferReceiver>,
    task: TaskSlot,
}
impl BufferedStream {
    pub(super) fn new(source: EventStream, capacity: Option<usize>) -> Self {
        Self {
            source: Some(source),
            capacity,
            receiver: None,
            task: TaskSlot::default(),
        }
    }
    /// A handle that ends the loop task and waits for it.
    pub(super) fn stopper(&self) -> BufferStop {
        BufferStop(self.task.clone())
    }
    fn set_task(&self, task: tokio::task::JoinHandle<()>) {
        *self.task.lock().expect("agent buffer mutex poisoned") = Some(task);
    }
    fn take_task(&self) -> Option<tokio::task::JoinHandle<()>> {
        self.task
            .lock()
            .expect("agent buffer mutex poisoned")
            .take()
    }
}

/// Ends a buffered run's loop task.
pub(super) struct BufferStop(TaskSlot);
impl BufferStop {
    /// Abort the loop task and wait until it is gone, with the model call and
    /// the tool futures it held.
    pub(super) async fn stop(&self) {
        let task = self.0.lock().expect("agent buffer mutex poisoned").take();
        if let Some(task) = task {
            task.abort();
            // The task ends cancelled, or with the panic that ended it first;
            // neither is delivered after a stop.
            let _ = task.await;
        }
    }
}
impl Stream for BufferedStream {
    type Item = Result<Event, Error>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        if let Some(mut source) = self.source.take() {
            let handle = match tokio::runtime::Handle::try_current() {
                Ok(handle) => handle,
                Err(_) => {
                    return Poll::Ready(Some(Err(Error::Configuration(
                        "buffered agent runs require a Tokio runtime".into(),
                    ))));
                }
            };
            if let Some(capacity) = self.capacity {
                let (sender, receiver) = tokio::sync::mpsc::channel(capacity.max(1));
                self.receiver = Some(BufferReceiver::Bounded(receiver));
                self.set_task(handle.spawn(async move {
                    while let Some(event) = source.next().await {
                        if sender.send(event).await.is_err() {
                            break;
                        }
                    }
                }));
            } else {
                let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
                self.receiver = Some(BufferReceiver::Unbounded(receiver));
                self.set_task(handle.spawn(async move {
                    while let Some(event) = source.next().await {
                        if sender.send(event).is_err() {
                            break;
                        }
                    }
                }));
            }
        }
        let event = match self.receiver.as_mut() {
            Some(BufferReceiver::Unbounded(receiver)) => receiver.poll_recv(cx),
            Some(BufferReceiver::Bounded(receiver)) => receiver.poll_recv(cx),
            None => return Poll::Ready(None),
        };
        if !matches!(event, Poll::Ready(None)) {
            return event;
        }
        let mut slot = self.task.lock().expect("agent buffer mutex poisoned");
        let Some(task) = slot.as_mut() else {
            return Poll::Ready(None);
        };
        match Pin::new(task).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                slot.take();
                drop(slot);
                self.receiver.take();
                match result {
                    Ok(()) => Poll::Ready(None),
                    Err(error) => Poll::Ready(Some(Err(Error::Configuration(format!(
                        "agent loop task failed: {error}"
                    ))))),
                }
            }
        }
    }
}
impl Drop for BufferedStream {
    fn drop(&mut self) {
        if let Some(task) = self.take_task() {
            task.abort()
        }
    }
}
