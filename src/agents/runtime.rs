//! Owned event sink and priority merge for an agent run.

use std::{
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
};

use futures::Stream;
use tokio::sync::mpsc;

use crate::{
    Error,
    types::{Event, HookPart, Message, MessageRole, events::HookEvent},
};

/// Cloneable producer handle. Dropping the consumer closes all producer handles.
#[derive(Clone, Debug)]
pub struct Runtime {
    sender: Arc<Mutex<Option<mpsc::UnboundedSender<Event>>>>,
}

impl Runtime {
    /// Create a run-local sink with the same immediate enqueue semantics as the Python runtime.
    #[must_use]
    pub fn new() -> (Self, RuntimeEvents) {
        let (sender, receiver) = mpsc::unbounded_channel();
        (
            Self {
                sender: Arc::new(Mutex::new(Some(sender))),
            },
            RuntimeEvents { receiver },
        )
    }

    /// Queue an event without blocking the consumer that resolves hooks.
    pub async fn put_event(&self, event: Event) -> Result<(), Error> {
        let sender = self
            .sender
            .lock()
            .expect("runtime sender mutex poisoned")
            .clone()
            .ok_or_else(closed)?;
        sender.send(event).map_err(|_| closed())
    }

    /// Emit a hook signal as an internal message and event.
    pub async fn put_hook(&self, hook: HookPart) -> Result<(), Error> {
        self.put_event(Event::Hook(HookEvent {
            replay: false,
            message: Message::new(MessageRole::Internal, vec![hook.clone().into()]),
            hook,
        }))
        .await
    }

    /// Close the producer side. Already queued events remain available.
    pub async fn signal_done(&self) {
        self.sender
            .lock()
            .expect("runtime sender mutex poisoned")
            .take();
    }
}

fn closed() -> Error {
    Error::Configuration("agent runtime event sink is closed".to_owned())
}

/// The single consumer of a runtime's queued events.
#[derive(Debug)]
pub struct RuntimeEvents {
    receiver: mpsc::UnboundedReceiver<Event>,
}

impl Stream for RuntimeEvents {
    type Item = Event;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.receiver.poll_recv(cx)
    }
}

impl RuntimeEvents {
    /// Merge a source in lockstep, prioritizing sink events emitted during its poll.
    #[must_use]
    pub fn merge<S>(self, source: S) -> RuntimeStream<S>
    where
        S: Stream<Item = Result<Event, Error>>,
    {
        RuntimeStream {
            events: self,
            source: Box::pin(source),
            pending: None,
            done: false,
        }
    }
}

/// A source and its runtime queue. Dropping this stream drops the source too.
pub struct RuntimeStream<S> {
    events: RuntimeEvents,
    source: Pin<Box<S>>,
    pending: Option<Result<Event, Error>>,
    done: bool,
}

impl<S: Stream<Item = Result<Event, Error>>> Stream for RuntimeStream<S> {
    type Item = Result<Event, Error>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match Pin::new(&mut this.events).poll_next(cx) {
            Poll::Ready(Some(event)) => return Poll::Ready(Some(Ok(event))),
            // Tokio may yield its cooperative budget while events remain queued.
            Poll::Pending if !this.events.receiver.is_empty() || this.done => {
                return Poll::Pending;
            }
            _ => {}
        }
        if let Some(event) = this.pending.take() {
            return Poll::Ready(Some(event));
        }
        if this.done {
            return Poll::Ready(None);
        }
        match this.source.as_mut().poll_next(cx) {
            Poll::Pending => match Pin::new(&mut this.events).poll_next(cx) {
                Poll::Ready(Some(event)) => Poll::Ready(Some(Ok(event))),
                // Closing the side channel does not finish a running source.
                Poll::Ready(None) | Poll::Pending => Poll::Pending,
            },
            Poll::Ready(event) => {
                this.pending = event;
                if this.pending.is_none() {
                    this.done = true;
                    this.events.receiver.close();
                }
                match Pin::new(&mut this.events).poll_next(cx) {
                    Poll::Ready(Some(event)) => Poll::Ready(Some(Ok(event))),
                    Poll::Pending if !this.events.receiver.is_empty() || this.done => Poll::Pending,
                    _ => Poll::Ready(this.pending.take()),
                }
            }
        }
    }
}
