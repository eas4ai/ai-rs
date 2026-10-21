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

//! Consumer buffering with owned task cancellation and failure propagation.
use crate::{Error, providers::EventStream, types::Event};
use futures::{Future, Stream, StreamExt};
use std::{
    pin::Pin,
    task::{Context as TaskContext, Poll},
};
enum BufferReceiver {
    Unbounded(tokio::sync::mpsc::UnboundedReceiver<Result<Event, Error>>),
    Bounded(tokio::sync::mpsc::Receiver<Result<Event, Error>>),
}
pub(super) struct BufferedStream {
    source: Option<EventStream>,
    capacity: Option<usize>,
    receiver: Option<BufferReceiver>,
    task: Option<tokio::task::JoinHandle<()>>,
}
impl BufferedStream {
    pub(super) fn new(source: EventStream, capacity: Option<usize>) -> Self {
        Self {
            source: Some(source),
            capacity,
            receiver: None,
            task: None,
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
                self.task = Some(handle.spawn(async move {
                    while let Some(event) = source.next().await {
                        if sender.send(event).await.is_err() {
                            break;
                        }
                    }
                }));
            } else {
                let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
                self.receiver = Some(BufferReceiver::Unbounded(receiver));
                self.task = Some(handle.spawn(async move {
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
        let Some(task) = self.task.as_mut() else {
            return Poll::Ready(None);
        };
        match Pin::new(task).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                self.task.take();
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
        if let Some(task) = self.task.take() {
            task.abort()
        }
    }
}
