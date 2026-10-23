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

//! Incremental server-sent event framing. This layer does not interpret JSON.

use bytes::Bytes;

use futures::{
    StreamExt,
    stream::{self, BoxStream},
};

use super::{Body, TransportError};

/// One dispatched SSE message. `id` and `retry` retain the latest valid values.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SseEvent {
    pub data: String,
    pub event: String,
    pub id: Option<String>,
    pub retry: Option<u64>,
}

/// Parse SSE across arbitrary byte boundaries, including split UTF-8 and CRLF.
/// Incomplete events at EOF are discarded, as required by SSE framing.
/// Each line and each event's data is limited to 8 MiB.
pub fn parse(body: Body) -> BoxStream<'static, Result<SseEvent, TransportError>> {
    parse_with_limit(body, MAX_EVENT_BYTES)
}

/// The default limit of a line and of an event's data: 8 MiB.
pub const MAX_EVENT_BYTES: usize = 8 * 1024 * 1024;

pub fn parse_with_limit(
    body: Body,
    max_bytes: usize,
) -> BoxStream<'static, Result<SseEvent, TransportError>> {
    stream::try_unfold(Parser::new(body, max_bytes), |mut parser| async move {
        loop {
            if let Some(event) = parser.ready.take() {
                return Ok(Some((event, parser)));
            }
            if parser.offset < parser.pending.len() {
                let byte = parser.pending[parser.offset];
                parser.offset += 1;
                parser.feed_byte(byte)?;
                continue;
            }
            match parser.body.next().await {
                Some(Ok(chunk)) => {
                    parser.pending = chunk;
                    parser.offset = 0;
                }
                Some(Err(error)) => return Err(error),
                None => return Ok(None),
            }
        }
    })
    .boxed()
}

struct Parser {
    body: Body,
    max_bytes: usize,
    line: Vec<u8>,
    data: String,
    event: String,
    id: Option<String>,
    retry: Option<u64>,
    ready: Option<SseEvent>,
    pending: Bytes,
    offset: usize,
    skip_lf: bool,
    first_line: bool,
}

impl Parser {
    fn new(body: Body, max_bytes: usize) -> Self {
        Self {
            body,
            max_bytes,
            line: Vec::new(),
            data: String::new(),
            event: String::new(),
            id: None,
            retry: None,
            ready: None,
            pending: Bytes::new(),
            offset: 0,
            skip_lf: false,
            first_line: true,
        }
    }

    fn feed_byte(&mut self, byte: u8) -> Result<(), TransportError> {
        if self.skip_lf {
            self.skip_lf = false;
            if byte == b'\n' {
                return Ok(());
            }
        }
        if byte == b'\r' || byte == b'\n' {
            self.finish_line()?;
            self.skip_lf = byte == b'\r';
        } else {
            if self.line.len() >= self.max_bytes {
                return Err(TransportError::Limit(self.max_bytes));
            }
            self.line.push(byte);
        }
        Ok(())
    }

    fn finish_line(&mut self) -> Result<(), TransportError> {
        let bytes = std::mem::take(&mut self.line);
        let line = std::str::from_utf8(&bytes)
            .map_err(|_| TransportError::Protocol("non-UTF-8 line".into()))?;
        let line = if self.first_line {
            self.first_line = false;
            line.strip_prefix('\u{feff}').unwrap_or(line)
        } else {
            line
        };
        if line.is_empty() {
            if !self.data.is_empty() {
                self.data.pop(); // Remove the last data line's newline.
                self.ready = Some(SseEvent {
                    data: std::mem::take(&mut self.data),
                    event: if self.event.is_empty() {
                        "message".into()
                    } else {
                        std::mem::take(&mut self.event)
                    },
                    id: self.id.clone(),
                    retry: self.retry,
                });
            }
            self.event.clear();
            return Ok(());
        }
        if line.starts_with(':') {
            return Ok(());
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "data" => {
                if self
                    .data
                    .len()
                    .saturating_add(value.len())
                    .saturating_add(1)
                    > self.max_bytes
                {
                    return Err(TransportError::Limit(self.max_bytes));
                }
                self.data.push_str(value);
                self.data.push('\n');
            }
            "event" => self.event = value.into(),
            "id" if !value.contains('\0') => self.id = Some(value.into()),
            "retry" if !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit()) => {
                if let Ok(retry) = value.parse() {
                    self.retry = Some(retry);
                }
            }
            _ => {}
        }
        Ok(())
    }
}
