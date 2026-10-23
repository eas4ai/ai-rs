//! `multipart/form-data` bodies, for the APIs that take files in a form.

use bytes::Bytes;

use crate::{transport::HttpRequest, types::messages::generate_id};

/// A `multipart/form-data` body built part by part.
pub(crate) struct Multipart {
    boundary: String,
    body: Vec<u8>,
}

/// A form name or filename with its quotes and line breaks escaped.
fn quoted(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace(['\r', '\n'], " ")
}

impl Multipart {
    pub(crate) fn new() -> Self {
        Self {
            boundary: format!("suprnova-{}", generate_id("form")),
            body: Vec::new(),
        }
    }

    fn header(&mut self, disposition: &str, media_type: Option<&str>) {
        self.body
            .extend_from_slice(format!("--{}\r\n", self.boundary).as_bytes());
        self.body.extend_from_slice(
            format!("Content-Disposition: form-data; {disposition}\r\n").as_bytes(),
        );
        if let Some(media_type) = media_type {
            self.body
                .extend_from_slice(format!("Content-Type: {media_type}\r\n").as_bytes());
        }
        self.body.extend_from_slice(b"\r\n");
    }

    /// A text field.
    pub(crate) fn text(&mut self, name: &str, value: &str) -> &mut Self {
        self.header(&format!("name=\"{}\"", quoted(name)), None);
        self.body.extend_from_slice(value.as_bytes());
        self.body.extend_from_slice(b"\r\n");
        self
    }

    /// A file field.
    pub(crate) fn file(
        &mut self,
        name: &str,
        filename: &str,
        media_type: &str,
        data: &[u8],
    ) -> &mut Self {
        self.header(
            &format!(
                "name=\"{}\"; filename=\"{}\"",
                quoted(name),
                quoted(filename)
            ),
            Some(media_type),
        );
        self.body.extend_from_slice(data);
        self.body.extend_from_slice(b"\r\n");
        self
    }

    /// Put the finished body and its content type on a request.
    pub(crate) fn into_request(mut self, mut request: HttpRequest) -> HttpRequest {
        self.body
            .extend_from_slice(format!("--{}--\r\n", self.boundary).as_bytes());
        request
            .headers
            .retain(|(name, _)| !name.eq_ignore_ascii_case("content-type"));
        request.headers.push((
            "content-type".into(),
            format!("multipart/form-data; boundary={}", self.boundary),
        ));
        request.body = Bytes::from(self.body);
        request
    }
}
