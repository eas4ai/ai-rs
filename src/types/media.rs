//! Media type detection, inference, and encoding helpers.

use std::borrow::Cow;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MediaError {
    #[error("cannot infer media_type from URL {url:?}; provide media_type explicitly")]
    CannotInfer { url: String },
    #[error("cannot detect media_type from bytes; provide media_type explicitly")]
    CannotDetect,
}

#[must_use]
pub fn is_url(data: &str) -> bool {
    data.starts_with("http://") || data.starts_with("https://") || data.starts_with("data:")
}

#[must_use]
pub fn is_downloadable_url(data: &str) -> bool {
    data.starts_with("http://") || data.starts_with("https://")
}

#[must_use]
pub fn split_data_url(url: &str) -> (Option<&str>, Option<&str>) {
    let Some(rest) = url.strip_prefix("data:") else {
        return (None, None);
    };
    let Some((header, content)) = rest.split_once(',') else {
        return (None, None);
    };
    let media_type = header.split(';').next().filter(|value| !value.is_empty());
    let content = (!content.is_empty()).then_some(content);
    (media_type, content)
}

#[must_use]
pub fn data_to_base64<'a>(data: impl Into<MediaDataRef<'a>>) -> String {
    match data.into() {
        MediaDataRef::Bytes(data) => STANDARD.encode(data),
        MediaDataRef::Text(data) => {
            if data.starts_with("data:") {
                split_data_url(data).1.unwrap_or(data).to_owned()
            } else {
                data.to_owned()
            }
        }
    }
}

#[must_use]
pub fn data_to_data_url<'a>(data: impl Into<MediaDataRef<'a>>, media_type: &str) -> String {
    let data = data.into();
    if let MediaDataRef::Text(text) = data
        && is_url(text)
    {
        return text.to_owned();
    }
    format!("data:{media_type};base64,{}", data_to_base64(data))
}

#[derive(Clone, Copy, Debug)]
pub enum MediaDataRef<'a> {
    Text(&'a str),
    Bytes(&'a [u8]),
}

impl<'a> From<&'a str> for MediaDataRef<'a> {
    fn from(value: &'a str) -> Self {
        Self::Text(value)
    }
}

impl<'a> From<&'a String> for MediaDataRef<'a> {
    fn from(value: &'a String) -> Self {
        Self::Text(value)
    }
}

impl<'a> From<&'a [u8]> for MediaDataRef<'a> {
    fn from(value: &'a [u8]) -> Self {
        Self::Bytes(value)
    }
}

impl<'a, const N: usize> From<&'a [u8; N]> for MediaDataRef<'a> {
    fn from(value: &'a [u8; N]) -> Self {
        Self::Bytes(value)
    }
}

pub fn infer_media_type(url: &str) -> Result<String, MediaError> {
    if url.starts_with("data:") {
        if let Some(media_type) = split_data_url(url).0 {
            return Ok(media_type.to_owned());
        }
    } else {
        let path = url.split(['?', '#']).next().unwrap_or(url);
        if let Some(media_type) = mime_guess::from_path(path).first_raw() {
            return Ok(media_type.to_owned());
        }
    }
    Err(MediaError::CannotInfer {
        url: url.to_owned(),
    })
}

type Signature = (&'static str, &'static [Option<u8>]);

const IMAGE_SIGNATURES: &[Signature] = &[
    ("image/gif", &[Some(0x47), Some(0x49), Some(0x46)]),
    (
        "image/png",
        &[Some(0x89), Some(0x50), Some(0x4e), Some(0x47)],
    ),
    ("image/jpeg", &[Some(0xff), Some(0xd8)]),
    (
        "image/webp",
        &[
            Some(0x52),
            Some(0x49),
            Some(0x46),
            Some(0x46),
            None,
            None,
            None,
            None,
            Some(0x57),
            Some(0x45),
            Some(0x42),
            Some(0x50),
        ],
    ),
    ("image/bmp", &[Some(0x42), Some(0x4d)]),
    (
        "image/tiff",
        &[Some(0x49), Some(0x49), Some(0x2a), Some(0x00)],
    ),
    (
        "image/tiff",
        &[Some(0x4d), Some(0x4d), Some(0x00), Some(0x2a)],
    ),
    (
        "image/avif",
        &[
            Some(0),
            Some(0),
            Some(0),
            Some(0x20),
            Some(0x66),
            Some(0x74),
            Some(0x79),
            Some(0x70),
            Some(0x61),
            Some(0x76),
            Some(0x69),
            Some(0x66),
        ],
    ),
    (
        "image/heic",
        &[
            Some(0),
            Some(0),
            Some(0),
            Some(0x20),
            Some(0x66),
            Some(0x74),
            Some(0x79),
            Some(0x70),
            Some(0x68),
            Some(0x65),
            Some(0x69),
            Some(0x63),
        ],
    ),
];

const AUDIO_SIGNATURES: &[Signature] = &[
    ("audio/mpeg", &[Some(0xff), Some(0xfb)]),
    ("audio/mpeg", &[Some(0xff), Some(0xfa)]),
    ("audio/mpeg", &[Some(0xff), Some(0xf3)]),
    ("audio/mpeg", &[Some(0xff), Some(0xf2)]),
    ("audio/mpeg", &[Some(0xff), Some(0xe3)]),
    ("audio/mpeg", &[Some(0xff), Some(0xe2)]),
    (
        "audio/wav",
        &[
            Some(0x52),
            Some(0x49),
            Some(0x46),
            Some(0x46),
            None,
            None,
            None,
            None,
            Some(0x57),
            Some(0x41),
            Some(0x56),
            Some(0x45),
        ],
    ),
    (
        "audio/ogg",
        &[Some(0x4f), Some(0x67), Some(0x67), Some(0x53)],
    ),
    (
        "audio/flac",
        &[Some(0x66), Some(0x4c), Some(0x61), Some(0x43)],
    ),
    ("audio/aac", &[Some(0x40), Some(0x15), Some(0), Some(0)]),
    (
        "audio/mp4",
        &[Some(0x66), Some(0x74), Some(0x79), Some(0x70)],
    ),
    (
        "audio/webm",
        &[Some(0x1a), Some(0x45), Some(0xdf), Some(0xa3)],
    ),
];

const VIDEO_SIGNATURES: &[Signature] = &[
    (
        "video/mp4",
        &[
            Some(0),
            Some(0),
            Some(0),
            None,
            Some(0x66),
            Some(0x74),
            Some(0x79),
            Some(0x70),
        ],
    ),
    (
        "video/webm",
        &[Some(0x1a), Some(0x45), Some(0xdf), Some(0xa3)],
    ),
    (
        "video/quicktime",
        &[
            Some(0),
            Some(0),
            Some(0),
            Some(0x14),
            Some(0x66),
            Some(0x74),
            Some(0x79),
            Some(0x70),
            Some(0x71),
            Some(0x74),
        ],
    ),
    (
        "video/x-msvideo",
        &[Some(0x52), Some(0x49), Some(0x46), Some(0x46)],
    ),
];

#[must_use]
pub fn detect_image_media_type<'a>(data: impl Into<MediaDataRef<'a>>) -> Option<&'static str> {
    detect_media_type(data.into(), IMAGE_SIGNATURES, false)
}

#[must_use]
pub fn detect_audio_media_type<'a>(data: impl Into<MediaDataRef<'a>>) -> Option<&'static str> {
    detect_media_type(data.into(), AUDIO_SIGNATURES, true)
}

#[must_use]
pub fn detect_video_media_type<'a>(data: impl Into<MediaDataRef<'a>>) -> Option<&'static str> {
    detect_media_type(data.into(), VIDEO_SIGNATURES, false)
}

fn detect_media_type(
    data: MediaDataRef<'_>,
    signatures: &[Signature],
    strip_id3: bool,
) -> Option<&'static str> {
    let raw = match data {
        MediaDataRef::Bytes(data) => Cow::Borrowed(data),
        MediaDataRef::Text(data) => Cow::Owned(decode_base64_prefix(data)),
    };
    let raw = if strip_id3 { strip_id3_tag(&raw) } else { &raw };
    signatures.iter().find_map(|(media_type, signature)| {
        matches_signature(raw, signature).then_some(*media_type)
    })
}

fn decode_base64_prefix(data: &str) -> Vec<u8> {
    let bytes = data.as_bytes();
    let end = bytes.len().min(48);
    let end = end - end % 4;
    STANDARD.decode(&bytes[..end]).unwrap_or_default()
}

fn matches_signature(data: &[u8], signature: &[Option<u8>]) -> bool {
    data.len() >= signature.len()
        && signature
            .iter()
            .zip(data)
            .all(|(expected, actual)| expected.is_none_or(|expected| expected == *actual))
}

fn strip_id3_tag(data: &[u8]) -> &[u8] {
    if data.len() < 10 || !data.starts_with(b"ID3") {
        return data;
    }
    let size = (usize::from(data[6] & 0x7f) << 21)
        | (usize::from(data[7] & 0x7f) << 14)
        | (usize::from(data[8] & 0x7f) << 7)
        | usize::from(data[9] & 0x7f);
    let offset = size + 10;
    if offset < data.len() {
        &data[offset..]
    } else {
        data
    }
}
