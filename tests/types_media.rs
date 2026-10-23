use base64::{Engine as _, engine::general_purpose::STANDARD};
use suprnova_ai::types::media;

const RIFF_WEBP: &[u8] = b"RIFF\0\0\0\0WEBP";
const RIFF_WAVE: &[u8] = b"RIFF\0\0\0\0WAVE";
const MP4: &[u8] = b"\0\0\0\x18ftypisom";

#[test]
fn data_url_helpers_preserve_python_behavior() {
    let payload = STANDARD.encode(b"hello");
    let data_url = format!("data:image/png;base64,{payload}");
    assert_eq!(
        media::split_data_url(&data_url),
        (Some("image/png"), Some(payload.as_str()))
    );
    assert_eq!(media::data_to_base64(data_url.as_str()), payload);
    assert_eq!(
        media::data_to_base64("/9j/4AAQSkZJRg=="),
        "/9j/4AAQSkZJRg=="
    );
    assert_eq!(media::split_data_url("data:nope"), (None, None));

    let result = media::data_to_data_url(b"\x89PNG", "image/png");
    let (_, encoded) = result.split_once(',').expect("data URL separator");
    assert_eq!(STANDARD.decode(encoded).expect("valid base64"), b"\x89PNG");
}

#[test]
fn detection_distinguishes_riff_containers_and_base64() {
    assert_eq!(
        media::detect_image_media_type(RIFF_WEBP),
        Some("image/webp")
    );
    assert_eq!(media::detect_image_media_type(RIFF_WAVE), None);
    assert_eq!(media::detect_audio_media_type(RIFF_WAVE), Some("audio/wav"));
    assert_eq!(media::detect_audio_media_type(RIFF_WEBP), None);
    assert_eq!(
        media::detect_image_media_type(STANDARD.encode(RIFF_WEBP).as_str()),
        Some("image/webp")
    );
}

#[test]
fn audio_detection_strips_id3_and_video_is_separate() {
    let mut tagged = b"ID3\x03\0\0\0\0\0\0".to_vec();
    tagged.extend_from_slice(&[0xff, 0xfb]);
    assert_eq!(
        media::detect_audio_media_type(tagged.as_slice()),
        Some("audio/mpeg")
    );
    assert_eq!(media::detect_video_media_type(MP4), Some("video/mp4"));
    assert_eq!(
        media::detect_video_media_type(STANDARD.encode(MP4).as_str()),
        Some("video/mp4")
    );
    assert_eq!(
        media::detect_video_media_type(&[0xff, 0xd8, 0xff, 0xe0]),
        None
    );
}

#[test]
fn unknown_and_short_values_return_none() {
    for value in [&[][..], &[0][..], &[0, 1, 2, 3][..]] {
        assert_eq!(media::detect_image_media_type(value), None);
        assert_eq!(media::detect_audio_media_type(value), None);
    }
}

#[test]
fn byte_base64_uses_standard_alphabet() {
    let encoded = media::data_to_base64(&[0xff, 0xd8, 0xff, 0xe0]);
    assert!(!encoded.contains(['-', '_']));
    assert_eq!(
        STANDARD.decode(encoded).expect("valid base64"),
        [0xff, 0xd8, 0xff, 0xe0]
    );
}

#[test]
fn unicode_base64_input_never_panics_at_the_prefix_boundary() {
    let text = format!("{}é", "A".repeat(47));
    assert_eq!(media::detect_image_media_type(text.as_str()), None);
    assert_eq!(media::detect_audio_media_type(text.as_str()), None);
    assert_eq!(media::detect_video_media_type(text.as_str()), None);
}
