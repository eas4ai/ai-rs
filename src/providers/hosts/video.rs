//! The video model of MiniMax (`minimax-video-model.ts`, HST-040 and
//! HST-041), at the root of the base URL, the base URL without its trailing
//! `/anthropic`, with the key as a bearer token. A job starts with
//! `POST <root>/v2/video_generation` and is checked with
//! `GET <root>/v2/query/video_generation/<task_id>`. Each call makes one
//! video, so `ops::generate_video` starts one job per video (OPS-003) and
//! polls by its `VideoOptions` (OPS-004); TypeScript's `pollIntervalMs` and
//! `pollTimeoutMs` fail with `Error::Validation` naming `VideoOptions`, and
//! its `webhookUrl` has no core counterpart. Warnings that TypeScript
//! returns come back in `Item.warnings`.
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};

use crate::{
    Error, Model, Provider,
    errors::{ProviderError, ProviderErrorKind},
    models::files,
    providers::{
        http::{self, ApiProfile, Auth},
        operations::{
            CallLimits, FrameType, Item, MediaInput, VideoParams, VideoPrompt, VideoStatus, Warning,
        },
    },
    types::{
        media,
        messages::{FileData, FilePart, generate_id},
    },
};

/// A host's video model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Videos {
    MiniMax,
}

/// MiniMax's video API: a bearer key and JSON errors with `error.message`.
static MINIMAX_API: ApiProfile = ApiProfile {
    label: "MiniMax",
    auth: Auth::Bearer,
    default_headers: &[],
    request_id_header: "request-id",
    collapse_server_errors: false,
    keep_error_object: false,
    top_level_first: false,
    code_prefixes_error: false,
    config_headers: &[],
};

/// The most a finished video may hold, the limit Google's videos get too
/// (GGL-032); TypeScript's core reads a result URL without one.
const VIDEO_MAX_BYTES: usize = 256 * 1024 * 1024;
const RESOLUTIONS: [&str; 3] = ["480P", "768P", "2K"];
const RATIOS: [&str; 7] = ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];
const DEFAULT_RESOLUTION: &str = "2K";
const DEFAULT_RATIO: &str = "16:9";
const DEFAULT_DURATION: u32 = 5;
const MAX_DURATION: u32 = 15;
const MAX_REFERENCE_IMAGES: usize = 9;
const MAX_REFERENCE_VIDEOS: usize = 3;
const MAX_REFERENCE_AUDIOS: usize = 3;
/// The tier of each frame size TypeScript maps (`RESOLUTION_MAP`), a closed
/// table since MiniMax publishes no tier dimensions.
const RESOLUTION_TABLE: [(&str, &str); 18] = [
    ("480x480", "480P"),
    ("1120x480", "480P"),
    ("854x480", "480P"),
    ("640x480", "480P"),
    ("480x854", "480P"),
    ("480x640", "480P"),
    ("768x768", "768P"),
    ("1792x768", "768P"),
    ("1366x768", "768P"),
    ("1024x768", "768P"),
    ("768x1366", "768P"),
    ("768x1024", "768P"),
    ("2048x2048", "2K"),
    ("2560x1080", "2K"),
    ("2560x1440", "2K"),
    ("2048x1536", "2K"),
    ("1440x2560", "2K"),
    ("1536x2048", "2K"),
];

/// Each call makes one video.
pub(super) fn limits() -> CallLimits {
    CallLimits {
        max_per_call: Some(1),
        parallel: true,
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Validation(message.into())
}

fn warning(feature: &str, details: String) -> Warning {
    Warning {
        kind: "unsupported".into(),
        feature: Some(feature.into()),
        details: Some(details),
        ..Warning::default()
    }
}

fn bad(option: &str, expected: &str, value: &Value) -> Error {
    invalid(format!(
        "minimax video option {option} must be {expected}, not {value}"
    ))
}

fn one_of(
    option: &str,
    value: &Value,
    allowed: &'static [&'static str],
) -> Result<&'static str, Error> {
    allowed
        .iter()
        .copied()
        .find(|allowed| value.as_str() == Some(allowed))
        .ok_or_else(|| bad(option, &format!("one of {}", allowed.join(", ")), value))
}

/// The `minimax` video options (`minimaxVideoModelOptionsSchema`).
#[derive(Debug, Default)]
struct Options {
    resolution: Option<&'static str>,
    ratio: Option<&'static str>,
    reference_audio_urls: Vec<String>,
    aigc_watermark: Option<bool>,
}

impl Options {
    fn parse(params: &VideoParams) -> Result<Self, Error> {
        let Some(entry) = params.provider_options.get("minimax") else {
            return Ok(Self::default());
        };
        let Value::Object(entry) = entry else {
            return Err(invalid("minimax video options must be an object"));
        };
        let mut options = Self::default();
        for (option, value) in entry {
            match option.as_str() {
                "resolution" => options.resolution = Some(one_of(option, value, &RESOLUTIONS)?),
                "ratio" => options.ratio = Some(one_of(option, value, &RATIOS)?),
                "referenceAudioUrls" => {
                    options.reference_audio_urls = value
                        .as_array()
                        .and_then(|items| {
                            items
                                .iter()
                                .map(|item| item.as_str().map(str::to_owned))
                                .collect::<Option<Vec<_>>>()
                        })
                        .ok_or_else(|| bad(option, "an array of strings", value))?;
                }
                "aigcWatermark" => {
                    options.aigc_watermark = Some(
                        value
                            .as_bool()
                            .ok_or_else(|| bad(option, "a boolean", value))?,
                    );
                }
                "pollIntervalMs" | "pollTimeoutMs" => {
                    return Err(invalid(format!(
                        "minimax video option {option} has no effect here: ops::generate_video polls by its VideoOptions poll_interval and poll_timeout"
                    )));
                }
                other => {
                    return Err(invalid(format!(
                        "minimax video option {other} is not known"
                    )));
                }
            }
        }
        Ok(options)
    }
}

/// The root MiniMax's video API lives at: the base URL without `/anthropic`.
fn root(provider: &dyn Provider) -> Result<String, Error> {
    let base = provider.config().base_url();
    base.trim_end_matches('/')
        .strip_suffix("/anthropic")
        .map(str::to_owned)
        .ok_or_else(|| {
            Error::Configuration(format!(
                "MiniMax video calls go to the root of the base URL, which is the base URL \
                 without its last segment anthropic; {base} does not end in /anthropic"
            ))
        })
}

fn transport(provider: &dyn Provider) -> Result<&dyn crate::transport::HttpTransport, Error> {
    provider.transport().ok_or_else(|| {
        Error::Configuration(format!(
            "provider {:?} has no transport for videos",
            provider.name()
        ))
    })
}

/// A video input as the URL MiniMax takes and its media type, when known
/// (`convertImageModelFileToDataUri`): a URL as given, a file's bytes or
/// base64 as a data URL of its type, and bare bytes as a data URL of the
/// image type their bytes show.
fn input_url(input: &MediaInput) -> Result<(String, Option<String>), Error> {
    let data_url = |media_type: &str, bytes: &[u8]| {
        format!("data:{media_type};base64,{}", STANDARD.encode(bytes))
    };
    Ok(match input {
        MediaInput::Url(url) => (url.clone(), None),
        MediaInput::Bytes(bytes) => {
            let media_type = media::detect_image_media_type(bytes.as_slice()).ok_or_else(|| {
                invalid(
                    "a MiniMax video input given as bare bytes must be an image whose type its \
                     bytes show; give other media as a file part with a media type",
                )
            })?;
            (data_url(media_type, bytes), Some(media_type.to_owned()))
        }
        MediaInput::File(file) => {
            let url = match &file.data {
                FileData::Text(text) if media::is_url(text) => text.clone(),
                FileData::Text(text) => format!("data:{};base64,{text}", file.media_type),
                FileData::Bytes(bytes) => data_url(&file.media_type, bytes),
            };
            (url, Some(file.media_type.clone()))
        }
    })
}

/// The top-level type of a media type, such as `image`.
fn top_level(media_type: Option<&str>) -> Option<&str> {
    media_type.map(|media_type| media_type.split('/').next().unwrap_or(media_type))
}

/// The URL a reference was given as, if it was given as one.
fn given_url(input: &MediaInput) -> Option<&str> {
    match input {
        MediaInput::File(file) => file
            .data
            .as_text()
            .filter(|text| media::is_downloadable_url(text)),
        MediaInput::Url(url) => Some(url),
        MediaInput::Bytes(_) => None,
    }
}

/// The tier a core resolution names: a tier in any case, or a frame size of
/// the table.
fn tier(resolution: &str) -> Option<&'static str> {
    RESOLUTIONS
        .iter()
        .copied()
        .find(|tier| tier.eq_ignore_ascii_case(resolution))
        .or_else(|| {
            RESOLUTION_TABLE
                .iter()
                .find(|(size, _)| *size == resolution)
                .map(|(_, tier)| *tier)
        })
}

/// Start a MiniMax video job (`doStart`): the job is its `task_id`, and
/// the item's `minimax` metadata holds the inputs that went into it.
pub(super) async fn minimax_start(
    model: &Model,
    prompt: VideoPrompt,
    params: VideoParams,
    provider: &dyn Provider,
) -> Result<Item<Value>, Error> {
    // The video comes as a URL on another origin, fetched through the
    // download transport; check for it before a job is started and billed,
    // as HST-031 checks it.
    if provider.download_transport().is_none() {
        return Err(invalid(
            "a MiniMax video call needs a download transport, since its video comes from \
             another origin; build the provider with with_download_transport or register it \
             with register_with_downloads",
        ));
    }
    // Polling the job needs a Tokio timer (OPS-004); check for it before the
    // job is started and billed, as the Fireworks Kontext call does.
    crate::models::call::require_timer()?;
    let options = Options::parse(&params)?;
    let root = root(provider)?;
    let id = model.id.as_str();
    let mut warnings = Vec::new();
    let mut warn = |feature: &str, details: String| warnings.push(warning(feature, details));
    if params.fps.is_some() {
        warn("fps", format!("{id} does not support a custom frame rate."));
    }
    if params.seed.is_some() {
        warn("seed", format!("{id} does not support a seed."));
    }
    if params.generate_audio.is_some() {
        warn(
            "generateAudio",
            format!(
                "The {id} API does not expose an audio parameter. The generateAudio option was ignored."
            ),
        );
    }
    // ops::generate_video starts one job per video (OPS-003), so this is
    // ops::start_video asking one job for several.
    if params.n > 1 {
        warn(
            "n",
            format!("{id} generates a single video per call. Only 1 video will be generated."),
        );
    }

    // Resolution: the API takes a tier, so the provider option wins and a
    // core value is mapped to one.
    let (supported, default): (&[&str], &str) = match id {
        "MiniMax-H3" => (&["768P", "2K"], DEFAULT_RESOLUTION),
        "MiniMax-H3-Max" => (&["480P", "768P"], "768P"),
        _ => (&RESOLUTIONS, DEFAULT_RESOLUTION),
    };
    let names = supported
        .iter()
        .map(|tier| format!("\"{tier}\""))
        .collect::<Vec<_>>()
        .join(" or ");
    let mut resolution = options.resolution;
    if let Some(chosen) = resolution
        && !supported.contains(&chosen)
    {
        warn(
            "resolution",
            format!("{id} supports {names}. The provider resolution \"{chosen}\" was ignored."),
        );
        resolution = None;
    }
    if let Some(core) = &params.resolution {
        match (resolution, tier(core)) {
            (Some(chosen), None) => warn(
                "resolution",
                format!(
                    "Unrecognized resolution \"{core}\". {id} supports {names}, so \
                     providerOptions.minimax.resolution (\"{chosen}\") was used instead."
                ),
            ),
            (Some(chosen), Some(mapped)) if mapped != chosen => warn(
                "resolution",
                format!(
                    "The resolution \"{core}\" selects {mapped}, but \
                     providerOptions.minimax.resolution (\"{chosen}\") was used instead."
                ),
            ),
            (Some(_), Some(_)) => {}
            (None, Some(mapped)) if supported.contains(&mapped) => resolution = Some(mapped),
            (None, Some(mapped)) => warn(
                "resolution",
                format!("{id} does not support the resolution \"{mapped}\". It supports {names}."),
            ),
            (None, None) => warn(
                "resolution",
                format!("Unrecognized resolution \"{core}\". {id} supports {names}."),
            ),
        }
    }
    let resolution = resolution.unwrap_or(default);

    // Content: the prompt, then the frames or the references.
    let mut content =
        vec![json!({"type": "text", "text": prompt.text.clone().unwrap_or_default()})];
    let mut image_count = 0u32;
    let mut reference_video_urls: Vec<String> = Vec::new();
    let first_frame_image = prompt
        .frame_images
        .iter()
        .find(|frame| frame.frame_type == FrameType::FirstFrame)
        .map(|frame| &frame.image);
    let first = first_frame_image.or(prompt.image.as_ref());
    let last = prompt
        .frame_images
        .iter()
        .find(|frame| frame.frame_type == FrameType::LastFrame)
        .map(|frame| &frame.image);
    let mut first_url = None;
    if let Some(input) = first {
        let (url, media_type) = input_url(input)?;
        match top_level(media_type.as_deref()) {
            None | Some("image") => first_url = Some(url),
            Some(kind) => {
                let feature = if first_frame_image.is_some() {
                    "frameImages"
                } else {
                    "image"
                };
                let details = if kind == "video" {
                    format!("{id} does not accept a video as a frame image. The video was ignored.")
                } else {
                    format!(
                        "{id} only accepts an image as a frame image; the \"{}\" file was ignored.",
                        media_type.unwrap_or_default()
                    )
                };
                warn(feature, details);
            }
        }
    }
    let mut last_url = None;
    if let Some(input) = last {
        if first_url.is_none() {
            warn(
                "frameImages",
                format!(
                    "{id} requires a first_frame when a last_frame is provided. The last_frame was ignored."
                ),
            );
        } else {
            let (url, media_type) = input_url(input)?;
            match top_level(media_type.as_deref()) {
                None | Some("image") => last_url = Some(url),
                Some("video") => warn(
                    "frameImages",
                    format!(
                        "{id} does not accept a video as a frame image. The last_frame video was ignored."
                    ),
                ),
                Some(_) => warn(
                    "frameImages",
                    format!(
                        "{id} only accepts an image as a frame image; the \"{}\" last_frame was ignored.",
                        media_type.unwrap_or_default()
                    ),
                ),
            }
        }
    }
    let uses_frames = first_url.is_some() || last_url.is_some();

    let supports_references = id != "MiniMax-H3-Max";
    let audio = &options.reference_audio_urls;
    if !supports_references && !prompt.references.is_empty() {
        warn(
            "inputReferences",
            format!(
                "{id} does not support reference-to-video inputs. The references were ignored."
            ),
        );
    }
    if !supports_references && !audio.is_empty() {
        warn(
            "referenceAudioUrls",
            format!("{id} does not support reference audio. The audio was ignored."),
        );
    }
    let uses_references =
        supports_references && (!prompt.references.is_empty() || !audio.is_empty());
    let media_part =
        |kind: &str, url: &str, role: &str| json!({"type": kind, kind: {"url": url}, "role": role});
    if uses_frames {
        if let Some(url) = &first_url {
            content.push(media_part("image_url", url, "first_frame"));
            image_count += 1;
        }
        if let Some(url) = &last_url {
            content.push(media_part("image_url", url, "last_frame"));
            image_count += 1;
        }
        if uses_references {
            warn(
                "inputReferences",
                format!(
                    "{id} cannot combine frame images with reference inputs. The references were ignored."
                ),
            );
        }
    } else if uses_references {
        let mut images = Vec::new();
        let mut videos = Vec::new();
        for input in &prompt.references {
            let (url, media_type) = input_url(input)?;
            match top_level(media_type.as_deref()) {
                Some("video") => videos.push((url, given_url(input).map(str::to_owned))),
                Some("image") => images.push(url),
                None => {
                    warn(
                        "inputReferences",
                        format!(
                            "{id} requires an explicit mediaType to route URL references as video \
                             or image. Pass {{ data: url, mediaType: \"video/mp4\" }} for video \
                             references. The reference was treated as an image."
                        ),
                    );
                    images.push(url);
                }
                Some(_) => warn(
                    "inputReferences",
                    format!(
                        "{id} only accepts image and video references; the \"{}\" reference was \
                         ignored. Pass reference audio via providerOptions.minimax.referenceAudioUrls.",
                        media_type.unwrap_or_default()
                    ),
                ),
            }
        }
        for url in images.iter().take(MAX_REFERENCE_IMAGES) {
            content.push(media_part("image_url", url, "reference_image"));
            image_count += 1;
        }
        if images.len() > MAX_REFERENCE_IMAGES {
            warn(
                "inputReferences",
                format!(
                    "{id} accepts at most {MAX_REFERENCE_IMAGES} reference images. Extra images were ignored."
                ),
            );
        }
        for (url, given) in videos.iter().take(MAX_REFERENCE_VIDEOS) {
            content.push(media_part("video_url", url, "reference_video"));
            if let Some(given) = given {
                reference_video_urls.push(given.clone());
            }
        }
        if videos.len() > MAX_REFERENCE_VIDEOS {
            warn(
                "inputReferences",
                format!(
                    "{id} accepts at most {MAX_REFERENCE_VIDEOS} reference videos. Extra videos were ignored."
                ),
            );
        }
        if !audio.is_empty() {
            if images.is_empty() && videos.is_empty() {
                warn(
                    "referenceAudioUrls",
                    format!(
                        "{id} reference audio must be paired with at least one reference image or video. The audio was ignored."
                    ),
                );
            } else {
                for url in audio.iter().take(MAX_REFERENCE_AUDIOS) {
                    content.push(media_part("audio_url", url, "reference_audio"));
                }
                if audio.len() > MAX_REFERENCE_AUDIOS {
                    warn(
                        "referenceAudioUrls",
                        format!(
                            "{id} accepts at most {MAX_REFERENCE_AUDIOS} reference audios. Extra audios were ignored."
                        ),
                    );
                }
            }
        }
    }

    // Aspect ratio: frames set it, and text-to-video takes no adaptive one.
    let text_to_video = content.len() == 1;
    let mut ratio = options.ratio;
    if ratio.is_none()
        && let Some(core) = &params.aspect_ratio
    {
        if let Some(allowed) = RATIOS.iter().copied().find(|ratio| ratio == core) {
            ratio = Some(allowed);
        } else {
            warn(
                "aspectRatio",
                if text_to_video {
                    format!(
                        "{id} does not support the aspect ratio \"{core}\". Using the default ({DEFAULT_RATIO})."
                    )
                } else {
                    format!(
                        "{id} does not support the aspect ratio \"{core}\". Using the provider default (adaptive)."
                    )
                },
            );
            if text_to_video {
                ratio = Some(DEFAULT_RATIO);
            }
        }
    }
    if ratio == Some("adaptive") && text_to_video {
        warn(
            "aspectRatio",
            format!(
                "{id} text-to-video does not support the adaptive aspect ratio. Using the default ({DEFAULT_RATIO})."
            ),
        );
        ratio = Some(DEFAULT_RATIO);
    }
    if uses_frames && ratio.is_some() {
        warn(
            "aspectRatio",
            format!(
                "{id} derives the aspect ratio from the frame image; the requested ratio was ignored."
            ),
        );
        ratio = None;
    }
    if ratio.is_none() && text_to_video {
        ratio = Some(DEFAULT_RATIO);
    }

    // Duration: whole seconds from 4 (H3) or 5 to 15.
    let min_duration = if id == "MiniMax-H3" { 4 } else { 5 };
    let mut duration = params.duration.unwrap_or(DEFAULT_DURATION);
    if let Some(asked) = params.duration {
        if asked > MAX_DURATION {
            warn(
                "duration",
                format!(
                    "{id} supports at most {MAX_DURATION} seconds. The requested duration of {asked} was clamped to {MAX_DURATION}."
                ),
            );
            duration = MAX_DURATION;
        } else if asked < min_duration {
            warn(
                "duration",
                format!(
                    "{id} requires at least {min_duration} seconds. The requested duration of {asked} was clamped to {min_duration}."
                ),
            );
            duration = min_duration;
        }
    }

    let mut body = json!({
        "model": id,
        "content": content,
        "resolution": resolution,
        "duration": duration,
    });
    if let Some(ratio) = ratio {
        body["ratio"] = json!(ratio);
    }
    if let Some(watermark) = options.aigc_watermark {
        body["aigc_watermark"] = json!(watermark);
    }
    let request = http::request_to(
        &MINIMAX_API,
        &root,
        provider.config(),
        "POST",
        &["v2", "video_generation"],
        Some(body),
        Default::default(),
        Default::default(),
    )?;
    let response = http::json(
        &MINIMAX_API,
        transport(provider)?,
        request,
        provider.name(),
        Some(id),
    )
    .await?;
    let task_id = response["task_id"]
        .as_str()
        .filter(|task_id| !task_id.is_empty())
        .ok_or_else(|| {
            http::response_error(provider.name(), "MiniMax video response has no task_id")
        })?;
    let mut item = Item::new(json!(task_id));
    item.warnings = warnings;
    item.provider_metadata = Some(
        [(
            "minimax".to_owned(),
            json!({
                "taskId": task_id,
                "resolvedInputs": {
                    "imageCount": image_count,
                    "referenceVideoUrls": reference_video_urls,
                },
            }),
        )]
        .into(),
    );
    Ok(item)
}

/// A task id as a URL path segment: one that names no segment or holds a
/// control character would send the check elsewhere.
fn task_id(operation: &Value) -> Result<&str, Error> {
    operation
        .as_str()
        .filter(|id| {
            !id.trim().is_empty() && *id != "." && *id != ".." && !id.chars().any(char::is_control)
        })
        .ok_or_else(|| {
            invalid(format!(
                "a MiniMax video job is its task id, which {operation} is not"
            ))
        })
}

/// Check a MiniMax video job once (`doStatus`): `succeeded` completes with
/// the video's URL, `failed`, `cancelled` and `expired` fail, and any other
/// status is pending.
pub(super) async fn minimax_status(
    model: &Model,
    operation: &Value,
    provider: &dyn Provider,
) -> Result<VideoStatus, Error> {
    let task_id = task_id(operation)?;
    let root = root(provider)?;
    let request = http::request_to(
        &MINIMAX_API,
        &root,
        provider.config(),
        "GET",
        &["v2", "query", "video_generation", task_id],
        None,
        Default::default(),
        Default::default(),
    )?;
    let response = http::json(
        &MINIMAX_API,
        transport(provider)?,
        request,
        provider.name(),
        Some(&model.id),
    )
    .await?;
    let task = &response["task"];
    match task["status"].as_str() {
        Some("succeeded") => {
            let url = task["content"]["url"]
                .as_str()
                .filter(|url| !url.is_empty())
                .ok_or_else(|| {
                    http::response_error(
                        provider.name(),
                        format!(
                            "MiniMax video generation completed but no video URL was returned. Task ID: {task_id}"
                        ),
                    )
                })?;
            let video = FilePart {
                id: generate_id("part"),
                data: FileData::Text(url.to_owned()),
                media_type: "video/mp4".into(),
                filename: None,
                provider_metadata: None,
            };
            let mut metadata = json!({"taskId": task_id, "videoUrl": url});
            for key in ["duration", "ratio", "resolution"] {
                if !task[key].is_null() {
                    metadata[key] = task[key].clone();
                }
            }
            if let Some(usage) = task["usage"].as_object() {
                let mut seconds = Map::new();
                for (from, to) in [
                    ("total_seconds", "totalSeconds"),
                    ("input_seconds", "inputSeconds"),
                    ("output_seconds", "outputSeconds"),
                ] {
                    if let Some(value) = usage.get(from).filter(|value| !value.is_null()) {
                        seconds.insert(to.into(), value.clone());
                    }
                }
                metadata["usage"] = Value::Object(seconds);
            }
            let mut item = Item::new(vec![video]);
            item.provider_metadata = Some([("minimax".to_owned(), metadata)].into());
            Ok(VideoStatus::Completed(item))
        }
        Some("failed") => {
            let message = task["error"]["message"]
                .as_str()
                .map(|message| format!(": {message}"))
                .unwrap_or_default();
            let code = match &task["error"]["code"] {
                Value::String(code) => format!(" ({code})"),
                Value::Number(code) => format!(" ({code})"),
                _ => String::new(),
            };
            Ok(VideoStatus::Failed {
                message: format!(
                    "MiniMax video generation failed{message}{code}. Task ID: {task_id}"
                ),
            })
        }
        Some("cancelled") => Ok(VideoStatus::Failed {
            message: format!("MiniMax video generation was cancelled. Task ID: {task_id}"),
        }),
        Some("expired") => Ok(VideoStatus::Failed {
            message: format!("MiniMax video generation request expired. Task ID: {task_id}"),
        }),
        _ => Ok(VideoStatus::Pending),
    }
}

/// Fetch a finished video through the download transport, which pins DNS
/// to public addresses and sends no credentials, as TypeScript's core
/// fetches a result URL, reading at most 256 MiB. Bytes come back as they
/// are.
pub(super) async fn minimax_download(
    video: FilePart,
    provider: &dyn Provider,
) -> Result<FilePart, Error> {
    let FileData::Text(url) = &video.data else {
        return Ok(video);
    };
    if !media::is_downloadable_url(url) {
        return Ok(video);
    }
    let downloads = provider.download_transport().ok_or_else(|| {
        invalid(
            "a MiniMax video comes from another origin; build the provider with \
             with_download_transport or register it with register_with_downloads to fetch it",
        )
    })?;
    let (bytes, content_type) = files::download(downloads.as_ref(), url, VIDEO_MAX_BYTES)
        .await
        .map_err(|error| {
            let message = "MiniMax video download failed".to_owned();
            let failed = match error.status_code {
                Some(status) => {
                    ProviderError::from_status(message, status, Some(provider.name().to_owned()))
                }
                None => ProviderError::new(ProviderErrorKind::Response, message)
                    .with_provider(provider.name()),
            };
            Error::from(failed.with_source(error))
        })?;
    let media_type = content_type
        .filter(|value| value.starts_with("video/"))
        .unwrap_or_else(|| video.media_type.clone());
    Ok(FilePart {
        data: FileData::Bytes(bytes.to_vec()),
        media_type,
        ..video
    })
}
