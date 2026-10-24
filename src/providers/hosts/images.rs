//! The image models of Together AI (`togetherai-image-model.ts`, HST-029).
//! Each call makes one image; `ops::generate_image` makes one call per
//! image (OPS-003). Warnings that TypeScript returns come back in
//! `Item.warnings`.
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Map, Value, json};

use crate::{
    Error, Model, Provider,
    providers::{
        http::{self, ApiProfile},
        operations::{ImageParams, ImagePrompt, Item, MediaInput, Warning},
    },
    types::{
        media,
        messages::{FileData, FilePart},
    },
};

/// A host's image model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Images {
    TogetherAI,
}

/// Generate the images of one call.
pub(crate) async fn generate(
    images: Images,
    api: &'static ApiProfile,
    model: &Model,
    prompt: ImagePrompt,
    params: ImageParams,
    provider: &dyn Provider,
) -> Result<Item<Vec<FilePart>>, Error> {
    match images {
        Images::TogetherAI => together(api, model, prompt, params, provider).await,
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Validation(message.into())
}

/// The host's entry of image options, as an object.
fn entry(params: &ImageParams, key: &str) -> Result<Map<String, Value>, Error> {
    match params.provider_options.get(key) {
        None => Ok(Map::new()),
        Some(Value::Object(entry)) => Ok(entry.clone()),
        Some(_) => Err(invalid(format!("{key} image options must be an object"))),
    }
}

/// Fail on an entry value of the wrong type. TypeScript's schemas take
/// `null` for these options too.
fn typed(
    entry: &Map<String, Value>,
    key: &str,
    option: &str,
    expected: &str,
    ok: fn(&Value) -> bool,
) -> Result<(), Error> {
    match entry.get(option) {
        Some(value) if !value.is_null() && !ok(value) => Err(invalid(format!(
            "{key} image option {option} must be {expected}, not {value}"
        ))),
        _ => Ok(()),
    }
}

/// A size `<width>x<height>`, as its two numbers.
fn size(size: &str) -> Result<(u64, u64), Error> {
    size.split_once('x')
        .and_then(|(width, height)| Some((width.parse().ok()?, height.parse().ok()?)))
        .ok_or_else(|| invalid(format!("image size must be <width>x<height>, not {size:?}")))
}

/// An input image as a URL: a URL as given, bytes or a file as a data URL
/// (`convertImageModelFileToDataUri`).
fn data_url(input: &MediaInput) -> String {
    let encoded = |media_type: &str, bytes: &[u8]| {
        format!("data:{media_type};base64,{}", STANDARD.encode(bytes))
    };
    match input {
        MediaInput::Url(url) => url.clone(),
        MediaInput::Bytes(bytes) => encoded(
            media::detect_image_media_type(bytes.as_slice()).unwrap_or("application/octet-stream"),
            bytes,
        ),
        MediaInput::File(file) => match &file.data {
            FileData::Text(text) if media::is_url(text) => text.clone(),
            FileData::Text(text) => format!("data:{};base64,{text}", file.media_type),
            FileData::Bytes(bytes) => encoded(&file.media_type, bytes),
        },
    }
}

fn warning(kind: &str, feature: Option<&str>, message: &str) -> Warning {
    let (details, message) = match feature {
        Some(_) => (Some(message.to_owned()), None),
        None => (None, Some(message.to_owned())),
    };
    Warning {
        kind: kind.into(),
        feature: feature.map(str::to_owned),
        details,
        message,
        ..Warning::default()
    }
}

/// An image the response holds as bytes, with the media type its bytes
/// show, PNG when they show none.
fn image(provider: &dyn Provider, bytes: Vec<u8>) -> Result<FilePart, Error> {
    let media_type = media::detect_image_media_type(bytes.as_slice()).unwrap_or("image/png");
    FilePart::from_bytes(bytes, Some(media_type), None)
        .map_err(|e| http::response_error(provider.name(), format!("image response: {e}")))
}

/// An image the response holds as base64 text.
fn base64_image(provider: &dyn Provider, text: &str) -> Result<FilePart, Error> {
    let bytes = STANDARD.decode(text).map_err(|_| {
        http::response_error(provider.name(), "image response holds invalid base64")
    })?;
    image(provider, bytes)
}

/// `POST <base>/images/generations`, as `togetherai-image-model.ts` sends it.
async fn together(
    api: &'static ApiProfile,
    model: &Model,
    prompt: ImagePrompt,
    params: ImageParams,
    provider: &dyn Provider,
) -> Result<Item<Vec<FilePart>>, Error> {
    if prompt.mask.is_some() {
        return Err(invalid(
            "Together AI does not support mask-based image editing; use a FLUX Kontext model, \
             such as black-forest-labs/FLUX.1-kontext-pro, with a reference image and a prompt",
        ));
    }
    let mut options = entry(&params, "togetherai")?;
    typed(
        &options,
        "togetherai",
        "steps",
        "a number",
        Value::is_number,
    )?;
    typed(
        &options,
        "togetherai",
        "guidance",
        "a number",
        Value::is_number,
    )?;
    typed(
        &options,
        "togetherai",
        "negative_prompt",
        "a string",
        Value::is_string,
    )?;
    typed(
        &options,
        "togetherai",
        "disable_safety_checker",
        "a boolean",
        Value::is_boolean,
    )?;
    let mut warnings = Vec::new();
    if params.aspect_ratio.is_some() {
        warnings.push(warning(
            "unsupported",
            Some("aspectRatio"),
            "This model does not support the `aspectRatio` option. Use `size` instead.",
        ));
    }
    // A model that does not diffuse takes no seed or diffusion settings.
    let diffusion = model.id != "google/gemini-3-pro-image";
    if !diffusion {
        for option in [
            "steps",
            "guidance",
            "negative_prompt",
            "disable_safety_checker",
        ] {
            options.remove(option);
        }
        if params.seed.is_some() {
            warnings.push(warning(
                "unsupported",
                Some("seed"),
                &format!("The {} model does not support the `seed` option.", model.id),
            ));
        }
    }
    if prompt.images.len() > 1 {
        warnings.push(warning(
            "other",
            None,
            "Together AI only supports a single input image. Additional images are ignored.",
        ));
    }
    let mut body = json!({"model": model.id});
    if let Some(text) = &prompt.text {
        body["prompt"] = json!(text);
    }
    if let Some(seed) = params.seed.filter(|_| diffusion) {
        body["seed"] = json!(seed);
    }
    if params.n > 1 {
        body["n"] = json!(params.n);
    }
    if let Some(value) = &params.size {
        let (width, height) = size(value)?;
        body["width"] = json!(width);
        body["height"] = json!(height);
    }
    if let Some(first) = prompt.images.first() {
        body["image_url"] = json!(data_url(first));
    }
    body["response_format"] = json!("base64");
    if let Value::Object(fields) = &mut body {
        fields.extend(options);
    }
    let request = http::request(
        api,
        provider.config(),
        "POST",
        &["images", "generations"],
        Some(body),
        Default::default(),
        Default::default(),
    )?;
    let response = http::json(
        api,
        transport(provider)?,
        request,
        provider.name(),
        Some(&model.id),
    )
    .await?;
    let images = response["data"]
        .as_array()
        .ok_or_else(|| {
            http::response_error(provider.name(), "Together AI image response has no data")
        })?
        .iter()
        .map(|item| {
            let text = item["b64_json"].as_str().ok_or_else(|| {
                http::response_error(
                    provider.name(),
                    "Together AI image response has no b64_json",
                )
            })?;
            base64_image(provider, text)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let mut item = Item::new(images);
    item.warnings = warnings;
    Ok(item)
}

fn transport(provider: &dyn Provider) -> Result<&dyn crate::transport::HttpTransport, Error> {
    provider.transport().ok_or_else(|| {
        Error::Configuration(format!(
            "provider {:?} has no transport for images",
            provider.name()
        ))
    })
}
