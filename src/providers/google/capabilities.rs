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

//! What a Gemini model id says about the request shape it takes, as
//! `google-model-capabilities.ts` and `google-language-model.ts` classify
//! ids. Unknown Gemini ids inherit the newest behavior.

/// The byte offsets where `name` starts a segment of `id`: at the start or
/// after `/` or `.`, ignoring ASCII case.
fn segment_starts<'a>(id: &'a str, name: &'a str) -> impl Iterator<Item = usize> + 'a {
    let lower = id.to_ascii_lowercase();
    let name = name.to_ascii_lowercase();
    lower
        .match_indices(&name)
        .map(|(at, _)| at)
        .filter(move |at| *at == 0 || matches!(id.as_bytes()[at - 1], b'/' | b'.'))
        .collect::<Vec<_>>()
        .into_iter()
}

/// `(^|[/.])<prefix>(?:[.-]|$)`, ignoring case.
fn has_version(id: &str, prefix: &str) -> bool {
    segment_starts(id, prefix).any(|at| {
        matches!(
            id.as_bytes().get(at + prefix.len()),
            None | Some(b'.' | b'-')
        )
    })
}

/// `(^|[/.])gemini-`, ignoring case.
pub(crate) fn is_gemini(id: &str) -> bool {
    segment_starts(id, "gemini-").next().is_some()
}

/// `(^|/)gemini-`: a Gemini id at the start or after a path segment.
pub(crate) fn is_gemini_path(id: &str) -> bool {
    id.starts_with("gemini-") || id.contains("/gemini-")
}

/// A Gemini 2.5 model: `gemini-2.5` followed by `.`, `-` or the end.
pub(crate) fn is_gemini_25(id: &str) -> bool {
    has_version(id, "gemini-2.5")
}

fn is_pre_gemini_2(id: &str) -> bool {
    has_version(id, "gemini-1")
        || segment_starts(id, "gemini-pro").any(|at| {
            let rest = &id[at + "gemini-pro".len()..];
            rest.is_empty() || rest.eq_ignore_ascii_case("-vision")
        })
        || has_version(id, "gemini-robotics-er-1.5")
}

/// A model with Gemini 3 features: a Gemini id that is not a Gemini 1 or 2
/// model, `gemini-pro`, `gemini-pro-vision` or `gemini-robotics-er-1.5`.
pub(crate) fn uses_gemini_3_features(id: &str) -> bool {
    is_gemini(id) && !is_pre_gemini_2(id) && !has_version(id, "gemini-2")
}

/// Whether the built-in tools of Gemini 2 work with the model.
pub(crate) fn supports_gemini_2_tools(id: &str) -> bool {
    (is_gemini(id) && !is_pre_gemini_2(id)) || id.to_ascii_lowercase().contains("nano-banana")
}

/// Whether the file search tool works with the model.
pub(crate) fn supports_file_search(id: &str) -> bool {
    is_gemini_25(id) || uses_gemini_3_features(id)
}

/// Whether Gemini fetches an `https` URL of a listed media type itself:
/// `gemini-` models other than Gemini 2.0.
pub(crate) fn fetches_external_urls(id: &str) -> bool {
    is_gemini_path(id) && !(id.starts_with("gemini-2.0") || id.contains("/gemini-2.0"))
}

/// A Gemma model takes no system instruction.
pub(crate) fn is_gemma(id: &str) -> bool {
    id.to_ascii_lowercase().starts_with("gemma-")
}

/// Whether reasoning is sent as a thinking level rather than a budget.
pub(crate) fn uses_thinking_level(id: &str) -> bool {
    uses_gemini_3_features(id) && !id.contains("gemini-3-pro-image")
}

/// The last `gemini-` name of an id, lowercased.
fn gemini_name(id: &str) -> Option<String> {
    let at = segment_starts(id, "gemini-").last()?;
    let name = &id[at..];
    (!name.contains('/')).then(|| name.to_ascii_lowercase())
}

/// The lowest thinking level of a model with Gemini 3 features: `low` for
/// `gemini-flash-latest`, Pro models from version 3 and Flash models (not
/// Lite) from version 3.7, `minimal` otherwise.
pub(crate) fn minimum_thinking_level(id: &str) -> &'static str {
    let Some(name) = gemini_name(id) else {
        return "minimal";
    };
    if name == "gemini-flash-latest" {
        return "low";
    }
    let rest = &name["gemini-".len()..];
    let digits = |text: &str| -> Option<(u32, usize)> {
        let end = text
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(text.len());
        text[..end].parse().ok().map(|n| (n, end))
    };
    // gemini-<n>[.<m>]-pro, then the end or `-`.
    if let Some((major, end)) = digits(rest) {
        let mut tail = &rest[end..];
        if let Some(after) = tail.strip_prefix('.')
            && let Some((_, minor_end)) = digits(after)
        {
            tail = &after[minor_end..];
        }
        if let Some(after) = tail.strip_prefix("-pro")
            && (after.is_empty() || after.starts_with('-'))
            && major >= 3
        {
            return "low";
        }
    }
    // gemini-<n>.<m>-flash, then the end or `-` not followed by `lite`.
    let Some((major, end)) = digits(rest) else {
        return "minimal";
    };
    let Some(after) = rest[end..].strip_prefix('.') else {
        return "minimal";
    };
    let Some((minor, minor_end)) = digits(after) else {
        return "minimal";
    };
    let Some(tail) = after[minor_end..].strip_prefix("-flash") else {
        return "minimal";
    };
    let lite = tail
        .strip_prefix('-')
        .is_some_and(|t| t == "lite" || t.starts_with("lite-"));
    if !(tail.is_empty() || tail.starts_with('-')) || lite {
        return "minimal";
    }
    if major > 3 || (major == 3 && minor >= 7) {
        "low"
    } else {
        "minimal"
    }
}

/// The thinking budget cap of a model that takes a budget.
pub(crate) fn max_thinking_budget(id: &str) -> u64 {
    let lower = id.to_ascii_lowercase();
    if lower.contains("2.5-pro") || lower.contains("gemini-3-pro-image") {
        32768
    } else {
        24576
    }
}

/// The path of a model in request URLs: an id with `/` as it is, otherwise
/// `models/<id>`.
pub(crate) fn model_path(id: &str) -> String {
    if id.contains('/') {
        id.to_owned()
    } else {
        format!("models/{id}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_3_features_exclude_older_generations() {
        for id in [
            "gemini-3-flash-preview",
            "gemini-3.1-pro-preview",
            "models/gemini-4-argon",
            "gemini-flash-latest",
        ] {
            assert!(uses_gemini_3_features(id), "{id}");
        }
        for id in [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2",
            "gemini-1.5-pro",
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-robotics-er-1.5-preview",
            "gemma-3-27b-it",
        ] {
            assert!(!uses_gemini_3_features(id), "{id}");
        }
    }

    #[test]
    fn the_lowest_thinking_level_follows_the_model_family() {
        assert_eq!(minimum_thinking_level("gemini-3-flash-preview"), "minimal");
        assert_eq!(minimum_thinking_level("gemini-3-pro-preview"), "low");
        assert_eq!(minimum_thinking_level("gemini-3.1-pro-preview"), "low");
        assert_eq!(minimum_thinking_level("gemini-flash-latest"), "low");
        assert_eq!(minimum_thinking_level("gemini-3.7-flash"), "low");
        assert_eq!(minimum_thinking_level("gemini-3.8-flash"), "low");
        assert_eq!(minimum_thinking_level("gemini-3.6-flash"), "minimal");
        assert_eq!(minimum_thinking_level("gemini-3.8-flash-lite"), "minimal");
        assert_eq!(minimum_thinking_level("gemini-4.0-flash"), "low");
    }

    #[test]
    fn gemini_2_5_matches_the_version_segment() {
        assert!(is_gemini_25("gemini-2.5-flash"));
        assert!(is_gemini_25("models/gemini-2.5-pro"));
        assert!(is_gemini_25("gemini-2.5"));
        assert!(!is_gemini_25("gemini-2.50-flash"));
        assert!(!is_gemini_25("gemini-3-flash"));
    }
}
