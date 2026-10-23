//! What an OpenAI model takes, by its id, as
//! `openai-language-model-capabilities.ts` decides it. A setting a model does
//! not take is left out and logged at debug (OAI-005).

/// What OpenAI's API takes for one model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Capabilities {
    /// `o<n>`, or `gpt-<n>` with `n` at least 5 other than a chat variant
    /// without a minor version.
    pub reasoning: bool,
    pub flex: bool,
    pub priority: bool,
    /// GPT-6 and later: `configuration_update` items and `async` tools.
    pub gpt6: bool,
    /// The reasoning efforts the model takes, when OpenAI lists them.
    pub efforts: Option<&'static [&'static str]>,
    /// Temperature, top-p and logprobs go out when the effort is `none`.
    pub non_reasoning_params: bool,
}

const SOL_LUNA_EFFORTS: &[&str] = &["none", "low", "medium", "high", "xhigh", "max"];
const GPT6_EFFORTS: &[&str] = &["low", "medium", "high", "xhigh", "max"];

/// A `gpt-<major>[.<minor>][-<variant>]` model id, split.
struct GptVersion<'a> {
    major: u64,
    minor: Option<u64>,
    variant: Option<&'a str>,
}

fn digits(text: &str) -> usize {
    text.len() - text.trim_start_matches(|c: char| c.is_ascii_digit()).len()
}

/// A number too long for u64 is still larger than any version compared here.
fn number(text: &str) -> u64 {
    text.parse().unwrap_or(u64::MAX)
}

/// The `n` of an `o<n>` model id followed by `-` or the end.
fn o_series(model: &str) -> Option<u64> {
    let rest = model.strip_prefix('o')?;
    let n = digits(rest);
    (n > 0 && (rest.len() == n || rest[n..].starts_with('-'))).then(|| number(&rest[..n]))
}

fn gpt_version(model: &str) -> Option<GptVersion<'_>> {
    let rest = model.strip_prefix("gpt-")?;
    let n = digits(rest);
    if n == 0 {
        return None;
    }
    let major = number(&rest[..n]);
    let mut rest = &rest[n..];
    let mut minor = None;
    if let Some(after) = rest.strip_prefix('.') {
        let n = digits(after);
        if n == 0 {
            return None;
        }
        minor = Some(number(&after[..n]));
        rest = &after[n..];
    }
    let variant = match rest.strip_prefix('-') {
        _ if rest.is_empty() => None,
        Some(variant) if !variant.is_empty() => Some(variant),
        _ => return None,
    };
    Some(GptVersion {
        major,
        minor,
        variant,
    })
}

impl Capabilities {
    pub(crate) fn of(model: &str) -> Self {
        let o = o_series(model);
        let gpt = gpt_version(model);
        let chat = gpt
            .as_ref()
            .is_some_and(|v| v.minor.is_none() && v.variant.is_some_and(|v| v.starts_with("chat")));
        let nano = gpt
            .as_ref()
            .is_some_and(|v| v.variant.is_some_and(|v| v.starts_with("nano")));
        let major = gpt.as_ref().map(|v| v.major);
        let gpt6 = major.is_some_and(|m| m >= 6);
        let sol_or_luna = model == "gpt-6-sol" || model == "gpt-6-luna";
        let gpt5_up = major.is_some_and(|m| m >= 5) && !chat;
        let o3_up = o.is_some_and(|n| n >= 3);
        Self {
            reasoning: o.is_some() || gpt5_up,
            flex: o3_up || gpt5_up,
            priority: model.starts_with("gpt-4") || (gpt5_up && !nano) || o3_up,
            gpt6,
            efforts: if sol_or_luna {
                Some(SOL_LUNA_EFFORTS)
            } else if gpt6 {
                Some(GPT6_EFFORTS)
            } else {
                None
            },
            non_reasoning_params: sol_or_luna
                || gpt.as_ref().is_some_and(|v| {
                    !gpt6 && (v.major > 5 || (v.major == 5 && v.minor.unwrap_or(0) >= 1))
                }),
        }
    }

    /// Whether the model takes this reasoning effort; any effort when
    /// OpenAI lists none for it.
    pub(crate) fn takes_effort(&self, effort: &str) -> bool {
        self.efforts.is_none_or(|efforts| efforts.contains(&effort))
    }
}

/// Log a setting left out of a request because the model does not take it.
pub(crate) fn dropped(setting: &str, model: &str, reason: &str) {
    tracing::debug!(
        target: "suprnova_ai::openai",
        setting,
        model,
        "{setting} is left out for {model}: {reason}"
    );
}

#[cfg(test)]
mod tests {
    use super::Capabilities;

    #[test]
    fn capabilities_follow_the_typescript_rules() {
        let caps = Capabilities::of;
        assert!(caps("o3").reasoning && caps("o3").flex && caps("o3").priority);
        assert!(caps("o1").reasoning && !caps("o1").flex && !caps("o1").priority);
        assert!(!caps("gpt-4.1").reasoning && caps("gpt-4.1").priority);
        assert!(!caps("gpt-4.1").flex);
        assert!(caps("gpt-5-nano").flex && !caps("gpt-5-nano").priority);
        assert!(!caps("gpt-5-chat-latest").reasoning && !caps("gpt-5-chat-latest").flex);
        assert!(caps("gpt-5.1-chat-latest").reasoning);
        assert!(caps("gpt-5.4").non_reasoning_params && !caps("gpt-5").non_reasoning_params);
        assert!(
            caps("gpt-6-sol").non_reasoning_params && !caps("gpt-6-astra").non_reasoning_params
        );
        assert!(!caps("gpt-6.1-sol").non_reasoning_params);
        assert!(caps("gpt-6-astra").gpt6 && !caps("gpt-5.6").gpt6);
        assert!(
            caps("gpt-6-sol").takes_effort("none") && !caps("gpt-6-astra").takes_effort("none")
        );
        assert!(
            !caps("gpt-6-astra").takes_effort("minimal") && caps("gpt-5.4").takes_effort("minimal")
        );
        assert!(!caps("gpt-5..1").reasoning && !caps("o3x").reasoning);
    }
}
