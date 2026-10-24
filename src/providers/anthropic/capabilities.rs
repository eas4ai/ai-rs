//! What a Claude model takes, by its id, as `getModelCapabilities` in
//! `anthropic-language-model.ts` gives it (ANT-004).

/// The capabilities of one model id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Capabilities {
    pub max_output_tokens: u64,
    pub structured_output: bool,
    pub adaptive_thinking: bool,
    pub rejects_sampling: bool,
    pub xhigh_effort: bool,
    pub rejects_disabled_above_high: bool,
    pub rejects_disabled_thinking: bool,
    pub rejects_budget_thinking: bool,
    pub rejects_forced_tool_use: bool,
    pub between_tools_thinking: bool,
    pub known: bool,
}

/// A newer model: adaptive thinking and structured output, no sampling.
const NEWER: Capabilities = Capabilities {
    max_output_tokens: 128_000,
    structured_output: true,
    adaptive_thinking: true,
    rejects_sampling: true,
    xhigh_effort: true,
    rejects_disabled_above_high: false,
    rejects_disabled_thinking: false,
    rejects_budget_thinking: false,
    rejects_forced_tool_use: false,
    between_tools_thinking: false,
    known: true,
};

/// An older model: budget thinking and sampling, no structured output.
const OLDER: Capabilities = Capabilities {
    max_output_tokens: 4096,
    structured_output: false,
    adaptive_thinking: false,
    rejects_sampling: false,
    xhigh_effort: false,
    rejects_disabled_above_high: false,
    rejects_disabled_thinking: false,
    rejects_budget_thinking: false,
    rejects_forced_tool_use: false,
    between_tools_thinking: false,
    known: true,
};

/// Whether `id` holds `prefix` followed by `-` or `@`.
fn family(id: &str, prefix: &str) -> bool {
    id.match_indices(prefix)
        .any(|(at, _)| matches!(id.as_bytes().get(at + prefix.len()), Some(b'-' | b'@')))
}

/// Whether `id` names a legacy Claude model that TypeScript does not know:
/// `claude-instant`, `claude-2` or `claude-v2`, or `claude-3` without a
/// known family (`/claude-(?:instant(?:-|$)|v?2(?=$|[-.:])|3(?=$|[-.]))/`).
fn legacy(id: &str) -> bool {
    id.match_indices("claude-").any(|(at, prefix)| {
        let rest = &id[at + prefix.len()..];
        let ends =
            |rest: &str, stops: &[u8]| rest.is_empty() || stops.contains(&rest.as_bytes()[0]);
        if let Some(rest) = rest.strip_prefix("instant") {
            return ends(rest, b"-");
        }
        let rest2 = rest.strip_prefix('v').unwrap_or(rest);
        if let Some(rest2) = rest2.strip_prefix('2')
            && ends(rest2, b"-.:")
        {
            return true;
        }
        rest.strip_prefix('3').is_some_and(|rest| ends(rest, b"-."))
    })
}

impl Capabilities {
    pub(crate) fn of(id: &str) -> Self {
        let has = |name: &str| id.contains(name);
        if has("claude-sonnet-5-5") {
            Self {
                rejects_disabled_above_high: true,
                rejects_disabled_thinking: true,
                rejects_budget_thinking: true,
                rejects_forced_tool_use: true,
                between_tools_thinking: true,
                ..NEWER
            }
        } else if has("claude-opus-5-5") {
            Self {
                rejects_disabled_above_high: true,
                rejects_disabled_thinking: true,
                rejects_budget_thinking: true,
                rejects_forced_tool_use: true,
                ..NEWER
            }
        } else if has("claude-haiku-5-5") {
            Self {
                rejects_disabled_above_high: true,
                rejects_budget_thinking: true,
                ..NEWER
            }
        } else if has("claude-opus-5") {
            Self {
                rejects_disabled_above_high: true,
                ..NEWER
            }
        } else if has("claude-fable-5-1") {
            Self {
                rejects_disabled_thinking: true,
                rejects_budget_thinking: true,
                rejects_forced_tool_use: true,
                ..NEWER
            }
        } else if has("claude-fable-5") {
            Self {
                rejects_disabled_thinking: true,
                rejects_budget_thinking: true,
                ..NEWER
            }
        } else if has("claude-opus-4-8") || has("claude-opus-4-7") || has("claude-sonnet-5") {
            NEWER
        } else if has("claude-sonnet-4-6") || has("claude-opus-4-6") {
            Self {
                rejects_sampling: false,
                xhigh_effort: false,
                ..NEWER
            }
        } else if has("claude-sonnet-4-5") || has("claude-opus-4-5") || has("claude-haiku-4-5") {
            Self {
                max_output_tokens: 64_000,
                structured_output: true,
                ..OLDER
            }
        } else if has("claude-opus-4-1") {
            Self {
                max_output_tokens: 32_000,
                structured_output: true,
                ..OLDER
            }
        } else if family(id, "claude-sonnet-4") {
            Self {
                max_output_tokens: 64_000,
                ..OLDER
            }
        } else if family(id, "claude-opus-4") {
            Self {
                max_output_tokens: 32_000,
                ..OLDER
            }
        } else if has("claude-3-haiku") {
            OLDER
        } else if legacy(id) {
            Self {
                known: false,
                ..OLDER
            }
        } else if has("claude-") {
            // A Claude id newer than the known list.
            Self {
                rejects_disabled_above_high: true,
                known: false,
                ..NEWER
            }
        } else {
            // A model of another vendor behind the Messages protocol.
            Self {
                known: false,
                ..OLDER
            }
        }
    }

    /// Whether TypeScript treats the id as Anthropic's own model.
    pub(crate) fn claude(id: &str) -> bool {
        Self::of(id).known || id.contains("claude-")
    }
}

/// The settings and history parts of one request that the crate changed or
/// left out where TypeScript warns about them, each logged at debug
/// (ANT-004), and its cache breakpoints, at most four (ANT-015).
pub(crate) struct Notes {
    model: String,
    changes: Vec<(String, String)>,
    breakpoints: usize,
}
impl Notes {
    pub(crate) fn new(model: &str) -> Self {
        Self {
            model: model.to_owned(),
            changes: Vec::new(),
            breakpoints: 0,
        }
    }
    /// Log a change and keep it.
    pub(crate) fn note(&mut self, setting: &str, details: impl Into<String>) {
        let details = details.into();
        let model = &self.model;
        tracing::debug!(
            target: "suprnova_ai::anthropic",
            setting,
            model = model.as_str(),
            "{setting} for {model}: {details}"
        );
        self.changes.push((setting.to_owned(), details));
    }
    /// The changes, as setting and details.
    pub(crate) fn into_changes(self) -> Vec<(String, String)> {
        self.changes
    }
    /// The breakpoint to send on a block of this kind, counting it; `None`
    /// with a note where the block takes none or four came before it.
    pub(crate) fn cache(
        &mut self,
        control: Option<serde_json::Value>,
        context: &str,
        can_cache: bool,
    ) -> Option<serde_json::Value> {
        let control = control.filter(|c| !c.is_null())?;
        if !can_cache {
            self.note(
                "cacheControl",
                format!("cache_control cannot be set on {context}; it is left out"),
            );
            return None;
        }
        self.breakpoints += 1;
        if self.breakpoints > 4 {
            let found = self.breakpoints;
            self.note(
                "cacheControl",
                format!(
                    "at most 4 cache breakpoints are taken (found {found}); this one is left out"
                ),
            );
            return None;
        }
        Some(control)
    }
}

#[cfg(test)]
mod tests {
    use super::Capabilities;

    #[test]
    fn capabilities_follow_the_typescript_table() {
        let of = Capabilities::of;
        assert!(of("claude-sonnet-5-5").between_tools_thinking);
        assert!(!of("claude-opus-5-5").between_tools_thinking);
        assert!(of("claude-opus-5-5").rejects_forced_tool_use);
        assert!(!of("claude-haiku-5-5").rejects_disabled_thinking);
        assert!(of("claude-haiku-5-5").rejects_disabled_above_high);
        assert!(
            of("claude-opus-5").rejects_sampling && !of("claude-opus-5").rejects_budget_thinking
        );
        assert!(of("claude-fable-5-1").rejects_forced_tool_use);
        assert!(!of("claude-fable-5").rejects_forced_tool_use);
        assert!(!of("claude-sonnet-4-6").rejects_sampling);
        assert!(!of("claude-sonnet-4-6").xhigh_effort);
        assert_eq!(of("claude-sonnet-4-5").max_output_tokens, 64_000);
        assert!(!of("claude-sonnet-4-5").adaptive_thinking);
        assert_eq!(of("claude-opus-4-1").max_output_tokens, 32_000);
        assert!(!of("claude-sonnet-4-20250514").structured_output);
        assert!(!of("claude-sonnet-4@20250514").structured_output);
        assert_eq!(of("claude-opus-4-20250514").max_output_tokens, 32_000);
        assert!(of("claude-3-haiku-20240307").known);
        assert!(!of("claude-3-opus-20240229").known);
        assert!(!of("claude-2.1").known);
        assert!(!of("claude-instant-1.2").known);
        assert!(!of("claude-v2").known);
        assert!(!of("claude-30").known && of("claude-30").adaptive_thinking);
        assert!(!of("claude-next").known && of("claude-next").max_output_tokens == 128_000);
        assert_eq!(of("MiniMax-M2").max_output_tokens, 4096);
        assert!(!of("MiniMax-M2").structured_output);
        assert!(Capabilities::claude("claude-next"));
        assert!(!Capabilities::claude("MiniMax-M2"));
    }
}
