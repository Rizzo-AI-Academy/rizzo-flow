//! Chat formats rendered by this port itself.
//!
//! Why this exists: llama.cpp's C function `llama_chat_apply_template` only renders
//! templates it *recognises* (chatml, llama3, …). Given anything else — including a model's
//! own custom Jinja template — it returns -1. `chatml`-style templates happen to be
//! detected and rendered; Spark-X2.5's template is not, so through that API it could not
//! be applied at all and the port silently fell back to plain text, sending the model a
//! prompt different from the reference's (prompt_sha256 differed on every question).
//!
//! The formats below are therefore rendered here. Their markers and the injected system
//! line were taken from the reference rendering of the model's own template, so a prompt
//! compiled here is byte-identical to the reference's — which is checked by comparing
//! `prompt_sha256` (see REPORT.md §3.11).

/// Format ids accepted by `--chat-template` (as the file's content).
pub const BUILTIN_FORMATS: [&str; 1] = ["spark-x2.5-nonthinking"];

// Spark-X2.5's sentence markers, written as escapes so the source stays ASCII-only.
const BOS: &str = "<\u{FF5C}start\u{2581}of\u{2581}sentence\u{FF5C}>";
const EOS: &str = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>";

/// Render `system` + `user` in one of the built-in formats.
pub fn render(format: &str, system: &str, user: &str) -> String {
    match format {
        // Spark-X2.5 with thinking disabled, the configuration the reference uses.
        // The template injects its own first system line before the caller's content, and
        // closes the thinking block right after the assistant marker.
        "spark-x2.5-nonthinking" => format!(
            "{BOS}<|System|>\nyou are a helpful assistant.\n\n{system}{EOS}\
             {BOS}<|User|>{user}{EOS}\
             {BOS}<|Bot|></think>"
        ),
        other => {
            // Only reachable if BUILTIN_FORMATS and this match disagree.
            eprintln!("rizzo: warning: unknown built-in chat format {other:?}; using plain text");
            format!("{system}\n\n{user}\n\n")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spark_format_matches_the_reference_rendering() {
        // Ground truth: the reference tokenizer's rendering of the same two messages with
        // add_generation_prompt=True and enable_thinking=False.
        let expected = format!(
            "{BOS}<|System|>\nyou are a helpful assistant.\n\nSYS{EOS}{BOS}<|User|>USER{EOS}{BOS}<|Bot|></think>"
        );
        assert_eq!(render("spark-x2.5-nonthinking", "SYS", "USER"), expected);
    }

    #[test]
    fn unknown_format_degrades_loudly_to_plain_text() {
        let rendered = render("does-not-exist", "SYS", "USER");
        assert_eq!(rendered, "SYS\n\nUSER\n\n");
        assert!(BUILTIN_FORMATS.contains(&"spark-x2.5-nonthinking"));
    }
}
