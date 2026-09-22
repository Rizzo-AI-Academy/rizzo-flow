//! JSON artefacts I/O — parity with the Python `cli.write_json` / `cli.read_jsonl`.
//!
//! `write_json` is **create-only**: results are evidence, they are never silently
//! overwritten (the Python side opens the file with mode "x").

use serde_json::Value;

use crate::schema::ValidationError;

pub type Result<T> = std::result::Result<T, ValidationError>;

/// Read a JSONL file into a list of values (blank lines ignored).
pub fn read_jsonl(path: &str) -> Result<Vec<Value>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| ValidationError(format!("{path}: {e}")))?;
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str(line).map_err(|e| ValidationError(format!("{path}: {e}")))
        })
        .collect()
}

/// Write a JSON value to stdout, or to a new file (fails if it already exists).
pub fn write_json(value: &Value, destination: Option<&str>) -> Result<()> {
    let text = format!(
        "{}\n",
        serde_json::to_string_pretty(value)
            .map_err(|e| ValidationError(format!("json: {e}")))?
    );
    let Some(path) = destination else {
        print!("{text}");
        return Ok(());
    };
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ValidationError(format!("{}: {e}", parent.display())))?;
        }
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| ValidationError(format!("{path}: {e}")))?;
    file.write_all(text.as_bytes())
        .map_err(|e| ValidationError(format!("{path}: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn write_json_refuses_to_overwrite() {
        let dir = std::env::temp_dir().join(format!("rizzo-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.json");
        let _ = std::fs::remove_file(&path);
        write_json(&json!({"a": 1}), Some(path.to_str().unwrap())).unwrap();
        // second write must fail: results are create-only
        assert!(write_json(&json!({"a": 2}), Some(path.to_str().unwrap())).is_err());
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text, "{\n  \"a\": 1\n}\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_jsonl_skips_blank_lines() {
        let dir = std::env::temp_dir().join(format!("rizzo-io2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rows.jsonl");
        std::fs::write(&path, "{\"a\":1}\n\n  \n{\"b\":2}\n").unwrap();
        let rows = read_jsonl(path.to_str().unwrap()).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["b"], 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
