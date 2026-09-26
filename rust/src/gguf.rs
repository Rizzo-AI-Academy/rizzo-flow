//! Minimal GGUF header reader.
//!
//! Enough to report the architecture a file declares, *before* loading it. This exists
//! because a failed load from the binding carries no useful detail: llama.cpp writes its
//! own message to stderr and the Rust error is a generic "null result", so a user with an
//! unsupported architecture gets no explanation (see REPORT.md §5.1 for the Spark case).
//!
//! Deliberately bounded and fail-open: on anything unexpected it returns `None` and the
//! caller falls back to the raw loader error.

use std::io::{BufReader, Read};

const MAGIC: &[u8; 4] = b"GGUF";
const MAX_STRING: u64 = 1 << 20;
/// The architecture is one of the first keys; stop scanning after this many.
const MAX_KEYS: u64 = 64;
/// Bound on array elements we are willing to skip over.
const MAX_ARRAY: u64 = 1 << 20;

const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;

fn read_u32<R: Read>(reader: &mut R) -> Option<u32> {
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer).ok()?;
    Some(u32::from_le_bytes(buffer))
}

fn read_u64<R: Read>(reader: &mut R) -> Option<u64> {
    let mut buffer = [0u8; 8];
    reader.read_exact(&mut buffer).ok()?;
    Some(u64::from_le_bytes(buffer))
}

fn read_string<R: Read>(reader: &mut R) -> Option<String> {
    let length = read_u64(reader)?;
    if length > MAX_STRING {
        return None;
    }
    let mut buffer = vec![0u8; length as usize];
    reader.read_exact(&mut buffer).ok()?;
    String::from_utf8(buffer).ok()
}

fn skip_bytes<R: Read>(reader: &mut R, count: u64) -> Option<()> {
    if count > MAX_STRING {
        return None;
    }
    let mut buffer = vec![0u8; count as usize];
    reader.read_exact(&mut buffer).ok()?;
    Some(())
}

/// Fixed width of the scalar GGUF value types, or `None` for string/array.
fn scalar_width(value_type: u32) -> Option<u64> {
    match value_type {
        0 | 1 | 7 => Some(1),  // uint8, int8, bool
        2 | 3 => Some(2),      // uint16, int16
        4 | 5 | 6 => Some(4),  // uint32, int32, float32
        10 | 11 | 12 => Some(8), // uint64, int64, float64
        _ => None,
    }
}

fn skip_value<R: Read>(reader: &mut R, value_type: u32) -> Option<()> {
    if let Some(width) = scalar_width(value_type) {
        return skip_bytes(reader, width);
    }
    if value_type == TYPE_STRING {
        let length = read_u64(reader)?;
        return skip_bytes(reader, length);
    }
    if value_type == TYPE_ARRAY {
        let element_type = read_u32(reader)?;
        let count = read_u64(reader)?;
        if count > MAX_ARRAY {
            return None;
        }
        for _ in 0..count {
            skip_value(reader, element_type)?;
        }
        return Some(());
    }
    None
}

/// The `general.architecture` value declared by a GGUF file, if it can be read.
pub fn architecture(path: &str) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = BufReader::new(file);
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).ok()?;
    if &magic != MAGIC {
        return None;
    }
    let version = read_u32(&mut reader)?;
    if !(2..=3).contains(&version) {
        return None;
    }
    let _tensor_count = read_u64(&mut reader)?;
    let kv_count = read_u64(&mut reader)?;
    for _ in 0..kv_count.min(MAX_KEYS) {
        let key = read_string(&mut reader)?;
        let value_type = read_u32(&mut reader)?;
        if key == "general.architecture" && value_type == TYPE_STRING {
            return read_string(&mut reader);
        }
        skip_value(&mut reader, value_type)?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn minimal_gguf(architecture: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor count
        bytes.extend_from_slice(&2u64.to_le_bytes()); // kv count
        // kv 1: general.name (string), which we must skip over
        let name = "test-model";
        bytes.extend_from_slice(&(4u64).to_le_bytes());
        bytes.extend_from_slice(b"name");
        bytes.extend_from_slice(&TYPE_STRING.to_le_bytes());
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        // kv 2: general.architecture (string)
        let key = "general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(&TYPE_STRING.to_le_bytes());
        bytes.extend_from_slice(&(architecture.len() as u64).to_le_bytes());
        bytes.extend_from_slice(architecture.as_bytes());
        bytes
    }

    fn write_temp(name: &str, bytes: &[u8]) -> String {
        let path = std::env::temp_dir().join(format!("rizzo-gguf-{}-{name}", std::process::id()));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn reads_the_declared_architecture() {
        let path = write_temp("spark.gguf", &minimal_gguf("spark2_5"));
        assert_eq!(architecture(&path).as_deref(), Some("spark2_5"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fails_open_on_garbage() {
        let path = write_temp("garbage.gguf", b"not a gguf file at all");
        assert_eq!(architecture(&path), None);
        assert_eq!(architecture("/nonexistent/file.gguf"), None);
        let _ = std::fs::remove_file(&path);
    }
}
