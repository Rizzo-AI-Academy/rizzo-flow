//! Pinned checkpoints and reference-CLI option mapping, shared by the subcommands.
//!
//! The reference project selects a checkpoint with `--size` and `--quant` (its own
//! GGUF conversions of Spark-X2.5, pinned by revision **and** sha256) and a compute
//! backend with `--backend` / `--device`. This port mirrors that surface:
//!
//! * `--size`/`--quant` resolve to the same pinned GGUF files the reference uses, at
//!   the same `models/<repo>/<file>` location;
//! * `--device` maps onto llama.cpp GPU offload layers — `auto`/`gpu`/`cuda` offload
//!   everything, `cpu` offloads nothing, `mlx` is refused with an explanation;
//! * `--backend mlx` is refused (declared difference: this port is llama.cpp only);
//! * `--bits` is MLX-only in the reference, so it is accepted and reported as ignored;
//! * `--threads` is real: it sets the llama.cpp CPU thread count.
//!
//! `--batch-size` is accepted and reported as ignored: this engine decodes the
//! questions of a request sequentially.

pub struct GgufSpec {
    pub size: &'static str,
    pub quant: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub file: &'static str,
    pub sha256: &'static str,
}

const _SPARK_4B: (&str, &str) = (
    "XHToken/Spark-X2.5-4B-GGUF",
    "9826e0be84e6e6e8b9668abc91421109a1df1e2d",
);
const _SPARK_17B: (&str, &str) = (
    "XHToken/Spark-X2.5-1.7B-GGUF",
    "1f7fa33b1245c14730da39e125714ad3a327901b",
);

/// The checkpoint family this port is validated against (see REPORT.md §3.6).
/// Revisions and hashes match the reference project's own pinned table.
pub const GGUF: &[GgufSpec] = &[
    GgufSpec {
        size: "4b",
        quant: "q8_0",
        repo: _SPARK_4B.0,
        revision: _SPARK_4B.1,
        file: "Spark-X2.5-4B-Q8_0.gguf",
        sha256: "5c2c3c190e4337e1016b8593ca8e26e8b18c972200b107385d4ec61a25d9dea2",
    },
    GgufSpec {
        size: "4b",
        quant: "q4_k_m",
        repo: _SPARK_4B.0,
        revision: _SPARK_4B.1,
        file: "Spark-X2.5-4B-Q4_K_M.gguf",
        sha256: "adfcfa19a4ed6a5985da8bf565fe15f8e1a7e131d79bae2d19d48d1c40109428",
    },
    GgufSpec {
        size: "4b",
        quant: "bf16",
        repo: _SPARK_4B.0,
        revision: _SPARK_4B.1,
        file: "Spark-X2.5-4B.gguf",
        sha256: "8cecf405a41a4a10f833530910c2e13fde9fb39c325c8afc3c5d10e4181e1a14",
    },
    GgufSpec {
        size: "1.7b",
        quant: "q8_0",
        repo: _SPARK_17B.0,
        revision: _SPARK_17B.1,
        file: "Spark-X2.5-1.7B-Q8_0.gguf",
        sha256: "cd77c03185a834bb1162a4b7713520be5838058bfc54873645beff470bb24442",
    },
    GgufSpec {
        size: "1.7b",
        quant: "q4_k_m",
        repo: _SPARK_17B.0,
        revision: _SPARK_17B.1,
        file: "Spark-X2.5-1.7B-Q4_K_M.gguf",
        sha256: "902bde2522394954ac17821b3e5fd0df02defbc6944f122253f2580acf0503f4",
    },
    GgufSpec {
        size: "1.7b",
        quant: "bf16",
        repo: _SPARK_17B.0,
        revision: _SPARK_17B.1,
        file: "Spark-X2.5-1.7B.gguf",
        sha256: "67d5f2f06e6d898efcf0dc40cab8528bc82b871c8dafb0936784183d2c10cdd9",
    },
];

pub const DEFAULT_SIZE: &str = "4b";
pub const DEFAULT_QUANT: &str = "q8_0";
/// The device names the reference CLI accepts.
pub const DEVICES: [&str; 5] = ["auto", "gpu", "mlx", "cuda", "cpu"];
/// Backends the reference CLI can select.
pub const BACKENDS: [&str; 2] = ["llama", "mlx"];
/// llama.cpp builds the reference can download; this port compiles its own runtime.
pub const ACCELERATORS: [&str; 4] = ["auto", "metal", "cuda", "vulkan"];

pub fn spec(size: &str, quant: &str) -> Option<&'static GgufSpec> {
    GGUF.iter()
        .find(|spec| spec.size == size && spec.quant == quant)
}

pub fn sizes() -> Vec<&'static str> {
    let mut list: Vec<&'static str> = Vec::new();
    for spec in GGUF {
        if !list.contains(&spec.size) {
            list.push(spec.size);
        }
    }
    list
}

pub fn quants() -> Vec<&'static str> {
    let mut list: Vec<&'static str> = Vec::new();
    for spec in GGUF {
        if !list.contains(&spec.quant) {
            list.push(spec.quant);
        }
    }
    list
}

/// Local path for a pinned checkpoint, matching the reference convention
/// (`models/<repo name>/<file>`).
pub fn default_path(spec: &GgufSpec) -> String {
    let repo_name = spec.repo.rsplit('/').next().unwrap_or(spec.repo);
    format!("models/{repo_name}/{}", spec.file)
}

/// Map `--device` onto llama.cpp GPU offload layers.
pub fn device_to_gpu_layers(device: &str) -> Result<u32, String> {
    match device {
        "auto" | "gpu" | "cuda" => Ok(99),
        "cpu" => Ok(0),
        "mlx" => Err(
            "this build has no MLX backend (it runs llama.cpp/GGUF): use --device auto, gpu, cuda or cpu"
                .to_string(),
        ),
        other => Err(format!(
            "Device must be one of: {} (got {other:?})",
            DEVICES.join(", ")
        )),
    }
}

/// `--backend`: this port implements the llama.cpp backend only.
pub fn check_backend(backend: &str) -> Result<(), String> {
    match backend {
        "llama" => Ok(()),
        "mlx" => Err(
            "this port has no MLX backend (declared difference: it is llama.cpp/GGUF only); use --backend llama"
                .to_string(),
        ),
        other => Err(format!(
            "Backend must be one of: {} (got {other:?})",
            BACKENDS.join(", ")
        )),
    }
}

/// `--accelerator` on `download`: the runtime is compiled into the binary here, so the
/// reference's prebuilt-build choice does not apply.
pub fn check_accelerator(accelerator: &str) -> Result<(), String> {
    match accelerator {
        "auto" => Ok(()),
        "cuda" => Ok(()),
        "metal" | "vulkan" => Err(format!(
            "this binary compiles its own llama.cpp runtime (built with the `cuda` feature): \
             --accelerator {accelerator} does not apply — rebuild with the matching feature"
        )),
        other => Err(format!(
            "Accelerator must be one of: {} (got {other:?})",
            ACCELERATORS.join(", ")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_mapping_covers_the_reference_choices() {
        assert_eq!(device_to_gpu_layers("auto").unwrap(), 99);
        assert_eq!(device_to_gpu_layers("gpu").unwrap(), 99);
        assert_eq!(device_to_gpu_layers("cuda").unwrap(), 99);
        assert_eq!(device_to_gpu_layers("cpu").unwrap(), 0);
        assert!(device_to_gpu_layers("mlx").is_err(), "MLX must be refused, not silently mapped");
        assert!(device_to_gpu_layers("tpu").is_err());
    }

    #[test]
    fn backend_is_llama_only() {
        assert!(check_backend("llama").is_ok());
        assert!(check_backend("mlx").is_err());
        assert!(check_backend("vulkan").is_err());
    }

    #[test]
    fn pinned_specs_resolve_by_size_and_quant() {
        let chosen = spec(DEFAULT_SIZE, DEFAULT_QUANT).expect("the default must be pinned");
        assert!(chosen.file.ends_with(".gguf"));
        assert_eq!(chosen.revision.len(), 40, "revisions are pinned by commit");
        assert_eq!(chosen.sha256.len(), 64, "files are pinned by sha256");
        assert!(default_path(chosen).starts_with("models/"));
        assert_eq!(sizes(), vec!["4b", "1.7b"]);
        assert_eq!(quants(), vec!["q8_0", "q4_k_m", "bf16"]);
        assert!(spec("4b", "q2_k").is_none(), "only published quants are offered");
    }
}
