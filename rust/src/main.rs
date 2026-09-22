#![cfg_attr(feature = "llama", allow(unused_imports))]
//! rizzo-flow-rs CLI — F1/F2: `devices`, `decide`, `dump`, `serve`.
//! Parity target with the original `rizzo` CLI (decide/serve/download/schema/calibrate follow).

use rizzo_flow_rs::pinned;

#[cfg(feature = "llama")]
use rizzo_flow_rs::schema::Request;

#[cfg(feature = "llama")]
use rizzo_flow_rs::backend::{self, Engine};
#[cfg(feature = "llama")]
use rizzo_flow_rs::calibration::Calibration;
#[cfg(feature = "llama")]
use rizzo_flow_rs::compat;
#[cfg(feature = "llama")]
use rizzo_flow_rs::prompts::compile_request;

fn usage() -> ! {
    eprintln!(
        "rizzo-flow-rs\n\nUSAGE:\n  rizzo-rs devices\n  rizzo-rs schema [--response] [--output <file>]\n  rizzo-rs download [--size 4b] [--destination <path>]\n  rizzo-rs calibrate <rows.jsonl> --fingerprint <fp> --output <file>\n  rizzo-rs decide <request.json> [--model <gguf> | --size 4b] [--ctx 8192] [--device auto|gpu|cuda|cpu] [--ngl 99] [--topk N] [--calibration <file>] [--output <file>]\n  rizzo-rs evaluate <fixtures.jsonl> [--model <gguf> | --size 4b] [--repeats 1] [--compare-modes] [--ctx 8192] [--calibration <file>] [--output <file>]\n  rizzo-rs dump <request.json> --model <gguf> [--ctx 8192]\n  rizzo-rs serve [--model <gguf> | --size 4b] [--ctx 8192] [--device auto|gpu|cuda|cpu] [--calibration <file>] [--host 127.0.0.1] [--port 8017] [--api-key <key>]\n\nReference-CLI options accepted for drop-in compatibility:\n  --size <s>       select a pinned checkpoint (default 4b; see `download`)\n  --device <d>     auto|gpu|cuda offload to GPU, cpu does not, mlx is refused\n  --bits <4|8>     accepted and ignored (a GGUF carries its own quantization)\n  --batch-size <n> accepted and ignored (questions are decoded sequentially)\n  --max-tokens     alias of --ctx"
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first().map(String::as_str) else {
        usage();
    };
    match command {
        "devices" => devices(),
        "schema" => schema(&args[1..]),
        "download" => download(&args[1..]),
        "calibrate" => calibrate(&args[1..]),
        "decide" => decide(&args[1..]),
        "evaluate" => evaluate(&args[1..]),
        "dump" => dump(&args[1..]),
        "serve" => serve(&args[1..]),
        _ => usage(),
    }
}

/// `devices`: JSON description of the compute backends this build can use.
fn devices() {
    #[cfg(feature = "llama")]
    {
        let described = backend::describe_devices();
        if let Err(e) = rizzo_flow_rs::jsonio::write_json(&described, None) {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
    #[cfg(not(feature = "llama"))]
    {
        let described = serde_json::json!({
            "backend": "none",
            "platform": std::env::consts::OS,
            "available": [],
            "auto_selects": null,
        });
        let _ = rizzo_flow_rs::jsonio::write_json(&described, None);
        eprintln!("built without the `llama` feature: no devices");
    }
}

/// `schema`: print the JSON Schema of the request (or the response) contract.
///
/// The schemas are vendored from the original Rizzo Flow project (Apache-2.0) so
/// that clients see exactly the documented wire contract.
fn schema(args: &[String]) {
    const REQUEST: &str = include_str!("../schema/request.schema.json");
    const RESPONSE: &str = include_str!("../schema/response.schema.json");
    let mut output: Option<String> = None;
    let mut want_response = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                i += 1;
                output = args.get(i).cloned();
            }
            "--response" => want_response = true,
            other => {
                eprintln!("unknown option: {other}");
                usage();
            }
        }
        i += 1;
    }
    let text = if want_response { RESPONSE } else { REQUEST };
    let value: serde_json::Value = serde_json::from_str(text).unwrap_or_else(|e| {
        eprintln!("embedded schema is invalid: {e}");
        std::process::exit(1);
    });
    if let Err(e) = rizzo_flow_rs::jsonio::write_json(&value, output.as_deref()) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

/// `download`: fetch a pinned GGUF checkpoint for the local decision engine.
///
/// The original project pins MLX checkpoints (Spark-X2.5); this port runs on
/// llama.cpp, so it pins the GGUF it was validated with. Downloads shell out to
/// `curl` on purpose: no TLS stack in the dependency tree, and curl ships with
/// every supported target.
fn download(args: &[String]) {
    let mut size = pinned::DEFAULT_SIZE.to_string();
    let mut quant = pinned::DEFAULT_QUANT.to_string();
    let mut backend = "llama".to_string();
    let mut accelerator = "auto".to_string();
    let mut only: Option<String> = None;
    let mut destination: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--size" => {
                i += 1;
                if let Some(value) = args.get(i) {
                    size = value.clone();
                }
            }
            "--quant" => {
                i += 1;
                if let Some(value) = args.get(i) {
                    quant = value.clone();
                }
            }
            "--backend" => {
                i += 1;
                if let Some(value) = args.get(i) {
                    backend = value.clone();
                }
            }
            "--accelerator" => {
                i += 1;
                if let Some(value) = args.get(i) {
                    accelerator = value.clone();
                }
            }
            "--only" => {
                i += 1;
                only = args.get(i).cloned();
            }
            "--destination" => {
                i += 1;
                destination = args.get(i).cloned();
            }
            other => {
                eprintln!("unknown option: {other}");
                usage();
            }
        }
        i += 1;
    }
    if let Err(error) = pinned::check_backend(&backend) {
        eprintln!("{error}");
        std::process::exit(2);
    }
    if let Err(error) = pinned::check_accelerator(&accelerator) {
        eprintln!("{error}");
        std::process::exit(2);
    }
    if let Some(only) = only.as_deref() {
        if only != "runtime" && only != "weights" {
            eprintln!("rizzo: --only must be runtime or weights");
            std::process::exit(2);
        }
        if only == "runtime" {
            eprintln!(
                "rizzo: nothing to download — this binary compiles its own llama.cpp runtime \
                 (the reference downloads a prebuilt one)"
            );
            return;
        }
    }
    let Some(spec) = pinned::spec(&size, &quant) else {
        eprintln!(
            "rizzo: unknown checkpoint {size:?}/{quant:?}; sizes: {}; quants: {}",
            pinned::sizes().join(", "),
            pinned::quants().join(", ")
        );
        std::process::exit(1);
    };
    let path = destination.unwrap_or_else(|| pinned::default_path(spec));
    if std::path::Path::new(&path).exists() {
        eprintln!("rizzo: {path} already exists (downloads never overwrite a checkpoint)");
        std::process::exit(1);
    }
    if let Some(parent) = std::path::Path::new(&path).parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!("rizzo: {}: {e}", parent.display());
                std::process::exit(1);
            }
        }
    }
    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        spec.repo, spec.revision, spec.file
    );
    eprintln!("rizzo: downloading {} -> {path}", spec.file);
    let status = std::process::Command::new("curl")
        .args(["--fail", "--location", "--retry", "3", "--output"])
        .arg(&path)
        .arg(&url)
        .status();
    match status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            let _ = std::fs::remove_file(&path);
            eprintln!("rizzo: download failed ({status})");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("rizzo: cannot run curl: {e}");
            std::process::exit(1);
        }
    }
    // The reference verifies the file against its pinned sha256; so do we.
    match sha256_file(&path) {
        Ok(digest) if digest == spec.sha256 => eprintln!("rizzo: sha256 verified"),
        Ok(digest) => {
            let _ = std::fs::remove_file(&path);
            eprintln!("rizzo: sha256 mismatch for {}: got {digest}, expected {}", spec.file, spec.sha256);
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("rizzo: cannot hash {path}: {e}");
            std::process::exit(1);
        }
    }
    println!("{path}");
}

/// SHA-256 of a file, in the same hex form the reference pins.
fn sha256_file(path: &str) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// `calibrate`: fit one temperature per question type on separate labeled logit rows.
fn calibrate(args: &[String]) {
    let Some(input) = args.first() else { usage() };
    let mut fingerprint: Option<String> = None;
    let mut output: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--fingerprint" => {
                i += 1;
                fingerprint = args.get(i).cloned();
            }
            "--output" => {
                i += 1;
                output = args.get(i).cloned();
            }
            other => {
                eprintln!("unknown option: {other}");
                usage();
            }
        }
        i += 1;
    }
    let (Some(fingerprint), Some(output)) = (fingerprint, output) else {
        eprintln!("--fingerprint and --output are required");
        usage();
    };
    let rows = rizzo_flow_rs::jsonio::read_jsonl(input).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let calibration = rizzo_flow_rs::calibration::fit_temperature(&rows, &fingerprint)
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
    if let Err(e) = rizzo_flow_rs::jsonio::write_json(&calibration.to_json(), Some(&output)) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

/// Shared CLI parser for --model/--ctx/--ngl/--topk/--calibration/--host/--port.
struct Cli {
    model: Option<String>,
    size: Option<String>,
    quant: Option<String>,
    backend: Option<String>,
    device: Option<String>,
    threads: Option<i32>,
    bits: Option<u32>,
    batch_size: Option<usize>,
    ctx: u32,
    ngl: Option<u32>,
    topk: u32,
    calibration: Option<String>,
    host: String,
    port: u16,
    api_key: Option<String>,
    output: Option<String>,
    repeats: usize,
    compare_modes: bool,
}

fn parse_cli(args: &[String], allow_host: bool) -> Cli {
    let mut cli = Cli {
        model: None,
        size: None,
        quant: None,
        backend: None,
        device: None,
        threads: None,
        bits: None,
        batch_size: None,
        // the reference CLI defaults to 8192
        ctx: 8192,
        ngl: None,
        topk: 0,
        calibration: None,
        host: "127.0.0.1".to_string(),
        port: 8017,
        api_key: None,
        output: None,
        repeats: 1,
        compare_modes: false,
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                cli.model = args.get(i).cloned();
            }
            // Reference-CLI options kept for drop-in compatibility.
            "--size" => {
                i += 1;
                cli.size = args.get(i).cloned();
            }
            "--quant" => {
                i += 1;
                cli.quant = args.get(i).cloned();
            }
            "--backend" => {
                i += 1;
                cli.backend = args.get(i).cloned();
            }
            "--threads" => {
                i += 1;
                cli.threads = args.get(i).and_then(|v| v.parse().ok());
            }
            "--device" => {
                i += 1;
                cli.device = args.get(i).cloned();
            }
            "--bits" => {
                i += 1;
                cli.bits = args.get(i).and_then(|v| v.parse().ok());
            }
            "--batch-size" => {
                i += 1;
                cli.batch_size = args.get(i).and_then(|v| v.parse().ok());
            }
            // --max-tokens is the former name, kept as an alias.
            "--ctx" | "--max-tokens" => {
                i += 1;
                cli.ctx = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(cli.ctx);
            }
            "--ngl" => {
                i += 1;
                cli.ngl = args.get(i).and_then(|v| v.parse().ok()).or(cli.ngl);
            }
            "--topk" => {
                i += 1;
                cli.topk = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(cli.topk);
            }
            "--calibration" => {
                i += 1;
                cli.calibration = args.get(i).cloned();
            }
            "--output" => {
                i += 1;
                cli.output = args.get(i).cloned();
            }
            "--repeats" => {
                i += 1;
                cli.repeats = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(cli.repeats);
            }
            "--compare-modes" => cli.compare_modes = true,
            "--api-key" if allow_host => {
                i += 1;
                cli.api_key = args.get(i).cloned();
            }
            "--host" if allow_host => {
                i += 1;
                if let Some(h) = args.get(i) {
                    cli.host = h.clone();
                }
            }
            "--port" if allow_host => {
                i += 1;
                if let Some(p) = args.get(i).and_then(|v| v.parse().ok()) {
                    cli.port = p;
                }
            }
            other => {
                eprintln!("unknown option: {other}");
                usage();
            }
        }
        i += 1;
    }
    cli
}

/// Resolve the checkpoint, the GPU offload and the CPU thread count from the
/// reference-CLI options.
///
/// `--model` wins over `--size`+`--quant`; `--ngl` wins over `--device`; `--backend`
/// must be `llama` (this port has no MLX backend).
fn resolve_model_and_ngl(cli: &Cli) -> (String, u32, Option<i32>) {
    if let Some(backend) = cli.backend.as_deref() {
        if let Err(error) = pinned::check_backend(backend) {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
    let ngl = match (cli.ngl, cli.device.as_deref()) {
        (Some(ngl), _) => ngl,
        (None, Some(device)) => pinned::device_to_gpu_layers(device).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(2);
        }),
        (None, None) => pinned::device_to_gpu_layers("auto").unwrap_or(99),
    };
    let threads = match cli.threads {
        Some(threads) if threads > 0 => Some(threads),
        Some(_) => {
            eprintln!("rizzo: --threads must be a positive integer");
            std::process::exit(2);
        }
        None => None,
    };
    if let Some(bits) = cli.bits {
        if bits != 4 && bits != 8 {
            eprintln!("rizzo: --bits must be 4 or 8");
            std::process::exit(2);
        }
        eprintln!("rizzo: note: --bits {bits} ignored — it is MLX-backend only in the reference");
    }
    if let Some(batch) = cli.batch_size {
        eprintln!(
            "rizzo: note: --batch-size {batch} ignored — this engine decodes the questions of a request sequentially"
        );
    }
    if let Some(model) = &cli.model {
        return (model.clone(), ngl, threads);
    }
    let size = cli.size.as_deref().unwrap_or(pinned::DEFAULT_SIZE);
    let quant = cli.quant.as_deref().unwrap_or(pinned::DEFAULT_QUANT);
    let Some(spec) = pinned::spec(size, quant) else {
        eprintln!(
            "rizzo: unknown checkpoint {size:?}/{quant:?}; sizes: {}; quants: {}",
            pinned::sizes().join(", "),
            pinned::quants().join(", ")
        );
        std::process::exit(2);
    };
    let path = pinned::default_path(spec);
    if !std::path::Path::new(&path).exists() {
        eprintln!(
            "rizzo: no checkpoint at {path} — run `rizzo-flow-rs download --size {size} --quant {quant}` or pass --model"
        );
        std::process::exit(2);
    }
    (path, ngl, threads)
}

#[cfg(feature = "llama")]
fn load_calibration(cli: &Cli) -> Option<Calibration> {
    cli.calibration
        .as_deref()
        .map(Calibration::from_file)
        .transpose()
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        })
}

#[cfg(feature = "llama")]
fn decide(args: &[String]) {
    let Some(path) = args.first() else { usage() };
    let cli = parse_cli(&args[1..], false);
    let (model_path, ngl, threads) = resolve_model_and_ngl(&cli);
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let request: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| {
        eprintln!("invalid JSON: {e}");
        std::process::exit(1);
    });
    let engine = Engine::load(&model_path, cli.ctx, ngl, threads, load_calibration(&cli))
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
    let response = engine.decide(request, cli.topk).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    if let Err(e) = rizzo_flow_rs::jsonio::write_json(&response, cli.output.as_deref()) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

/// `evaluate`: reproducible benchmark report over a fixtures JSONL file.
#[cfg(feature = "llama")]
fn evaluate(args: &[String]) {
    let Some(path) = args.first() else { usage() };
    let cli = parse_cli(&args[1..], false);
    let (model_path, ngl, threads) = resolve_model_and_ngl(&cli);
    let fixtures = rizzo_flow_rs::jsonio::read_jsonl(path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let engine = Engine::load(&model_path, cli.ctx, ngl, threads, load_calibration(&cli))
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
    let report = rizzo_flow_rs::evaluate::evaluate(
        &engine,
        &fixtures,
        cli.repeats,
        cli.compare_modes,
    )
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    if let Err(e) = rizzo_flow_rs::jsonio::write_json(&report, cli.output.as_deref()) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "llama"))]
fn evaluate(_args: &[String]) {
    eprintln!("built without the `llama` feature — rebuild with --features llama");
    std::process::exit(2);
}

/// HTTP service with the Python `rizzo serve` contract.
///   GET  /health        -> {"status":"ready","model": <metadata>}
///   POST /v1/decisions  -> full decision response (200) or {"detail": ...} (422)
///   GET  /v1/models     -> model catalogue (Bearer auth when RIZZO_API_KEY is set)
///   POST /v1/systemone  -> TypeSafe-compatible wire format (Bearer auth as above)
#[cfg(feature = "llama")]
fn serve(args: &[String]) {
    let cli = parse_cli(args, true);
    let (model_path, ngl, threads) = resolve_model_and_ngl(&cli);
    let engine = Engine::load(&model_path, cli.ctx, ngl, threads, load_calibration(&cli))
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
    let api_key: Option<String> = cli
        .api_key
        .clone()
        .or_else(|| std::env::var("RIZZO_API_KEY").ok())
        .filter(|k| !k.is_empty());
    let addr = format!("{}:{}", cli.host, cli.port);
    let server = tiny_http::Server::http(&addr).unwrap_or_else(|e| {
        eprintln!("cannot bind {addr}: {e}");
        std::process::exit(1);
    });
    println!(
        "rizzo-flow-rs serve: listening on http://{addr} (model {model_path}, auth {})",
        if api_key.is_some() { "on" } else { "off" }
    );
    let json_header: tiny_http::Header = "Content-Type: application/json".parse().unwrap();

    for mut request in server.incoming_requests() {
        let method = request.method().clone();
        let url = request.url().to_string();
        let authorized = bearer_ok(&request, api_key.as_deref());
        let detail = |msg: &str| serde_json::to_string(&serde_json::json!({"detail": msg})).unwrap();
        let (code, body) = match (method, url.as_str()) {
            (tiny_http::Method::Get, "/health") => (
                200,
                serde_json::to_string(&serde_json::json!({
                    "status": "ready",
                    "model": engine.metadata,
                }))
                .unwrap(),
            ),
            (tiny_http::Method::Post, "/v1/decisions") => {
                let mut raw = String::new();
                use std::io::Read;
                if let Err(e) = request.as_reader().read_to_string(&mut raw) {
                    (400, detail(&format!("bad body: {e}")))
                } else {
                    match serde_json::from_str::<serde_json::Value>(&raw) {
                        Ok(value) => match engine.decide(value, 0) {
                            Ok(response) => (200, serde_json::to_string(&response).unwrap()),
                            Err(e) => (422, detail(&e.0)),
                        },
                        Err(e) => (422, detail(&format!("invalid JSON: {e}"))),
                    }
                }
            }
            (tiny_http::Method::Get, "/v1/models") if !authorized => {
                (401, detail("Missing or invalid API key"))
            }
            (tiny_http::Method::Get, "/v1/models") => (
                200,
                serde_json::to_string(&compat::list_models(&engine.metadata)).unwrap(),
            ),
            (tiny_http::Method::Post, "/v1/systemone") if !authorized => {
                (401, detail("Missing or invalid API key"))
            }
            (tiny_http::Method::Post, "/v1/systemone") => {
                let mut raw = String::new();
                use std::io::Read;
                if let Err(e) = request.as_reader().read_to_string(&mut raw) {
                    (400, detail(&format!("bad body: {e}")))
                } else {
                    match serde_json::from_str::<serde_json::Value>(&raw) {
                        Ok(value) => match compat::SystemOneRequest::parse(value) {
                            Err(e) => (422, detail(&e.0)),
                            Ok(wire) => {
                                match compat::resolve_model(&wire.model, &engine.metadata) {
                                    Err(e) => (422, detail(&e.0)),
                                    Ok(served) => match compat::to_native(&wire) {
                                        Err(e) => (422, detail(&e.0)),
                                        Ok((native, options)) => {
                                            let native_value =
                                                serde_json::to_value(&native).unwrap();
                                            match engine.decide(native_value, 0) {
                                                Err(e) => (422, detail(&e.0)),
                                                Ok(response) => {
                                                    match compat::from_native(
                                                        &wire, &response, &options, &served,
                                                    ) {
                                                        Ok(out) => (
                                                            200,
                                                            serde_json::to_string(&out).unwrap(),
                                                        ),
                                                        Err(e) => (422, detail(&e.0)),
                                                    }
                                                }
                                            }
                                        }
                                    },
                                }
                            }
                        },
                        Err(e) => (422, detail(&format!("invalid JSON: {e}"))),
                    }
                }
            }
            // Development UI of the Python service (playground/snake/logo): not ported.
            (_, "/playground") | (_, "/snake") | (_, "/playground/logo.png") | (_, "/") => {
                (501, detail("playground UI not ported"))
            }
            _ => (404, detail("not found")),
        };
        let response = tiny_http::Response::from_string(body)
            .with_header(json_header.clone())
            .with_status_code(code);
        if let Err(e) = request.respond(response) {
            eprintln!("response error: {e}");
        }
    }
}

/// Bearer check (constant-time); enforced only when a key is configured.
#[cfg(feature = "llama")]
fn bearer_ok(request: &tiny_http::Request, api_key: Option<&str>) -> bool {
    let Some(key) = api_key else { return true };
    let expected = format!("Bearer {key}");
    request.headers().iter().any(|h| {
        h.field.equiv("Authorization")
            && constant_time_eq(h.value.as_str().as_bytes(), expected.as_bytes())
    })
}

#[cfg(feature = "llama")]
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(not(feature = "llama"))]
fn decide(_args: &[String]) {
    eprintln!("built without the `llama` feature — rebuild with --features llama");
    std::process::exit(2);
}

#[cfg(not(feature = "llama"))]
fn serve(_args: &[String]) {
    eprintln!("built without the `llama` feature — rebuild with --features llama");
    std::process::exit(2);
}

#[cfg(not(feature = "llama"))]
fn dump(_args: &[String]) {
    eprintln!("built without the `llama` feature — rebuild with --features llama");
    std::process::exit(2);
}

/// Debug helper: print the compiled prompt for each question (mechanics parity checks).
#[cfg(feature = "llama")]
fn dump(args: &[String]) {
    let Some(path) = args.first() else { usage() };
    let mut model: Option<String> = None;
    let mut ctx: u32 = 4096;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model = args.get(i).cloned();
            }
            "--ctx" => {
                i += 1;
                ctx = args.get(i).and_then(|v| v.parse().ok()).unwrap_or(ctx);
            }
            other => {
                eprintln!("unknown option: {other}");
                usage();
            }
        }
        i += 1;
    }
    let Some(model_path) = model else {
        eprintln!("--model is required");
        usage();
    };
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let request = Request::from_json_str(&raw).unwrap_or_else(|e| {
        eprintln!("invalid request: {e}");
        std::process::exit(1);
    });
    let backend_handle = llama_cpp_2::llama_backend::LlamaBackend::init().unwrap_or_else(|e| {
        eprintln!("backend init: {e}");
        std::process::exit(1);
    });
    let model = llama_cpp_2::model::LlamaModel::load_from_file(
        &backend_handle,
        &model_path,
        &llama_cpp_2::model::params::LlamaModelParams::default().with_n_gpu_layers(0),
    )
    .unwrap_or_else(|e| {
        eprintln!("model load: {e}");
        std::process::exit(1);
    });
    let template = model.chat_template(None).ok();
    let tokenizer = rizzo_flow_rs::backend::LlamaTokenizer { model: &model, template };
    let (prefix, jobs) = compile_request(&tokenizer, &request, ctx as usize).unwrap_or_else(|e| {
        eprintln!("compile: {e}");
        std::process::exit(1);
    });
    println!("PREFIX_TOKENS {}", prefix.len());
    for job in &jobs {
        println!(
            "=== {} ({} tokens, slots {:?}) ===",
            job.id,
            job.tokens.len(),
            job.slots
        );
        println!("{}", job.prompt);
    }
}