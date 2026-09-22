//! llama.cpp backend (GGUF) — the engine that replaces MLX in this port.
//! Feature-gated: `--features llama` builds it (and llama.cpp itself); `--features llama,cuda`
//! builds with GPU offload.
//!
//! Strategy (parity with the original `SparkBackend.score`):
//!   1. decode the shared state prefix **once** (KV cache keeps it);
//!   2. per question: decode only the suffix, read the logits of the answer-letter slots;
//!   3. restore the prefix state (snapshot/restore: on hybrid recurrent models the KV
//!      sequence cannot be rewound with `clear_kv_cache_seq`, but llama.cpp can serialize
//!      and restore the full sequence state) and move to the next question.
//!
//! Long-lived serving: `Engine` keeps the model + tokenizer + calibration resident and
//! creates a fresh context per request under a lock (serialized GPU access, no
//! self-referential lifetime games).

use std::sync::Mutex;

use serde_json::{json, Map, Value};

use crate::calibration::{sha256_canonical, Calibration};
use crate::prompts::{compile_request, Compiled, Tokenizer};
use crate::schema::{Request, ValidationError};

pub type Result<T> = std::result::Result<T, ValidationError>;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(ValidationError(msg.into()))
}

/// One decoded answer: the raw logits of the answer-letter slots, in candidate order.
pub struct Scored {
    pub id: String,
    pub logits: Vec<f32>,
}

/// Tokenizer adapter over a loaded llama.cpp model.
pub struct LlamaTokenizer<'m> {
    pub model: &'m llama_cpp_2::model::LlamaModel,
    pub template: Option<llama_cpp_2::model::LlamaChatTemplate>,
}

impl<'m> Tokenizer for LlamaTokenizer<'m> {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.model
            .str_to_token(text, llama_cpp_2::model::AddBos::Never)
            .map(|tokens| tokens.into_iter().map(|t| t.0 as u32).collect())
            .unwrap_or_default()
    }

    fn chat_prompt(&self, system: &str, user: &str) -> String {
        if let Some(tmpl) = &self.template {
            let messages = match (
                llama_cpp_2::model::LlamaChatMessage::new("system".into(), system.into()),
                llama_cpp_2::model::LlamaChatMessage::new("user".into(), user.into()),
            ) {
                (Ok(sys), Ok(usr)) => Some([sys, usr]),
                _ => None,
            };
            if let Some(messages) = messages {
                if let Ok(rendered) = self.model.apply_chat_template(tmpl, &messages, true) {
                    return rendered;
                }
            }
        }
        // Fallback (should not trigger on models with a chat template).
        format!("{system}\n\n{user}\n\n")
    }
}

impl<'m> LlamaTokenizer<'m> {
    /// Decode a single token id to its piece text (UTF-8, non-special).
    pub fn decode_token(&self, token: u32) -> String {
        use encoding_rs::UTF_8;
        let mut decoder = UTF_8.new_decoder();
        self.model
            .token_to_piece(
                llama_cpp_2::token::LlamaToken(token as i32),
                &mut decoder,
                false,
                None,
            )
            .unwrap_or_else(|_| format!("<tok{token}>"))
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Timing {
    pub load_seconds: f64,
    pub prefill_seconds: f64,
    pub inference_seconds: f64,
    pub shared_prefix_tokens: usize,
    pub logical_input_tokens: usize,
    pub batches: usize,
    pub generated_tokens: u32,
    pub queue_seconds: f64,
    pub compile_seconds: f64,
    pub total_seconds: f64,
}

/// Resident decision engine: model, tokenizer, calibration, metadata. Thread-safe:
/// requests serialize on one lock (one resident model, serialized GPU access).
pub struct Engine {
    backend: llama_cpp_2::llama_backend::LlamaBackend,
    model: llama_cpp_2::model::LlamaModel,
    template: Option<llama_cpp_2::model::LlamaChatTemplate>,
    calibration: Option<Calibration>,
    pub metadata: Value,
    n_ctx: u32,
    n_threads: Option<i32>,
    lock: Mutex<()>,
}

/// Quantization tag detected from the file name (part of the model identity).
fn detect_precision(name: &str) -> &'static str {
    let lower = name.to_lowercase();
    for q in [
        "q4_k_m", "q4_k_s", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "q4_0", "q5_0", "q3_k_m", "q2_k",
        "iq4_xs", "f16", "fp16", "bf16",
    ] {
        if lower.contains(q) {
            return q;
        }
    }
    "unknown"
}

/// Model identity seen by clients (parity with the Python backend metadata).
pub fn model_metadata(
    model_path: &str,
    n_gpu_layers: u32,
    load_seconds: f64,
) -> Result<Value> {
    let name = std::path::Path::new(model_path)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| model_path.to_string());
    let precision = detect_precision(&name);
    let identity = json!({
        "source": name,
        "backend": "llama.cpp",
        "device": if n_gpu_layers > 0 { "cuda" } else { "cpu" },
        "n_gpu_layers": n_gpu_layers,
        "precision": precision,
        "prompt_version": "rizzo-decisions-v1",
    });
    let fingerprint = sha256_canonical(&identity);
    let mut map = identity_as_map(&identity);
    map.insert("name".to_string(), Value::String(name.clone()));
    map.insert("path".to_string(), Value::String(model_path.to_string()));
    map.insert("fingerprint".to_string(), Value::String(fingerprint));
    map.insert("load_seconds".to_string(), json!(load_seconds));
    Ok(Value::Object(map))
}

fn identity_as_map(value: &Value) -> Map<String, Value> {
    value
        .as_object()
        .cloned()
        .unwrap_or_default()
}

impl Engine {
    /// Load the model once and prepare a resident engine.
    pub fn load(
        model_path: &str,
        n_ctx: u32,
        n_gpu_layers: u32,
        n_threads: Option<i32>,
        calibration: Option<Calibration>,
    ) -> Result<Engine> {
        use llama_cpp_2::model::params::LlamaModelParams;
        use llama_cpp_2::model::LlamaModel;

        let t_start = std::time::Instant::now();
        if !std::path::Path::new(model_path).exists() {
            return err(format!("model not found: {model_path}"));
        }
        let backend = llama_cpp_2::llama_backend::LlamaBackend::init()
            .map_err(|e| ValidationError(format!("backend init: {e}")))?;
        let params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, model_path, &params).map_err(|e| {
            // The binding's error carries no detail (llama.cpp writes its own message to
            // stderr), so read the architecture from the file to explain the failure.
            let message = format!("{e}");
            match crate::gguf::architecture(model_path).as_deref() {
                Some("spark2_5") => ValidationError(format!(
                    "model load: {message} — the file declares architecture `spark2_5` \
                     (Spark-X2.5), which this build's llama.cpp does not support: the \
                     `llama-cpp-2` crate vendors a llama.cpp older than the Spark support \
                     (upstream uses release b11081). See REPORT.md §5.1. \
                     Pass --model with a GGUF this build supports."
                )),
                Some(other) => ValidationError(format!(
                    "model load: {message} — the file declares architecture `{other}`"
                )),
                None => ValidationError(format!("model load: {message}")),
            }
        })?;
        let template = model.chat_template(None).ok();
        let metadata = model_metadata(model_path, n_gpu_layers, t_start.elapsed().as_secs_f64())?;
        if let Some(cal) = &calibration {
            let fp = metadata
                .get("fingerprint")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if cal.fingerprint != fp {
                return err(format!(
                    "Calibration was fitted for a different model/runtime/prompt configuration (expected fingerprint {fp})"
                ));
            }
        }
        Ok(Engine {
            backend,
            model,
            template,
            calibration,
            metadata,
            n_ctx,
            n_threads,
            lock: Mutex::new(()),
        })
    }

    fn tokenizer(&self) -> LlamaTokenizer<'_> {
        LlamaTokenizer {
            model: &self.model,
            template: self.template.clone(),
        }
    }

    /// Context parameters, including the CPU thread count when the caller set one.
    fn ctx_params(&self, n_batch: u32) -> llama_cpp_2::context::params::LlamaContextParams {
        use llama_cpp_2::context::params::LlamaContextParams;
        use std::num::NonZeroU32;
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.n_ctx))
            .with_n_batch(n_batch);
        match self.n_threads {
            Some(threads) => params.with_n_threads(threads).with_n_threads_batch(threads),
            None => params,
        }
    }

    /// `mode: direct` — each job gets its own context and re-prefills the full prompt,
    /// with no shared-prefix reuse (`shared_prefix_tokens: 0`, one batch per job).
    fn score_direct(
        &self,
        jobs: &[Compiled],
        queue_seconds: f64,
        compile_seconds: f64,
    ) -> Result<(Vec<Scored>, Timing)> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use std::num::NonZeroU32;

        let t_infer = std::time::Instant::now();
        let mut logical_input_tokens = 0usize;
        let mut scored = Vec::with_capacity(jobs.len());
        for job in jobs {
            let tokens: Vec<llama_cpp_2::token::LlamaToken> = job
                .tokens
                .iter()
                .map(|t| llama_cpp_2::token::LlamaToken(*t as i32))
                .collect();
            logical_input_tokens += tokens.len();
            let ctx_params = self
                .ctx_params(tokens.len().max(512) as u32);
            let mut ctx = self
                .model
                .new_context(&self.backend, ctx_params)
                .map_err(|e| ValidationError(format!("context: {e}")))?;
            let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
            // `logits_all: false` -> logits only for the last position, so the buffer
            // holds one row and slot (token id) indices address it correctly.
            batch
                .add_sequence(&tokens, 0, false)
                .map_err(|e| ValidationError(format!("direct batch {}: {e}", job.id)))?;
            ctx.decode(&mut batch)
                .map_err(|e| ValidationError(format!("direct decode {}: {e}", job.id)))?;
            let all_logits = ctx.get_logits();
            let logits: Vec<f32> = job
                .slots
                .iter()
                .map(|slot| {
                    all_logits
                        .get(*slot as usize)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY)
                })
                .collect();
            scored.push(Scored {
                id: job.id.clone(),
                logits,
            });
        }
        let inference_seconds = t_infer.elapsed().as_secs_f64();
        Ok((
            scored,
            Timing {
                load_seconds: self
                    .metadata
                    .get("load_seconds")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                prefill_seconds: 0.0,
                inference_seconds,
                shared_prefix_tokens: 0,
                logical_input_tokens,
                batches: jobs.len(),
                generated_tokens: 0,
                queue_seconds,
                compile_seconds,
                total_seconds: 0.0, // filled by decide()
            },
        ))
    }

    /// `mode: shared` — decode the shared state prefix once, then one suffix per question.
    fn score_shared(
        &self,
        prefix: &[u32],
        jobs: &[Compiled],
        queue_seconds: f64,
        compile_seconds: f64,
    ) -> Result<(Vec<Scored>, Timing)> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use std::num::NonZeroU32;

        let t_prefill = std::time::Instant::now();
        // Every batch submitted must fit in n_batch: the prefix *and* the longest suffix.
        // Sizing it on the prefix alone aborts llama.cpp on a long question
        // (GGML_ASSERT(n_tokens_all <= cparams.n_batch)).
        let longest_suffix = jobs
            .iter()
            .map(|job| job.tokens.len().saturating_sub(prefix.len()))
            .max()
            .unwrap_or(0);
        let ctx_params = self.ctx_params(prefix.len().max(longest_suffix).max(512) as u32);
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| ValidationError(format!("context: {e}")))?;

        // 1) prefix once
        let prefix_tokens: Vec<llama_cpp_2::token::LlamaToken> = prefix
            .iter()
            .map(|t| llama_cpp_2::token::LlamaToken(*t as i32))
            .collect();
        let mut batch = LlamaBatch::new(prefix_tokens.len().max(1), 1);
        batch
            .add_sequence(&prefix_tokens, 0, false)
            .map_err(|e| ValidationError(format!("prefix batch: {e}")))?;
        ctx.decode(&mut batch)
            .map_err(|e| ValidationError(format!("prefix decode: {e}")))?;
        let prefill_seconds = t_prefill.elapsed().as_secs_f64();

        // 2) snapshot the prefix state (the Rizzo-style "cache clone")
        let prefix_state = ctx
            .state_seq_get(0, llama_cpp_2::LlamaStateSeqFlags::empty())
            .map_err(|e| ValidationError(format!("state snapshot: {e}")))?;

        // 3) one suffix per question
        let t_infer = std::time::Instant::now();
        let base = prefix.len() as i32;
        let mut logical_input_tokens = prefix.len();
        let mut scored = Vec::with_capacity(jobs.len());
        for job in jobs {
            ctx.state_seq_set(&prefix_state, 0)
                .map_err(|e| ValidationError(format!("state restore {}: {e}", job.id)))?;
            let suffix_tokens: Vec<llama_cpp_2::token::LlamaToken> = job.tokens[prefix.len()..]
                .iter()
                .map(|t| llama_cpp_2::token::LlamaToken(*t as i32))
                .collect();
            logical_input_tokens += suffix_tokens.len();
            let mut batch = LlamaBatch::new(suffix_tokens.len().max(1), 1);
            for (i, token) in suffix_tokens.iter().enumerate() {
                batch
                    .add(*token, base + i as i32, &[0], i + 1 == suffix_tokens.len())
                    .map_err(|e| ValidationError(format!("suffix batch {}: {e}", job.id)))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| ValidationError(format!("suffix decode {}: {e}", job.id)))?;
            let all_logits = ctx.get_logits();
            let logits: Vec<f32> = job
                .slots
                .iter()
                .map(|slot| {
                    all_logits
                        .get(*slot as usize)
                        .copied()
                        .unwrap_or(f32::NEG_INFINITY)
                })
                .collect();
            scored.push(Scored {
                id: job.id.clone(),
                logits,
            });
        }
        let inference_seconds = t_infer.elapsed().as_secs_f64();

        Ok((
            scored,
            Timing {
                load_seconds: self
                    .metadata
                    .get("load_seconds")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0),
                prefill_seconds,
                inference_seconds,
                shared_prefix_tokens: prefix.len(),
                logical_input_tokens,
                batches: 1 + jobs.len(),
                generated_tokens: 0,
                queue_seconds,
                compile_seconds,
                total_seconds: 0.0, // filled by decide()
            },
        ))
    }

    /// Score one request, honouring the execution mode (serialized by the engine lock).
    /// An empty prefix always falls back to `direct`, like the Python backend.
    fn score_request(
        &self,
        prefix: &[u32],
        jobs: &[Compiled],
        mode: crate::schema::Mode,
        queue_seconds: f64,
        compile_seconds: f64,
    ) -> Result<(Vec<Scored>, Timing)> {
        if mode == crate::schema::Mode::Direct || prefix.is_empty() {
            self.score_direct(jobs, queue_seconds, compile_seconds)
        } else {
            self.score_shared(prefix, jobs, queue_seconds, compile_seconds)
        }
    }

    /// Full decision pipeline over a raw request value (re-validates the snapshot,
    /// like the Python `Engine.decide`).
    pub fn decide(&self, raw: Value, topk: u32) -> Result<Value> {
        let started = std::time::Instant::now();
        let request = Request::from_value(raw)?;

        let _guard = self.lock.lock().map_err(|_| {
            ValidationError("inference lock poisoned".into())
        })?;
        let acquired = started.elapsed().as_secs_f64();

        let tokenizer = self.tokenizer();
        let (prefix, jobs) = compile_request(&tokenizer, &request, self.n_ctx as usize)
            .map_err(|e| ValidationError(format!("compile: {e}")))?;
        let compile_seconds = started.elapsed().as_secs_f64() - acquired;

        let (scored, mut timing) =
            self.score_request(&prefix, &jobs, request.mode, acquired, compile_seconds)?;

        // debug: show the top candidate positions (indices into the candidate list)
        if topk > 0 {
            for s in &scored {
                let mut ranked: Vec<(usize, f32)> =
                    s.logits.iter().enumerate().map(|(i, l)| (i, *l)).collect();
                ranked.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let top: Vec<String> = ranked
                    .iter()
                    .take(topk as usize)
                    .map(|(idx, lg)| format!("{}:{:.3}", idx, lg))
                    .collect();
                println!("TOPK {}: {}", s.id, top.join("  "));
            }
        }

        let mut answers = Map::new();
        for job in &jobs {
            let question = &request.questions[&job.id];
            let logits: Vec<f64> = scored
                .iter()
                .find(|s| s.id == job.id)
                .map(|s| s.logits.iter().map(|x| *x as f64).collect())
                .unwrap_or_default();
            let temperature = self
                .calibration
                .as_ref()
                .map(|c| c.temperature(question.type_name()))
                .unwrap_or(1.0);
            let mut answer = crate::decisions::decode(question, &logits, temperature)?;
            let sha = {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(job.prompt.as_bytes());
                let digest = hasher.finalize();
                digest.iter().map(|b| format!("{b:02x}")).collect::<String>()
            };
            if let Value::Object(map) = &mut answer {
                map.insert("prompt_sha256".into(), Value::String(sha));
                map.insert(
                    "input_tokens".into(),
                    Value::Number(serde_json::Number::from(job.tokens.len() as u64)),
                );
            }
            answers.insert(job.id.clone(), answer);
        }
        let total_seconds = started.elapsed().as_secs_f64();
        timing.total_seconds = total_seconds;

        let response = json!({
            "model": self.metadata,
            "mode": request.mode,
            "answers": Value::Object(answers),
            "calibration": self.calibration.as_ref().map(|c| c.to_json()),
            "timing": timing,
        });
        Ok(response)
    }
}

/// The engine is a `Decider`, so the evaluation harness runs on real inference.
impl crate::evaluate::Decider for Engine {
    fn decide(&self, request: Value) -> crate::evaluate::Result<Value> {
        Engine::decide(self, request, 0)
    }
}

/// JSON device description (parity with the Python `rizzo devices` output shape).
pub fn describe_devices() -> Value {
    let devices = llama_cpp_2::list_llama_ggml_backend_devices();
    let mut available: Vec<String> = Vec::new();
    let mut names: Vec<Value> = Vec::new();
    for device in &devices {
        let kind = format!("{:?}", device.device_type).to_lowercase();
        names.push(json!({"name": device.name, "description": device.description, "type": kind}));
        let label = if kind.contains("gpu") { "cuda" } else { "cpu" };
        if !available.iter().any(|a| a == label) {
            available.push(label.to_string());
        }
    }
    let auto = if available.iter().any(|a| a == "cuda") {
        "cuda"
    } else {
        "cpu"
    };
    json!({
        "backend": "llama.cpp",
        "platform": std::env::consts::OS,
        "available": available,
        "auto_selects": auto,
        "devices": names,
    })
}

/// Human-readable backend devices (parity with `rizzo devices`).
pub fn devices() -> String {    let devices = llama_cpp_2::list_llama_ggml_backend_devices();
    let mut lines = vec![format!("llama.cpp {} device(s):", devices.len())];
    for d in devices {
        lines.push(format!("  - {:?} {} ({})", d.device_type, d.name, d.description));
    }
    lines.join("\n")
}

/// Expose the raw error mapping used by `main` when a model path is missing.
pub fn check_model_path(path: &str) -> Result<()> {
    if !std::path::Path::new(path).exists() {
        return err(format!("model not found: {path}"));
    }
    Ok(())
}