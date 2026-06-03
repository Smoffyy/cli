use std::sync::Arc;
use anyhow::Result;
use axum::{
    extract::{State, Json, Path as AxumPath},
    http::{StatusCode, Request, HeaderMap},
    middleware::{self, Next},
    response::{IntoResponse, sse::{KeepAlive, Sse}},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::Path;
use tokio::sync::Mutex;
use uuid::Uuid;

use gguf_rs::model::llama::{KvCache, LlamaModel};
use gguf_rs::tokenizer::bpe::Tokenizer;
use gguf_rs::tokenizer::chat::ChatTemplate;
use gguf_rs::gpu::VkCtx;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelConfig {
    id: String,
    name: String,
    path: String,
    #[serde(default)]
    gpu: bool,
    temperature: Option<f32>,
    ctx_len: Option<usize>,
    max_tokens: Option<usize>,
    system_prompt: Option<String>,
    stop: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ServerConfig {
    server: ServerSettings,
    defaults: SamplingDefaults,
    models: Vec<ModelConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ServerSettings {
    host: String,
    port: u16,
    api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SamplingDefaults {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    max_tokens: usize,
    repetition_penalty: f32,
    seed: u64,
    ctx_len: usize,
    smart_context: bool,
    prefill_batch: usize,
}

struct LoadedModel {
    model:     LlamaModel,
    tokenizer: Tokenizer,
    template:  ChatTemplate,
    config:    ModelConfig,
    // GPU context lives here — only ever accessed from spawn_blocking
    gpu:       Option<VkCtx>,
}

// SAFETY: VkCtx holds a *mut u8 staging pointer. Access is serialised by the
// Mutex<Option<LoadedModel>> — only one spawn_blocking task holds the guard at a time.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

struct AppState {
    loaded:     Mutex<Option<LoadedModel>>,
    // Prevents concurrent model loads — requesters queue here then recheck if already loaded
    load_lock:  Mutex<()>,
    all_models: Vec<ModelConfig>,
    defaults:   SamplingDefaults,
    api_key:    Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role:    String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatCompletionRequest {
    model:              Option<String>,
    messages:           Vec<ChatMessage>,
    temperature:        Option<f32>,
    top_k:              Option<usize>,
    top_p:              Option<f32>,
    max_tokens:         Option<usize>,
    repetition_penalty: Option<f32>,
    stream:             Option<bool>,
    stop:               Option<serde_json::Value>, // string or array per OpenAI spec
}

#[derive(Debug, Serialize)]
struct ChatCompletionChunk {
    id:      String,
    object:  String,
    created: u64,
    model:   String,
    choices: Vec<ChoiceDelta>,
}

#[derive(Debug, Serialize)]
struct ChoiceDelta {
    index:         usize,
    delta:         Delta,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role:    Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChatCompletion {
    id:      String,
    object:  String,
    created: u64,
    model:   String,
    choices: Vec<Choice>,
    usage:   Usage,
}

#[derive(Debug, Serialize)]
struct Choice {
    index:         usize,
    message:       ChatMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens:     usize,
    completion_tokens: usize,
    total_tokens:      usize,
}

#[derive(Debug, Serialize)]
struct ModelListResponse {
    object: String,
    data:   Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id:         String,
    object:     String,
    created:    u64,
    owned_by:   String,
    permission: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LoadModelRequest {
    model: String,
}

// OpenAI-compatible error body
fn openai_error(status: StatusCode, err_type: &str, message: String) -> impl IntoResponse {
    (status, Json(json!({
        "error": {
            "message": message,
            "type":    err_type,
            "code":    status.as_u16()
        }
    })))
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn ts() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    format!("{:.3}", ms as f64 / 1000.0)
}

fn load_config(path: &str) -> Result<ServerConfig> {
    let content = std::fs::read_to_string(path)?;
    Ok(serde_yaml::from_str(&content)?)
}

// Find a model config by id, name, or path substring (case-insensitive)
fn find_model<'a>(all: &'a [ModelConfig], req_model: &str) -> Option<&'a ModelConfig> {
    let lower = req_model.to_lowercase();
    all.iter().find(|m| {
        m.id.to_lowercase() == lower
            || m.name.to_lowercase() == lower
            || m.path.to_lowercase().contains(&lower)
            || lower.contains(&m.id.to_lowercase())
    })
}

// Parse OpenAI stop field: can be a string or array of strings
fn parse_stop(val: Option<&serde_json::Value>) -> Vec<String> {
    match val {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(arr)) => arr.iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect(),
        _ => vec![],
    }
}

// Runs entirely on a blocking thread — no await, no Send requirement on VkCtx.
fn load_model_blocking(cfg: ModelConfig, ctx_len: usize) -> Result<LoadedModel> {
    let mut gpu: Option<VkCtx> = if cfg.gpu {
        match VkCtx::init() {
            Ok(g)  => Some(g),
            Err(e) => { eprintln!("GPU init failed: {e}, using CPU"); None }
        }
    } else { None };

    let (model, gguf) = LlamaModel::load(Path::new(&cfg.path), ctx_len, gpu.as_mut())?;
    let tokenizer = Tokenizer::from_gguf(&gguf)?;
    let tmpl_str  = gguf.metadata.get("tokenizer.chat_template")
        .and_then(|v| v.as_str()).map(|s| s.to_string());
    let template  = ChatTemplate::detect(&tokenizer, tmpl_str.as_deref());

    Ok(LoadedModel { model, tokenizer, template, config: cfg, gpu })
}

async fn do_load_model(state: &Arc<AppState>, model_id: &str) -> Result<()> {
    let cfg = find_model(&state.all_models, model_id)
        .ok_or_else(|| anyhow::anyhow!("Model not found: {}. Available: {}",
            model_id,
            state.all_models.iter().map(|m| m.id.as_str()).collect::<Vec<_>>().join(", ")))?
        .clone();

    // Acquire exclusive load lock — other requests queue here
    eprintln!("[{}] [load] Waiting for load lock (model={})", ts(), model_id);
    let _load_guard = state.load_lock.lock().await;
    eprintln!("[{}] [load] Got load lock", ts());

    // Double-check: another request may have loaded this model while we waited
    {
        let guard = state.loaded.lock().await;
        if let Some(m) = guard.as_ref() {
            let matches = m.config.id.to_lowercase() == model_id.to_lowercase()
                || model_id.to_lowercase().contains(&m.config.id.to_lowercase());
            if matches {
                eprintln!("[{}] [load] Model already loaded by another request, reusing", ts());
                return Ok(());
            }
        }
    }

    eprintln!("[{}] [load] Dropping current model to free VRAM", ts());
    *state.loaded.lock().await = None;

    eprintln!("[{}] [load] Loading: {} ({})", ts(), cfg.id, cfg.path);
    let ctx_len = cfg.ctx_len.unwrap_or(state.defaults.ctx_len);

    let loaded = tokio::task::spawn_blocking(move || {
        eprintln!("[load] spawn_blocking started");
        let result = load_model_blocking(cfg, ctx_len);
        eprintln!("[load] spawn_blocking finished: {}", if result.is_ok() { "ok" } else { "err" });
        result
    }).await??;

    eprintln!("[{}] [load] Model ready: {} | layers={} embd={} vocab={}",
        ts(), loaded.config.id,
        loaded.model.config.n_layers, loaded.model.config.n_embd, loaded.model.config.n_vocab);
    *state.loaded.lock().await = Some(loaded);
    Ok(())
}

// Ensure the right model is loaded, hot-swapping if needed.
async fn ensure_model(state: &Arc<AppState>, req_model: Option<&str>) -> Result<(), (StatusCode, String)> {
    let req_id = match req_model {
        None => return Ok(()), // use whatever is loaded
        Some(m) => m,
    };

    let already_loaded = {
        let guard = state.loaded.lock().await;
        guard.as_ref().map(|l| {
            l.config.id.to_lowercase() == req_id.to_lowercase()
            || req_id.to_lowercase().contains(&l.config.id.to_lowercase())
        }).unwrap_or(false)
    };

    if !already_loaded {
        eprintln!("[server] Hot-swapping to model: {}", req_id);
        do_load_model(state, req_id).await
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    }
    Ok(())
}

// Auth middleware — pass-through when no api_key configured
async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: Request<axum::body::Body>,
    next: Next,
) -> impl IntoResponse {
    if let Some(expected) = &state.api_key {
        let provided = headers.get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        if provided != Some(expected.as_str()) {
            return openai_error(
                StatusCode::UNAUTHORIZED,
                "invalid_api_key",
                "Invalid API key".to_string(),
            ).into_response();
        }
    }
    next.run(request).await
}

async fn models_list(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let loaded_id = state.loaded.lock().await
        .as_ref().map(|m| m.config.id.clone());

    let data = state.all_models.iter().map(|m| {
        let is_loaded = loaded_id.as_deref() == Some(m.id.as_str());
        ModelInfo {
            id:         m.id.clone(),
            object:     "model".to_string(),
            created:    now_secs(),
            owned_by:   "gguf-rs".to_string(),
            permission: if is_loaded { vec![json!("loaded")] } else { vec![] },
        }
    }).collect();
    Json(ModelListResponse { object: "list".to_string(), data })
}

async fn model_info(
    State(state): State<Arc<AppState>>,
    AxumPath(model_id): AxumPath<String>,
) -> impl IntoResponse {
    match find_model(&state.all_models, &model_id) {
        Some(m) => Json(json!({
            "id":       m.id,
            "object":   "model",
            "created":  now_secs(),
            "owned_by": "gguf-rs",
        })).into_response(),
        None => openai_error(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("Model '{}' not found", model_id),
        ).into_response(),
    }
}

async fn load_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LoadModelRequest>,
) -> impl IntoResponse {
    match do_load_model(&state, &req.model).await {
        Ok(_)  => (StatusCode::OK, Json(json!({"status":"loaded","model":req.model}))).into_response(),
        Err(e) => openai_error(StatusCode::BAD_REQUEST, "load_error", e.to_string()).into_response(),
    }
}

// Shared generation parameters passed into the blocking thread
struct GenParams {
    prompt_ids:  Vec<u32>,
    max_tokens:  usize,
    temperature: f32,
    top_k:       usize,
    top_p:       f32,
    rep_penalty: f32,
    stop_strings: Vec<String>,
    ctx_len:     usize,
    prefill_batch: usize,
}

// Return value from the generation loop
struct GenResult {
    text:          String,
    tokens_gen:    usize,
    finish_reason: &'static str,
}

// Run prefill then decode; shared by stream and non-stream paths.
// Must be called from spawn_blocking — takes &mut LoadedModel directly.
fn run_generation(loaded: &mut LoadedModel, p: &GenParams) -> GenResult {
    let c       = &loaded.model.config;
    let ctx_len = p.ctx_len.min(c.n_ctx);
    let mut cpu_cache = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());
    let mut tok_stops = loaded.template.stop_tokens(&loaded.tokenizer);

    // encode any request-level stop strings and add to stop set
    for s in &p.stop_strings {
        let ids = loaded.tokenizer.encode(s, false);
        if ids.len() == 1 { tok_stops.push(ids[0]); }
    }

    let mut logits = if !p.prompt_ids.is_empty() {
        match loaded.gpu.as_mut() {
            Some(g) => {
                let toks: Vec<usize> = p.prompt_ids.iter().map(|&id| id as usize).collect();
                loaded.model.forward_gpu_prefill(&toks, 0, g, p.prefill_batch)
            }
            None => {
                let mut l = vec![0f32; c.n_vocab];
                for (i, &id) in p.prompt_ids.iter().enumerate() {
                    l = loaded.model.forward_cpu(id as usize, i, &mut cpu_cache);
                }
                l
            }
        }
    } else { vec![0f32; c.n_vocab] };

    let mut pos    = p.prompt_ids.len();
    // Fixed-size ring buffer for repetition penalty context
    let rep_window = 64usize;
    let mut recent: Vec<u32> = Vec::with_capacity(rep_window);
    let mut out    = String::new();
    let mut count  = 0usize;
    let mut finish_reason = "length";

    let t0 = std::time::Instant::now();
    eprintln!("[{}] [gen] start pos={} max={}", ts(), pos, p.max_tokens);

    for _ in 0..p.max_tokens {
        let next = gguf_rs::sampler::sample(
            &mut logits, p.temperature, p.top_k, p.top_p, p.rep_penalty, &recent,
        );
        if tok_stops.contains(&(next as u32)) {
            finish_reason = "stop";
            break;
        }

        let word = loaded.tokenizer.decode(next as u32);

        // O(1) ring-buffer push
        if recent.len() == rep_window { recent.remove(0); }
        recent.push(next as u32);

        logits = match loaded.gpu.as_mut() {
            Some(g) => loaded.model.forward_gpu(next, pos, g),
            None    => loaded.model.forward_cpu(next, pos, &mut cpu_cache),
        };
        pos   += 1;
        count += 1;
        if !word.is_empty() { out.push_str(&word); }
    }

    let elapsed = t0.elapsed().as_secs_f32();
    eprintln!("[{}] [gen] done {} tok {:.2}s ({:.1} tok/s) reason={} preview={:?}",
        ts(), count, elapsed, count as f32 / elapsed.max(0.001),
        finish_reason, &out[..out.len().min(60)]);

    GenResult { text: out, tokens_gen: count, finish_reason }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let req_model = req.model.clone();
    eprintln!("[{}] [req] POST /v1/chat/completions model={:?} stream={:?} msgs={}",
        ts(), req_model, req.stream, req.messages.len());

    // Hot-swap if a different model is requested
    if let Err((status, msg)) = ensure_model(&state, req_model.as_deref()).await {
        return Err(openai_error(status, "model_error", msg).into_response());
    }

    let temperature  = req.temperature.unwrap_or(state.defaults.temperature);
    let top_k        = req.top_k.unwrap_or(state.defaults.top_k);
    let top_p        = req.top_p.unwrap_or(state.defaults.top_p);
    let max_tokens   = req.max_tokens.unwrap_or(state.defaults.max_tokens);
    let rep_penalty  = req.repetition_penalty.unwrap_or(state.defaults.repetition_penalty);
    let stream       = req.stream.unwrap_or(false);
    let messages     = req.messages;
    let stop_strings = parse_stop(req.stop.as_ref());
    let defaults     = state.defaults.clone();

    // Brief lock: build prompt + tokenize, release before inference
    let (prompt_ids, model_id, prompt_tokens) = {
        let guard = state.loaded.lock().await;
        let loaded = guard.as_ref().ok_or_else(|| {
            openai_error(StatusCode::SERVICE_UNAVAILABLE, "no_model", "No model loaded".to_string())
                .into_response()
        })?;

        // Build full prompt using chat template
        // user_turn() already appends the assistant header so generation starts correctly
        let mut full = String::new();

        // system message from request or model config
        let sys_content = messages.iter()
            .find(|m| m.role == "system")
            .map(|m| m.content.clone())
            .or_else(|| loaded.config.system_prompt.clone())
            .unwrap_or_else(|| "You are a helpful assistant.".to_string());

        full.push_str(&loaded.template.system_prompt(&sys_content));

        for msg in messages.iter().filter(|m| m.role != "system") {
            match msg.role.as_str() {
                "user"      => full.push_str(&loaded.template.user_turn(&msg.content)),
                "assistant" => full.push_str(&format!("{}\n", msg.content)),
                _           => {}
            }
        }

        eprintln!("[{}] [prompt] {} chars, preview: {:?}...",
            ts(), full.len(), &full[..full.len().min(100)]);

        let add_bos = loaded.template.uses_bos() && loaded.tokenizer.add_bos_token;
        let ids     = loaded.tokenizer.encode(&full, add_bos);
        let n       = ids.len();
        let mid     = loaded.config.id.clone();
        eprintln!("[{}] [prompt] {} tokens (model={})", ts(), n, mid);
        (ids, mid, n)
    }; // lock released — VkCtx never held across an await

    let gen_params = GenParams {
        prompt_ids,
        max_tokens,
        temperature,
        top_k,
        top_p,
        rep_penalty,
        stop_strings,
        ctx_len:      defaults.ctx_len,
        prefill_batch: defaults.prefill_batch,
    };

    if stream {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<
            Result<axum::response::sse::Event, std::convert::Infallible>
        >(256);
        let state2   = state.clone();
        let chunk_id = format!("chatcmpl-{}", Uuid::new_v4());
        let mid2     = model_id.clone();

        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            let mut guard = rt.block_on(state2.loaded.lock());
            let loaded = match guard.as_mut() { Some(l) => l, None => return };

            let send = |chunk: ChatCompletionChunk| {
                if let Ok(json) = serde_json::to_string(&chunk) {
                    let _ = rt.block_on(tx.send(
                        Ok(axum::response::sse::Event::default().data(json))
                    ));
                }
            };

            // opening role delta
            send(ChatCompletionChunk {
                id: chunk_id.clone(), object: "chat.completion.chunk".to_string(),
                created: now_secs(), model: mid2.clone(),
                choices: vec![ChoiceDelta { index: 0, finish_reason: None,
                    delta: Delta { role: Some("assistant".to_string()), content: None } }],
            });

            // Stream tokens from shared generation loop
            let c       = &loaded.model.config;
            let ctx_len = gen_params.ctx_len.min(c.n_ctx);
            let mut cpu_cache = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());
            let mut tok_stops = loaded.template.stop_tokens(&loaded.tokenizer);
            for s in &gen_params.stop_strings {
                let ids = loaded.tokenizer.encode(s, false);
                if ids.len() == 1 { tok_stops.push(ids[0]); }
            }

            let mut logits = if !gen_params.prompt_ids.is_empty() {
                match loaded.gpu.as_mut() {
                    Some(g) => {
                        let toks: Vec<usize> = gen_params.prompt_ids.iter().map(|&id| id as usize).collect();
                        loaded.model.forward_gpu_prefill(&toks, 0, g, gen_params.prefill_batch)
                    }
                    None => {
                        let mut l = vec![0f32; c.n_vocab];
                        for (i, &id) in gen_params.prompt_ids.iter().enumerate() {
                            l = loaded.model.forward_cpu(id as usize, i, &mut cpu_cache);
                        }
                        l
                    }
                }
            } else { vec![0f32; c.n_vocab] };

            let rep_window = 64usize;
            let mut pos       = gen_params.prompt_ids.len();
            let mut recent    = Vec::with_capacity(rep_window);
            let mut gen_count = 0usize;
            let mut finish_reason = "length";
            let t0 = std::time::Instant::now();
            eprintln!("[{}] [gen] stream start pos={} max={}", ts(), pos, gen_params.max_tokens);

            for _ in 0..gen_params.max_tokens {
                let next = gguf_rs::sampler::sample(
                    &mut logits, gen_params.temperature, gen_params.top_k,
                    gen_params.top_p, gen_params.rep_penalty, &recent,
                );
                if tok_stops.contains(&(next as u32)) { finish_reason = "stop"; break; }

                let word = loaded.tokenizer.decode(next as u32);
                if recent.len() == rep_window { recent.remove(0); }
                recent.push(next as u32);

                logits = match loaded.gpu.as_mut() {
                    Some(g) => loaded.model.forward_gpu(next, pos, g),
                    None    => loaded.model.forward_cpu(next, pos, &mut cpu_cache),
                };
                pos       += 1;
                gen_count += 1;

                if !word.is_empty() {
                    send(ChatCompletionChunk {
                        id: chunk_id.clone(), object: "chat.completion.chunk".to_string(),
                        created: now_secs(), model: mid2.clone(),
                        choices: vec![ChoiceDelta { index: 0, finish_reason: None,
                            delta: Delta { role: None, content: Some(word) } }],
                    });
                }
            }

            let elapsed = t0.elapsed().as_secs_f32();
            eprintln!("[{}] [gen] stream done {} tok {:.2}s ({:.1} tok/s) reason={}",
                ts(), gen_count, elapsed, gen_count as f32 / elapsed.max(0.001), finish_reason);

            send(ChatCompletionChunk {
                id: chunk_id, object: "chat.completion.chunk".to_string(),
                created: now_secs(), model: mid2,
                choices: vec![ChoiceDelta { index: 0,
                    finish_reason: Some(finish_reason.to_string()),
                    delta: Delta { role: None, content: None } }],
            });
            let _ = rt.block_on(tx.send(
                Ok(axum::response::sse::Event::default().data("[DONE]"))
            ));
        });

        let out = async_stream::stream! {
            while let Some(ev) = rx.recv().await { yield ev; }
        };
        Ok((StatusCode::OK, Sse::new(out).keep_alive(KeepAlive::default())).into_response())

    } else {
        let state2 = state.clone();
        let (completion, comp_tokens, finish_reason) =
            tokio::task::spawn_blocking(move || -> Result<(String, usize, &'static str)> {
                let rt = tokio::runtime::Handle::current();
                let mut guard = rt.block_on(state2.loaded.lock());
                let loaded = guard.as_mut().ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
                let r = run_generation(loaded, &gen_params);
                Ok((r.text, r.tokens_gen, r.finish_reason))
            }).await
                .map_err(|e| openai_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()).into_response())?
                .map_err(|e: anyhow::Error| openai_error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", e.to_string()).into_response())?;

        let guard      = state.loaded.lock().await;
        let model_name = guard.as_ref().map(|m| m.config.id.clone()).unwrap_or(model_id);

        let resp = ChatCompletion {
            id:      format!("chatcmpl-{}", Uuid::new_v4()),
            object:  "chat.completion".to_string(),
            created: now_secs(),
            model:   model_name,
            choices: vec![Choice {
                index:         0,
                message:       ChatMessage { role: "assistant".to_string(), content: completion },
                finish_reason: finish_reason.to_string(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens: comp_tokens,
                total_tokens:      prompt_tokens + comp_tokens,
            },
        };
        Ok((StatusCode::OK, Json(resp)).into_response())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = load_config("config.yaml")
        .or_else(|_| load_config("config.example.yaml"))
        .unwrap_or_else(|_| { eprintln!("Config not found. Create config.yaml"); std::process::exit(1) });

    gguf_rs::sampler::set_seed(config.defaults.seed);

    let api_key = config.server.api_key.clone().filter(|k| !k.is_empty());
    if api_key.is_some() {
        eprintln!("[{}] API key auth enabled", ts());
    }

    let state = Arc::new(AppState {
        loaded:     Mutex::new(None),
        load_lock:  Mutex::new(()),
        all_models: config.models.clone(),
        defaults:   config.defaults.clone(),
        api_key,
    });

    // No model loaded on startup — hot-swap on first request
    eprintln!("[{}] Server starting. Available models:", ts());
    for m in &config.models {
        eprintln!("  - {} | {} | gpu={} ctx={:?}", m.id, m.path, m.gpu, m.ctx_len);
    }
    eprintln!("[{}] Defaults: temp={} top_k={} top_p={} max_tokens={} prefill_batch={}",
        ts(), config.defaults.temperature, config.defaults.top_k, config.defaults.top_p,
        config.defaults.max_tokens, config.defaults.prefill_batch);

    let app = Router::new()
        .route("/v1/models",                get(models_list))
        .route("/v1/models/:model_id",      get(model_info))
        .route("/v1/chat/completions",      post(chat_completions))
        .route("/v1/models/load",           post(load_handler))
        .with_state(state.clone())
        .layer(middleware::from_fn_with_state(state, auth_middleware))
        .layer(tower_http::cors::CorsLayer::permissive());

    let addr = format!("{}:{}", config.server.host, config.server.port);
    eprintln!("[{}] Server on http://{}", ts(), addr);
    axum::serve(tokio::net::TcpListener::bind(&addr).await?, app).await?;
    Ok(())
}