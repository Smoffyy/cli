use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;
use anyhow::Result;
use clap::Parser;
use gguf_rs::model::llama::{KvCache, LlamaModel};
use gguf_rs::tokenizer::bpe::Tokenizer;
use gguf_rs::tokenizer::chat::ChatTemplate;
use gguf_rs::gpu::VkCtx;

#[derive(Parser)]
#[command(name="gguf-rs-cli", about="Pure Rust local LLM inference engine with Vulkan GPU acceleration")]
struct Args {
    #[arg(short, long)]
    model: String,
    #[arg(short, long)]
    prompt: Option<String>,
    #[arg(short, long)]
    system: Option<String>,
    #[arg(short='n', long, default_value_t=512)]
    max_tokens: usize,
    #[arg(short='t', long, default_value_t=0.7)]
    temperature: f32,
    #[arg(long, default_value_t=40)]
    top_k: usize,
    #[arg(long, default_value_t=0.9)]
    top_p: f32,
    #[arg(long, default_value_t=1.1)]
    rep_penalty: f32,
    #[arg(short='c', long, default_value_t=8192)]
    ctx_len: usize,
    #[arg(long, default_value_t=false)]
    gpu: bool,
    #[arg(long, default_value_t=false)]
    smart_context: bool,
    #[arg(long, default_value_t=false)]
    stats: bool,
    #[arg(long, default_value_t=false)]
    debug_tokens: bool,
    #[arg(long, default_value_t=false)]
    debug_gpu: bool,
    #[arg(long, default_value_t=42)]
    seed: u64,
    /// Number of prompt tokens to process per GPU submit during prefill (GPU only)
    #[arg(long, default_value_t=512)]
    prefill_batch: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    gguf_rs::sampler::set_seed(args.seed);

    let mut gpu: Option<VkCtx> = if args.gpu {
        match VkCtx::init() {
            Ok(mut g) => {
                g.debug_gpu = args.debug_gpu;
                eprintln!("GPU ready.");
                Some(g)
            }
            Err(e) => { eprintln!("No GPU — using CPU. ({e})"); None }
        }
    } else { None };

    let path = Path::new(&args.model);
    let (model, gguf) = LlamaModel::load(path, args.ctx_len, gpu.as_mut())?;
    let tok = Tokenizer::from_gguf(&gguf)?;

    let model_name = gguf.metadata.get("general.name").and_then(|v| v.as_str()).unwrap_or("Unknown");
    let arch       = gguf.metadata.get("general.architecture").and_then(|v| v.as_str()).unwrap_or("unknown");
    eprintln!("Model: {} ({})", model_name, arch);
    eprintln!("EOS tokens: {:?}", tok.eos_ids);

    let tmpl_str = gguf.metadata.get("tokenizer.chat_template")
        .and_then(|v| v.as_str()).map(|s| s.to_string());
    let tmpl = ChatTemplate::detect(&tok, tmpl_str.as_deref());

    let system = args.system.unwrap_or_else(|| {
        extract_default_system(tmpl_str.as_deref())
            .unwrap_or_else(|| "You are a helpful assistant.".to_string())
    });

    eprintln!("Tokenizer: {:?} | Template: {:?} | add_bos: {}",
        tok.tok_model, tmpl, tok.add_bos_token);
    eprintln!("System: {}", &system[..system.len().min(80)]);
    eprintln!("Params: temp={} top_k={} top_p={} rep_penalty={} smart_context={} prefill_batch={}",
        args.temperature, args.top_k, args.top_p, args.rep_penalty, args.smart_context, args.prefill_batch);

    let c       = &model.config;
    let ctx_len = args.ctx_len.min(c.n_ctx);
    eprintln!("Context: {} tokens | Mode: {}\n",
        ctx_len, if gpu.is_some() { "GPU (Vulkan)" } else { "CPU" });

    let mut cpu_cache = KvCache::new(c.n_layers, ctx_len, c.n_kv_heads, c.head_dim());
    let stops         = tmpl.stop_tokens(&tok);
    let mut recent: Vec<u32> = Vec::with_capacity(64);

    let sys_text = tmpl.system_prompt(&system);
    let sys_ids: Vec<u32> = if sys_text.is_empty() { vec![] }
    else { tok.encode(&sys_text, tmpl.uses_bos() && tok.add_bos_token) };

    match args.prompt {
        Some(ref p) => {
            print!("{}", p); io::stdout().flush()?;
            let ids = tok.encode(p, true);
            let t0  = Instant::now();
            let (mut pos, mut logits) = prefill_split(&model, &ids, 0, &mut gpu, &mut cpu_cache, args.prefill_batch);
            let pm  = t0.elapsed().as_millis();
            let t1  = Instant::now();
            let mut dummy: Vec<u32> = Vec::new();
            let gen = generate_collect(&model, &tok, &mut pos, &mut logits,
                args.max_tokens, args.temperature, args.top_k, args.top_p,
                args.rep_penalty, ctx_len, &stops, &mut gpu, &mut cpu_cache,
                &mut recent, &sys_ids, &mut dummy, args.smart_context, args.prefill_batch);
            println!();
            if args.stats {
                let gs = t1.elapsed().as_secs_f32();
                eprintln!("[Stats] prefill: {} tok in {}ms | generated: {} tok in {:.2}s ({:.1} tok/s)",
                    ids.len(), pm, gen.len(), gs, gen.len() as f32 / gs.max(0.001));
            }
        }
        None => {
            eprintln!("Type your message. /quit to exit.");

            if args.debug_tokens {
                eprintln!("[system prompt: {:?}]", &sys_text);
                eprintln!("[system ids: {:?}]", &sys_ids[..sys_ids.len().min(15)]);
            }

            let pt = Instant::now();
            let (mut pos, _) = prefill_split(&model, &sys_ids, 0, &mut gpu, &mut cpu_cache, args.prefill_batch);
            if args.stats {
                eprintln!("[Stats] system prefill: {} tok in {}ms",
                    sys_ids.len(), pt.elapsed().as_millis());
            }

            let mut history: Vec<u32> = Vec::new();
            let stdin = io::stdin();

            loop {
                eprint!("\nYou: "); io::stderr().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 { break; }
                let msg = line.trim();
                if msg.is_empty()                   { continue; }
                if msg == "/quit" || msg == "/exit" { break; }

                let turn     = tmpl.user_turn(msg);
                let turn_ids = tok.encode(&turn, false);
                if args.debug_tokens {
                    eprintln!("[user turn ids: {:?}]", &turn_ids[..turn_ids.len().min(15)]);
                }

                let reserve = if args.smart_context {
                    (ctx_len / 8).max(32).min(args.max_tokens)
                } else {
                    32.min(args.max_tokens)
                };

                if pos + turn_ids.len() + reserve >= ctx_len {
                    if history.is_empty() {
                        eprintln!("[Context: too small, clearing conversation]");
                        let (p, _) = prefill_split(&model, &sys_ids, 0, &mut gpu, &mut cpu_cache, args.prefill_batch);
                        pos = p;
                    } else {
                        pos = rebuild_cache(&model, &sys_ids, &mut history,
                                            &mut gpu, &mut cpu_cache, args.prefill_batch);
                    }
                }

                let pt0 = Instant::now();
                let (new_pos, mut logits) = prefill_split(&model, &turn_ids, pos,
                                                           &mut gpu, &mut cpu_cache, args.prefill_batch);
                let pm = pt0.elapsed().as_millis();
                pos = new_pos;
                history.extend_from_slice(&turn_ids);

                eprint!("Assistant: "); io::stderr().flush()?;
                let gt0 = Instant::now();
                let gen_ids = generate_collect(
                    &model, &tok, &mut pos, &mut logits,
                    args.max_tokens, args.temperature, args.top_k, args.top_p,
                    args.rep_penalty, ctx_len, &stops, &mut gpu, &mut cpu_cache,
                    &mut recent, &sys_ids, &mut history, args.smart_context, args.prefill_batch,
                );
                history.extend_from_slice(&gen_ids);
                println!();

                if args.stats {
                    let gs = gt0.elapsed().as_secs_f32();
                    eprintln!("[Stats] prefill: {} tok in {}ms | generated: {} tok in {:.2}s ({:.1} tok/s) | ctx: {}/{}",
                        turn_ids.len(), pm,
                        gen_ids.len(), gs, gen_ids.len() as f32 / gs.max(0.001),
                        pos, ctx_len);
                }
                recent.clear();
            }
        }
    }
    Ok(())
}

fn rebuild_cache(
    model:        &LlamaModel,
    sys_ids:      &[u32],
    history:      &mut Vec<u32>,
    gpu:          &mut Option<VkCtx>,
    cpu_cache:    &mut KvCache,
    prefill_batch: usize,
) -> usize {
    let drop_n = (history.len() / 4).max(1).min(history.len());
    history.drain(..drop_n);

    let t0 = std::time::Instant::now();
    eprintln!("[Context: dropped {} tokens, replaying system({}) + history({})...]",
        drop_n, sys_ids.len(), history.len());

    if gpu.is_none() {
        let c = &model.config;
        for l in 0..c.n_layers { cpu_cache.k[l].fill(0.0); cpu_cache.v[l].fill(0.0); }
    }

    let (mut pos, _) = prefill_split(model, sys_ids, 0, gpu, cpu_cache, prefill_batch);
    for (i, &id) in history.iter().enumerate() {
        let _ = match gpu.as_mut() {
            Some(g) => model.forward_gpu(id as usize, pos + i, g),
            None    => model.forward_cpu(id as usize, pos + i, cpu_cache),
        };
    }
    pos += history.len();
    eprintln!("[Context: rebuilt in {}ms, pos now {}]",
        t0.elapsed().as_millis(), pos);
    pos
}

fn prefill_split(
    model:        &LlamaModel,
    ids:          &[u32],
    start:        usize,
    gpu:          &mut Option<VkCtx>,
    cpu_cache:    &mut KvCache,
    prefill_batch: usize,
) -> (usize, Vec<f32>) {
    if ids.is_empty() { return (start, vec![0f32; model.config.n_vocab]); }
    let logits = match gpu.as_mut() {
        Some(g) => {
            let tokens: Vec<usize> = ids.iter().map(|&id| id as usize).collect();
            model.forward_gpu_prefill(&tokens, start, g, prefill_batch)
        }
        None => {
            let mut logits = vec![0f32; model.config.n_vocab];
            for (i, &id) in ids.iter().enumerate() {
                logits = model.forward_cpu(id as usize, start + i, cpu_cache);
            }
            logits
        }
    };
    (start + ids.len(), logits)
}

fn generate_collect(
    model:         &LlamaModel,
    tok:           &Tokenizer,
    pos:           &mut usize,
    last:          &mut Vec<f32>,
    max:           usize,
    temperature:   f32,
    top_k:         usize,
    top_p:         f32,
    rep_penalty:   f32,
    ctx:           usize,
    stops:         &[u32],
    gpu:           &mut Option<VkCtx>,
    cpu_cache:     &mut KvCache,
    recent:        &mut Vec<u32>,
    sys_ids:       &[u32],
    history:       &mut Vec<u32>,
    smart_context: bool,
    prefill_batch: usize,
) -> Vec<u32> {
    let mut generated: Vec<u32> = Vec::new();

    loop {
        let next = gguf_rs::sampler::sample(last, temperature, top_k, top_p, rep_penalty, recent);
        if stops.contains(&(next as u32)) { break; }
        if generated.len() >= max         { break; }

        if *pos >= ctx - 1 {
            if !smart_context { break; }
            history.extend_from_slice(&generated);
            generated.clear();
            *pos = rebuild_cache(model, sys_ids, history, gpu, cpu_cache, prefill_batch);
            if let Some(&last_id) = history.last().or(sys_ids.last()) {
                *last = match gpu.as_mut() {
                    Some(g) => model.forward_gpu(last_id as usize, *pos - 1, g),
                    None    => model.forward_cpu(last_id as usize, pos.saturating_sub(1), cpu_cache),
                };
            }
            continue;
        }

        let word = tok.decode(next as u32);
        if !word.is_empty() { print!("{}", word); io::stdout().flush().ok(); }

        recent.push(next as u32);
        if recent.len() > 64 { recent.remove(0); }
        generated.push(next as u32);

        *last = match gpu.as_mut() {
            Some(g) => model.forward_gpu(next, *pos, g),
            None    => model.forward_cpu(next, *pos, cpu_cache),
        };
        *pos += 1;
    }
    generated
}

fn extract_default_system(template: Option<&str>) -> Option<String> {
    let tmpl = template?;
    let mut sf = 0;
    while let Some(rel) = tmpl[sf..].find("<|im_start|>system\n") {
        let start = sf + rel + "<|im_start|>system\n".len();
        if let Some(end) = tmpl[start..].find("<|im_end|>") {
            let c = tmpl[start..start+end].trim();
            if !c.is_empty() && !c.contains("{{") && !c.contains("{%") && !c.contains("messages") {
                return Some(c.to_string());
            }
        }
        sf += rel + "<|im_start|>system\n".len();
    }
    for (open, close) in &[
        ("<<SYS>>\n",      "\n<</SYS>>"),
        ("<|system|>\n",   "<|end|>"),
        ("<|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>"),
    ] {
        if let Some(start) = tmpl.find(open) {
            let after = &tmpl[start + open.len()..];
            if let Some(end) = after.find(close) {
                let c = after[..end].trim();
                if !c.is_empty() && !c.contains("{{") && !c.contains("messages") {
                    return Some(c.to_string());
                }
            }
        }
    }
    None
}
