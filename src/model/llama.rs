use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use crate::gguf::{reader, types::GgufFile};
use crate::tensor::{dequant::QuantTensor, storage::TensorStorage};
use crate::model::config::ModelConfig;
use crate::math::{ops, rope};
use crate::gpu::{VkCtx, GpuTensor, ActBuf};

pub struct Weights {
    pub token_embd:  QuantTensor,
    pub output_norm: Vec<f32>,
    pub output:      QuantTensor,
    pub attn_norm:   Vec<Vec<f32>>,
    pub ffn_norm:    Vec<Vec<f32>>,
    pub attn_q:      Vec<QuantTensor>,
    pub attn_k:      Vec<QuantTensor>,
    pub attn_v:      Vec<QuantTensor>,
    pub attn_out:    Vec<QuantTensor>,
    pub ffn_gate:    Vec<QuantTensor>,
    pub ffn_up:      Vec<QuantTensor>,
    pub ffn_down:    Vec<QuantTensor>,
    pub attn_q_bias: Vec<Option<Vec<f32>>>,
    pub attn_k_bias: Vec<Option<Vec<f32>>>,
    pub attn_v_bias: Vec<Option<Vec<f32>>>,
    // F32 fallback for fused QKV (e.g. Qwen3.5 attn_qkv.weight); None = use quant tensor
    pub attn_q_f32:  Vec<Option<Vec<f32>>>,
    pub attn_k_f32:  Vec<Option<Vec<f32>>>,
    pub attn_v_f32:  Vec<Option<Vec<f32>>>,
}

pub struct GpuWeights {
    pub output:   Option<GpuTensor>,
    // Fused QKV buffers for architectures like Qwen3.5 (kept alive so slices stay valid)
    pub fused_qkv: Vec<Option<GpuTensor>>,
    pub attn_q:   Vec<Option<GpuTensor>>,
    pub attn_k:   Vec<Option<GpuTensor>>,
    pub attn_v:   Vec<Option<GpuTensor>>,
    pub attn_out: Vec<Option<GpuTensor>>,
    pub ffn_gate: Vec<Option<GpuTensor>>,
    pub ffn_up:   Vec<Option<GpuTensor>>,
    pub ffn_down: Vec<Option<GpuTensor>>,
}

pub struct GpuActs {
    pub x:           ActBuf,
    pub xn:          ActBuf,
    pub q:           ActBuf,
    pub k:           ActBuf,
    pub v:           ActBuf,
    pub attn_out:    ActBuf,
    pub proj:        ActBuf,
    pub gate:        ActBuf,
    pub up:          ActBuf,
    pub ff:          ActBuf,
    pub logits:      ActBuf,
    pub logits_rb:   ActBuf,
    pub k_cache:     Vec<ActBuf>,
    pub v_cache:     Vec<ActBuf>,
    pub scores:      ActBuf,
    pub ctx_len:     usize,
    pub attn_norms:  Vec<ActBuf>,
    pub ffn_norms:   Vec<ActBuf>,
    pub out_norm:    ActBuf,
    pub q_bias:      Vec<Option<ActBuf>>,
    pub k_bias:      Vec<Option<ActBuf>>,
    pub v_bias:      Vec<Option<ActBuf>>,
    // Batch activation buffers for GEMM-based prefill (sized for max_batch tokens)
    pub max_batch:   usize,
    pub x_batch:     ActBuf,
    pub xn_batch:    ActBuf,
    pub q_batch:     ActBuf,
    pub k_batch:     ActBuf,
    pub v_batch:     ActBuf,
    pub ao_batch:    ActBuf,
    pub proj_batch:  ActBuf,
    pub gate_batch:  ActBuf,
    pub up_batch:    ActBuf,
    pub ff_batch:    ActBuf,
}

pub struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}
impl KvCache {
    pub fn new(n_layers: usize, n_ctx: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let sz = n_ctx * n_kv_heads * head_dim;
        Self { k: vec![vec![0f32; sz]; n_layers], v: vec![vec![0f32; sz]; n_layers] }
    }
}

// Simple f32 matrix-vector multiply: out[r] = dot(weight_row_r, inp)
fn f32_matvec(w: &[f32], inp: &[f32], out: &mut [f32], cols: usize) {
    use rayon::prelude::*;
    out.par_iter_mut().enumerate().for_each(|(r, o)| {
        *o = w[r*cols..(r+1)*cols].iter().zip(inp.iter()).map(|(a,b)| a*b).sum();
    });
}

pub struct LlamaModel {
    pub config:   ModelConfig,
    pub weights:  Weights,
    pub gpu_w:    Option<GpuWeights>,
    pub gpu_acts: Option<GpuActs>,
}

impl LlamaModel {
    pub fn load(path: &Path, ctx_len: usize,
                gpu: Option<&mut VkCtx>) -> Result<(Self, GgufFile)> {
        eprintln!("Parsing GGUF...");
        let f    = std::fs::File::open(path)?;
        let gguf = reader::parse(std::io::BufReader::new(f))?;
        let cfg  = ModelConfig::from_gguf(&gguf)?;
        let stor = TensorStorage::new(path, gguf.data_offset)?;

        eprintln!("Config: {} layers | embd {} | heads {}/{} | ff {} | rope_base {}",
            cfg.n_layers, cfg.n_embd, cfg.n_heads, cfg.n_kv_heads,
            cfg.n_ff, cfg.rope_freq_base);
        // Print blk.0 tensor names to help diagnose unknown architectures
        let mut blk0: Vec<&str> = gguf.tensors.iter()
            .filter(|t| t.name.starts_with("blk.0."))
            .map(|t| t.name.as_str()).collect();
        blk0.sort();
        eprintln!("Tensors in blk.0: {}", blk0.join(", "));

        let tmap: HashMap<&str, _> = gguf.tensors.iter()
            .map(|t| (t.name.as_str(), t)).collect();

        // Helper: on missing tensor, print similar names for diagnostics
        let similar_names = |name: &str| -> String {
            let prefix = name.split('.').take(2).collect::<Vec<_>>().join(".");
            let mut found: Vec<&str> = tmap.keys()
                .filter(|k| k.starts_with(&prefix))
                .copied().take(8).collect();
            found.sort();
            found.join(", ")
        };
        let get_q = |name: &str| -> Result<QuantTensor> {
            let info = tmap.get(name).ok_or_else(|| {
                anyhow::anyhow!("Missing tensor: {}\n  Similar: {}", name, similar_names(name))
            })?;
            Ok(QuantTensor::new(stor.mmap.clone(), stor.tensor_offset(info),
                                info.byte_size(), info.typ, &info.dims))
        };
        let get_f = |name: &str| -> Result<Vec<f32>> {
            let info = tmap.get(name).ok_or_else(|| {
                anyhow::anyhow!("Missing tensor: {}\n  Similar: {}", name, similar_names(name))
            })?;
            let s = stor.tensor_offset(info);
            crate::tensor::dequant::dequantize(
                info.typ, &stor.mmap[s..s+info.byte_size()], info.n_elements())
        };
        // get_f_any: tries each name in order, returns first found
        let get_f_any = |names: &[&str]| -> Result<Vec<f32>> {
            for name in names {
                if let Some(info) = tmap.get(*name) {
                    let s = stor.tensor_offset(info);
                    return crate::tensor::dequant::dequantize(
                        info.typ, &stor.mmap[s..s+info.byte_size()], info.n_elements());
                }
            }
            let similar = similar_names(names[0]);
            Err(anyhow::anyhow!("Missing tensor (tried: {})\n  Similar: {}",
                names.join(", "), similar))
        };
        let get_q_any = |names: &[&str]| -> Result<QuantTensor> {
            for name in names {
                if let Some(info) = tmap.get(*name) {
                    return Ok(QuantTensor::new(stor.mmap.clone(), stor.tensor_offset(info),
                                               info.byte_size(), info.typ, &info.dims));
                }
            }
            let similar = similar_names(names[0]);
            Err(anyhow::anyhow!("Missing tensor (tried: {})\n  Similar: {}",
                names.join(", "), similar))
        };
        let get_bias = |name: &str| -> Option<Vec<f32>> {
            tmap.get(name).and_then(|info| {
                let s = stor.tensor_offset(info);
                crate::tensor::dequant::dequantize(
                    info.typ, &stor.mmap[s..s+info.byte_size()], info.n_elements()).ok()
            })
        };

        let mb = gguf.tensors.iter().map(|t| t.byte_size()).sum::<usize>() / 1_000_000;
        eprintln!("Loading weights (~{} MB)...", mb);

        let token_embd  = get_q("token_embd.weight")?;
        let output_norm = get_f("output_norm.weight")?;
        let output      = get_q("output.weight").or_else(|_| get_q("token_embd.weight"))?;

        let (mut an, mut fn_) = (vec![], vec![]);
        let (mut aq, mut ak, mut av, mut ao) = (vec![], vec![], vec![], vec![]);
        let (mut fg, mut fu, mut fd) = (vec![], vec![], vec![]);
        let (mut aqb, mut akb, mut avb) = (vec![], vec![], vec![]);
        let (mut aq_f32, mut ak_f32, mut av_f32): (Vec<Option<Vec<f32>>>, Vec<Option<Vec<f32>>>, Vec<Option<Vec<f32>>>) = (vec![], vec![], vec![]);

        for i in 0..cfg.n_layers {
            // Attention norm: standard or Qwen3/Gemma variant names
            an.push(get_f_any(&[
                &format!("blk.{}.attn_norm.weight",             i),
                &format!("blk.{}.pre_attention_layernorm.weight",i),
            ])?);
            // FFN norm: standard or post-attention-layernorm variant
            fn_.push(get_f_any(&[
                &format!("blk.{}.ffn_norm.weight",               i),
                &format!("blk.{}.post_attention_layernorm.weight",i),
                &format!("blk.{}.post_attention_norm.weight",     i),
                &format!("blk.{}.pre_ff_layernorm.weight",        i),
            ])?);
            // Weight tensors — try separate Q/K/V first, then fused QKV (Qwen3.5, etc.)
            let has_fused_qkv = tmap.contains_key(format!("blk.{}.attn_qkv.weight", i).as_str())
                && !tmap.contains_key(format!("blk.{}.attn_q.weight", i).as_str());
            if has_fused_qkv {
                // Fused QKV: dequantize all rows using get_row(), split into Q/K/V
                let fused = get_q_any(&[&format!("blk.{}.attn_qkv.weight", i)])?;
                let n_q   = cfg.n_heads    * cfg.head_dim();
                let n_k   = cfg.n_kv_heads * cfg.head_dim();
                let n_v   = cfg.n_kv_heads * cfg.head_dim();
                let cols  = cfg.n_embd;
                let mut q_f32 = Vec::with_capacity(n_q * cols);
                let mut k_f32 = Vec::with_capacity(n_k * cols);
                let mut v_f32 = Vec::with_capacity(n_v * cols);
                for r in 0..n_q           { q_f32.extend_from_slice(&fused.get_row(r)); }
                for r in n_q..n_q+n_k     { k_f32.extend_from_slice(&fused.get_row(r)); }
                for r in n_q+n_k..n_q+n_k+n_v { v_f32.extend_from_slice(&fused.get_row(r)); }
                eprintln!("[load] Layer {i}: fused QKV dequantized Q={n_q} K={n_k} V={n_v} rows");
                // Placeholder QuantTensors (CPU matvec will use f32 fallback below)
                // Push placeholder QuantTensors pointing to fused buffer (f32 fallback used for actual compute)
                aq.push(get_q_any(&[&format!("blk.{}.attn_qkv.weight", i)])?);
                ak.push(get_q_any(&[&format!("blk.{}.attn_qkv.weight", i)])?);
                av.push(get_q_any(&[&format!("blk.{}.attn_qkv.weight", i)])?);
                aq_f32.push(Some(q_f32));
                ak_f32.push(Some(k_f32));
                av_f32.push(Some(v_f32));
            } else {
                aq.push(get_q_any(&[
                    &format!("blk.{}.attn_q.weight",          i),
                    &format!("blk.{}.self_attn.q_proj.weight",i),
                ])?);
                ak.push(get_q_any(&[
                    &format!("blk.{}.attn_k.weight",          i),
                    &format!("blk.{}.self_attn.k_proj.weight",i),
                ])?);
                av.push(get_q_any(&[
                    &format!("blk.{}.attn_v.weight",          i),
                    &format!("blk.{}.self_attn.v_proj.weight",i),
                ])?);
                aq_f32.push(None); ak_f32.push(None); av_f32.push(None);
            }
            ao.push(get_q_any(&[
                &format!("blk.{}.attn_output.weight",            i),
                &format!("blk.{}.self_attn.o_proj.weight",       i),
            ])?);
            fg.push(get_q_any(&[
                &format!("blk.{}.ffn_gate.weight",               i),
                &format!("blk.{}.mlp.gate_proj.weight",          i),
            ])?);
            fu.push(get_q_any(&[
                &format!("blk.{}.ffn_up.weight",                 i),
                &format!("blk.{}.mlp.up_proj.weight",            i),
            ])?);
            fd.push(get_q_any(&[
                &format!("blk.{}.ffn_down.weight",               i),
                &format!("blk.{}.mlp.down_proj.weight",          i),
            ])?);
            aqb.push(get_bias(&format!("blk.{}.attn_q.bias",    i)));
            akb.push(get_bias(&format!("blk.{}.attn_k.bias",    i)));
            avb.push(get_bias(&format!("blk.{}.attn_v.bias",    i)));
        }
        eprintln!("Weights ready.");

        let weights = Weights {
            token_embd, output_norm, output,
            attn_norm: an, ffn_norm: fn_,
            attn_q: aq, attn_k: ak, attn_v: av, attn_out: ao,
            ffn_gate: fg, ffn_up: fu, ffn_down: fd,
            attn_q_bias: aqb, attn_k_bias: akb, attn_v_bias: avb,
            attn_q_f32: aq_f32, attn_k_f32: ak_f32, attn_v_f32: av_f32,
        };

        let (gpu_w, gpu_acts) = if let Some(g) = gpu {
            eprintln!("Uploading weight tensors to GPU...");
            let n_gpu = |o: &Option<GpuTensor>| if o.is_some() { 1usize } else { 0 };
            let output_gt = g.upload(&weights.output);
            // Upload Q/K/V weights; for fused QKV layers use pre-dequantized f32
            let attn_q: Vec<_> = (0..cfg.n_layers).map(|i| {
                if let Some(ref f32d) = weights.attn_q_f32[i] {
                    g.upload_f32_weight(f32d, (cfg.n_heads * cfg.head_dim()) as u32, cfg.n_embd as u32)
                } else { g.upload(&weights.attn_q[i]) }
            }).collect();
            let attn_k: Vec<_> = (0..cfg.n_layers).map(|i| {
                if let Some(ref f32d) = weights.attn_k_f32[i] {
                    g.upload_f32_weight(f32d, (cfg.n_kv_heads * cfg.head_dim()) as u32, cfg.n_embd as u32)
                } else { g.upload(&weights.attn_k[i]) }
            }).collect();
            let attn_v: Vec<_> = (0..cfg.n_layers).map(|i| {
                if let Some(ref f32d) = weights.attn_v_f32[i] {
                    g.upload_f32_weight(f32d, (cfg.n_kv_heads * cfg.head_dim()) as u32, cfg.n_embd as u32)
                } else { g.upload(&weights.attn_v[i]) }
            }).collect();
            let attn_out: Vec<_> = weights.attn_out.iter().map(|w| g.upload(w)).collect();
            let ffn_gate: Vec<_> = weights.ffn_gate.iter().map(|w| g.upload(w)).collect();
            let ffn_up:   Vec<_> = weights.ffn_up.iter().map(|w| g.upload(w)).collect();
            let ffn_down: Vec<_> = weights.ffn_down.iter().map(|w| g.upload(w)).collect();
            let on = [&attn_q,&attn_k,&attn_v,&attn_out,&ffn_gate,&ffn_up,&ffn_down]
                .iter().flat_map(|v|v.iter()).map(n_gpu).sum::<usize>() + n_gpu(&output_gt);
            eprintln!("{}/{} weight tensors on GPU, {} on CPU rayon",
                on, cfg.n_layers*7+1, cfg.n_layers*7+1 - on);

            let gw = GpuWeights {
                output: output_gt, attn_q, attn_k, attn_v, attn_out,
                ffn_gate, ffn_up, ffn_down,
                fused_qkv: vec![], // fused QKV handled via CPU dequant path
            };

            eprintln!("Allocating GPU activation buffers (ctx={})...", ctx_len);
            let hd      = cfg.head_dim();
            let kvd     = cfg.n_kv_heads * hd;
            let kv_size = (ctx_len * kvd * 4) as u64;

            let mut k_cache = Vec::with_capacity(cfg.n_layers);
            let mut v_cache = Vec::with_capacity(cfg.n_layers);
            for _ in 0..cfg.n_layers {
                k_cache.push(g.alloc_act(kv_size)?);
                v_cache.push(g.alloc_act(kv_size)?);
            }

            let scores = g.alloc_act((cfg.n_heads * ctx_len) as u64 * 4)?;

            let mut attn_norms = Vec::with_capacity(cfg.n_layers);
            let mut ffn_norms  = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                let ab = g.alloc_act(cfg.n_embd as u64 * 4)?;
                g.write_act(&ab, &weights.attn_norm[i]);
                attn_norms.push(ab);
                let ab = g.alloc_act(cfg.n_embd as u64 * 4)?;
                g.write_act(&ab, &weights.ffn_norm[i]);
                ffn_norms.push(ab);
            }
            let out_norm = g.alloc_act(cfg.n_embd as u64 * 4)?;
            g.write_act(&out_norm, &weights.output_norm);

            let mut q_bias_bufs = Vec::with_capacity(cfg.n_layers);
            let mut k_bias_bufs = Vec::with_capacity(cfg.n_layers);
            let mut v_bias_bufs = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                q_bias_bufs.push(if let Some(ref b) = weights.attn_q_bias[i] {
                    let ab = g.alloc_act(b.len() as u64 * 4)?;
                    g.write_act(&ab, b); Some(ab)
                } else { None });
                k_bias_bufs.push(if let Some(ref b) = weights.attn_k_bias[i] {
                    let ab = g.alloc_act(b.len() as u64 * 4)?;
                    g.write_act(&ab, b); Some(ab)
                } else { None });
                v_bias_bufs.push(if let Some(ref b) = weights.attn_v_bias[i] {
                    let ab = g.alloc_act(b.len() as u64 * 4)?;
                    g.write_act(&ab, b); Some(ab)
                } else { None });
            }

            // Allocate batch buffers for GEMM prefill (sized for prefill_batch tokens)
            let mb = ctx_len.min(512); // default batch size; caller can override
            let b64 = mb as u64;
            let acts = GpuActs {
                x:        g.alloc_act(cfg.n_embd as u64 * 4)?,
                xn:       g.alloc_act(cfg.n_embd as u64 * 4)?,
                q:        g.alloc_act((cfg.n_heads * hd) as u64 * 4)?,
                k:        g.alloc_act(kvd as u64 * 4)?,
                v:        g.alloc_act(kvd as u64 * 4)?,
                attn_out: g.alloc_act(cfg.n_embd as u64 * 4)?,
                proj:     g.alloc_act(cfg.n_embd as u64 * 4)?,
                gate:     g.alloc_act(cfg.n_ff as u64 * 4)?,
                up:       g.alloc_act(cfg.n_ff as u64 * 4)?,
                ff:       g.alloc_act(cfg.n_embd as u64 * 4)?,
                logits:   g.alloc_act(cfg.n_vocab as u64 * 4)?,
                logits_rb:g.alloc_readback(cfg.n_vocab as u64 * 4)?,
                k_cache, v_cache, scores,
                ctx_len,
                attn_norms, ffn_norms, out_norm,
                q_bias: q_bias_bufs,
                k_bias: k_bias_bufs,
                v_bias: v_bias_bufs,
                max_batch:  mb,
                x_batch:    g.alloc_act(b64 * cfg.n_embd as u64 * 4)?,
                xn_batch:   g.alloc_act(b64 * cfg.n_embd as u64 * 4)?,
                q_batch:    g.alloc_act(b64 * (cfg.n_heads * hd) as u64 * 4)?,
                k_batch:    g.alloc_act(b64 * kvd as u64 * 4)?,
                v_batch:    g.alloc_act(b64 * kvd as u64 * 4)?,
                ao_batch:   g.alloc_act(b64 * cfg.n_embd as u64 * 4)?,
                proj_batch: g.alloc_act(b64 * cfg.n_embd as u64 * 4)?,
                gate_batch: g.alloc_act(b64 * cfg.n_ff as u64 * 4)?,
                up_batch:   g.alloc_act(b64 * cfg.n_ff as u64 * 4)?,
                ff_batch:   g.alloc_act(b64 * cfg.n_embd as u64 * 4)?,
            };
            eprintln!("GPU buffers ready.");
            (Some(gw), Some(acts))
        } else {
            (None, None)
        };

        Ok((Self { config: cfg, weights, gpu_w, gpu_acts }, gguf))
    }

    // Records all GPU commands for one transformer layer (no begin/submit).
    // Used by both forward_gpu and forward_gpu_prefill.
    fn record_layer_gpu(&self, l: usize, pos: usize, gpu: &mut VkCtx) {
        let c   = &self.config;
        let gw  = self.gpu_w.as_ref().unwrap();
        let ga  = self.gpu_acts.as_ref().unwrap();
        let hd  = c.head_dim();
        let kvd = c.n_kv_heads * hd;

        gpu.cmd_rmsnorm(&ga.x, &ga.attn_norms[l], &ga.xn, c.n_embd as u32, c.rms_norm_eps);
        gpu.barrier();

        if let Some(t) = gw.attn_q[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.q); }
        if let Some(t) = gw.attn_k[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.k); }
        if let Some(t) = gw.attn_v[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.v); }
        gpu.barrier();

        if let Some(ref b) = ga.q_bias[l] { gpu.cmd_add(&ga.q, b, (c.n_heads * hd) as u32); }
        if let Some(ref b) = ga.k_bias[l] { gpu.cmd_add(&ga.k, b, kvd as u32); }
        if let Some(ref b) = ga.v_bias[l] { gpu.cmd_add(&ga.v, b, kvd as u32); }
        if ga.q_bias[l].is_some() || ga.k_bias[l].is_some() || ga.v_bias[l].is_some() {
            gpu.barrier();
        }

        gpu.cmd_rope(&ga.q, &ga.k, c.n_heads as u32, c.n_kv_heads as u32,
                     hd as u32, pos as u32, c.rope_freq_base);
        gpu.barrier();

        gpu.cmd_kv_write(&ga.k, &ga.v, &ga.k_cache[l], &ga.v_cache[l],
                         pos as u32, c.n_kv_heads as u32, hd as u32);
        gpu.barrier();

        gpu.cmd_attention(&ga.q, &ga.k_cache[l], &ga.v_cache[l],
                          &ga.attn_out, &ga.scores,
                          c.n_heads as u32, c.n_kv_heads as u32,
                          hd as u32, (pos + 1) as u32, ga.ctx_len as u32);
        gpu.barrier();

        if let Some(t) = gw.attn_out[l].as_ref() { gpu.cmd_gemv(t, &ga.attn_out, &ga.proj); }
        gpu.barrier();

        gpu.cmd_add(&ga.x, &ga.proj, c.n_embd as u32);
        gpu.barrier();

        gpu.cmd_rmsnorm(&ga.x, &ga.ffn_norms[l], &ga.xn, c.n_embd as u32, c.rms_norm_eps);
        gpu.barrier();

        if let Some(t) = gw.ffn_gate[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.gate); }
        if let Some(t) = gw.ffn_up[l].as_ref()   { gpu.cmd_gemv(t, &ga.xn, &ga.up); }
        gpu.barrier();

        gpu.cmd_swiglu(&ga.gate, &ga.up, c.n_ff as u32);
        gpu.barrier();

        if let Some(t) = gw.ffn_down[l].as_ref() { gpu.cmd_gemv(t, &ga.gate, &ga.ff); }
        gpu.barrier();

        gpu.cmd_add(&ga.x, &ga.ff, c.n_embd as u32);
        gpu.barrier();
    }

    /// GEMM-based batched prefill: processes all N prompt tokens using matrix×matrix ops.
    /// Replaces N×GEMV dispatches with ceil(N/chunk)×GEMM dispatches — massive speedup.
    /// Falls back to the GEMV path per-token if any layer weight is missing from GPU.
    pub fn forward_gpu_prefill_gemm(&self, tokens: &[usize], start_pos: usize,
                                     gpu: &mut VkCtx, chunk: usize) -> Vec<f32> {
        if tokens.is_empty() { return vec![0f32; self.config.n_vocab]; }
        let c   = &self.config;
        let gw  = self.gpu_w.as_ref().unwrap();
        let ga  = self.gpu_acts.as_ref().unwrap();
        let hd  = c.head_dim();
        let kvd = c.n_kv_heads * hd;

        // Pre-stage all embeddings in host-visible memory (no VRAM used)
        let mut all_embs = Vec::with_capacity(tokens.len() * c.n_embd);
        for &tok in tokens { all_embs.extend_from_slice(&self.weights.token_embd.get_row(tok)); }
        gpu.ensure_batch_emb_staging(all_embs.len() as u64 * 4);
        gpu.write_batch_emb(&all_embs);
        let emb_src = gpu.batch_emb_buf();

        let n        = tokens.len();
        let chunk_sz = chunk.max(1).min(ga.max_batch);
        let mut chunk_start = 0;

        while chunk_start < n {
            let chunk_end  = (chunk_start + chunk_sz).min(n);
            let b          = (chunk_end - chunk_start) as u32;
            let last_chunk = chunk_end == n;

            // DS estimate: per token × per layer ≈ 22 DS, ×2 headroom
            let ds_needed = b * (c.n_layers as u32) * 22 * 2 + b * c.n_layers as u32 + 64;
            gpu.begin_for_sets(ds_needed);

            // Load this chunk's embeddings into x_batch
            let emb_byte_off = (chunk_start * c.n_embd * 4) as u64;
            gpu.cmd_copy_to_act(&ga.x_batch, emb_src, emb_byte_off);

            for l in 0..c.n_layers {
                // ── Attention block ──────────────────────────────────────────
                gpu.cmd_batch_rmsnorm(&ga.x_batch, &ga.attn_norms[l], &ga.xn_batch,
                                      c.n_embd as u32, c.rms_norm_eps, b);
                gpu.barrier();

                if let Some(t) = gw.attn_q[l].as_ref() { gpu.cmd_gemm(t, &ga.xn_batch, &ga.q_batch, b); }
                if let Some(t) = gw.attn_k[l].as_ref() { gpu.cmd_gemm(t, &ga.xn_batch, &ga.k_batch, b); }
                if let Some(t) = gw.attn_v[l].as_ref() { gpu.cmd_gemm(t, &ga.xn_batch, &ga.v_batch, b); }
                gpu.barrier();

                if let Some(ref bias) = ga.q_bias[l] {
                    gpu.cmd_batch_bias_add(&ga.q_batch, bias, (c.n_heads * hd) as u32, b); }
                if let Some(ref bias) = ga.k_bias[l] {
                    gpu.cmd_batch_bias_add(&ga.k_batch, bias, kvd as u32, b); }
                if let Some(ref bias) = ga.v_bias[l] {
                    gpu.cmd_batch_bias_add(&ga.v_batch, bias, kvd as u32, b); }
                if ga.q_bias[l].is_some() || ga.k_bias[l].is_some() || ga.v_bias[l].is_some() {
                    gpu.barrier(); }

                gpu.cmd_batch_rope(&ga.q_batch, &ga.k_batch,
                                   c.n_heads as u32, c.n_kv_heads as u32, hd as u32,
                                   (start_pos + chunk_start) as u32, c.rope_freq_base, b);
                gpu.barrier();

                gpu.cmd_batch_kv_write(&ga.k_batch, &ga.v_batch,
                                       &ga.k_cache[l], &ga.v_cache[l],
                                       (start_pos + chunk_start) as u32,
                                       c.n_kv_heads as u32, hd as u32, b);
                gpu.barrier();

                // Attention: causal, sequential per token within the chunk
                for i in 0..b as usize {
                    let seq_len  = (start_pos + chunk_start + i + 1) as u32;
                    let q_off    = (i * c.n_heads * hd) as u32;
                    let ao_off   = (i * c.n_embd) as u32;
                    gpu.cmd_batch_attn_item(
                        &ga.q_batch, &ga.k_cache[l], &ga.v_cache[l],
                        &ga.ao_batch, &ga.scores,
                        c.n_heads as u32, c.n_kv_heads as u32, hd as u32,
                        seq_len, ga.ctx_len as u32, q_off, ao_off,
                    );
                    if i + 1 < b as usize { gpu.barrier(); }
                }
                gpu.barrier();

                if let Some(t) = gw.attn_out[l].as_ref() {
                    gpu.cmd_gemm(t, &ga.ao_batch, &ga.proj_batch, b); }
                gpu.barrier();

                gpu.cmd_batch_add(&ga.x_batch, &ga.proj_batch, c.n_embd as u32, b);
                gpu.barrier();

                // ── FFN block ────────────────────────────────────────────────
                gpu.cmd_batch_rmsnorm(&ga.x_batch, &ga.ffn_norms[l], &ga.xn_batch,
                                      c.n_embd as u32, c.rms_norm_eps, b);
                gpu.barrier();

                if let Some(t) = gw.ffn_gate[l].as_ref() { gpu.cmd_gemm(t, &ga.xn_batch, &ga.gate_batch, b); }
                if let Some(t) = gw.ffn_up[l].as_ref()   { gpu.cmd_gemm(t, &ga.xn_batch, &ga.up_batch,   b); }
                gpu.barrier();

                gpu.cmd_batch_swiglu(&ga.gate_batch, &ga.up_batch, c.n_ff as u32, b);
                gpu.barrier();

                if let Some(t) = gw.ffn_down[l].as_ref() {
                    gpu.cmd_gemm(t, &ga.gate_batch, &ga.ff_batch, b); }
                gpu.barrier();

                gpu.cmd_batch_add(&ga.x_batch, &ga.ff_batch, c.n_embd as u32, b);
                gpu.barrier();
            }

            if last_chunk {
                // Extract last token's state → compute logits
                let last_off = ((b as usize - 1) * c.n_embd * 4) as u64;
                gpu.cmd_copy_to_act(&ga.xn, ga.x_batch.buf, last_off);
                gpu.barrier();
                gpu.cmd_rmsnorm(&ga.xn, &ga.out_norm, &ga.xn, c.n_embd as u32, c.rms_norm_eps);
                gpu.barrier();
                if let Some(t) = gw.output.as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.logits); }
                gpu.submit_with_readback(&ga.logits, &ga.logits_rb);
            } else {
                gpu.submit_no_readback();
            }
            chunk_start = chunk_end;
        }

        gpu.read_logits(&ga.logits, &ga.logits_rb)
    }

        /// Batched GPU prefill: processes all tokens in chunks, one GPU submit per chunk.
    /// Eliminates the per-token fence-wait overhead that makes prefill slow.
    /// Only reads logits back from the final token.
    pub fn forward_gpu_prefill(&self, tokens: &[usize], start_pos: usize, gpu: &mut VkCtx, chunk: usize) -> Vec<f32> {
        if tokens.is_empty() { return vec![0f32; self.config.n_vocab]; }

        let c = &self.config;
        let chunk_sz = chunk.max(1);

        // Pre-stage all embeddings into one device-local buffer (uploaded before begin())
        let mut all_embs = Vec::with_capacity(tokens.len() * c.n_embd);
        for &tok in tokens { all_embs.extend_from_slice(&self.weights.token_embd.get_row(tok)); }
        let (emb_buf, emb_mem) = gpu.upload_f32_host_visible(&all_embs);

        let n = tokens.len();
        let mut chunk_start = 0;
        while chunk_start < n {
            let chunk_end  = (chunk_start + chunk_sz).min(n);
            let last_chunk = chunk_end == n;

            // DS count per token: ~20 dispatches/layer (rmsnorm,Q,K,V,biases,rope,kv_write,
            // attn,proj,add,rmsnorm2,gate,up,swiglu,down,add) + 2 for output = ~22/layer.
            // Over-allocate by 2x so any model fits without pool exhaustion.
            let tokens_in_chunk = (chunk_end - chunk_start) as u32;
            let ds_per_token    = (c.n_layers as u32) * 22 + 4;
            gpu.begin_for_sets(tokens_in_chunk * ds_per_token * 2);
            for i in chunk_start..chunk_end {
                // Load this token's pre-staged embedding into ga.x
                let ga = self.gpu_acts.as_ref().unwrap();
                gpu.cmd_copy_to_act(&ga.x, emb_buf, (i * c.n_embd * 4) as u64);
                for l in 0..c.n_layers {
                    self.record_layer_gpu(l, start_pos + i, gpu);
                }
            }

            if last_chunk {
                // Final token: compute output norm + logits, then readback
                let ga = self.gpu_acts.as_ref().unwrap();
                let gw = self.gpu_w.as_ref().unwrap();
                gpu.cmd_rmsnorm(&ga.x, &ga.out_norm, &ga.xn, c.n_embd as u32, c.rms_norm_eps);
                gpu.barrier();
                if let Some(t) = gw.output.as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.logits); }
                gpu.submit_with_readback(&ga.logits, &ga.logits_rb);
            } else {
                gpu.submit_no_readback();
            }
            chunk_start = chunk_end;
        }

        gpu.free_buffer(emb_buf, emb_mem);
        let ga = self.gpu_acts.as_ref().unwrap();
        gpu.read_logits(&ga.logits, &ga.logits_rb)
    }

    pub fn forward_gpu(&self, token: usize, pos: usize, gpu: &mut VkCtx) -> Vec<f32> {
        let c   = &self.config;
        let w   = &self.weights;
        let gw  = self.gpu_w.as_ref().unwrap();
        let ga  = self.gpu_acts.as_ref().unwrap();
        let hd  = c.head_dim();
        let kvd = c.n_kv_heads * hd;

        let emb = w.token_embd.get_row(token);

        gpu.begin();
        gpu.cmd_upload_act(&ga.x, &emb);
        gpu.timestamp(); // ts0: after upload

        for l in 0..c.n_layers {
            gpu.cmd_rmsnorm(&ga.x, &ga.attn_norms[l], &ga.xn,
                            c.n_embd as u32, c.rms_norm_eps);
            gpu.barrier();
            if l == 0 { gpu.timestamp(); } // ts1: after rmsnorm

            if let Some(t) = gw.attn_q[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.q); }
            if let Some(t) = gw.attn_k[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.k); }
            if let Some(t) = gw.attn_v[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.v); }
            gpu.barrier();
            if l == 0 { gpu.timestamp(); } // ts2: after QKV gemv

            if let Some(ref b) = ga.q_bias[l] {
                gpu.cmd_add(&ga.q, b, (c.n_heads * hd) as u32); }
            if let Some(ref b) = ga.k_bias[l] {
                gpu.cmd_add(&ga.k, b, kvd as u32); }
            if let Some(ref b) = ga.v_bias[l] {
                gpu.cmd_add(&ga.v, b, kvd as u32); }
            if ga.q_bias[l].is_some() || ga.k_bias[l].is_some() || ga.v_bias[l].is_some() {
                gpu.barrier(); }

            gpu.cmd_rope(&ga.q, &ga.k,
                         c.n_heads as u32, c.n_kv_heads as u32,
                         hd as u32, pos as u32, c.rope_freq_base);
            gpu.barrier();

            gpu.cmd_kv_write(&ga.k, &ga.v, &ga.k_cache[l], &ga.v_cache[l],
                             pos as u32, c.n_kv_heads as u32, hd as u32);
            gpu.barrier();

            gpu.cmd_attention(&ga.q, &ga.k_cache[l], &ga.v_cache[l],
                              &ga.attn_out, &ga.scores,
                              c.n_heads as u32, c.n_kv_heads as u32,
                              hd as u32, (pos + 1) as u32, ga.ctx_len as u32);
            gpu.barrier();
            if l == 0 { gpu.timestamp(); } // ts3: after attention

            if let Some(t) = gw.attn_out[l].as_ref() {
                gpu.cmd_gemv(t, &ga.attn_out, &ga.proj); }
            gpu.barrier();

            gpu.cmd_add(&ga.x, &ga.proj, c.n_embd as u32);
            gpu.barrier();

            gpu.cmd_rmsnorm(&ga.x, &ga.ffn_norms[l], &ga.xn,
                            c.n_embd as u32, c.rms_norm_eps);
            gpu.barrier();

            if let Some(t) = gw.ffn_gate[l].as_ref() { gpu.cmd_gemv(t, &ga.xn, &ga.gate); }
            if let Some(t) = gw.ffn_up[l].as_ref()   { gpu.cmd_gemv(t, &ga.xn, &ga.up); }
            gpu.barrier();
            if l == 0 { gpu.timestamp(); } // ts4: after gate+up gemv

            gpu.cmd_swiglu(&ga.gate, &ga.up, c.n_ff as u32);
            gpu.barrier();

            if let Some(t) = gw.ffn_down[l].as_ref() {
                gpu.cmd_gemv(t, &ga.gate, &ga.ff); }
            gpu.barrier();
            if l == 0 { gpu.timestamp(); } // ts5: after ffn_down

            gpu.cmd_add(&ga.x, &ga.ff, c.n_embd as u32);
            if l < c.n_layers - 1 { gpu.barrier(); }
        }

        gpu.timestamp(); // ts6: after all layers
        gpu.cmd_rmsnorm(&ga.x, &ga.out_norm, &ga.xn,
                        c.n_embd as u32, c.rms_norm_eps);
        gpu.barrier();
        if let Some(t) = gw.output.as_ref() {
            gpu.cmd_gemv(t, &ga.xn, &ga.logits);
        }
        gpu.timestamp(); // ts7: after output gemv

        gpu.submit_with_readback(&ga.logits, &ga.logits_rb);
        gpu.print_timestamps();
        gpu.read_logits(&ga.logits, &ga.logits_rb)
    }

    pub fn forward_cpu(&self, token: usize, pos: usize, cache: &mut KvCache) -> Vec<f32> {
        let c   = &self.config;
        let w   = &self.weights;
        let hd  = c.head_dim();
        let kvd = c.n_kv_heads * hd;

        let mut x      = w.token_embd.get_row(token);
        let mut xn     = vec![0f32; c.n_embd];
        let mut q      = vec![0f32; c.n_heads * hd];
        let mut k      = vec![0f32; kvd];
        let mut v      = vec![0f32; kvd];
        let mut scores = vec![0f32; c.n_heads * (pos + 1)];
        let mut attn   = vec![0f32; c.n_embd];
        let mut proj   = vec![0f32; c.n_embd];
        let mut gate   = vec![0f32; c.n_ff];
        let mut up     = vec![0f32; c.n_ff];
        let mut ff     = vec![0f32; c.n_embd];

        for l in 0..c.n_layers {
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.attn_norm[l], c.rms_norm_eps);
            // Use f32 fallback for fused QKV layers, otherwise use quantized matvec
            if let Some(ref qw) = w.attn_q_f32[l] { f32_matvec(qw, &xn, &mut q, c.n_embd); }
            else { w.attn_q[l].matvec(&mut q, &xn); }
            if let Some(ref kw) = w.attn_k_f32[l] { f32_matvec(kw, &xn, &mut k, c.n_embd); }
            else { w.attn_k[l].matvec(&mut k, &xn); }
            if let Some(ref vw) = w.attn_v_f32[l] { f32_matvec(vw, &xn, &mut v, c.n_embd); }
            else { w.attn_v[l].matvec(&mut v, &xn); }
            if let Some(ref b) = w.attn_q_bias[l] { ops::add_into(&mut q, b); }
            if let Some(ref b) = w.attn_k_bias[l] { ops::add_into(&mut k, b); }
            if let Some(ref b) = w.attn_v_bias[l] { ops::add_into(&mut v, b); }
            rope::apply_rope(&mut q, &mut k, pos, hd, c.rope_freq_base, c.n_heads, c.n_kv_heads);
            let cb = pos * kvd;
            cache.k[l][cb..cb+kvd].copy_from_slice(&k);
            cache.v[l][cb..cb+kvd].copy_from_slice(&v);
            let kv_ratio = c.n_heads / c.n_kv_heads;
            attn.fill(0.0);
            for h in 0..c.n_heads {
                let kv_h = h / kv_ratio;
                let qh   = &q[h*hd..(h+1)*hd];
                let sc   = &mut scores[h*(pos+1)..(h+1)*(pos+1)];
                let scale= (hd as f32).sqrt();
                for p in 0..=pos {
                    let ko = p*kvd+kv_h*hd;
                    sc[p] = qh.iter().zip(cache.k[l][ko..ko+hd].iter())
                               .map(|(a,b)| a*b).sum::<f32>() / scale;
                }
                ops::softmax(sc);
                let ah = &mut attn[h*hd..(h+1)*hd];
                ah.fill(0.0);
                for p in 0..=pos {
                    let vo = p*kvd+kv_h*hd;
                    let sp = sc[p];
                    for (o,vi) in ah.iter_mut().zip(cache.v[l][vo..vo+hd].iter()) { *o+=sp*vi; }
                }
            }
            w.attn_out[l].matvec(&mut proj, &attn);
            ops::add_into(&mut x, &proj);
            xn.copy_from_slice(&x);
            ops::rmsnorm(&mut xn, &w.ffn_norm[l], c.rms_norm_eps);
            w.ffn_gate[l].matvec(&mut gate, &xn);
            w.ffn_up[l].matvec(&mut up, &xn);
            for i in 0..c.n_ff { gate[i] = ops::silu(gate[i]) * up[i]; }
            w.ffn_down[l].matvec(&mut ff, &gate);
            ops::add_into(&mut x, &ff);
        }
        ops::rmsnorm(&mut x, &w.output_norm, c.rms_norm_eps);
        let mut logits = vec![0f32; c.n_vocab];
        w.output.matvec(&mut logits, &x);
        logits
    }
}