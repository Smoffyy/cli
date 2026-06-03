use std::sync::atomic::{AtomicU64, Ordering};

static SEED: AtomicU64 = AtomicU64::new(12345);

pub fn set_seed(s: u64) { SEED.store(s, Ordering::Relaxed); }

fn rand_f32() -> f32 {
    let s = SEED.fetch_update(Ordering::Relaxed, Ordering::Relaxed,
        |v| Some(v.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)))
        .unwrap();
    ((s >> 33) as f32) / (u32::MAX as f32)
}

/// Greedy decode — always picks the highest-probability token
fn greedy(logits: &[f32]) -> usize {
    logits.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i).unwrap_or(0)
}

/// Apply repetition penalty to tokens seen recently.
/// Divides positive logits and multiplies negative ones — penalty > 1.0 suppresses repeats.
fn apply_rep_penalty(logits: &mut [f32], recent: &[u32], penalty: f32) {
    if penalty == 1.0 { return; }
    for &tok in recent {
        if let Some(l) = logits.get_mut(tok as usize) {
            if *l > 0.0 { *l /= penalty; } else { *l *= penalty; }
        }
    }
}

/// Full sampler: temperature + top_k + top_p (nucleus) + repetition penalty.
/// With the defaults (temp=0.7, top_k=40, top_p=0.9, rep=1.1) this produces
/// clean output for most instruction-tuned models without any extra CLI flags.
pub fn sample(logits: &mut Vec<f32>, temperature: f32, top_k: usize,
              top_p: f32, rep_penalty: f32, recent: &[u32]) -> usize {
    // Sanitise any NaN/Inf that the GPU may produce on edge cases
    for v in logits.iter_mut() {
        if !v.is_finite() { *v = -1e9; }
    }

    apply_rep_penalty(logits, recent, rep_penalty);

    if temperature <= 0.0 { return greedy(logits); }

    for v in logits.iter_mut() { *v /= temperature; }

    // Sort (index, scaled_logit) descending — total_cmp is NaN-safe and branchless
    let mut pairs: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    pairs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    // top_k: discard everything outside the top k candidates
    if top_k > 0 && top_k < pairs.len() {
        pairs.truncate(top_k);
    }

    // Softmax over survivors
    let max = pairs[0].1;
    let mut sum = 0.0f32;
    for p in pairs.iter_mut() { p.1 = (p.1 - max).exp(); sum += p.1; }
    for p in pairs.iter_mut() { p.1 /= sum; }

    // top_p (nucleus): keep the smallest prefix whose cumulative prob >= top_p
    if top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut cut = pairs.len();
        for (i, p) in pairs.iter().enumerate() {
            cum += p.1;
            if cum >= top_p { cut = i + 1; break; }
        }
        pairs.truncate(cut);
        let s: f32 = pairs.iter().map(|p| p.1).sum();
        for p in pairs.iter_mut() { p.1 /= s; }
    }

    // Multinomial sample
    let r = rand_f32();
    let mut cum = 0.0f32;
    for (idx, prob) in &pairs {
        cum += prob;
        if r < cum { return *idx; }
    }
    pairs.last().map(|(i, _)| *i).unwrap_or(0)
}