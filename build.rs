use std::path::PathBuf;

fn compile(src: &str, out_name: &str, out_dir: &str) {
    let src_text = std::fs::read_to_string(src)
        .unwrap_or_else(|_| panic!("Cannot read {}", src));
    let compiler = shaderc::Compiler::new()
        .unwrap_or_else(|| panic!("Failed to create shaderc compiler"));
    let mut options = shaderc::CompileOptions::new()
        .unwrap_or_else(|| panic!("Failed to create compile options"));
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_1 as u32);
    options.set_source_language(shaderc::SourceLanguage::GLSL);
    let result = compiler.compile_into_spirv(
        &src_text,
        shaderc::ShaderKind::Compute,
        src,
        "main",
        Some(&options),
    ).unwrap_or_else(|e| panic!("Compile {}: {}", src, e));
    let out = PathBuf::from(out_dir).join(format!("{}.spv", out_name));
    std::fs::write(&out, result.as_binary_u8()).unwrap();
    println!("cargo:rerun-if-changed={}", src);
}

fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rerun-if-changed=build.rs");
    for (src, name) in &[
        // ── existing GEMV + ops ──────────────────────────────────────────────
        ("src/gpu/q4_0_gemv.glsl",       "q4_0_gemv"),
        ("src/gpu/q4_1_gemv.glsl",       "q4_1_gemv"),
        ("src/gpu/q4k_gemv.glsl",        "q4k_gemv"),
        ("src/gpu/q3k_gemv.glsl",        "q3k_gemv"),
        ("src/gpu/q5k_gemv.glsl",        "q5k_gemv"),
        ("src/gpu/q6k_gemv.glsl",        "q6k_gemv"),
        ("src/gpu/q8_0_gemv.glsl",       "q8_0_gemv"),
        ("src/gpu/f32_gemv.glsl",        "f32_gemv"),
        ("src/gpu/rmsnorm.glsl",         "rmsnorm"),
        ("src/gpu/rope.glsl",            "rope"),
        ("src/gpu/kv_write.glsl",        "kv_write"),
        ("src/gpu/attention.glsl",       "attention"),
        ("src/gpu/swiglu.glsl",          "swiglu"),
        ("src/gpu/add.glsl",             "add"),
        ("src/gpu/add_rmsnorm.glsl",     "add_rmsnorm"),
        // ── GEMM (batched prefill) ───────────────────────────────────────────
        ("src/gpu/f32_gemm.glsl",        "f32_gemm"),
        ("src/gpu/q4_0_gemm.glsl",       "q4_0_gemm"),
        ("src/gpu/q4_1_gemm.glsl",       "q4_1_gemm"),
        ("src/gpu/q4k_gemm.glsl",        "q4k_gemm"),
        ("src/gpu/q3k_gemm.glsl",        "q3k_gemm"),
        ("src/gpu/q5k_gemm.glsl",        "q5k_gemm"),
        ("src/gpu/q6k_gemm.glsl",        "q6k_gemm"),
        ("src/gpu/q8_0_gemm.glsl",       "q8_0_gemm"),
        // ── Batch ops ────────────────────────────────────────────────────────
        ("src/gpu/batch_rmsnorm.glsl",   "batch_rmsnorm"),
        ("src/gpu/batch_rope.glsl",      "batch_rope"),
        ("src/gpu/batch_kv_write.glsl",  "batch_kv_write"),
        ("src/gpu/batch_swiglu.glsl",    "batch_swiglu"),
        ("src/gpu/batch_add.glsl",       "batch_add"),
        ("src/gpu/batch_bias_add.glsl",  "batch_bias_add"),
        ("src/gpu/batch_attn_item.glsl", "batch_attn_item"),
    ] { compile(src, name, &out); }
}