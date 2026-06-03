#version 450
layout(local_size_x=128) in;
layout(set=0,binding=0) readonly buffer Q  { float q[];  };
layout(set=0,binding=1) readonly buffer KC { float kc[]; };
layout(set=0,binding=2) readonly buffer VC { float vc[]; };
layout(set=0,binding=3) buffer AO { float ao[]; };
layout(set=0,binding=4) buffer SC { float sc[]; };
layout(push_constant) uniform PC {
    uint n_heads; uint n_kv_heads; uint head_dim; uint seq_len; uint n_ctx;
    uint q_offset; uint ao_offset;
} pc;

shared float sdata[128];

void main() {
    uint h = gl_WorkGroupID.x;
    if (h >= pc.n_heads) return;
    uint tid     = gl_LocalInvocationID.x;
    uint kv_h    = h / (pc.n_heads / pc.n_kv_heads);
    uint kvd     = pc.n_kv_heads * pc.head_dim;
    float inv_sq = 1.0 / sqrt(float(pc.head_dim));
    uint q_base  = pc.q_offset + h * pc.head_dim;
    uint sc_base = h * pc.n_ctx;

    for (uint p = tid; p < pc.seq_len; p += 128u) {
        float s = 0.0;
        uint k_base = p * kvd + kv_h * pc.head_dim;
        for (uint d = 0u; d < pc.head_dim; d++)
            s += q[q_base + d] * kc[k_base + d];
        sc[sc_base + p] = s * inv_sq;
    }
    barrier();

    float local_max = -1e30;
    for (uint p = tid; p < pc.seq_len; p += 128u) {
        float v = sc[sc_base + p];
        if (v > local_max) local_max = v;
    }
    sdata[tid] = local_max;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (tid < s && sdata[tid+s] > sdata[tid]) sdata[tid] = sdata[tid+s];
        barrier();
    }
    float max_s = sdata[0];
    barrier();

    float local_sum = 0.0;
    for (uint p = tid; p < pc.seq_len; p += 128u) {
        float e = exp(sc[sc_base + p] - max_s);
        sc[sc_base + p] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    barrier();
    for (uint s = 64u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    float total = sdata[0];
    barrier();

    for (uint p = tid; p < pc.seq_len; p += 128u)
        sc[sc_base + p] /= total;
    barrier();

    uint ao_base = pc.ao_offset + h * pc.head_dim;
    for (uint d = tid; d < pc.head_dim; d += 128u) {
        float val = 0.0;
        for (uint p = 0u; p < pc.seq_len; p++)
            val += sc[sc_base + p] * vc[p * kvd + kv_h * pc.head_dim + d];
        ao[ao_base + d] = val;
    }
}
