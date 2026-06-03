#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer X   { float x[];   };
layout(set=0,binding=1) readonly buffer W   { float w[];   };
layout(set=0,binding=2) buffer Out { float out_data[]; };
layout(push_constant) uniform PC { uint n; uint eps_bits; } pc;

shared float sdata[256];

void main() {
    uint b   = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;
    uint n   = pc.n;
    float eps = uintBitsToFloat(pc.eps_bits);
    uint base = b * n;

    float ss = 0.0;
    for (uint i = tid; i < n; i += 256u)
        ss += x[base + i] * x[base + i];

    sdata[tid] = ss;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }

    float scale = 1.0 / sqrt(sdata[0] / float(n) + eps);
    for (uint i = tid; i < n; i += 256u)
        out_data[base + i] = x[base + i] * scale * w[i];
}
