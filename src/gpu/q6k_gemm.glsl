#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Mat { uint data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint bpr; uint batch; } pc;

// Q6K block = 64 u32s per 256 weights

shared float sdata[256];

void main() {
    uint row      = gl_WorkGroupID.x;
    uint tok_b    = gl_WorkGroupID.y;
    if (row >= pc.rows || tok_b >= pc.batch) return;
    uint tid      = gl_LocalInvocationID.x;
    uint in_base  = tok_b * pc.bpr * 256u;
    float sum     = 0.0;
    uint row_base = row * pc.bpr;

    for (uint b = 0u; b < pc.bpr; b++) {
        uint base = (row_base + b) * 64u;
        uint vb   = b * 256u;
        uint hf   = tid >> 7u;
        uint local_ = tid & 127u;
        uint quad = local_ >> 5u;
        uint li   = local_ & 31u;

        uint ql_off = base + 16u + hf * 16u;
        uint qh_off = base + 48u + hf * 8u;
        uint si     = hf * 8u;
        uint vo     = vb + hf * 128u;

        uint qh_word = mat.data[qh_off + li/4u];
        uint qh_byte = (qh_word >> ((li%4u)*8u)) & 0xFFu;

        float sc_val;
        int q_val;
        if (quad == 0u) {
            uint ql_word = mat.data[ql_off + li/4u];
            uint ql_byte = (ql_word >> ((li%4u)*8u)) & 0xFFu;
            q_val  = int((ql_byte & 0xFu) | ((qh_byte & 3u) << 4u)) - 32;
            sc_val = uintBitsToFloat(mat.data[base + si + li/16u]);
            sum   += sc_val * float(q_val) * vin.data[in_base + vo + li];
        } else if (quad == 1u) {
            uint ql_word = mat.data[ql_off + 8u + li/4u];
            uint ql_byte = (ql_word >> ((li%4u)*8u)) & 0xFFu;
            q_val  = int((ql_byte & 0xFu) | (((qh_byte >> 2u) & 3u) << 4u)) - 32;
            sc_val = uintBitsToFloat(mat.data[base + si + li/16u + 2u]);
            sum   += sc_val * float(q_val) * vin.data[in_base + vo + li + 32u];
        } else if (quad == 2u) {
            uint ql_word = mat.data[ql_off + li/4u];
            uint ql_byte = (ql_word >> ((li%4u)*8u)) & 0xFFu;
            q_val  = int(((ql_byte >> 4u) & 0xFu) | (((qh_byte >> 4u) & 3u) << 4u)) - 32;
            sc_val = uintBitsToFloat(mat.data[base + si + li/16u + 4u]);
            sum   += sc_val * float(q_val) * vin.data[in_base + vo + li + 64u];
        } else {
            uint ql_word = mat.data[ql_off + 8u + li/4u];
            uint ql_byte = (ql_word >> ((li%4u)*8u)) & 0xFFu;
            q_val  = int(((ql_byte >> 4u) & 0xFu) | (((qh_byte >> 6u) & 3u) << 4u)) - 32;
            sc_val = uintBitsToFloat(mat.data[base + si + li/16u + 6u]);
            sum   += sc_val * float(q_val) * vin.data[in_base + vo + li + 96u];
        }
    }

    sdata[tid] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0u) vout.data[tok_b * pc.rows + row] = sdata[0];
}
