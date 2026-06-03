#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Mat { uint data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint bpr; uint batch; } pc;

// Q5_K superblock = 44 u32s per 256 weights

shared float sdata[256];

void get_scale_min(uint j, uint s0, uint s1, uint s2,
                   out float sc_out, out float mn_out) {
    uint scales_bytes[12];
    scales_bytes[0]  = s0 & 0xFFu; scales_bytes[1]  = (s0 >> 8u) & 0xFFu;
    scales_bytes[2]  = (s0 >> 16u) & 0xFFu; scales_bytes[3]  = (s0 >> 24u) & 0xFFu;
    scales_bytes[4]  = s1 & 0xFFu; scales_bytes[5]  = (s1 >> 8u) & 0xFFu;
    scales_bytes[6]  = (s1 >> 16u) & 0xFFu; scales_bytes[7]  = (s1 >> 24u) & 0xFFu;
    scales_bytes[8]  = s2 & 0xFFu; scales_bytes[9]  = (s2 >> 8u) & 0xFFu;
    scales_bytes[10] = (s2 >> 16u) & 0xFFu; scales_bytes[11] = (s2 >> 24u) & 0xFFu;
    if (j < 4u) {
        sc_out = float(scales_bytes[j] & 63u);
        mn_out = float(scales_bytes[j + 4u] & 63u);
    } else {
        uint off = j - 4u;
        sc_out = float((scales_bytes[off + 4u] >> 6u) | ((scales_bytes[off + 8u] & 0xFu) << 2u));
        mn_out = float((scales_bytes[off + 4u + 4u] >> 6u) | ((scales_bytes[off + 8u] >> 4u) << 2u));
    }
}

void main() {
    uint row      = gl_WorkGroupID.x;
    uint tok_b    = gl_WorkGroupID.y;
    if (row >= pc.rows || tok_b >= pc.batch) return;
    uint tid      = gl_LocalInvocationID.x;
    uint in_base  = tok_b * pc.bpr * 256u;
    float sum     = 0.0;
    uint row_base = row * pc.bpr;

    for (uint b = 0u; b < pc.bpr; b++) {
        uint blk   = (row_base + b) * 44u;
        uint vb    = b * 256u;
        float df   = uintBitsToFloat(mat.data[blk]);
        float dmin = uintBitsToFloat(mat.data[blk + 1u]);
        uint s0    = mat.data[blk + 2u];
        uint s1    = mat.data[blk + 3u];
        uint s2    = mat.data[blk + 4u];

        uint i       = tid;
        uint quarter = i / 64u;
        uint li      = i % 64u;
        uint sub     = li / 32u;
        uint pos     = li % 32u;
        uint is      = quarter * 2u + sub;
        float sc_val, mn_val;
        get_scale_min(is, s0, s1, s2, sc_val, mn_val);

        uint qs_idx  = quarter * 32u + pos;
        uint qs_word = mat.data[blk + 13u + qs_idx / 4u];
        uint qs_byte = (qs_word >> ((qs_idx % 4u) * 8u)) & 0xFFu;
        uint lo      = (sub == 0u) ? (qs_byte & 0xFu) : (qs_byte >> 4u);

        uint qh_word = mat.data[blk + 5u + pos / 4u];
        uint qh_byte = (qh_word >> ((pos % 4u) * 8u)) & 0xFFu;
        uint hbit    = (qh_byte >> (quarter * 2u + sub)) & 1u;

        float val = float(lo | (hbit << 4u));
        sum += (df * sc_val * val - dmin * mn_val) * vin.data[in_base + vb + i];
    }

    sdata[tid] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0u) vout.data[tok_b * pc.rows + row] = sdata[0];
}
