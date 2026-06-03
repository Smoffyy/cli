#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Mat { uint data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint bpr; uint batch; } pc;

// Q3_K superblock = 28 u32s per 256 weights:
//   [0]       = d (f32)
//   [1..8]    = hmask (32 bytes = 8 u32s)
//   [9..24]   = qs (64 bytes = 16 u32s, 2-bit values)
//   [25..27]  = scales (12 bytes = 3 u32s, 16 scales in ggml Q3_K layout)

shared float sdata[256];

uint scale_byte(uint blk, uint k) {
    uint word = mat.data[blk + 25u + k / 4u];
    return (word >> ((k % 4u) * 8u)) & 0xFFu;
}

float extract_scale(uint blk, uint si) {
    uint j = si % 4u;
    uint group = si / 4u;

    uint s_j  = scale_byte(blk, j);
    uint s_j4 = scale_byte(blk, j + 4u);
    uint s_j8 = scale_byte(blk, j + 8u);

    uint raw;
    if (group == 0u) {
        raw = s_j & 0x3Fu;
    } else if (group == 1u) {
        raw = (s_j4 & 0xFu) | ((s_j >> 4u) << 4u);
    } else if (group == 2u) {
        raw = s_j8 & 0x3Fu;
    } else {
        raw = (s_j8 >> 4u) | ((s_j4 >> 4u) << 4u);
    }
    return float(int(raw) - 32);
}

void main() {
    uint row   = gl_WorkGroupID.x;
    uint tok_b = gl_WorkGroupID.y;
    if (row >= pc.rows || tok_b >= pc.batch) return;
    uint tid      = gl_LocalInvocationID.x;
    uint in_base  = tok_b * pc.bpr * 256u;
    float sum     = 0.0;
    uint row_base = row * pc.bpr;

    for (uint b = 0u; b < pc.bpr; b++) {
        uint blk    = (row_base + b) * 28u;
        uint vb     = b * 256u;
        float d_val = uintBitsToFloat(mat.data[blk]);
        uint i      = tid;

        uint qs_byte_idx = i / 4u;
        uint qs_word = mat.data[blk + 9u + qs_byte_idx / 4u];
        uint qs_byte = (qs_word >> ((qs_byte_idx % 4u) * 8u)) & 0xFFu;
        uint qs_2bit = (qs_byte >> ((i % 4u) * 2u)) & 3u;

        uint hm_byte_idx = i % 32u;
        uint hm_word = mat.data[blk + 1u + hm_byte_idx / 4u];
        uint hm_byte = (hm_word >> ((hm_byte_idx % 4u) * 8u)) & 0xFFu;
        uint hbit    = (hm_byte >> (i / 32u)) & 1u;

        int q = int(qs_2bit | (hbit << 2u)) - 4;

        float sc = extract_scale(blk, i / 16u);
        sum += d_val * sc * float(q) * vin.data[in_base + vb + i];
    }

    sdata[tid] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0u) vout.data[tok_b * pc.rows + row] = sdata[0];
}
