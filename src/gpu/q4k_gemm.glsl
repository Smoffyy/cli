#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Mat { uint data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint bpr; uint batch; uint row_start; } pc;

// Q4K block = 48 u32s per 256 weights

shared float sdata[256];

void main() {
    uint row      = gl_WorkGroupID.x;
    uint tok_b    = gl_WorkGroupID.y;
    if (row >= pc.rows || tok_b >= pc.batch) return;
    uint tid      = gl_LocalInvocationID.x;
    uint in_base  = tok_b * pc.bpr * 256u;
    float sum     = 0.0;
    uint row_base = (row + pc.row_start) * pc.bpr;

    for (uint b = 0u; b < pc.bpr; b++) {
        uint blk = (row_base + b) * 48u;
        uint vb  = b * 256u;
        uint sub = tid >> 6u;
        uint p   = tid & 63u;
        float sc, mn;
        uint nib;
        if (p < 32u) {
            sc  = uintBitsToFloat(mat.data[blk + sub * 2u]);
            mn  = uintBitsToFloat(mat.data[blk + 8u + sub * 2u]);
            nib = (mat.data[blk + 16u + sub * 8u + (p >> 2u)] >> ((p & 3u) * 8u)) & 0xFu;
        } else {
            sc  = uintBitsToFloat(mat.data[blk + sub * 2u + 1u]);
            mn  = uintBitsToFloat(mat.data[blk + 8u + sub * 2u + 1u]);
            uint lp = p - 32u;
            nib = (mat.data[blk + 16u + sub * 8u + (lp >> 2u)] >> ((lp & 3u) * 8u + 4u)) & 0xFu;
        }
        sum += (sc * float(nib) - mn) * vin.data[in_base + vb + tid];
    }

    sdata[tid] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0u) vout.data[tok_b * pc.rows + row] = sdata[0];
}
