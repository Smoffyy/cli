#version 450
layout(local_size_x=256) in;
layout(set=0,binding=0) readonly buffer Mat { float data[]; } mat;
layout(set=0,binding=1) readonly buffer In  { float data[]; } vin;
layout(set=0,binding=2) buffer Out { float data[]; } vout;
layout(push_constant) uniform PC { uint rows; uint cols; uint batch; uint row_start; } pc;

shared float sdata[256];

void main() {
    uint row   = gl_WorkGroupID.x;
    uint tok_b = gl_WorkGroupID.y;
    if (row >= pc.rows || tok_b >= pc.batch) return;
    uint tid  = gl_LocalInvocationID.x;
    uint in_base = tok_b * pc.cols;
    float sum = 0.0;

    uint base = (row + pc.row_start) * pc.cols;
    for (uint i = tid; i < pc.cols; i += 256u)
        sum += mat.data[base + i] * vin.data[in_base + i];

    sdata[tid] = sum;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier();
    }
    if (tid == 0u) vout.data[tok_b * pc.rows + row] = sdata[0];
}
