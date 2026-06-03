#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0) buffer A    { float a[]; };
layout(set=0,binding=1) readonly buffer Bias { float bias[]; };
layout(set=0,binding=2) buffer _Out { float dummy[]; };
layout(push_constant) uniform PC { uint n; uint batch; } pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= pc.batch * pc.n) return;
    a[gid] += bias[gid % pc.n];
}
