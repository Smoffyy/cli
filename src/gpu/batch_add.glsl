#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0) buffer A { float a[]; };
layout(set=0,binding=1) readonly buffer B { float b[]; };
layout(set=0,binding=2) buffer _Out { float dummy[]; };
layout(push_constant) uniform PC { uint total; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.total) a[i] += b[i];
}
