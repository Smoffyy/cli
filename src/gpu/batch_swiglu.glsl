#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0) buffer Gate { float gate[]; };
layout(set=0,binding=1) readonly buffer Up { float up[]; };
layout(set=0,binding=2) buffer _Out { float dummy[]; };
layout(push_constant) uniform PC { uint total; } pc;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.total) return;
    float g = gate[i];
    gate[i] = (g / (1.0 + exp(-g))) * up[i];
}
