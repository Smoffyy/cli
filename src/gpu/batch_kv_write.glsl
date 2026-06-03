#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0) readonly buffer K  { float k[]; };
layout(set=0,binding=1) readonly buffer V  { float v[]; };
layout(set=0,binding=2) buffer KC { float kc[]; };
layout(set=0,binding=3) buffer VC { float vc[]; };
layout(push_constant) uniform PC {
    uint start_pos; uint n_kv_heads; uint head_dim; uint batch;
} pc;

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint kvd = pc.n_kv_heads * pc.head_dim;
    if (gid >= pc.batch * kvd) return;

    uint b = gid / kvd;
    uint d = gid % kvd;
    uint cache_off = (pc.start_pos + b) * kvd + d;
    kc[cache_off] = k[b * kvd + d];
    vc[cache_off] = v[b * kvd + d];
}
