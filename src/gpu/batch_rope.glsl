#version 450
layout(local_size_x=64) in;
layout(set=0,binding=0) buffer Q  { float q[]; };
layout(set=0,binding=1) buffer K  { float k[]; };
layout(set=0,binding=2) buffer _Unused { float dummy[]; };
layout(push_constant) uniform PC {
    uint n_heads; uint n_kv_heads; uint head_dim;
    uint start_pos; uint freq_bits; uint batch;
} pc;

void main() {
    uint gid         = gl_GlobalInvocationID.x;
    uint hd2         = pc.head_dim / 2u;
    uint q_pairs     = pc.n_heads    * hd2;
    uint k_pairs     = pc.n_kv_heads * hd2;
    uint per_token   = q_pairs + k_pairs;
    if (gid >= pc.batch * per_token) return;

    uint tok    = gid / per_token;
    uint within = gid % per_token;
    uint pos    = pc.start_pos + tok;
    float freq_base = uintBitsToFloat(pc.freq_bits);

    bool is_q = within < q_pairs;
    uint h, pair;
    if (is_q) {
        h    = within / hd2;
        pair = within % hd2;
    } else {
        uint kw = within - q_pairs;
        h       = kw / hd2;
        pair    = kw % hd2;
    }

    float theta = float(pos) / pow(freq_base, float(pair * 2u) / float(pc.head_dim));
    float c = cos(theta), s = sin(theta);

    if (is_q) {
        uint base = tok * pc.n_heads * pc.head_dim + h * pc.head_dim;
        float a = q[base + pair], b = q[base + pair + hd2];
        q[base + pair]      = a * c - b * s;
        q[base + pair + hd2]= a * s + b * c;
    } else {
        uint base = tok * pc.n_kv_heads * pc.head_dim + h * pc.head_dim;
        float a = k[base + pair], b = k[base + pair + hd2];
        k[base + pair]      = a * c - b * s;
        k[base + pair + hd2]= a * s + b * c;
    }
}
