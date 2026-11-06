// NCHW sliding-window kernels. One thread per output element; every sum runs
// in a fixed ascending order, so results are deterministic (no atomics) and
// match the CPU reference order: fold sums taps in (r, q) order, max-pool
// backward sums cells in (y, x) order.
// MODE 0: unfold [n,c,h,w] -> [c*kh*kw, n*oh*ow]
// MODE 1: fold (adjoint gather) [c*kh*kw, n*oh*ow] -> [n,c,h,w]
// MODE 2: max pool forward, writing the in-plane argmax offset as a float
// MODE 3: max pool backward (gather over the windows covering each pixel)
struct Geom { int n, c, h, w, kh, kw, sh, sw, ph, pw, dh, dw, oh, ow; };

extern "C" __global__ void ferro_kernel(const float* __restrict__ x, const float* aux, float* __restrict__ out,
    float* arg_out, unsigned long long total, Geom g) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    long long plane_out = (long long)g.oh * g.ow;
    long long cols = (long long)g.n * plane_out;
#if MODE == 0
    long long t = (long long)(i / cols), p = (long long)(i % cols);
    int ci = (int)(t / (g.kh * g.kw)), r = (int)((t / g.kw) % g.kh), q = (int)(t % g.kw);
    long long img = p / plane_out;
    int y = (int)((p / g.ow) % g.oh), xx = (int)(p % g.ow);
    int iy = y * g.sh + r * g.dh - g.ph, ix = xx * g.sw + q * g.dw - g.pw;
    out[i] = (iy >= 0 && iy < g.h && ix >= 0 && ix < g.w) ? x[((img * g.c + ci) * g.h + iy) * g.w + ix] : 0.0f;
#elif MODE == 1
    int ix = (int)(i % g.w), iy = (int)((i / g.w) % g.h), ci = (int)((i / ((unsigned long long)g.w * g.h)) % g.c);
    long long img = (long long)(i / ((unsigned long long)g.w * g.h * g.c));
    float acc = 0.0f;
    for (int r = 0; r < g.kh; ++r) {
        int ty = iy + g.ph - r * g.dh;
        if (ty < 0 || ty % g.sh != 0 || ty / g.sh >= g.oh) continue;
        int y = ty / g.sh;
        for (int q = 0; q < g.kw; ++q) {
            int tx = ix + g.pw - q * g.dw;
            if (tx < 0 || tx % g.sw != 0 || tx / g.sw >= g.ow) continue;
            long long row = ((long long)ci * g.kh + r) * g.kw + q;
            acc += x[row * cols + img * plane_out + (long long)y * g.ow + tx / g.sw];
        }
    }
    out[i] = acc;
#elif MODE == 2
    int xx = (int)(i % g.ow), y = (int)((i / g.ow) % g.oh);
    long long plane = (long long)(i / plane_out);
    const float* src = x + plane * g.h * g.w;
    int arg = y * g.sh * g.w + xx * g.sw;
    for (int r = 0; r < g.kh; ++r) {
        for (int q = 0; q < g.kw; ++q) {
            int idx = (y * g.sh + r) * g.w + xx * g.sw + q;
            float v = src[idx];
            if (v != v) { arg = idx; r = g.kh; break; }
            if (v > src[arg]) arg = idx;
        }
    }
    out[i] = src[arg];
    arg_out[i] = (float)arg;
#elif MODE == 3
    int ix = (int)(i % g.w), iy = (int)((i / g.w) % g.h);
    long long plane = (long long)(i / ((unsigned long long)g.w * g.h));
    int self = iy * g.w + ix;
    int y0 = iy >= g.kh ? (iy - g.kh) / g.sh + 1 : 0, y1 = min(g.oh - 1, iy / g.sh);
    int x0 = ix >= g.kw ? (ix - g.kw) / g.sw + 1 : 0, x1 = min(g.ow - 1, ix / g.sw);
    float acc = 0.0f;
    for (int y = y0; y <= y1; ++y)
        for (int xx = x0; xx <= x1; ++xx) {
            long long cell = plane * plane_out + (long long)y * g.ow + xx;
            if ((int)aux[cell] == self) acc += x[cell];
        }
    out[i] = acc;
#endif
}
