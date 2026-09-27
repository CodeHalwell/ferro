extern "C" __global__ void ferro_kernel(const float* x, const float* y, const long long* csr, float* out, int* status,
    int groups, int width, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= groups * width) return;
    int s = i / width, f = i % width;
    long long begin = csr[s], end = csr[s + 1];
    const long long* edge = csr + groups + 1;
    if (mode == 0) {
        float sum = 0.0f;
        for (long long p = begin; p < end; ++p) sum += x[edge[p] * width + f];
        out[i] = sum;
    } else if (mode == 2) {
        float m = -__int_as_float(0x7f800000);
        for (long long p = begin; p < end; ++p) {
            float v = x[edge[p] * width + f];
            if (!isfinite(v)) atomicExch(status, 1);
            m = fmaxf(m, v);
        }
        float denom = 0.0f;
        for (long long p = begin; p < end; ++p) {
            long long j = edge[p] * width + f;
            float v = expf(x[j] - m);
            out[j] = v;
            denom += v;
        }
        for (long long p = begin; p < end; ++p) out[edge[p] * width + f] /= denom;
    } else if (mode == 3) {
        float dot = 0.0f;
        for (long long p = begin; p < end; ++p) {
            long long j = edge[p] * width + f;
            dot += x[j] * y[j];
        }
        for (long long p = begin; p < end; ++p) {
            long long j = edge[p] * width + f;
            out[j] = y[j] * (x[j] - dot);
        }
    } else {
        for (long long p = begin; p < end; ++p) out[edge[p] * width + f] = x[i];
    }
}
