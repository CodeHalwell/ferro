// Center around the first element before summing: avoids cancellation from
// large offsets. Double accumulation keeps wide rows stable without E[x*x]-E[x]^2.
extern "C" __global__ void ferro_kernel(const float* x, const float* w, const float* b,
    float* y,
#ifndef OUTPUT_ONLY
    float* h, float* stds,
#endif
    unsigned cols, float eps, unsigned has_w, unsigned has_b) {
    __shared__ double sums[256];
    unsigned t = threadIdx.x;
    unsigned base = blockIdx.x * cols;
    double origin = (double)x[base];
    double sum = 0.0;
    for (unsigned long long i=t; i<cols; i+=256) sum += (double)x[base+i] - origin;
    sums[t] = sum;
    __syncthreads();
    for (unsigned s=128; s; s>>=1) {
        if (t<s) sums[t] += sums[t+s];
        __syncthreads();
    }
    double mean = origin + sums[0] / cols;
    __syncthreads();
    double var = 0.0;
    for (unsigned long long i=t; i<cols; i+=256) { double c=(double)x[base+i]-mean; var += c*c; }
    sums[t] = var;
    __syncthreads();
    for (unsigned s=128; s; s>>=1) {
        if (t<s) sums[t] += sums[t+s];
        __syncthreads();
    }
    double std = sqrt(sums[0] / cols + (double)eps);
#ifndef OUTPUT_ONLY
    if (t==0) stds[blockIdx.x] = (float)std;
#endif
    for (unsigned long long i=t; i<cols; i+=256) {
        float norm = (float)(((double)x[base+i]-mean) / std);
#ifndef OUTPUT_ONLY
        h[base+i] = norm;
#endif
        y[base+i] = norm * (has_w ? w[i] : 1.0f) + (has_b ? b[i] : 0.0f);
    }
}
