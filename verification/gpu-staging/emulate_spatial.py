"""Host emulation of crates/ferro-cuda/src/spatial.cu.

Compiles each MODE of the kernel as host C++ (one loop iteration per CUDA
thread, so blockIdx/threadIdx are derived from the flat index) and checks
it against an independent numpy reference, including overlapping and
strided/dilated/padded windows and NaN/tie handling for max pooling. This
proves the indexing and summation logic without a GPU; it does not prove
CUDA execution, which the gated GPU tests do.

    python3 verification/gpu-staging/emulate_spatial.py
"""
import ctypes
import itertools
import os
import pathlib
import subprocess
import tempfile

import numpy as np

SRC = pathlib.Path(__file__).resolve().parents[2] / "crates/ferro-cuda/src/spatial.cu"
PRELUDE = r"""
#define __global__
#define __restrict__
static int min(int a, int b) { return a < b ? a : b; }
struct Dim { unsigned int x; };
static Dim blockIdx, threadIdx, blockDim;
"""
DRIVER = r"""
extern "C" void run(const float* x, const float* aux, float* out, float* arg_out, unsigned long long total, Geom g) {
    blockDim.x = 256;
    for (unsigned long long i = 0; i < total; ++i) {
        blockIdx.x = (unsigned int)(i / 256); threadIdx.x = (unsigned int)(i % 256);
        ferro_kernel(x, aux, out, arg_out, total, g);
    }
}
"""


class Geom(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in "n c h w kh kw sh sw ph pw dh dw oh ow".split()]


def build(mode, tmp):
    src = SRC.read_text()
    path = pathlib.Path(tmp) / f"m{mode}.cpp"
    path.write_text(f"#define MODE {mode}\n{PRELUDE}\n{src}\n{DRIVER}")
    lib = pathlib.Path(tmp) / f"m{mode}.so"
    subprocess.run(["g++", "-O1", "-shared", "-fPIC", "-o", str(lib), str(path)], check=True)
    fn = ctypes.CDLL(str(lib)).run
    fp = ctypes.POINTER(ctypes.c_float)
    fn.argtypes = [fp, fp, fp, fp, ctypes.c_ulonglong, Geom]
    return fn


def call(fn, x, aux, total, g, with_arg=False):
    out = np.zeros(total, np.float32)
    arg = np.zeros(total, np.float32)
    p = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    x = np.ascontiguousarray(x, np.float32)
    aux = x if aux is None else np.ascontiguousarray(aux, np.float32)
    fn(p(x), p(aux), p(out), p(arg), total, g)
    return (out, arg) if with_arg else out


def geom(n, c, h, w, k, s, p, d):
    oh = (h + 2 * p[0] - (d[0] * (k[0] - 1) + 1)) // s[0] + 1
    ow = (w + 2 * p[1] - (d[1] * (k[1] - 1) + 1)) // s[1] + 1
    return Geom(n, c, h, w, k[0], k[1], s[0], s[1], p[0], p[1], d[0], d[1], oh, ow)


def unfold_ref(x, g):
    col = np.zeros((g.c * g.kh * g.kw, g.n * g.oh * g.ow), np.float32)
    for img, ci, r, q, y, xx in itertools.product(range(g.n), range(g.c), range(g.kh), range(g.kw), range(g.oh), range(g.ow)):
        iy, ix = y * g.sh + r * g.dh - g.ph, xx * g.sw + q * g.dw - g.pw
        if 0 <= iy < g.h and 0 <= ix < g.w:
            col[(ci * g.kh + r) * g.kw + q, (img * g.oh + y) * g.ow + xx] = x[img, ci, iy, ix]
    return col


def maxpool_ref(x, g):
    out = np.zeros((g.n, g.c, g.oh, g.ow), np.float32)
    arg = np.zeros_like(out)
    for img, ci, y, xx in itertools.product(range(g.n), range(g.c), range(g.oh), range(g.ow)):
        plane = x[img, ci].reshape(-1)
        best = y * g.sh * g.w + xx * g.sw
        for r, q in itertools.product(range(g.kh), range(g.kw)):
            i = (y * g.sh + r) * g.w + xx * g.sw + q
            if np.isnan(plane[i]):
                best = i
                break
            if plane[i] > plane[best]:
                best = i
        out[img, ci, y, xx], arg[img, ci, y, xx] = plane[best], best
    return out, arg


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        fns = [build(m, tmp) for m in range(4)]
        convs = [((2, 3, 7, 6), (3, 3), (1, 1), (1, 1), (1, 1)), ((1, 2, 9, 8), (3, 2), (2, 1), (0, 2), (1, 2)),
                 ((3, 1, 5, 5), (1, 1), (1, 1), (0, 0), (1, 1)), ((2, 4, 11, 10), (5, 3), (3, 2), (2, 1), (2, 1))]
        for shape, k, s, p, d in convs:
            g = geom(*shape, k, s, p, d)
            x = rng.standard_normal(shape).astype(np.float32)
            col = call(fns[0], x, None, g.c * g.kh * g.kw * g.n * g.oh * g.ow, g).reshape(g.c * g.kh * g.kw, -1)
            want = unfold_ref(x, g)
            assert np.array_equal(col, want), f"unfold {shape} {k} {s} {p} {d}"
            # fold is the adjoint: <unfold(x), c> == <x, fold(c)>, and fold(c)
            # equals the dense transpose applied to c.
            c = rng.standard_normal(want.shape).astype(np.float32)
            folded = call(fns[1], c, None, x.size, g)
            dense = np.zeros((want.size, x.size), np.float64)
            for j in range(x.size):
                e = np.zeros(x.size, np.float32)
                e[j] = 1
                dense[:, j] = unfold_ref(e.reshape(shape), g).reshape(-1)
            assert np.allclose(folded, dense.T @ c.reshape(-1), atol=1e-5), f"fold {shape}"
        for shape, k, s in [((2, 3, 7, 8), 2, 2), ((1, 2, 7, 8), 3, 2), ((2, 1, 6, 6), 3, 1)]:
            g = geom(*shape, (k, k), (s, s), (0, 0), (1, 1))
            x = rng.standard_normal(shape).astype(np.float32)
            x[0, 0, 1, 1] = np.nan
            x[-1, -1, 2, :] = 0.5
            out, arg = call(fns[2], x, None, g.n * g.c * g.oh * g.ow, g, with_arg=True)
            want, warg = maxpool_ref(x, g)
            assert np.array_equal(out, want.reshape(-1), equal_nan=True) and np.array_equal(arg, warg.reshape(-1)), f"maxpool {shape} {k} {s}"
            gy = rng.standard_normal(out.size).astype(np.float32)
            gx = call(fns[3], gy, arg, x.size, g)
            ref = np.zeros(x.size, np.float32)
            planes = np.repeat(np.arange(g.n * g.c), g.oh * g.ow)
            for cell, (pl, a) in enumerate(zip(planes, arg.astype(np.int64))):
                ref[pl * g.h * g.w + a] += gy[cell]
            assert np.allclose(gx, ref, atol=1e-6), f"maxpool backward {shape} {k} {s}"
    print("spatial.cu host emulation: unfold, fold, max_pool2d forward/backward match the references")


if __name__ == "__main__":
    main()
