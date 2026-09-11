"""Strict binary32 ULP distances, retaining subnormals and near-zero tails."""
import numpy as np


def ulp_distance(actual, expected):
    a, b = np.broadcast_arrays(np.asarray(actual, np.float32), np.asarray(expected, np.float32))
    def ordered(x):
        bits = x.view(np.uint32)
        magnitude = bits & np.uint32(0x7fffffff)
        # A single key for both zeros; adjacent subnormals straddling zero are 2 ULP.
        return np.where(bits >> np.uint32(31), np.uint32(0x80000000) - magnitude,
                        np.uint32(0x80000000) + magnitude).astype(np.int64)
    distance = np.abs(ordered(a) - ordered(b))
    distance = np.where(a == b, 0, distance)
    distance = np.where(np.isnan(a) & np.isnan(b), 0, distance)
    mismatch = (np.isnan(a) != np.isnan(b)) | ((np.isinf(a) | np.isinf(b)) & (a != b))
    return np.where(mismatch, 2**32 - 1, distance)


def compare(actual, expected, atol, rtol):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} != {b.shape}")
    finite = np.isfinite(a) & np.isfinite(b)
    error = np.full(a.shape, np.inf, dtype=np.float64)
    error[finite] = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
    nonzero = finite & (b != 0)
    relative = error[nonzero] / np.abs(b[nonzero].astype(np.float64))
    ulp = ulp_distance(a, b)
    def percentiles(values):
        values = np.asarray(values).ravel()
        if not values.size:
            return None
        return {f"p{q}": float(np.percentile(values, q, method="higher")) for q in (50, 95, 99, 100)}
    near = finite & (np.abs(b) < 1e-5)
    worst = int(np.argmax(ulp))
    return dict(count=int(a.size), passed=bool(np.all(finite & (error <= atol + rtol * np.abs(b)))),
                atol=float(atol), rtol=float(rtol), nonfinite_count=int((~finite).sum()),
                absolute=percentiles(error), relative_nonzero=percentiles(relative),
                relative_reference_zero_count=int((b == 0).sum()), ulp=percentiles(ulp),
                near_zero_count=int(near.sum()), near_zero_ulp=percentiles(ulp[near]),
                away_from_zero_ulp=percentiles(ulp[finite & ~near]),
                worst_ulp=dict(index=list(np.unravel_index(worst, a.shape)),
                               actual=float(a.flat[worst]), expected=float(b.flat[worst]),
                               absolute=float(error.flat[worst]), ulp=int(ulp.flat[worst])))
