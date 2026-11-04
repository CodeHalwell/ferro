"""Native eager CPU f32/f64 linear algebra with first-order analytic gradients.

LU uses partial pivoting and f64 working arithmetic, including for f32 inputs;
output and gradient dtype match the inputs. Coefficient matrices must be finite.
Complex tensors, out= buffers and inference capture are unsupported. Native
factorizations provide first-order adjoints; compositions retain their kernels'
autograd support.
"""
from collections import namedtuple
import math
import sys
from numbers import Real
from . import _native


CholeskyExResult = namedtuple('CholeskyExResult', ('L', 'info'))
SlogdetResult = namedtuple('SlogdetResult', ('sign', 'logabsdet'))
QRResult = namedtuple('QRResult', ('Q', 'R'))
EighResult = namedtuple('EighResult', ('eigenvalues', 'eigenvectors'))
SVDResult = namedtuple('SVDResult', ('U', 'S', 'Vh'))
LstsqResult = namedtuple('LstsqResult', ('solution', 'residuals', 'rank', 'singular_values'))
SolveExResult = namedtuple('SolveExResult', ('result', 'info'))
InvExResult = namedtuple('InvExResult', ('inverse', 'info'))
LUFactorResult = namedtuple('LUFactorResult', ('LU', 'pivots'))
LUFactorExResult = namedtuple('LUFactorExResult', ('LU', 'pivots', 'info'))
LUResult = namedtuple('LUResult', ('P', 'L', 'U'))


def _options(out=None, **flags):
    if out is not None:
        raise NotImplementedError('linalg out= buffers are unsupported')
    if any(type(value) is not bool for value in flags.values()):
        raise TypeError('linalg flags must be bool')


def solve(A, B, *, left=True, out=None):
    """Solve AX=B or XA=B, with broadcast batches and Torch's vector-RHS rules."""
    _options(out, left=left)
    return _native._linalg_solve(A, B, left)


def inv(A, *, out=None):
    """Invert square matrices by partial-pivot LU; singular matrices reject."""
    _options(out)
    return _native._linalg_inv(A)


def solve_ex(A, B, *, left=True, check_errors=False, out=None):
    """Solve AX=B or XA=B and return the first zero LU pivot per original A batch.

    info=0 indicates success. Positive info identifies a singular factorization;
    its result and gradients are undefined, with no guarantee of particular
    NaN/Inf patterns. check_errors=True raises instead. info is untracked int64,
    rather than Torch's int32. Finite CPU float32/float64 coefficients only.
    Backward saves A and the result; later RHS/status mutations are permitted.
    """
    _options(out, left=left, check_errors=check_errors)
    return SolveExResult(*_native._linalg_solve_ex(A, B, left, check_errors))


def inv_ex(A, *, check_errors=False, out=None):
    """Inverse and original-batch LU status, with output-only saved values.

    info uses untracked int64 instead of Torch's int32. Positive info means the
    result and gradients are undefined; check_errors=True raises. Finite eager
    CPU float32/float64 coefficients and first-order gradients are supported.
    """
    _options(out, check_errors=check_errors)
    return InvExResult(*_native._linalg_inv_ex(A, check_errors))


def _tensor_system_inputs(*inputs):
    if _native._is_capturing():
        raise ValueError('tensor linear algebra does not support capture')
    for value in inputs:
        if not isinstance(value, _native.Tensor):
            raise TypeError('tensor linear algebra inputs must be Tensors')
        if value.device != 'cpu' or value.dtype not in ('float32', 'float64'):
            raise ValueError('tensor linear algebra requires CPU float32 or float64 tensors')
        if math.prod(max(size, 1) for size in value.shape) > sys.maxsize or value.numel() > sys.maxsize // 8:
            raise ValueError('tensor linear algebra exceeds addressable working capacity')
    if any(value.dtype != inputs[0].dtype for value in inputs):
        raise ValueError('tensor linear algebra inputs must have matching dtypes')


def tensorinv(A, ind=2, *, out=None):
    """Invert a tensor by grouping dimensions around ind into a square matrix.

    The output shape is A.shape[ind:] + A.shape[:ind]. Uses native LU with
    first-order CPU float32/float64 gradients, including strided inputs.
    """
    _options(out)
    _tensor_system_inputs(A)
    if type(ind) is not int:
        raise TypeError('tensorinv ind must be an integer')
    shape = A.shape
    if ind <= 0 or ind > len(shape):
        raise ValueError('tensorinv ind must be positive and no greater than the input rank')
    size = math.prod(shape[:ind])
    if math.prod(shape[ind:]) != size:
        raise ValueError('tensorinv split dimension products must be equal')
    if size * size > sys.maxsize // 8:
        raise ValueError('tensorinv matrix exceeds addressable working capacity')
    return inv(A.reshape(size, size)).reshape(shape[ind:] + shape[:ind])


def tensorsolve(A, B, dims=None, *, out=None):
    """Solve a tensor equation by moving dims last and flattening to a square LU solve.

    The solution shape is the permuted A.shape[B.ndim:]. B is flattened, so
    its individual dimensions need not equal A's leading dimensions. Uses
    first-order CPU float32/float64 gradients for both tensor inputs.
    """
    _options(out)
    _tensor_system_inputs(A, B)
    rank = len(A.shape)
    if len(B.shape) > rank:
        raise ValueError('tensorsolve B cannot have more dimensions than A')
    order = list(range(rank))
    if dims is not None:
        if not isinstance(dims, (tuple, list)) or any(type(axis) is not int for axis in dims):
            raise TypeError('tensorsolve dims must be a sequence of integers')
        if any(axis < -rank or axis >= rank for axis in dims):
            raise ValueError('tensorsolve dimension is out of range')
        axes = [axis % rank for axis in dims]
        if len(set(axes)) != len(axes):
            raise ValueError('tensorsolve dimensions must be distinct')
        order = [axis for axis in order if axis not in axes] + axes
    shape = [A.shape[axis] for axis in order][len(B.shape):]
    size = math.prod(shape)
    if A.numel() != size * size or B.numel() != size:
        raise ValueError('tensorsolve requires a square flattened system and matching RHS size')
    if size * size > sys.maxsize // 8:
        raise ValueError('tensorsolve matrix exceeds addressable working capacity')
    matrix = A.permute(*order) if order != list(range(rank)) else A
    return solve(matrix.reshape(size, size), B.reshape(size)).reshape(shape)


def solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None):
    """Solve from one triangle; ignored entries and a unit diagonal have zero gradients.

    A singular selected triangle rejects instead of returning nonfinite values.
    """
    _options(out, upper=upper, left=left, unitriangular=unitriangular)
    return _native._linalg_solve_triangular(A, B, upper, left, unitriangular)


def cholesky(A, *, upper=False, out=None):
    """Factor positive-definite matrices, reading only the requested triangle."""
    _options(out, upper=upper)
    return _native._linalg_cholesky_ex(A, upper, True)[0]


def cholesky_ex(A, *, upper=False, check_errors=False, out=None):
    """Return factor and int64 failure minor (0 on success); gradients require SPD input.

    Partial factors with nonzero info are not a valid factorization. Info uses
    Ferro's int64 storage rather than Torch's int32 storage.
    """
    _options(out, upper=upper, check_errors=check_errors)
    return CholeskyExResult(*_native._linalg_cholesky_ex(A, upper, check_errors))


def cholesky_solve(input, input2, upper=False, *, out=None):
    """Solve a positive-definite system from a triangular Cholesky factor.

    input is a matrix RHS; input2 is the lower (or upper) square factor. Batch
    dimensions broadcast. CPU float32/float64 only, with finite selected factor
    entries. Singular factors return undefined nonfinite values and gradients.

    Backward matches Torch's full unmasked factor formula, using all original
    factor entries. This is not the entrywise derivative of discarding arbitrary
    values in the unused triangle. All operands and output are version-checked.
    """
    _options(out, upper=upper)
    return _native._linalg_cholesky_solve(input, input2, upper)


def cholesky_inverse(input, upper=False, *, out=None):
    """Invert from a nonsingular triangular Cholesky factor on real CPU tensors.

    float32/float64 output is exactly symmetric. Only the selected triangle is
    read in forward; its entries must be finite. Backward matches Torch's full
    unmasked factor formula, including original unused entries, rather than
    finite differences of discarding them. Factor/output mutations are guarded.
    """
    _options(out, upper=upper)
    return _native._linalg_cholesky_inverse(input, upper)


def det(A, *, out=None):
    """Batched determinant with true polynomial gradients, including singular matrices.

    Singular and numerically degenerate adjoints use signed minors, costing O(n^5).
    This preserves nonzero cofactors where an inverse-based formula is undefined.
    """
    _options(out)
    return _native._linalg_det(A)


def slogdet(A, *, out=None):
    """Sign and log absolute determinant; log gradients require nonsingular matrices.

    Singular matrices return sign=0 and logabsdet=-inf, with undefined NaN log
    gradients. Sign-only backward leaves the input gradient unused (None).
    """
    _options(out)
    return SlogdetResult(*_native._linalg_slogdet(A))


def _norm_input(A, dtype):
    if A.dtype not in ('float32', 'float64'):
        raise TypeError('linalg norms require float32 or float64 tensors')
    if dtype is None or dtype == A.dtype:
        return A
    if dtype != 'float64' or A.dtype != 'float32':
        raise ValueError('norm dtype may preserve or widen float32 to float64')
    return A.to(dtype=dtype)


def _dims(dim):
    if dim is None:
        return None
    if type(dim) is int:
        return [dim]
    if not isinstance(dim, (tuple, list)) or any(type(axis) is not int for axis in dim):
        raise TypeError('norm dim must be an int or sequence of ints')
    return list(dim)


def vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None):
    """Stable real p-norm, with flattened dim=None/() and tie-split gradients.

    ord=0 counts nonzero values and leaves ordinary input gradients unused.
    Spectral decompositions are not involved in this vector reduction.
    """
    _options(out, keepdim=keepdim)
    if isinstance(ord, bool) or not isinstance(ord, Real) or math.isnan(ord):
        raise TypeError('vector norm ord must be a real number other than NaN')
    return _native._linalg_vector_norm(_norm_input(x, dtype), float(ord), _dims(dim), keepdim)


def matrix_norm(A, ord='fro', dim=(-2, -1), keepdim=False, *, dtype=None, out=None):
    """Real matrix norms, with spectral/nuclear orders using native singular values.

    Spectral ties split the singular-value cotangent evenly, including zero.
    Nuclear gradients at zero use the SVD's selected orthogonal bases.
    """
    _options(out, keepdim=keepdim)
    axes = _dims(dim)
    rank = len(A.shape)
    if rank < 2 or axes is None or len(axes) != 2:
        raise ValueError('matrix norm requires two distinct matrix dimensions')
    if any(axis < -rank or axis >= rank for axis in axes):
        raise ValueError('matrix norm dimension is out of range')
    axes = [axis % rank for axis in axes]
    if axes[0] == axes[1]:
        raise ValueError('matrix norm dimensions must be distinct')
    A = _norm_input(A, dtype)
    if ord == 'fro':
        return vector_norm(A, 2, axes, keepdim)
    if ord in ('nuc', 2, -2):
        order = [axis for axis in range(rank) if axis not in axes] + axes
        singular = svdvals(A.permute(*order))
        if ord == 'nuc':
            result = singular.sum_dim(-1, False)
        elif ord == 2 and singular.shape[-1] == 0:
            result = singular.sum_dim(-1, False).detach()
        else:
            extreme = vector_norm(singular.detach(), float('inf') if ord == 2 else -float('inf'), -1, True)
            selected = (singular == extreme).to(dtype=A.dtype)
            result = (singular * selected).sum_dim(-1, False) / selected.sum_dim(-1, False)
        return result.reshape([1 if axis in axes else n for axis, n in enumerate(A.shape)]) if keepdim else result
    if isinstance(ord, bool) or not isinstance(ord, Real) or math.isnan(ord):
        raise ValueError('unsupported matrix norm order')
    if ord not in (1, -1, float('inf'), -float('inf')):
        raise ValueError('unsupported matrix norm order')
    inner_axis, outer_axis = axes if abs(ord) == 1 else axes[::-1]
    sums = vector_norm(A, 1, inner_axis, True)
    result = vector_norm(sums, float('inf') if ord > 0 else -float('inf'), outer_axis, True)
    return result if keepdim else result.reshape([n for axis, n in enumerate(A.shape) if axis not in axes])


def norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None):
    """Torch linalg.norm routing for real vector and matrix norms."""
    axes = _dims(dim)
    if axes is not None and len(axes) not in (1, 2):
        raise ValueError('norm dim must contain one or two dimensions')
    if axes is None:
        if ord is None:
            return vector_norm(A, 2, None, keepdim, dtype=dtype, out=out)
        if len(A.shape) not in (1, 2):
            raise ValueError('norm with an order and no dim requires a vector or matrix')
        axes = list(range(len(A.shape)))
    if len(axes) == 1 and not isinstance(ord, str):
        return vector_norm(A, 2 if ord is None else ord, axes, keepdim, dtype=dtype, out=out)
    return matrix_norm(A, 'fro' if ord is None else ord, axes, keepdim, dtype=dtype, out=out)


def diagonal(A, *, offset=0, dim1=-2, dim2=-1):
    """Shared-storage CPU float32/float64 diagonal view, with scatter gradients."""
    if any(type(value) is not int for value in (offset, dim1, dim2)):
        raise TypeError('diagonal offset and dimensions must be ints')
    return _native._linalg_diagonal(A, offset, dim1, dim2)


def qr(A, mode='reduced', *, out=None):
    """Householder QR with independent leading min(m,n) columns.

    Both factors have first-order analytic input gradients. Factorization uses
    f64 working arithmetic and requires finite, representable factors; exact
    zero R pivots reject. Reduced mode supports every matrix shape. Complete
    mode supports square/wide matrices; tall complete, r and higher-order AD reject.
    """
    _options(out)
    if mode == 'r':
        raise NotImplementedError('QR mode=r is not implemented')
    if mode not in ('reduced', 'complete'):
        raise ValueError('invalid QR mode')
    if mode == 'complete' and len(A.shape) >= 2 and A.shape[-2] > A.shape[-1]:
        raise NotImplementedError('complete QR currently requires rows <= columns')
    return QRResult(*_native._linalg_qr(A))


def eigh(A, UPLO='L', *, out=None):
    """Real symmetric eigenvalues and column eigenvectors, in ascending order.

    The native Jacobi solver reads only the selected triangle and uses f64
    working arithmetic. Eigenvector signs are arbitrary. Input gradients are
    symmetric, matching Torch's symmetric-matrix convention; eigenvector
    gradients are undefined at repeated eigenvalues and become ill-conditioned
    as gaps shrink. Extreme finite gaps can overflow in backward, as in Torch.
    CPU f32/f64 and first-order AD only; out= and capture are unsupported.
    """
    _options(out)
    if not isinstance(UPLO, str):
        raise TypeError('UPLO must be a string')
    if UPLO.upper() not in ('L', 'U'):
        raise ValueError('UPLO must be L or U')
    return EighResult(*_native._linalg_eigh(A, UPLO.upper() == 'U'))


def eigvalsh(A, UPLO='L', *, out=None):
    """Eigenvalues of a real symmetric matrix; see eigh for numeric scope."""
    return eigh(A, UPLO, out=out).eigenvalues


def svd(A, full_matrices=True, *, driver=None, out=None):
    """Real CPU one-sided Jacobi SVD with descending singular values.

    Supports full/reduced batched factors in f32/f64, using f64 working
    arithmetic without forming A.T @ A. Column signs and complementary bases
    are arbitrary. Full-factor complementary-vector gradients are ignored,
    matching Torch. First-order vector gradients require nonzero, separated
    singular values; repeated/zero values make vector derivatives undefined.
    Finite coefficients and convergence within 100 sweeps are required.
    Unresolved correlated subnormal values below the original matrix precision
    may be deflated; orthogonal tiny singular values are preserved. Exact ranks
    at zero tolerance can therefore differ from other solvers near underflow.
    Complex, driver overrides, capture and higher-order AD are unsupported.
    """
    _options(out, full_matrices=full_matrices)
    if driver is not None:
        raise NotImplementedError('CPU SVD driver overrides are unsupported')
    return SVDResult(*_native._linalg_svd(A, full_matrices))


def svdvals(A, *, driver=None, out=None):
    """Singular values with their own adjoint, avoiding vector gap reciprocals.

    At nonsmooth zero/repeated values, gradients select the returned bases.
    """
    return svd(A, False, driver=driver, out=out).S


def _tolerance(value):
    if value is None or isinstance(value, _native.Tensor):
        return value
    if not isinstance(value, Real):
        raise TypeError('atol and rtol must be real scalars or tensors')
    from . import tensor
    return tensor(float(value), dtype='float64')


def pinv(A, rcond=None, hermitian=False, *, atol=None, rtol=None, out=None):
    """Real Moore-Penrose inverse with scalar or broadcast-batch tolerances.

    Uses the native SVD, or the lower triangle and symmetric eigenvalues when
    hermitian=True. The dense Moore-Penrose adjoint avoids singular-vector
    gap divisions. It follows Torch's convention when a tolerance discards
    nonzero singular values and can then differ from finite differences of
    the truncated forward map. Cutoff crossings are not differentiable. Tolerances are
    constant inputs. Numerical scalars use f64 threshold arithmetic, while
    tensor tolerances preserve their dtype. First-order CPU f32/f64 only.
    """
    _options(out, hermitian=hermitian)
    if rcond is not None:
        if atol is not None or rtol is not None:
            raise TypeError('rcond cannot be combined with atol or rtol')
        rtol = rcond
    return _native._linalg_pinv(A, _tolerance(atol), _tolerance(rtol), hermitian)


def matrix_rank(A, tol=None, hermitian=False, *, atol=None, rtol=None, out=None):
    """Numerical rank as an untracked int64 tensor; tolerances broadcast batches.

    At an exact zero cutoff, rank can differ from LAPACK when one solver
    preserves an exact zero singular value and another returns roundoff.
    """
    _options(out, hermitian=hermitian)
    if tol is not None:
        if atol is not None or rtol is not None:
            raise TypeError('tol cannot be combined with atol or rtol')
        atol = tol
        rtol = 0.
    return _native._linalg_matrix_rank(A, _tolerance(atol), _tolerance(rtol), hermitian)


def lstsq(A, B, rcond=None, *, driver=None, out=None):
    """Minimum-norm least squares using native Jacobi SVD and real adjoints.

    Returns CPU-default fields: solution, numerical rank, empty residuals and
    empty singular_values. The algorithm is SVD, not LAPACK's default pivoted
    QR driver; truncated solutions and borderline ranks may therefore differ.
    Named driver overrides are unsupported. rcond must be None or strictly
    between zero and one. Matrix RHS
    tensors have the same rank as A and broadcast batch axes; vector RHS
    tensors match A's complete batch and row shape. First-order CPU f32/f64.

    Backward follows Torch's default-cutoff pseudoinverse convention even
    when explicit rcond truncates the forward solution. In that case it need
    not equal finite differences of the truncated forward map.
    """
    _options(out)
    if driver is not None:
        raise NotImplementedError('lstsq uses native Jacobi SVD; LAPACK driver overrides are unsupported')
    if rcond is not None and (isinstance(rcond, bool) or not isinstance(rcond, Real)):
        raise TypeError('rcond must be a real scalar or None')
    return LstsqResult(*_native._linalg_lstsq(A, B, rcond))


def lu_factor(A, *, pivot=True, out=None):
    """Rectangular native partial-pivot LU with one-based int64 pivots.

    Packed LU uses the input float32/float64 dtype; pivots use int64 rather
    than Torch's int32. Singular factors raise. CPU only, with first-order
    gradients at full-rank matrices away from pivot-selection boundaries.
    """
    _options(out, pivot=pivot)
    if not pivot:
        raise NotImplementedError('CPU LU without pivoting is unsupported')
    return LUFactorResult(*_native._linalg_lu_factor(A))


def lu_factor_ex(A, *, pivot=True, check_errors=False, out=None):
    """Packed LU, one-based pivots and first-zero-pivot status, both int64.

    The metadata dtype differs from Torch's int32. Positive info identifies a
    singular factor with undefined gradients; check_errors=True raises.
    Backward saves packed LU and pivots, permitting input/status mutations.
    """
    _options(out, pivot=pivot, check_errors=check_errors)
    if not pivot:
        raise NotImplementedError('CPU LU without pivoting is unsupported')
    return LUFactorExResult(*_native._linalg_lu_factor_ex(A, check_errors))


def lu(A, *, pivot=True, out=None):
    """Return real CPU P,L,U with A=P@L@U and rectangular L/U shapes.

    P is nondifferentiable; first-order L/U gradients require full rank and
    locally constant pivots. Backward saves all three returned tensor values.
    """
    _options(out, pivot=pivot)
    if not pivot:
        raise NotImplementedError('CPU LU without pivoting is unsupported')
    return LUResult(*_native._linalg_lu(A))


def lu_solve(LU, pivots, B, *, left=True, adjoint=False, out=None):
    """Solve from square packed LU and one-based int64 pivots, differentiating LU/B.

    B must have matrix dimensions; its batches broadcast with LU. pivots must
    match LU.shape[:-1] exactly. adjoint applies the real transpose. LU and B
    share a CPU float32/float64 dtype; singular factors have undefined results
    and gradients. Backward saves LU, pivots and output, allowing B mutation.
    """
    _options(out, left=left, adjoint=adjoint)
    return _native._linalg_lu_solve(LU, pivots, B, left, adjoint)


def matrix_power(A, n, *, out=None):
    """Integer powers of batched square real CPU matrices by repeated squaring.

    Zero powers return identity with zero input gradients, including nonfinite
    inputs. Power one makes an owning contiguous copy. Negative powers use a
    checked native inverse and require finite nonsingular coefficient matrices.
    """
    _options(out)
    _tensor_system_inputs(A)
    if type(n) is not int:
        raise TypeError('matrix_power n must be an integer')
    if n < -(1 << 63) or n >= 1 << 63:
        raise ValueError('matrix_power n must fit a signed 64-bit integer')
    if len(A.shape) < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError('matrix_power requires square matrix dimensions')
    if not A.numel():
        return A.clone(memory_format='contiguous_format')
    if n == 0:
        from . import eye
        zero = A.narrow(-1, 0, 0).sum_dim(-1, True).expand(A.shape)
        return zero + eye(A.shape[-1], dtype=A.dtype)
    if n < 0:
        A = inv_ex(A, check_errors=True).inverse
        n = -n
    elif n == 1:
        return A.clone(memory_format='contiguous_format')
    if n == 1:
        return A
    if n == 2:
        return A @ A
    if n == 3:
        return (A @ A) @ A
    result = None
    while n:
        if n & 1:
            result = A if result is None else result @ A
        n >>= 1
        if n:
            A = A @ A
    return result


def multi_dot(tensors, *, out=None):
    """Multiply a real CPU matrix chain, allowing vector first/last operands.

    Dynamic programming minimizes scalar multiplications among plans whose
    intermediate shapes fit addressable storage. This is an arithmetic-cost
    plan, not a measured runtime or peak-memory guarantee. No batch dimensions.
    """
    _options(out)
    if not isinstance(tensors, (tuple, list)):
        raise TypeError('multi_dot tensors must be a tuple or list')
    count = len(tensors)
    if count < 2:
        raise ValueError('multi_dot requires at least two tensors')
    if count * count > sys.maxsize // 8:
        raise ValueError('multi_dot planning capacity exceeded')
    _tensor_system_inputs(*tensors)
    shapes = [value.shape for value in tensors]
    if any(len(shape) != 2 and not (i in (0, count - 1) and len(shape) == 1) for i, shape in enumerate(shapes)):
        raise ValueError('multi_dot requires matrices, with optional vector endpoints')
    first_vector, last_vector = len(shapes[0]) == 1, len(shapes[-1]) == 1
    if first_vector:
        shapes[0] = [1, shapes[0][0]]
    if last_vector:
        shapes[-1] = [shapes[-1][0], 1]
    if any(left[1] != right[0] for left, right in zip(shapes, shapes[1:])):
        raise ValueError('multi_dot adjacent matrix dimensions must match')
    sizes = [shapes[0][0]] + [shape[1] for shape in shapes]
    costs = [[None] * count for _ in range(count)]
    splits = [[None] * count for _ in range(count)]
    for i in range(count):
        costs[i][i] = 0
    for width in range(2, count + 1):
        for i in range(count - width + 1):
            j = i + width - 1
            rows, columns = sizes[i], sizes[j + 1]
            if rows * columns > sys.maxsize // 8 or max(rows, 1) * max(columns, 1) > sys.maxsize:
                continue
            for k in range(i, j):
                if costs[i][k] is None or costs[k + 1][j] is None:
                    continue
                cost = costs[i][k] + costs[k + 1][j] + rows * sizes[k + 1] * columns
                # Torch's three-operand fast path chooses left association on ties.
                if costs[i][j] is None or cost < costs[i][j] or (count == 3 and width == 3 and cost == costs[i][j]):
                    costs[i][j], splits[i][j] = cost, k
    if costs[0][-1] is None:
        raise ValueError('multi_dot has no addressable intermediate allocation plan')
    matrices = list(tensors)
    if first_vector:
        matrices[0] = matrices[0].unsqueeze(0)
    if last_vector:
        matrices[-1] = matrices[-1].unsqueeze(-1)
    pending, values = [(0, count - 1, False)], {}
    while pending:
        i, j, ready = pending.pop()
        if i == j:
            values[i, j] = matrices[i]
            continue
        k = splits[i][j]
        if ready:
            values[i, j] = values.pop((i, k)) @ values.pop((k + 1, j))
        else:
            pending.extend(((i, j, True), (k + 1, j, False), (i, k, False)))
    shape = ([] if first_vector else [sizes[0]]) + ([] if last_vector else [sizes[-1]])
    return values[0, count - 1].reshape(shape)


def cond(A, p=None, *, out=None):
    """Real CPU matrix condition number for spectral and induced norm orders.

    None and +/-2 use the first and last sorted singular values, including
    their selected gradients at ties. 'nuc' uses sum(S)*sum(1/S); other orders
    use norm(A)*norm(inv(A)). Non-spectral orders require square nonempty
    matrices; string orders also require square empty matrices. Empty inputs
    return detached zeros. Singular gradients are
    undefined; near-zero singular values retain the native SVD's numerical
    rank limits. Only finite float32/float64 inputs and first-order gradients.

    Inverse-norm gradients follow the unfused norm/inverse formula. Torch 2.13
    cond backward currently raises a saved-inverse version error for these
    orders; Ferro does not reproduce that reference implementation defect.
    """
    _options(out)
    _tensor_system_inputs(A)
    if len(A.shape) < 2:
        raise ValueError('cond requires at least two dimensions')
    if isinstance(p, str):
        if p not in ('fro', 'nuc'):
            raise ValueError('cond received an invalid norm order')
        if A.shape[-2] != A.shape[-1]:
            raise ValueError('cond string norm orders require square matrices')
    elif p is not None:
        if not isinstance(p, Real):
            raise TypeError('cond p must be a real scalar, fro, nuc or None')
        if p not in (1, -1, 2, -2, math.inf, -math.inf):
            raise ValueError('cond received an invalid norm order')
        if isinstance(p, bool):
            p = int(p)
    shape = A.shape[:-2]
    if math.prod(shape) > sys.maxsize // 8:
        raise ValueError('cond output exceeds addressable working capacity')
    if not A.numel():
        from . import zeros
        return zeros(shape, dtype=A.dtype)
    if p not in (None, 2, -2) and A.shape[-2] != A.shape[-1]:
        raise ValueError('cond non-spectral norm orders require square matrices')
    if p in (None, 2, -2, 'nuc'):
        singular = svdvals(A)
        if p == 'nuc':
            return singular.sum_dim(-1, False) * (1. / singular).sum_dim(-1, False)
        largest = singular.narrow(-1, 0, 1).squeeze(-1)
        smallest = singular.narrow(-1, singular.shape[-1] - 1, 1).squeeze(-1)
        return smallest / largest if p == -2 else largest / smallest
    from . import full, where
    inverse, info = inv_ex(A)
    value = matrix_norm(A, p) * matrix_norm(inverse, p)
    return where(info != 0, full(value.shape, math.inf, dtype=A.dtype), value)


def _vector_operation_shape(input, other, dim, *, integer=False):
    if _native._is_capturing():
        raise ValueError('vector linear algebra does not support capture')
    dtypes = ('float32', 'float64', 'int64') if integer else ('float32', 'float64')
    for value in (input, other):
        if not isinstance(value, _native.Tensor):
            raise TypeError('vector linear algebra inputs must be Tensors')
        if value.device != 'cpu' or value.dtype not in dtypes:
            raise ValueError('vector linear algebra received an unsupported dtype or device')
        width = 4 if value.dtype == 'float32' else 8
        if value.numel() > sys.maxsize // width or math.prod(max(size, 1) for size in value.shape) > sys.maxsize:
            raise ValueError('vector linear algebra input exceeds addressable capacity')
    if input.dtype != other.dtype:
        raise ValueError('vector linear algebra inputs must have matching dtypes')
    if type(dim) is not int:
        raise TypeError('vector linear algebra dim must be an integer')
    rank = max(len(input.shape), len(other.shape))
    if dim < -max(rank, 1) or dim >= max(rank, 1):
        raise ValueError('vector linear algebra dimension is out of range')
    dim %= max(rank, 1)
    left = [1] * (rank - len(input.shape)) + input.shape
    right = [1] * (rank - len(other.shape)) + other.shape
    shape = []
    for a, b in zip(left, right):
        if a == b or b == 1:
            shape.append(a)
        elif a == 1:
            shape.append(b)
        else:
            raise ValueError('vector linear algebra inputs do not broadcast')
    width = 4 if input.dtype == 'float32' else 8
    if math.prod(shape) > sys.maxsize // width or math.prod(max(size, 1) for size in shape) > sys.maxsize:
        raise ValueError('vector linear algebra output exceeds addressable capacity')
    return shape, dim


def cross(input, other, *, dim=-1, out=None):
    """Cross product of same-rank real CPU tensors, broadcasting other axes.

    The selected axis has length three in both inputs. float32/float64 retain
    their ordinary gradient support; int64 uses exact wrapping arithmetic.
    """
    _options(out)
    shape, dim = _vector_operation_shape(input, other, dim, integer=True)
    if not shape or len(input.shape) != len(other.shape):
        raise ValueError('cross inputs must have the same positive rank')
    if input.shape[dim] != 3 or other.shape[dim] != 3:
        raise ValueError('cross requires length three along the selected axis')
    from . import cat
    parts = []
    for first, second in ((1, 2), (2, 0), (0, 1)):
        parts.append(input.narrow(dim, first, 1) * other.narrow(dim, second, 1)
                     - input.narrow(dim, second, 1) * other.narrow(dim, first, 1))
    return cat(parts, dim=dim)


def vecdot(input, other, *, dim=-1, out=None):
    """Real CPU float32/float64 vector dot products after broadcasting both inputs.

    The selected axis follows the broadcast shape, including size-one axes.
    Scalar inputs accept dim=-1 or 0 and return their scalar product. Complex
    conjugation, out= buffers and inference capture are unsupported.
    """
    _options(out)
    shape, dim = _vector_operation_shape(input, other, dim)
    output_shape = shape[:dim] + shape[dim + 1:] if shape else []
    width = 4 if input.dtype == 'float32' else 8
    if math.prod(output_shape) > sys.maxsize // width:
        raise ValueError('vecdot reduction output exceeds addressable capacity')
    product = input * other
    return product.sum_dim(dim, False) if shape else product


def vander(input, N=None):
    """Ascending real CPU Vandermonde powers, using native cumulative products.

    N defaults to the last input dimension and must exceed one. Scalars form
    one row. Supports float32, float64 and int64; integer powers wrap. The
    cumulative-product higher-derivative work bound applies when inputs are
    zero. This is the linalg convention, which differs from torch.vander.
    """
    from operator import index
    from . import ones, cat
    if not isinstance(input, _native.Tensor):
        raise TypeError('vander input must be a Tensor')
    if input.device != 'cpu' or input.dtype not in ('float32', 'float64', 'int64'):
        raise ValueError('vander requires real CPU float32, float64 or int64 input')
    if _native._is_capturing():
        raise ValueError('vander does not support inference capture')
    shape = input.shape or [1]
    if N is None:
        N = shape[-1]
    else:
        N = _native._real_scalar(N, scalar=True)
        if isinstance(N, bool):
            raise TypeError('vander N must be an integer, not bool')
        N = index(N)
    if N <= 1:
        raise ValueError('vander N must be greater than one')
    output_shape = [*shape, N]
    if math.prod(output_shape) > sys.maxsize // 8 or math.prod(max(size, 1) for size in output_shape) > sys.maxsize:
        raise ValueError('vander output exceeds addressable allocation or layout capacity')
    values = input.reshape([1]) if input.ndim == 0 else input
    repeated = values.unsqueeze(-1).expand([*shape, N - 1])
    powers = repeated.cumprod(-1)
    return cat([ones([*shape, 1], dtype=input.dtype), powers], dim=-1)

