"""Actual CPU F64/I64 DLPack, precise values and descriptor ownership."""
import ctypes as C
import gc
import unittest

import numpy as np
import torch
import ferro as fr
from test_boolean_interop import Device, Dtype, Descriptor, Managed, Deleter, new_capsule, capsule_name


class TypedProducer:
    def __init__(self, values, shape, dtype, strides=None, offset=0):
        scalar = C.c_double if dtype == 'float64' else C.c_int64
        self.data = (scalar * len(values))(*values)
        self.shape = (C.c_int64 * len(shape))(*shape)
        self.strides = None if strides is None else (C.c_int64 * len(strides))(*strides)
        self.calls = 0
        self.deleter = Deleter(self.release)
        pointer = C.addressof(self.data) if values else None
        self.managed = Managed(Descriptor(pointer, Device(1, 0), len(shape), Dtype(2 if dtype == 'float64' else 0, 64, 1),
                                          self.shape, self.strides, offset), None, self.deleter)
        self.capsule = new_capsule(C.addressof(self.managed), b'dltensor', None)

    def release(self, pointer):
        self.calls += 1

    def __dlpack__(self):
        return self.capsule


class TypedInterop(unittest.TestCase):
    def same(self, actual, expected):
        self.assertEqual(actual.dtype, str(expected.dtype).removeprefix('torch.'))
        self.assertEqual(actual.shape, list(expected.shape))
        values = torch.tensor(actual.tolist(), dtype=expected.dtype).reshape(expected.shape)
        torch.testing.assert_close(values, expected, rtol=0, atol=0, equal_nan=True)
        self.assertFalse(actual.requires_grad)

    def test_exact_torch_numpy_round_trips_views_scalars_and_empty(self):
        for dtype, raw in [('float64', [1.0000000000000002, 1e300, -1e-300, -0., float('inf'), float('nan')]),
                           ('int64', [-(1 << 63), (1 << 63) - 1, (1 << 53) + 1, -(1 << 53) - 1, 0, 7])]:
            base = torch.tensor(raw, dtype=getattr(torch, dtype)).reshape(2, 3)
            cases = [base, base.T, base.reshape(-1)[1:4], base[:1, :1].reshape(()),
                     torch.empty(2, 0, 3, dtype=getattr(torch, dtype))]
            for reference in cases:
                with self.subTest(dtype=dtype, shape=reference.shape, strides=reference.stride()):
                    imported = fr.from_dlpack(reference)
                    self.same(imported, reference)
                    output = torch.from_dlpack(imported)
                    self.assertEqual(output.dtype, reference.dtype)
                    torch.testing.assert_close(output, reference, rtol=0, atol=0, equal_nan=True)
                    array = np.from_dlpack(imported)
                    self.assertEqual(array.dtype, getattr(np, dtype))
                    np.testing.assert_equal(array, reference.numpy())
                    self.same(fr.from_dlpack(array.copy()), reference)

    def test_exports_and_imports_are_owning_and_safe_after_producer_drop(self):
        for dtype in ('float64', 'int64'):
            expected = torch.tensor([[1., 2.], [3., 4.]], dtype=getattr(torch, dtype))
            value = fr.tensor(expected.tolist(), dtype=dtype).transpose(0, 1)
            output = torch.from_dlpack(value)
            output[0, 0] = 9
            self.same(value, expected.T)
            del value
            gc.collect()
            self.assertEqual(output.tolist(), [[9, 3], [2, 4]])
            source = expected.clone()
            value = fr.from_dlpack(source)
            source.fill_(7)
            self.same(value, expected)

    def test_foreign_strides_offsets_zero_broadcast_and_single_consumption(self):
        for dtype, values in [('float64', [1e300, -1e-300, 1.0000000000000002, 9.]),
                              ('int64', [-(1 << 63), (1 << 53) + 1, (1 << 63) - 1, 9])]:
            for shape, strides, offset, expected in [([2, 2], [1, 2], 0, [[values[0], values[2]], [values[1], values[3]]]),
                                                    ([2], None, 8, values[1:3]),
                                                    ([2, 2], [0, 0], 0, [[values[0]] * 2] * 2)]:
                source = TypedProducer(values, shape, dtype, strides, offset)
                actual = fr.from_dlpack(source)
                self.same(actual, torch.tensor(expected, dtype=getattr(torch, dtype)))
                self.assertEqual(source.calls, 1)
                self.assertEqual(capsule_name(source.capsule), b'used_dltensor')
                with self.assertRaisesRegex(ValueError, 'unconsumed'):
                    fr.from_dlpack(source)
                self.assertEqual(source.calls, 1)

    def test_invalid_typed_descriptors_reject_before_consuming_or_cuda_initialization(self):
        for dtype in ('float64', 'int64'):
            for change, text in [(lambda t: setattr(t, 'byte_offset', 4), 'aligned'),
                                 (lambda t: setattr(t, 'data', t.data + 4), 'aligned'),
                                 (lambda t: setattr(t, 'device', Device(2, 0)), 'CPU storage'),
                                 (lambda t: setattr(t, 'dtype', Dtype(0, 32, 1)), 'unsupported DLPack dtype')]:
                source = TypedProducer([1, 2], [2], dtype)
                change(source.managed.tensor)
                with self.assertRaisesRegex(ValueError, text):
                    fr.from_dlpack(source)
                self.assertEqual(source.calls, 0)
                self.assertEqual(capsule_name(source.capsule), b'dltensor')
            source = TypedProducer([1], [2**61], dtype, [0])
            with self.assertRaisesRegex(ValueError, 'allocation capacity'):
                fr.from_dlpack(source)
            self.assertEqual(source.calls, 0)


if __name__ == '__main__':
    unittest.main()
