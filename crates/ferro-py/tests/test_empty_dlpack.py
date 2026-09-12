"""Empty CUDA descriptors must preserve residency without reading data."""
import ctypes as C
import gc
import os
import unittest

import ferro
import torch


class Device(C.Structure):
    _fields_ = [("kind", C.c_int), ("ordinal", C.c_int)]


class Dtype(C.Structure):
    _fields_ = [("code", C.c_uint8), ("bits", C.c_uint8), ("lanes", C.c_uint16)]


class Tensor(C.Structure):
    _fields_ = [("data", C.c_void_p), ("device", Device), ("ndim", C.c_int),
                ("dtype", Dtype), ("shape", C.POINTER(C.c_int64)),
                ("strides", C.POINTER(C.c_int64)), ("byte_offset", C.c_uint64)]


class Managed(C.Structure):
    pass


Deleter = C.CFUNCTYPE(None, C.POINTER(Managed))
Managed._fields_ = [("tensor", Tensor), ("ctx", C.c_void_p), ("deleter", Deleter)]
new_capsule = C.pythonapi.PyCapsule_New
new_capsule.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
new_capsule.restype = C.py_object
capsule_name = C.pythonapi.PyCapsule_GetName
capsule_name.argtypes = [C.py_object]
capsule_name.restype = C.c_char_p


class Producer:
    def __init__(self, shape, strides):
        self.shape = (C.c_int64 * len(shape))(*shape)
        self.strides = None if strides is None else (C.c_int64 * len(strides))(*strides)
        self.calls = 0
        self.deleter = Deleter(self.release)
        self.managed = Managed(Tensor(None, Device(2, 0), len(shape), Dtype(2, 32, 1),
                                      self.shape, self.strides, 0), None, self.deleter)
        # Fixture owns its descriptor, not the capsule; explicit failure cleanup
        # below models an unconsumed producer without Python destructor callbacks.
        self.capsule = new_capsule(C.addressof(self.managed), b"dltensor", None)

    def release(self, pointer):
        assert C.addressof(pointer.contents) == C.addressof(self.managed)
        self.calls += 1

    def __dlpack__(self):
        return self.capsule


class EmptyDlpackTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not ferro.cuda_is_available() or not ferro.cuda_init(0):
            if os.environ.get("FERRO_REQUIRE_CUDA"):
                self.fail("CUDA required")
            self.skipTest("CUDA unavailable")

    def check_empty(self, value, shape):
        self.assertEqual(tuple(value.shape), tuple(shape))
        self.assertEqual(value.__dlpack_device__(), (2, 0))
        self.assertEqual(value.tolist(), torch.empty(shape).tolist())
        consumed = torch.from_dlpack(value)
        self.assertEqual(consumed.device, torch.device("cuda:0"))
        self.assertEqual(tuple(consumed.shape), tuple(shape))
        self.assertEqual(consumed.numel(), 0)
        return consumed

    def test_raw_null_empty_strided_and_contiguous(self):
        for shape, strides in [((0, 3), (3, 1)), ((2, 0), (1, 1)),
                               ((2, 0, 4), (4, 4, 1)), ((0, 3), None),
                               ((0, 3), (-(1 << 63), (1 << 63) - 1))]:
            with self.subTest(shape=shape, strides=strides):
                source = Producer(shape, strides)
                value = ferro.from_dlpack(source)
                self.assertEqual(source.calls, 1)
                self.assertEqual(capsule_name(source.capsule), b"used_dltensor")
                consumed = self.check_empty(value, shape)
                del value
                gc.collect()
                self.assertEqual(consumed.numel(), 0)
                with self.assertRaisesRegex(ValueError, "unconsumed"):
                    ferro.from_dlpack(source)
                del consumed
                gc.collect()
                self.assertEqual(source.calls, 1)

    def test_torch_empty_cuda_roundtrip_and_snapshot(self):
        for shape in [(0,), (0, 3), (2, 0), (2, 0, 4), (1, 2, 0, 3)]:
            with self.subTest(shape=shape):
                source = torch.empty(shape, device="cuda")
                value = ferro.from_dlpack(source)
                self.check_empty(value, shape)
                root = ferro.capture(lambda: value.relu())
                compiled = root.compile_fused()
                graph = compiled.prepare_static()
                graph.replay()
                snapshot = graph.snapshot()
                consumed = self.check_empty(snapshot, shape)
                self.check_empty(ferro.from_dlpack(consumed), shape)
                del graph, compiled, root, snapshot, value, source
                gc.collect()
                self.assertEqual(tuple(consumed.shape), shape)
                self.assertEqual(consumed.device.type, "cuda")

    def test_invalid_descriptors_before_empty_shortcut(self):
        cases = [((1,), None, None, "null data"),
                 ((0, -1), None, None, "negative extent"),
                 ((0, 1 << 62, 1 << 62), None, None, "overflows"),
                 ((1 << 62, 1 << 62, 0), None, None, "overflows"),
                 ((0,), None, ("ndim", -1), "negative ndim"),
                 ((0,), None, ("byte_offset", 1), "aligned"),
                 ((0,), None, ("dtype", Dtype(2, 64, 1)), "float32"),
                 ((0,), None, ("device", Device(2, -1)), "device_id")]
        for shape, strides, change, message in cases:
            with self.subTest(shape=shape, message=message):
                source = Producer(shape, strides)
                if change:
                    setattr(source.managed.tensor, *change)
                with self.assertRaisesRegex(ValueError, message):
                    ferro.from_dlpack(source)
                self.assertEqual(source.calls, 0)
                self.assertEqual(capsule_name(source.capsule), b"dltensor")
                source.release(C.pointer(source.managed))
                self.assertEqual(source.calls, 1)


if __name__ == "__main__":
    unittest.main()
