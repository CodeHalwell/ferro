"""Real bool storage at CPU DLPack and safetensors boundaries."""
import ctypes as C
import gc
import json
from pathlib import Path
import struct
import tempfile
import unittest

import ferro as fr
import torch
from safetensors.torch import load_file, save_file


class Device(C.Structure):
    _fields_ = [('kind', C.c_int), ('ordinal', C.c_int)]


class Dtype(C.Structure):
    _fields_ = [('code', C.c_uint8), ('bits', C.c_uint8), ('lanes', C.c_uint16)]


class Descriptor(C.Structure):
    _fields_ = [('data', C.c_void_p), ('device', Device), ('ndim', C.c_int),
                ('dtype', Dtype), ('shape', C.POINTER(C.c_int64)),
                ('strides', C.POINTER(C.c_int64)), ('byte_offset', C.c_uint64)]


class Managed(C.Structure):
    pass


Deleter = C.CFUNCTYPE(None, C.POINTER(Managed))
Managed._fields_ = [('tensor', Descriptor), ('ctx', C.c_void_p), ('deleter', Deleter)]
new_capsule = C.pythonapi.PyCapsule_New
new_capsule.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
new_capsule.restype = C.py_object
capsule_name = C.pythonapi.PyCapsule_GetName
capsule_name.argtypes = [C.py_object]
capsule_name.restype = C.c_char_p


class BoolProducer:
    def __init__(self, values, shape, strides=None, offset=0):
        self.data = (C.c_uint8 * len(values))(*values)
        self.shape = (C.c_int64 * len(shape))(*shape)
        self.strides = None if strides is None else (C.c_int64 * len(strides))(*strides)
        self.calls = 0
        self.deleter = Deleter(self.release)
        pointer = C.addressof(self.data) if values else None
        self.managed = Managed(Descriptor(pointer, Device(1, 0), len(shape), Dtype(6, 8, 1),
                                          self.shape, self.strides, offset), None, self.deleter)
        # The fixture owns the descriptor; failed imports leave its capsule unconsumed.
        self.capsule = new_capsule(C.addressof(self.managed), b'dltensor', None)

    def release(self, pointer):
        assert C.addressof(pointer.contents) == C.addressof(self.managed)
        self.calls += 1

    def __dlpack__(self):
        return self.capsule


class BooleanInteropTests(unittest.TestCase):
    def check_bool(self, actual, expected):
        self.assertEqual(actual.dtype, 'bool')
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertFalse(actual.requires_grad)

    def test_torch_dlpack_roundtrip_views_empty_scalars_and_copy_ownership(self):
        base = torch.tensor([[True, False, True], [False, True, False]])
        cases = [base, base.T, base.reshape(-1)[1:4], torch.tensor(True),
                 torch.tensor(False).expand(2, 3), torch.empty(2, 0, 3, dtype=torch.bool)]
        for source in cases:
            with self.subTest(shape=source.shape, stride=source.stride()):
                value = fr.from_dlpack(source)
                self.check_bool(value, source)
                consumer = torch.from_dlpack(value)
                self.assertEqual(consumer.dtype, torch.bool)
                torch.testing.assert_close(consumer, source)
                saved = source.clone()
                consumer.logical_not_()
                self.check_bool(value, saved)
                del value
                gc.collect()
                torch.testing.assert_close(consumer, saved.logical_not())
        source = torch.tensor([False, True])
        imported = fr.from_dlpack(source)
        source.logical_not_()
        self.assertEqual(imported.tolist(), [False, True])

    def test_bool_export_materializes_strided_native_values_exactly(self):
        value = fr.tensor([[True, False, True], [False, True, False]], dtype='bool').transpose(0, 1)
        consumer = torch.from_dlpack(value)
        self.assertEqual(consumer.dtype, torch.bool)
        self.assertTrue(consumer.is_contiguous())
        self.assertEqual(consumer.tolist(), value.tolist())
        for value in [fr.tensor((1 << 53) + 1), fr.tensor(1.0000000000000002, dtype='float64')]:
            exported = torch.from_dlpack(value)
            self.assertEqual(str(exported.dtype).removeprefix('torch.'), value.dtype)
            self.assertEqual(exported.item(), value.item())

    def test_foreign_bool_bytes_and_offsets_release_producer_once(self):
        for values, shape, strides, offset, expected in [
            ([0, 2, 255, 1], [2, 2], [1, 2], 0, [[False, True], [True, True]]),
            ([55, 0, 255], [2], None, 1, [False, True]),
            ([], [2, 0, 3], [100, -100, 1], 0, [[], []]),
        ]:
            producer = BoolProducer(values, shape, strides, offset)
            value = fr.from_dlpack(producer)
            self.assertEqual(value.dtype, 'bool')
            self.assertEqual(value.tolist(), expected)
            self.assertEqual(producer.calls, 1)
            self.assertEqual(capsule_name(producer.capsule), b'used_dltensor')
            with self.assertRaisesRegex(ValueError, 'unconsumed'):
                fr.from_dlpack(producer)
            self.assertEqual(producer.calls, 1)

    def test_bad_bool_descriptors_reject_before_consumption_or_cuda_init(self):
        cases = [('bits', lambda t: setattr(t, 'dtype', Dtype(6, 1, 1)), 'unsupported DLPack dtype'),
                 ('lanes', lambda t: setattr(t, 'dtype', Dtype(6, 8, 2)), 'unsupported DLPack dtype'),
                 ('cuda', lambda t: setattr(t, 'device', Device(2, 0)), 'CPU storage'),
                 ('null', lambda t: setattr(t, 'data', None), 'null data'),
                 ('offset', lambda t: setattr(t, 'byte_offset', (1 << 64) - 1), 'pointer capacity')]
        for name, change, message in cases:
            with self.subTest(case=name):
                producer = BoolProducer([1], [1])
                change(producer.managed.tensor)
                with self.assertRaisesRegex(ValueError, message):
                    fr.from_dlpack(producer)
                self.assertEqual(producer.calls, 0)
                self.assertEqual(capsule_name(producer.capsule), b'dltensor')
        for shape in [(0, 1 << 62, 1 << 62), (1 << 62, 1 << 62, 0)]:
            with self.assertRaisesRegex(ValueError, 'overflows'):
                fr.from_dlpack(BoolProducer([], shape))
        with self.assertRaisesRegex(ValueError, 'allocation capacity'):
            fr.from_dlpack(BoolProducer([1], [2, 1 << 62], [0, 0]))

    def test_safetensors_cross_library_roundtrip_bool_and_typed_empty(self):
        cases = {'mask': torch.tensor([[True, False, True], [False, True, False]]),
                 'scalar': torch.tensor(True), 'empty': torch.empty(2, 0, 3, dtype=torch.bool)}
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'bool.safetensors')
            save_file(cases, path)
            loaded = fr.load_safetensors(path)
            for name, tensor in loaded.items():
                self.check_bool(tensor, cases[name])
            loaded['mask'] = loaded['mask'].transpose(0, 1)
            fr.save_safetensors(path, loaded)
            expected = dict(cases, mask=cases['mask'].T)
            restored = load_file(path)
            for name, tensor in restored.items():
                self.assertEqual(tensor.dtype, torch.bool)
                torch.testing.assert_close(tensor, expected[name])
            raw = Path(path).read_bytes()
            header_size = struct.unpack('<Q', raw[:8])[0]
            header = json.loads(raw[8:8 + header_size])
            self.assertTrue(all(item['dtype'] == 'BOOL' for item in header.values()))


if __name__ == '__main__':
    unittest.main()
