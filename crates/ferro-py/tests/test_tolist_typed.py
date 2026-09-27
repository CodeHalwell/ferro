"""CPU public tolist contracts: exact storage values and Python scalar types."""
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import ferro as fr


class TypedToList(unittest.TestCase):
    def check_values(self, actual, expected, scalar_type):
        if isinstance(expected, list):
            self.assertIs(type(actual), list)
            self.assertEqual(len(actual), len(expected))
            for got, want in zip(actual, expected):
                self.check_values(got, want, scalar_type)
        else:
            self.assertIs(type(actual), scalar_type)
            if isinstance(expected, float) and math.isnan(expected):
                self.assertTrue(math.isnan(actual))
            else:
                self.assertEqual(actual, expected)
                if expected == 0 and scalar_type is float:
                    self.assertEqual(math.copysign(1., actual), math.copysign(1., expected))

    def test_i64_extrema_and_strided_layouts(self):
        values = [-(1 << 63), -(1 << 63) + 1, -(1 << 54) - 1,
                  (1 << 53) + 1, (1 << 54) + 1, (1 << 63) - 1]
        tensor = fr.Tensor.from_i64(values, [2, 3])
        cases = [(tensor, [values[:3], values[3:]]),
                 (tensor.transpose(0, 1), [[values[i], values[i + 3]] for i in range(3)]),
                 (tensor.reshape([1, 2, 3]).transpose(0, 2),
                  [[[values[i]], [values[i + 3]]] for i in range(3)])]
        for index, (view, expected) in enumerate(cases):
            with self.subTest(layout=index):
                self.check_values(view.tolist(), expected, int)
                self.check_values(view.cpu().tolist(), expected, int)

    def test_i64_scalars_are_python_ints(self):
        for value in [-(1 << 63), -(1 << 63) + 1, -1, 0, 1, (1 << 53) + 1, (1 << 63) - 1]:
            with self.subTest(value=value):
                self.check_values(fr.Tensor.from_i64([value], []).tolist(), value, int)

    def test_empty_shapes_and_transposes(self):
        for factory in [fr.Tensor, fr.Tensor.from_i64]:
            for shape, expected in [([0], []), ([0, 3], []), ([2, 0], [[], []]),
                                    ([2, 0, 3], [[], []])]:
                with self.subTest(factory=factory, shape=shape):
                    tensor = factory([], shape)
                    self.assertEqual(tensor.tolist(), expected)
                    self.assertEqual(tensor.cpu().tolist(), expected)
            self.assertEqual(factory([], [0, 2]).transpose(0, 1).tolist(), [[], []])

    def test_f32_values_and_strides_remain_floats(self):
        values = [1.25, -0., float('inf'), float('-inf'), float('nan'), -3.5]
        tensor = fr.Tensor(values, [2, 3])
        self.check_values(tensor.tolist(), [values[:3], values[3:]], float)
        self.check_values(tensor.transpose(0, 1).tolist(),
                          [[values[i], values[i + 3]] for i in range(3)], float)
        self.check_values(fr.Tensor([-0.], []).tolist(), -0., float)

    def test_checkpoint_float_dtypes_are_not_narrowed(self):
        cases = [('F64', struct.pack('<6d', 1.0000000000000002, 1e300, 1e-300, -0., float('inf'), float('nan')),
                  [1.0000000000000002, 1e300, 1e-300, -0., float('inf'), float('nan')]),
                 ('F16', struct.pack('<6H', 0x3c00, 0x0001, 0x7bff, 0x8000, 0x7c00, 0x7e00),
                  [1., 2. ** -24, 65504., -0., float('inf'), float('nan')]),
                 ('BF16', struct.pack('<6H', 0x3f80, 0x0001, 0x7f7f, 0x8000, 0x7f80, 0x7fc0),
                  [1., 2. ** -133, struct.unpack('<f', struct.pack('<I', 0x7f7f0000))[0],
                   -0., float('inf'), float('nan')])]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'typed.safetensors'
            for dtype, data, values in cases:
                with self.subTest(dtype=dtype):
                    header = json.dumps({'x': {'dtype': dtype, 'shape': [2, 3],
                                              'data_offsets': [0, len(data)]}}).encode()
                    path.write_bytes(struct.pack('<Q', len(header)) + header + data)
                    tensor = fr.load_safetensors(str(path))['x']
                    self.check_values(tensor.tolist(), [values[:3], values[3:]], float)
                    self.check_values(tensor.transpose(0, 1).tolist(),
                                      [[values[i], values[i + 3]] for i in range(3)], float)


if __name__ == '__main__':
    unittest.main()
