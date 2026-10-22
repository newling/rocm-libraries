# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only tests: python -m unittest discover -s clients/tests -p 'test_hipblaslt_gentest.py'."""

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hipblaslt_gentest as gentest


class CharArrayBytesTests(unittest.TestCase):
    FIELD = ctypes.c_char * 32

    def test_value_that_fits_round_trips(self):
        data = gentest.char_array_bytes("gpu_arch", "9(0a|42|50)|1250", self.FIELD)
        self.assertEqual(data, b"9(0a|42|50)|1250")
        self.assertEqual(self.FIELD(*data).value, b"9(0a|42|50)|1250")

    def test_longest_terminated_value_is_accepted(self):
        value = "x" * 31
        self.assertEqual(gentest.char_array_bytes("gpu_arch", value, self.FIELD), value.encode())

    def test_value_that_fills_the_array_is_rejected(self):
        # ctypes would store all 32 bytes with no terminator.
        with self.assertRaises(SystemExit) as ctx:
            gentest.char_array_bytes("gpu_arch", "x" * 32, self.FIELD)
        self.assertIn("gpu_arch", str(ctx.exception.code))
        self.assertIn("at most 31", str(ctx.exception.code))

    def test_longer_value_is_rejected(self):
        with self.assertRaises(SystemExit):
            gentest.char_array_bytes("gpu_arch_exclude", "x" * 40, self.FIELD)


class GeneratorTests(unittest.TestCase):
    def generate(self, value):
        tests_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "test.yaml"
            output = Path(directory) / "test.data"
            source.write_text(
                "include: hipblaslt_common.yaml\n"
                "Tests:\n"
                "- function:\n"
                "    matmul: *single_precision\n"
                "  name: char_array_boundary\n"
                f"  gpu_arch: '{value}'\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(tests_dir / "hipblaslt_gentest.py"),
                    "-I",
                    str(tests_dir / "data"),
                    str(source),
                    "-o",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            return result, output.read_bytes()

    def test_generator_accepts_terminated_strings(self):
        for value in ("", "x" * 31, "é" * 15 + "x"):
            with self.subTest(value=value):
                result, data = self.generate(value)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(data.startswith(b"hipBLASLt\0"))
                if value:
                    self.assertIn(value.encode("utf-8") + b"\0", data)

    def test_generator_rejects_unterminated_strings(self):
        for value in ("x" * 32, "é" * 16):
            with self.subTest(value=value):
                result, _ = self.generate(value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("gpu_arch", result.stderr)
                self.assertIn("at most 31", result.stderr)


if __name__ == "__main__":
    unittest.main()
