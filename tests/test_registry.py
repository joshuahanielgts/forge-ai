import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.registry import attach, calculator, csv_lookup  # noqa: E402


class CalculatorTests(unittest.TestCase):
    def test_arithmetic(self):
        self.assertEqual(calculator("2 + 3 * 4"), {"ok": True, "value": 14})

    def test_rejects_code_injection(self):
        for bad in ["__import__('os').system('id')", "open('x')", "[1][0]"]:
            self.assertFalse(calculator(bad)["ok"], bad)

    def test_division_by_zero_is_data(self):
        self.assertFalse(calculator("1/0")["ok"])


class CsvLookupTests(unittest.TestCase):
    def setUp(self):
        self.f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="")
        self.f.write("part_no,desc,qty\nP-100,Bearing,4\np-200,Seal,9\n")
        self.f.close()

    def tearDown(self):
        os.unlink(self.f.name)

    def test_case_insensitive_match(self):
        r = csv_lookup(self.f.name, "part_no", "p-100", ["desc"])
        self.assertEqual(r, {"ok": True, "rows": [{"desc": "Bearing"}], "count": 1})

    def test_missing_column_is_error(self):
        self.assertFalse(csv_lookup(self.f.name, "nope", "x")["ok"])


class AttachTests(unittest.TestCase):
    def test_only_requested_tools_attached(self):
        self.assertEqual(set(attach(["calculator"])), {"calculator"})

    def test_unknown_tool_raises(self):
        with self.assertRaises(KeyError):
            attach(["calculator", "teleport"])


if __name__ == "__main__":
    unittest.main()
