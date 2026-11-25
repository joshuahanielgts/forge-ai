"""Groundability guard tests. The threshold comes from the fixture file."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from forgeai.guard import (  # noqa: E402
    calibrate,
    decompose,
    evaluate,
    format_separation,
    load_threshold,
    main,
)

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = FIXTURES / "corpus"
THRESHOLD = FIXTURES / "groundability_threshold.json"


class DecompositionTests(unittest.TestCase):
    def test_splits_and_drops_stopwords(self):
        self.assertEqual(decompose("The quick, brown fox"), "quick brown fox")
        self.assertEqual(decompose("must-answer for the team"), "answer team")
        self.assertEqual(decompose("the and of"), "")

    def test_guard_source_does_not_call_a_model(self):
        source = Path(__file__).resolve().parents[1].joinpath("forgeai", "guard.py").read_text(encoding="utf-8")
        self.assertNotIn("LLMClient", source)
        self.assertNotIn("forgeai.llm", source)
        self.assertNotIn("chat(", source)


class ThresholdTests(unittest.TestCase):
    def test_loads_number_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cutoff.json"
            path.write_text(json.dumps({"threshold": 0.25}) + "\n", encoding="utf-8")
            self.assertEqual(load_threshold(path), 0.25)
        self.assertIsInstance(load_threshold(THRESHOLD), float)


class SeparationTests(unittest.TestCase):
    def test_fixture_lists_separate(self):
        stats = calibrate(CORPUS, FIXTURES / "supported_topics.txt", FIXTURES / "unsupported_topics.txt")
        print(format_separation(stats))
        self.assertGreaterEqual(len((FIXTURES / "supported_topics.txt").read_text(encoding="utf-8").splitlines()), 10)
        self.assertGreaterEqual(len((FIXTURES / "unsupported_topics.txt").read_text(encoding="utf-8").splitlines()), 10)
        self.assertEqual(stats["overlap_count"], 0)
        self.assertGreater(stats["min_supported_score"], stats["max_unsupported_score"])
        self.assertEqual(stats["unseparated"], [])
        stored = json.loads(THRESHOLD.read_text(encoding="utf-8"))
        self.assertEqual(stored["overlap_count"], 0)
        self.assertAlmostEqual(load_threshold(THRESHOLD), stats["threshold"])
        self.assertGreater(stored["threshold"], stats["max_unsupported_score"])
        self.assertLessEqual(stored["threshold"], stats["min_supported_score"])


class RunTests(unittest.TestCase):
    def _requirements(self, folder, topics):
        path = folder / "requirements.json"
        path.write_text(json.dumps({"must_answer": topics}) + "\n", encoding="utf-8")
        return path

    def test_unsupported_topic_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requirements = self._requirements(root, ["zephyrnine quollbit"])
            code = main([
                "--requirements", str(requirements),
                "--docs", str(CORPUS),
                "--run-id", "guard1",
                "--runs-dir", str(root / "runs"),
                "--threshold", str(THRESHOLD),
            ])
            report = json.loads((root / "runs" / "guard1" / "groundability.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 1)
        self.assertTrue(report["stopped"])
        self.assertEqual(report["topics"][0]["status"], "unsupported")
        self.assertEqual(report["topics"][0]["score"], 0.0)
        self.assertIsNone(report["topics"][0]["chunk_id"])

    def test_accept_gap_is_logged_and_allows_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            topic = "zephyrnine quollbit"
            requirements = self._requirements(root, [topic])
            code = main([
                "--requirements", str(requirements),
                "--docs", str(CORPUS),
                "--run-id", "guard2",
                "--runs-dir", str(root / "runs"),
                "--threshold", str(THRESHOLD),
                "--accept-gap", topic,
            ])
            report = json.loads((root / "runs" / "guard2" / "groundability.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertFalse(report["stopped"])
        self.assertEqual(report["accepted_gaps"], [topic])
        self.assertEqual(report["topics"][0]["status"], "unsupported")
        self.assertTrue(report["topics"][0]["gap_accepted"])

    def test_supported_topic_keeps_chunk_id(self):
        report = evaluate(
            {"must_answer": ["alphaquartz label"]},
            CORPUS,
            load_threshold(THRESHOLD),
        )
        row = report["topics"][0]
        self.assertEqual(row["status"], "supported")
        self.assertEqual(row["query"], "alphaquartz label")
        self.assertGreater(row["score"], 0)
        self.assertTrue(row["chunk_id"].startswith("archive_north.txt#"))
        self.assertFalse(report["stopped"])

    def test_empty_corpus_is_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            docs.mkdir()
            requirements = self._requirements(root, ["alphaquartz label"])
            code = main([
                "--requirements", str(requirements),
                "--docs", str(docs),
                "--run-id", "empty",
                "--runs-dir", str(root / "runs"),
                "--threshold", str(THRESHOLD),
            ])
            report = json.loads((root / "runs" / "empty" / "groundability.json").read_text(encoding="utf-8"))
        self.assertNotEqual(code, 0)
        self.assertEqual(report["topics"][0]["status"], "unsupported")
        self.assertIsNone(report["topics"][0]["chunk_id"])
        self.assertEqual(report["topics"][0]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
