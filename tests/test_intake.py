"""Intake tests. Stdin is scripted. The model is a fake, so Ollama is not required."""
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from forgeai.intake import IntakeError, _ask, inspect_folder, main, run_intake  # noqa: E402

DRAFT = {
    "scope": "Answer questions from the supplied documents.",
    "audience": "unspecified",
    "tone": "unspecified",
    "must_answer": ["the topics named in the brief"],
    "must_refuse": ["topics outside the documents"],
    "questions": [
        "Who is the audience?",
        "What tone should answers use?",
        "Which topics must be answered?",
        "Which topics must be refused?",
        "What is explicitly out of scope?",
        "Should the agent guess when a document is silent?",
        "Which language should it answer in?",
    ],
}

MERGED = {
    "scope": "Answer questions from the supplied documents for the named audience.",
    "audience": "shore staff",
    "tone": "formal",
    "must_answer": ["leave accrual"],
    "must_refuse": ["medical advice"],
}


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, temperature=0, seed=0):
        self.calls.append({"messages": messages, "temperature": temperature, "seed": seed})
        if not self.responses:
            raise AssertionError("no fake response left")
        return self.responses.pop(0)


class InventoryTests(unittest.TestCase):
    def test_pages_rows_and_sizes(self):
        from pypdf import PdfWriter

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "notes.txt").write_text("hello\n", encoding="utf-8")
            (folder / "parts.csv").write_text("part_no,desc\nA,one\nB,two\n", encoding="utf-8")
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            writer.add_blank_page(width=612, height=792)
            with (folder / "policy.pdf").open("wb") as handle:
                writer.write(handle)
            (folder / "other.docx").write_bytes(b"not a pdf")
            entries = {entry["name"]: entry for entry in inspect_folder(folder)}
        self.assertEqual(entries["policy.pdf"]["pages"], 2)
        self.assertEqual(entries["policy.pdf"]["rows"], None)
        self.assertGreater(entries["policy.pdf"]["bytes"], 0)
        self.assertEqual(entries["parts.csv"]["rows"], 2)
        self.assertEqual(entries["parts.csv"]["pages"], None)
        self.assertEqual(entries["notes.txt"]["pages"], None)
        self.assertEqual(entries["notes.txt"]["rows"], None)
        self.assertEqual(entries["other.docx"]["type"], "docx")
        self.assertIsNone(entries["other.docx"]["error"])

    def test_missing_folder_raises(self):
        with self.assertRaises(IntakeError):
            inspect_folder(Path(tempfile.gettempdir()) / "forgeai-missing-docs-dir")

    def test_unreadable_pdf_is_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "bad.pdf").write_bytes(b"%PDF-1.4\nnot a pdf")
            entry = inspect_folder(folder)[0]
        self.assertEqual(entry["name"], "bad.pdf")
        self.assertIsNone(entry["pages"])
        self.assertTrue(entry["error"])


class InterviewTests(unittest.TestCase):
    def _run(self, responses, answers, **kwargs):
        self.reads = []

        def input_fn():
            self.reads.append(1)
            if not answers:
                raise AssertionError("stdin read with no scripted answer left")
            return answers.pop(0)

        printed = []
        llm = FakeLLM(responses)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "docs"
            folder.mkdir()
            (folder / "notes.txt").write_text("A record number is RQ-14.\n", encoding="utf-8")
            runs = Path(tmp) / "runs"
            result = run_intake(
                "Answer questions about the supplied notes.",
                folder,
                "run1",
                runs_dir=runs,
                llm=llm,
                input_fn=input_fn,
                output_fn=printed.append,
                **kwargs,
            )
            json_path = runs / "run1" / "requirements.json"
            md_path = runs / "run1" / "requirements.md"
            saved = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = md_path.read_text(encoding="utf-8")
        return result, saved, markdown, llm, printed

    def test_asks_at_most_five_and_writes_requirements(self):
        answers = ["shore staff", "formal", "leave accrual", "medical advice", "nothing else", "too many"]
        result, saved, markdown, llm, printed = self._run(
            [json.dumps(DRAFT), json.dumps(MERGED)],
            answers,
        )
        self.assertEqual(len(self.reads), 5)
        self.assertEqual(len(result["questions"]), 5)
        self.assertTrue(result["questions_truncated"])
        self.assertEqual(result["questions_dropped"], 2)
        self.assertEqual(saved["audience"], "shore staff")
        self.assertEqual(saved["must_answer"], ["leave accrual"])
        self.assertEqual(saved["must_refuse"], ["medical advice"])
        self.assertIn("shore staff", markdown)
        self.assertIn("notes.txt", markdown)
        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(llm.calls[0]["temperature"], 0)
        self.assertEqual(llm.calls[0]["seed"], 0)
        merge_prompt = llm.calls[1]["messages"][1]["content"]
        self.assertIn("shore staff", merge_prompt)
        self.assertNotIn("Which language should it answer in?", merge_prompt)
        self.assertIn("notes.txt", llm.calls[0]["messages"][1]["content"])
        self.assertEqual(printed[0], "Q1: Who is the audience?\n")
        self.assertEqual(result["checkpoint"], "skipped")

    def test_zero_questions_does_not_read_stdin(self):
        draft = dict(DRAFT, questions=[])
        result, saved, markdown, llm, printed = self._run([json.dumps(draft)], [])
        self.assertEqual(self.reads, [])
        self.assertEqual(printed, [])
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(saved["audience"], "unspecified")
        self.assertIn("## Scope", markdown)

    def test_non_interactive_records_unanswered_questions(self):
        result, saved, markdown, llm, printed = self._run(
            [json.dumps(DRAFT)],
            [],
            interactive=False,
        )
        self.assertEqual(self.reads, [])
        self.assertEqual(len(result["questions"]), 5)
        self.assertTrue(all(item["answer"] is None for item in result["questions"]))
        self.assertIn("(not asked)", markdown)
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(saved["checkpoint"], "skipped")

    def test_invalid_json_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "docs"
            folder.mkdir()
            (folder / "notes.txt").write_text("hello\n", encoding="utf-8")
            runs = Path(tmp) / "runs"
            with self.assertRaises(IntakeError):
                run_intake(
                    "Answer from the notes.",
                    folder,
                    "bad",
                    runs_dir=runs,
                    llm=FakeLLM(["this is not json"]),
                    input_fn=lambda: "x",
                    output_fn=lambda text: None,
                )
            self.assertFalse((runs / "bad" / "requirements.json").exists())

    def test_checkpoint_approve_is_an_extra_read(self):
        answers = ["shore staff", "formal", "leave accrual", "medical advice", "nothing else", "approve"]
        result, saved, markdown, llm, printed = self._run(
            [json.dumps(DRAFT), json.dumps(MERGED)],
            answers,
            checkpoint=True,
        )
        self.assertEqual(len(self.reads), 6)
        self.assertEqual(result["checkpoint"], "approved")
        self.assertEqual(saved["scope"], MERGED["scope"])
        self.assertTrue(any(line.startswith("# Requirements") for line in printed))

    def test_checkpoint_can_edit_one_field(self):
        answers = ["shore staff", "formal", "leave accrual", "medical advice", "nothing else",
                   "audience=warehouse crew"]
        result, saved, markdown, llm, printed = self._run(
            [json.dumps(DRAFT), json.dumps(MERGED)],
            answers,
            checkpoint=True,
        )
        self.assertEqual(result["checkpoint"], "edited")
        self.assertEqual(saved["audience"], "warehouse crew")
        self.assertIn("warehouse crew", markdown)

    def test_checkpoint_refuses_non_interactive(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "docs"
            folder.mkdir()
            with self.assertRaises(IntakeError):
                run_intake("A brief.", folder, "run1", llm=FakeLLM([]), interactive=False, checkpoint=True)


class CapTests(unittest.TestCase):
    def test_ask_refuses_a_sixth_question(self):
        with self.assertRaises(IntakeError):
            _ask(["q1", "q2", "q3", "q4", "q5", "q6"], lambda: "a", lambda text: None)


class CliTests(unittest.TestCase):
    def test_non_interactive_command(self):
        draft = dict(DRAFT, questions=["Who is the audience?"])

        class RecordingClient:
            def __init__(self, run_id, runs_dir):
                self.run_id = run_id
                self.runs_dir = runs_dir

            def chat(self, messages, temperature=0, seed=0):
                return json.dumps(draft)

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "docs"
            folder.mkdir()
            (folder / "notes.txt").write_text("hello\n", encoding="utf-8")
            runs = Path(tmp) / "runs"
            with patch("forgeai.intake.LLMClient", RecordingClient), redirect_stdout(StringIO()):
                code = main([
                    "--brief", "Answer from the notes.",
                    "--docs", str(folder),
                    "--run-id", "cli1",
                    "--runs-dir", str(runs),
                    "--non-interactive",
                ])
            saved = json.loads((runs / "cli1" / "requirements.json").read_text(encoding="utf-8"))
            self.assertTrue((runs / "cli1" / "requirements.md").is_file())
        self.assertEqual(code, 0)
        self.assertIsNone(saved["questions"][0]["answer"])


if __name__ == "__main__":
    unittest.main()
