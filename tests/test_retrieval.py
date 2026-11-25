import logging
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from forgeai.retrieval import BM25Index, chunk_text, load_documents  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "retrieval"


class RankingTests(unittest.TestCase):
    def test_rare_term_ranks_its_document_first(self):
        chunks = load_documents([FIXTURES / "alpha.txt", FIXTURES / "beta.md"])
        result = BM25Index(chunks).search("xylophonequartz policy")
        self.assertEqual(result.status, "ok")
        self.assertTrue(result.hits[0].source.endswith("alpha.txt"))
        self.assertGreater(result.hits[0].score, 0)
        if len(result.hits) > 1:
            self.assertGreater(result.hits[0].score, result.hits[1].score)
            self.assertTrue(result.hits[1].source.endswith("beta.md"))


class ChunkBoundaryTests(unittest.TestCase):
    def test_long_paragraph_windows_overlap(self):
        text = "".join(str(i % 10) for i in range(50))
        chunks = chunk_text(text, "long.txt", chunk_size=20, overlap=5)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, text[0:20])
        self.assertEqual(chunks[1].text, text[15:35])
        self.assertEqual(chunks[2].text, text[30:50])
        self.assertEqual(chunks[0].text[-5:], chunks[1].text[:5])
        self.assertEqual(chunks[1].text[-5:], chunks[2].text[:5])

    def test_short_paragraph_is_not_split(self):
        text = "Shore staff accrue leave monthly."
        chunks = chunk_text(text, "note.txt", chunk_size=80, overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)

    def test_packed_chunk_repeats_previous_tail(self):
        chunks = chunk_text("AAAA\n\nBBBB", "p.txt", chunk_size=4, overlap=2)
        self.assertEqual(chunks[0].text, "AAAA")
        self.assertEqual(chunks[1].text, "AABBBB")


class NoMatchTests(unittest.TestCase):
    def test_empty_corpus(self):
        result = BM25Index(load_documents([])).search("xylophonequartz")
        self.assertEqual(result.status, "no_match")
        self.assertEqual(result.hits, [])

    def test_zero_shared_tokens_is_no_match(self):
        chunks = load_documents([FIXTURES / "alpha.txt", FIXTURES / "beta.md"])
        self.assertGreater(len(chunks), 0)
        result = BM25Index(chunks).search("qqqqzzzznotindocs")
        self.assertEqual(result.status, "no_match")
        self.assertEqual(result.hits, [])


class LoadTests(unittest.TestCase):
    def test_csv_one_chunk_per_row_keeps_header(self):
        chunks = load_documents([FIXTURES / "parts.csv"])
        self.assertEqual(len(chunks), 2)
        self.assertIn("part_no", chunks[0].text)
        self.assertIn("P-100", chunks[0].text)
        self.assertIn("desc", chunks[0].text)
        self.assertIn("Bearing", chunks[0].text)
        self.assertIn("P-200", chunks[1].text)
        self.assertIn("Seal", chunks[1].text)

    def test_pdf_text_is_retrieved(self):
        path = Path(self._write_temp(b"", ".pdf"))
        path.write_bytes(_pdf_with_text("The xylophonequartz frequency is filed under policy."))
        try:
            chunks = load_documents([path])
            self.assertTrue(chunks)
            self.assertIn("xylophonequartz", chunks[0].text)
            result = BM25Index(chunks).search("xylophonequartz")
            self.assertEqual(result.status, "ok")
            self.assertTrue(result.hits[0].source.endswith(".pdf"))
        finally:
            path.unlink()

    def test_corrupt_pdf_raises(self):
        path = self._write_temp(b"%PDF-1.4\nnot a pdf", ".pdf")
        logger = logging.getLogger("pypdf")
        previous = logger.level
        logger.setLevel(logging.CRITICAL)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaises(ValueError) as caught:
                    load_documents([path])
            self.assertIn("could not read PDF", str(caught.exception))
        finally:
            logger.setLevel(previous)
            os.unlink(path)

    def test_other_suffix_is_rejected(self):
        path = self._write_temp(b"hello", ".docx")
        try:
            with self.assertRaises(ValueError) as caught:
                load_documents([path])
            self.assertIn(".docx", str(caught.exception))
        finally:
            os.unlink(path)

    def _write_temp(self, data: bytes, suffix: str) -> str:
        handle = tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False)
        handle.write(data)
        handle.close()
        return handle.name


def _pdf_with_text(text: str) -> bytes:
    """One-page PDF with a Helvetica text operator. No extra library."""
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET\n".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


if __name__ == "__main__":
    unittest.main()
