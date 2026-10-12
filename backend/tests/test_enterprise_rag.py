"""Offline tests for the enterprise_rag dataset preparation and benchmark helpers."""

from __future__ import annotations

import io
import sys
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.knowledge_bases.enterprise_rag.bench import doc_recall, summarize
from backend.knowledge_bases.enterprise_rag.prepare import corpus_row, read_member


class BytesRemote:
    """Stands in for RemoteFile: serves byte ranges of an in-memory zip."""

    def __init__(self, data: bytes) -> None:
        self.data = data

    def fetch(self, start: int, length: int) -> bytes:
        return self.data[start : start + length]


class PrepareTest(unittest.TestCase):
    def test_reads_one_member_by_range(self) -> None:
        buffer = io.BytesIO()
        documents = {
            "slack/dsid_" + "a" * 32 + "__eng.txt": "eng-security\n\nnoah: rotate keys " * 50,
            "github/dsid_" + "b" * 32 + "__pr.txt": "Fix prefill thresholds\n\nBody",
        }
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, text in documents.items():
                archive.writestr(name, text, compress_type=zipfile.ZIP_DEFLATED)
            archive.writestr("notes/stored.txt", "plain", compress_type=zipfile.ZIP_STORED)
        remote = BytesRemote(buffer.getvalue())
        archive = zipfile.ZipFile(io.BytesIO(buffer.getvalue()))

        for info in archive.infolist():
            with self.subTest(name=info.filename):
                self.assertEqual(
                    read_member(remote, info), documents.get(info.filename, "plain")
                )

    def test_corpus_row_takes_title_and_source_from_the_document(self) -> None:
        info = zipfile.ZipInfo("jira/internal/dsid_" + "c" * 32 + "__INT-1.txt")
        row = corpus_row(info, "dsid_" + "c" * 32, "\n  Cleanup stale instances \nBody\n")
        self.assertEqual(row["source_type"], "jira")
        self.assertEqual(row["title"], "Cleanup stale instances")
        self.assertEqual(row["document"], "Cleanup stale instances \nBody")


class BenchTest(unittest.TestCase):
    def test_doc_recall(self) -> None:
        self.assertEqual(doc_recall(["a", "b"], ["b", "c"]), 0.5)
        self.assertEqual(doc_recall(["a"], []), 0.0)
        self.assertEqual(doc_recall([], ["a"]), 0.0)

    def test_summarize_groups_by_type_with_overall_last(self) -> None:
        summary = summarize([("basic", 1.0), ("semantic", 0.0), ("basic", 0.5)])
        self.assertEqual(list(summary), ["basic", "semantic", "all"])
        self.assertEqual(summary["basic"], (2, 0.75))
        self.assertEqual(summary["all"], (3, 0.5))


if __name__ == "__main__":
    unittest.main()
