"""Integrity of source labels, preserved v1 cases and retrieval/evidence units."""

import csv
import hashlib
import json
import unittest

from backend.knowledge_bases.maintenance.prepare import read_source, validate_source, questions, transcript, SOURCE
from backend.knowledge_bases.maintenance.bench import deduplicate_case_hits, validate_run_source
from backend.knowledge_bases.maintenance.audit import structure_checks, content_only


class MaintenanceBenchmarkTests(unittest.TestCase):
    def test_all_forty_long_cases_have_grounded_annotations_and_preserve_v1(self):
        source = read_source()
        validate_source(source)
        with (SOURCE / "legacy_conversations.csv").open(encoding="utf-8-sig") as handle:
            old = {r["case_id"]:r["raw_conversation"] for r in csv.DictReader(handle)}
        for case in source["cases"]:
            raw = "\n".join(f'{t["speaker"]}: {t["text"]}' for t in case["turns"])
            self.assertGreaterEqual(sum(len(t["text"].split()) for t in case["turns"]), 280)
            if case["case_id"] in old:
                self.assertTrue(raw.startswith(old[case["case_id"]]))
            ingested = transcript(case)
            self.assertNotIn('gold_card', ingested)
            self.assertNotIn('sample_answer', ingested)

    def test_questions_cover_all_cases_and_separate_abstention_from_positive_qrels(self):
        rows = questions(read_source())
        self.assertEqual(len(rows), 164)
        positive = [r for r in rows if r["answerable"] == "true"]
        self.assertEqual(len(positive), 156)
        self.assertEqual({cid for r in positive for cid in r["gold_case_ids"].split(';')},
                         {str(i) for i in range(1,41)})
        self.assertEqual(sum(r["retrieval_type"] == "paraphrase_no_asset_id" for r in rows),20)
        self.assertEqual(sum(r["retrieval_type"] == "cross_case_comparison" for r in rows),12)

    def test_source_case_rank_is_deduplicated_without_reordering(self):
        self.assertEqual(deduplicate_case_hits([{"case_id":x} for x in ('2','2','7','2','8')],2),['2','7'])

    def test_assessment_refuses_source_revisions_after_extraction(self):
        source = {"turns": ["Expert: Repair pending."]}
        run = {"source_sha256": hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest()}
        validate_run_source(run,source)
        source["turns"] = ["Expert: Repair completed."]
        with self.assertRaisesRegex(ValueError,"Source changed"):
            validate_run_source(run,source)

    def test_evidence_quote_must_match_the_claimed_source_turn(self):
        case = {"turns":[{"turn_id":"T01","text":"Clean the lens."},
                         {"turn_id":"T02","text":"Restore the recipe."}]}
        record = {"id":"item", "context":{"case_card":{
            "context":"M01", "observation":"unstable", "reasoning":"oil", "action":"clean",
            "outcome":None, "evidence":[{"turn_id":"T01","quote":"Clean the lens."},
                                         {"turn_id":"T02","quote":"Clean the lens."}]}}}
        result = structure_checks(case,[record])
        self.assertEqual(result['valid_quotes'],1)
        self.assertEqual(result['quotes'],2)
        self.assertNotIn('evidence',content_only(record)['context']['case_card'])
        self.assertIn('evidence',record['context']['case_card'])


if __name__ == '__main__':
    unittest.main()
