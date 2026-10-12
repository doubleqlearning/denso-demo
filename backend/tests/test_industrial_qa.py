import asyncio
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4

from pydantic import ValidationError

from backend.knowledge_bases.industrial_qa import CATALOG, SPEC
from backend.knowledge_bases.industrial_qa.catalog import Asset, Catalog
from backend.knowledge_bases.industrial_qa import pilot
from backend.knowledge_bases.industrial_qa.pilot import (
    build_parser,
    cap_per_conversation,
    cited_ranking,
    gold_questions,
    record_turns,
    select,
)
from backend.knowledge_bases.industrial_qa.schema import CaseEvidence, IndustrialQaExtraction
from backend.src.services.knowledge_extraction import KnowledgeType
from backend.src.storage import KnowledgeRecord


SCOPE = dict.fromkeys(
    ("machine_id", "machine_model", "component_id", "component_model", "line_id", "station",
     "product_model", "configuration", "serial_range", "condition", "failure_mode")
)


class IndustrialQaSpecTests(unittest.TestCase):
    def test_spec_owns_its_extraction_contract(self):
        self.assertEqual(SPEC.database, "kb_industrial_qa")
        self.assertIs(SPEC.extraction_output_model, IndustrialQaExtraction)
        self.assertEqual(SPEC.extraction_version, "industrial-qa-case-lesson-v3")
        self.assertIn("{RAW_CONVERSATION}", SPEC.extract_prompt)
        self.assertIn("industrial maintenance knowledge assistant", SPEC.agent_prompt)

    def test_spec_ontology_adds_the_catalog_to_the_failure_modes(self):
        keys = SPEC.ontology.keys
        self.assertIn("electrical_fault", keys["failure_mode"].values)
        self.assertEqual(
            set(keys), {"failure_mode", "machine_id", "machine_model", "component_model",
                        "site", "line_id", "station"},
        )
        # Only failure_mode is closed, so the catalog never reaches the schema.
        self.assertEqual([name for name, key in keys.items() if key.closed], ["failure_mode"])
        self.assertIn("FX5", keys["machine_model"].values)
        self.assertEqual(SPEC.record_normalizer, CATALOG.normalize_records)

    def test_generated_assets_cover_the_three_families(self):
        families = {asset.machine_model for asset in CATALOG.assets.values()}
        self.assertEqual(families, {"FX5", "MR-JE", "i4L"})
        context, _ = SPEC.ontology.normalize({"machine_model": "MELSEC iQ-F"})
        self.assertEqual(context["machine_model"], "FX5")


def catalog(*assets: Asset) -> Catalog:
    types = Catalog.from_files(assets=None)
    return Catalog(
        list(types.families.values()), list(types.components.values()), types.sites,
        list(types.processes.values()), assets={asset.machine_id: asset for asset in assets},
    )


def record(type_=KnowledgeType.RULE, claim="Check the cable.", **context) -> KnowledgeRecord:
    return KnowledgeRecord(
        subject="cable", claim=claim, type=type_, evidence_id=uuid4(), context=context
    )


class CatalogParsingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = catalog()

    def test_asset_ids_in_any_spelling(self):
        self.assertEqual(
            self.catalog.asset_ids("fx5 plc963 then MR-JE-AX 549, again FX5-PLC963"),
            ["FX5-PLC963", "MR-JE-AX549"],
        )
        self.assertEqual(self.catalog.family_of("i4L-R186"), "i4L")
        self.assertEqual(self.catalog.without_asset_ids("FX5-PLC963 CPU"), "CPU")

    def test_family_and_component_names(self):
        self.assertEqual(self.catalog.family("MELSEC iQ-F"), "FX5")
        self.assertEqual(self.catalog.family("FX5 CPU unit"), "FX5")
        self.assertEqual(self.catalog.component("FX5U-32MT/ES"), "FX5U")
        self.assertEqual(self.catalog.component("HG-KN13 motor"), "HG-KN")
        self.assertEqual(self.catalog.component("FX5-ENET module"), "FX5-ENET")
        self.assertIsNone(self.catalog.component("FX5"))

    def test_locations(self):
        self.assertEqual(self.catalog.line("Dong Nai Line 1"), ("Dong Nai", "Line 1"))
        self.assertEqual(self.catalog.line("LINE11"), (None, "Line 11"))
        self.assertEqual(self.catalog.station("palletizing cell d"), "Palletizing cell D")
        self.assertEqual(self.catalog.station("cell D palletizing"), "Palletizing cell D")
        self.assertEqual(self.catalog.station("cell C"), "cell C")
        self.assertEqual(self.catalog.station("station 04"), "Station 4")

    def test_an_asset_is_located_by_the_turn_introducing_it(self):
        texts = [
            "[T01] FX5-PLC992 over at Line 7, cell D palletizing keeps alarming.",
            "[T03] Back in 2019 Hanoi Line 2 had the same.",
            "[T06] at Hai Phong Line 7, that palletizing VFD.",
        ]
        asset = self.catalog.locate(texts, "FX5-PLC992")
        self.assertEqual(
            (asset.machine_model, asset.site, asset.line_id, asset.station),
            ("FX5", "Hai Phong", "Line 7", "Palletizing cell D"),
        )

    def test_build_keeps_first_asset_location_without_conversation_ids(self):
        built = self.catalog.build([
            {"conversation_id": "c1", "turns": [{"text": "MR-JE-AX549 on Line 3, Palletizing cell D"}]},
            {"conversation_id": "c2", "turns": [{"text": "MR-JE-AX549 again, now Line 9"}]},
        ])
        self.assertEqual(built["assets"]["MR-JE-AX549"], {
            "machine_model": "MR-JE", "line_id": "Line 3", "station": "Palletizing cell D",
        })
        self.assertEqual(built["lines"], ["Line 3", "Line 9"])


class CatalogNormalizeTests(unittest.TestCase):
    AX549 = Asset("MR-JE-AX549", "MR-JE", None, "Line 3", "Palletizing cell D")
    TRANSCRIPT = "[T01] MR-JE-AX549 on Line 3, Palletizing cell D trips AL. 16."

    def setUp(self):
        self.catalog = catalog(self.AX549)

    def normalize(self, *records, transcript=TRANSCRIPT):
        self.catalog.normalize_records(transcript, list(records))
        return records

    def test_asset_id_moves_out_of_the_model_keys(self):
        case, = self.normalize(record(
            KnowledgeType.CASE, component_model="MR-JE-AX549 amplifier", component_id="MR-JE-AX549",
        ))
        self.assertEqual(case.context, {
            "machine_id": "MR-JE-AX549", "machine_model": "MR-JE", "component_model": "amplifier",
            "line_id": "Line 3", "station": "Palletizing cell D",
        })
        self.assertEqual(case.revision["catalog"]["changed"]["component_id"], "MR-JE-AX549")

    def test_component_written_as_machine_model(self):
        rule, = self.normalize(record(machine_model="HG-KN13", line_id="line 03"))
        self.assertEqual(
            (rule.context["machine_model"], rule.context["component_model"]), ("MR-JE", "HG-KN")
        )

    def test_specific_model_is_preserved_separately_from_family(self):
        item, = self.normalize(record(machine_id="MR-JE-AX549", machine_model="MR-JE-70A"))
        self.assertEqual(item.context["machine_model"], "MR-JE")
        self.assertEqual(item.context["machine_model_specific"], "MR-JE-70A")
        self.assertEqual(item.revision["catalog"]["changed"]["machine_model"], "MR-JE-70A")
        before = dict(item.context)
        self.normalize(item)
        self.assertEqual(item.context, before)

    def test_specific_model_can_determine_family_without_an_asset(self):
        item, = self.normalize(record(machine_model="MR-JE-70A"), transcript="")
        self.assertEqual(item.context, {"machine_model": "MR-JE", "machine_model_specific": "MR-JE-70A"})

    def test_catalog_asset_corrects_an_inconsistent_known_family(self):
        item, = self.normalize(record(machine_id="MR-JE-AX549", machine_model="FX5"))
        self.assertEqual(item.context["machine_model"], "MR-JE")
        self.assertNotIn("machine_model_specific", item.context)

    def test_new_specific_field_does_not_overwrite_a_source_model(self):
        item, = self.normalize(record(machine_id="MR-JE-AX549", machine_model="MR-JE",
                                     machine_model_specific="MR-JE-70A"))
        self.assertEqual(item.context["machine_model_specific"], "MR-JE-70A")


    def test_locations_are_split_and_moved_to_their_keys(self):
        rule, = self.normalize(record(line_id="Dong Nai Line 1", station="LINE11"), transcript="")
        self.assertEqual(rule.context, {"site": "Dong Nai", "line_id": "Line 1"})

    def test_bare_cell_takes_the_process_of_the_conversation_asset(self):
        rule, = self.normalize(record(station="cell D"))
        self.assertEqual(rule.context["station"], "Palletizing cell D")
        self.assertEqual(rule.context["machine_id"], "MR-JE-AX549")
        self.assertEqual(rule.revision["catalog"]["inherited"], "conversation")

    def test_lesson_inherits_the_asset_of_the_case_it_cites(self):
        case = record(KnowledgeType.CASE, machine_id="mr-je-ax 549")
        lesson = record(line_id="Line 3")
        lesson.supported_by = [case.id]
        self.normalize(case, lesson, transcript="[T01] MR-JE-AX549 and MR-JE-AX550 on Line 3")
        self.assertEqual(case.context["machine_id"], "MR-JE-AX549")
        self.assertEqual(lesson.context["machine_id"], "MR-JE-AX549")
        self.assertEqual(lesson.revision["catalog"]["inherited"], "case")

    def test_general_lesson_stays_unscoped(self):
        case = record(KnowledgeType.CASE, machine_id="MR-JE-AX549")
        general = record(claim="Always photograph the wiring before removing a terminal block.")
        general.supported_by = [case.id]
        self.normalize(case, general)
        self.assertEqual(general.context, {})
        self.assertNotIn("catalog", general.revision)

    def test_unlocated_lesson_naming_the_asset_is_scoped(self):
        lesson, = self.normalize(record(claim="On MR-JE-AX549, check the battery first."))
        self.assertEqual(lesson.context["machine_id"], "MR-JE-AX549")
        self.assertEqual(lesson.revision["catalog"]["inherited"], "claim")

    def test_record_of_another_location_is_not_given_the_asset(self):
        rule, = self.normalize(record(line_id="Line 9"))
        self.assertNotIn("machine_id", rule.context)


class IndustrialLookupPolicyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.catalog = catalog(Asset("MR-JE-AX549", "MR-JE"))

    @staticmethod
    def matches(filters, context):
        if not filters:
            return True
        key, value = next(iter(filters.items()))
        if key == "$and":
            return all(IndustrialLookupPolicyTests.matches(part, context) for part in value)
        if key == "$or":
            return any(IndustrialLookupPolicyTests.matches(part, context) for part in value)
        field = key.removeprefix("context.")
        if "$eq" in value:
            return context.get(field) == value["$eq"]
        if "$exists" in value:
            return (context.get(field) is not None) == value["$exists"]
        raise AssertionError(f"Unsupported test filter: {filters}")

    def test_known_asset_survives_wrong_family_without_matching_other_assets(self):
        filters = self.catalog.lookup_scope(["MR-JE-AX549"], "FX5")
        self.assertTrue(self.matches(filters, {"machine_id": "MR-JE-AX549", "machine_model": "MR-JE-70A"}))
        self.assertTrue(self.matches(filters, {"machine_id": "MR-JE-AX549", "machine_model": "FX5"}))
        self.assertTrue(self.matches(filters, {"machine_model": "MR-JE"}))
        self.assertFalse(self.matches(filters, {"machine_id": "MR-JE-AX550", "machine_model": "MR-JE"}))
        self.assertFalse(self.matches(filters, {"machine_model": "FX5"}))

    def test_variant_query_uses_family_and_keeps_specific_model_filter(self):
        filters = self.catalog.knowledge_filter(context={"machine_model": "MR-JE-70A"})
        self.assertTrue(self.matches(filters, {"machine_model": "MR-JE"}))
        self.assertFalse(self.matches(filters, {"machine_model": "FX5"}))
        filters = self.catalog.knowledge_filter(context={"machine_id": "MR-JE-AX549",
                                                         "machine_model_specific": "MR-JE-70A"},
                                                exact_context_keys=("machine_model_specific",))
        self.assertTrue(self.matches(filters, {"machine_id": "MR-JE-AX549", "machine_model": "bad",
                                              "machine_model_specific": "MR-JE-70A"}))
        self.assertFalse(self.matches(filters, {"machine_id": "MR-JE-AX549", "machine_model": "MR-JE"}))

    def test_family_only_queries_still_exclude_other_families(self):
        filters = self.catalog.lookup_scope([], "MR-JE")
        self.assertTrue(self.matches(filters, {"machine_model": "MR-JE"}))
        self.assertFalse(self.matches(filters, {"machine_model": "FX5"}))

    async def test_online_agent_uses_the_same_asset_policy(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from backend.knowledge_bases.industrial_qa.service import IndustrialQaService
        from backend.src.agents.agent.service import KnowledgeAgentService
        from backend.tests.test_knowledge_agent import FakeStore

        store = FakeStore(hits=[])
        documents = SimpleNamespace(search=AsyncMock(return_value=[]))
        written = '{"answer":"No evidence.","sources":[],"checklist":[]}'
        with mock.patch("backend.src.services.llm.chat_client_from_env", return_value=object()), \
                mock.patch("backend.src.services.llm.ainvoke", AsyncMock(return_value=written)), \
                mock.patch("backend.knowledge_bases.industrial_qa.service.open_document_store",
                           AsyncMock(return_value=documents)):
            service = KnowledgeAgentService.from_kb(SPEC, store)
            self.assertIsInstance(service, IndustrialQaService)
            await service.ask("Check MR-JE-AX549 encoder")
        filters = store.searches[0][1]
        self.assertTrue(self.matches(filters, {"machine_id": "MR-JE-AX549", "machine_model": "MR-JE-70A"}))
        self.assertFalse(self.matches(filters, {"machine_id": "MR-JE-AX550", "machine_model": "MR-JE"}))



class IndustrialQaSchemaTests(unittest.TestCase):
    def test_turn_marker_in_a_quote_becomes_its_turn_id(self):
        evidence = CaseEvidence.model_validate(
            {"turn_id": None, "quote": "No link light means it's not a software problem. [T05]"}
        )
        self.assertEqual(evidence.turn_id, "T05")
        self.assertEqual(evidence.quote, "No link light means it's not a software problem.")
        self.assertEqual(CaseEvidence.model_validate({"turn_id": "9", "quote": "x"}).turn_id, "T09")
        with self.assertRaises(ValidationError):
            CaseEvidence.model_validate({"turn_id": None, "quote": "no turn"})

    def test_lessons_keep_scope_and_source_turns(self):
        result = IndustrialQaExtraction.model_validate({
            "lessons": [{
                "subject": "unmanaged switch", "claim": "Power it from 24 V.", "mandatory": True,
                "scope": {**SCOPE, "line_id": "Dong Nai Line 1"}, "source_turns": ["T9"],
                "case_refs": [0],
            }],
            "cases": [{
                "subject": "i4L-R186", "scope": {**SCOPE, "machine_id": "i4L-R186"},
                "case_card": {"context": "c", "observation": "o", "reasoning": "r", "action": "a",
                              "evidence": [{"turn_id": "T03", "quote": "q"}]},
            }],
        }).to_result()
        case, lesson = result.knowledge_items
        self.assertEqual(case.type, KnowledgeType.CASE)
        self.assertEqual(case.context["machine_id"], "i4L-R186")
        self.assertEqual(lesson.context, {"line_id": "Dong Nai Line 1", "source_turns": ["T09"]})
        self.assertEqual(lesson.supported_by, [0])

    def test_scope_and_source_turns_are_required(self):
        with self.assertRaises(ValidationError):
            IndustrialQaExtraction.model_validate(
                {"lessons": [{"subject": "s", "claim": "c", "scope": SCOPE}]}
            )
        with self.assertRaises(ValidationError):
            IndustrialQaExtraction.model_validate(
                {"lessons": [{"subject": "s", "claim": "c", "source_turns": ["T01"]}]}
            )


class IndustrialQaPilotTests(unittest.TestCase):
    def test_records_point_to_the_turns_they_cite(self):
        turns = [{"turn_id": n, "text": text} for n, text in
                 ((2, "check the switch power LED"), (5, "no link light"), (9, "use 24 V supply"))]
        lesson = KnowledgeRecord(
            subject="switch", claim="Power the switch from 24 V.", type=KnowledgeType.RULE,
            evidence_id=uuid4(), context={"source_turns": ["T09", "T05"]},
        )
        self.assertEqual(record_turns(lesson, turns, "which supply powers the switch? 24 V"), [9, 5])
        unlabelled = lesson.model_copy(update={"context": {}})
        self.assertEqual(sorted(record_turns(unlabelled, turns)), [2, 9])

    def test_cited_sources_lead_the_ranking(self):
        sources = {"K1": {"items": [{"id": "c", "turn_id": 1}]},
                   "D1": {"items": [{"type": "document", "id": "d1"}]}}
        uncited = [{"id": "c", "turn_id": 1}, {"type": "document", "id": "d1"}]
        self.assertEqual(cited_ranking(sources, ["D1"], uncited),
                         [{"type": "document", "id": "d1"}, {"id": "c", "turn_id": 1}])

    def test_caps_records_per_conversation(self):
        a, b = uuid4(), uuid4()
        hits = [SimpleNamespace(record=SimpleNamespace(evidence_id=e)) for e in (a, a, a, b, a, b)]
        kept = cap_per_conversation(hits, 2, 3)
        self.assertEqual([h.record.evidence_id for h in kept], [a, a, b])
        self.assertEqual(len(cap_per_conversation(hits, 0, 10)), 6)

    def test_reruns_listed_question_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            ids = [q["question_id"] for q in gold_questions("train")[:3]][::-1]
            listed = Path(directory) / "ids.txt"
            listed.write_text("\n".join(ids))
            args = build_parser().parse_args(
                ["select", "--run-dir", str(Path(directory) / "run"), "--question-ids", str(listed)]
            )
            self.assertEqual(select(args)["question_ids"], ids)

    def test_selects_questions_and_distractors_only_from_train(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args([
                "select", "--run-dir", directory, "--sample", "10", "--distractors", "5",
            ])
            run = select(args)
            train = gold_questions("train")
            self.assertEqual(run["split"], "train")
            self.assertEqual(len(run["question_ids"]), 10)
            self.assertEqual(len(run["distractor_conversation_ids"]), 5)
            self.assertTrue(set(run["question_ids"]) <= {q["question_id"] for q in train})
            self.assertTrue(set(run["conversation_ids"]) <= {q["conversation_id"] for q in train})
            self.assertEqual(select(args), run)

    def test_rejects_test_split_and_existing_test_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            parser = build_parser()
            with self.assertRaises(SystemExit):
                parser.parse_args(["select", "--run-dir", directory, "--split", "test"])
            args = parser.parse_args(["select", "--run-dir", directory])
            checkpoint = Path(directory) / "run.json"
            original = json.dumps({"split": "test"})
            checkpoint.write_text(original)
            with self.assertRaisesRegex(SystemExit, "train split"):
                select(args)
            self.assertEqual(checkpoint.read_text(), original)

    def test_rejects_test_questions_in_train_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            args = build_parser().parse_args(["select", "--run-dir", directory])
            question = gold_questions("test")[0]
            (Path(directory) / "run.json").write_text(json.dumps({
                "split": "train", "question_ids": [question["question_id"]],
                "conversation_ids": [question["conversation_id"]],
            }))
            with self.assertRaisesRegex(SystemExit, "outside the train split"):
                select(args)


if __name__ == "__main__":
    unittest.main()


class FakeScopeStore:
    def __init__(self, records):
        self.records = {record.id: record for record in records}
        self.updates = []

    async def get_knowledge(self, ids):
        return [self.records[i].model_copy(deep=True) for i in ids if i in self.records]

    async def update_knowledge_scope(self, records, *, revision_keys=()):
        self.updates.append(([record.id for record in records], tuple(revision_keys)))
        return len(records)

    async def aclose(self):
        pass


class RenormalizeTests(unittest.TestCase):
    TRANSCRIPT = "[T01] MR-JE-AX549 on Line 3, Palletizing cell D trips AL. 16."

    def test_normalize_scope_keeps_old_spellings_and_replaces_review_items(self):
        rule = record(machine_model="melsec iq-f", failure_mode="Electrical fault")
        rule.revision = {"ontology": {"normalized": {"failure_mode": "electrical"},
                                      "unknown": {"machine_model": "melsec iq-f"}}}
        with mock.patch.object(pilot, "SPEC", SPEC):
            pilot.normalize_scope("", [rule])
        self.assertEqual(rule.context["machine_model"], "FX5")
        report = rule.revision["ontology"]
        self.assertEqual(report["normalized"]["failure_mode"], "Electrical fault")
        self.assertNotIn("unknown", report)

    def test_renormalize_rescopes_stored_records_in_a_copied_run(self):
        case = record(KnowledgeType.CASE, component_model="MR-JE-AX549")
        lesson = record(line_id="line 3")
        lesson.supported_by = [case.id]
        unchanged = record(claim="Always photograph the wiring first.")
        merged = record(line_id="LINE 3")
        conversation = {
            "status": "processed",
            "records": [r.model_dump(mode="json") for r in (case, lesson, unchanged)],
            "merged": [merged.model_dump(mode="json")],
        }
        store = FakeScopeStore([case, lesson, unchanged])
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp) / "old", Path(tmp) / "new"
            old.mkdir()
            run = {"run_id": "r1", "split": "train", "question_ids": [],
                   "conversation_ids": ["c1"], "conversations": {"c1": conversation}}
            (old / "run.json").write_text(json.dumps(run))
            args = SimpleNamespace(run_dir=new, from_run_dir=old)
            pilot.start_renormalized_run(args)
            copied = json.loads((new / "run.json").read_text())
            self.assertEqual(copied["renormalized_from"], str(old))
            with (
                mock.patch.object(pilot.KnowledgeStore, "for_kb", mock.AsyncMock(return_value=store)),
                mock.patch.object(pilot, "public_conversations", return_value={"c1": {}}),
                mock.patch.object(pilot, "render_transcript", return_value=self.TRANSCRIPT),
                mock.patch.object(pilot, "CATALOG", catalog()),
                mock.patch.object(pilot, "SPEC", SPEC),
            ):
                self.assertEqual(asyncio.run(pilot.renormalize(args, copied)), 0)
            saved = json.loads((new / "run.json").read_text())
            # The old run's checkpoint is left alone.
            self.assertNotIn("renormalized", json.loads((old / "run.json").read_text()))

        (ids, keys), = store.updates
        self.assertEqual(ids, [case.id, lesson.id])  # merged ones are not stored
        self.assertEqual(keys, ("catalog", "ontology"))
        rows = saved["conversations"]["c1"]
        by_id = {row["id"]: row for row in [*rows["records"], *rows["merged"]]}
        self.assertEqual(by_id[str(case.id)]["context"]["machine_id"], "MR-JE-AX549")
        self.assertEqual(by_id[str(lesson.id)]["context"]["line_id"], "Line 3")
        self.assertEqual(by_id[str(lesson.id)]["revision"]["catalog"]["inherited"], "case")
        self.assertEqual(by_id[str(merged.id)]["context"]["line_id"], "Line 3")
        self.assertEqual(by_id[str(unchanged.id)]["context"], {})
        self.assertEqual(saved["renormalized"]["changed"], 3)
        self.assertEqual(saved["renormalized"]["updated"], 2)
