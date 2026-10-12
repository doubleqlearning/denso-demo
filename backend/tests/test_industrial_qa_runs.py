"""Single-DB runner policy and transactional overwrite in disposable schemas."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text

from backend.knowledge_bases.industrial_qa import SPEC, pilot, rewrite, train
from backend.knowledge_bases.industrial_qa.benchmark_adapter import sha
from backend.knowledge_bases.industrial_qa.runs import overwrite_run, replace_previous_runs, require_official_database
from backend.src.services.knowledge_extraction import KnowledgeType
from backend.src.storage import KnowledgeRecord, KnowledgeStore
from backend.src.storage.databases import database_url


class RunnerDatabaseTests(unittest.TestCase):
    def test_database_override_is_rejected_before_opening_another_store(self):
        self.assertIs(require_official_database(SPEC), SPEC)
        with self.assertRaisesRegex(ValueError, 'must use kb_industrial_qa'):
            require_official_database(replace(SPEC, name='industrial_qa_exp_other'))

    def test_nested_reuse_retains_all_source_run_ids(self):
        baseline = {'split':'train','run_id':'full','reused_run_ids':['base'],
            'conversation_ids':['c'],'conversations':{'c':{'status':'processed','records':[]}},'extraction':{}}
        run = train.make_run([{'question_id':'q','conversation_id':'c'}], baseline, Path('new'))
        self.assertEqual(run['reused_run_ids'], ['full','base'])

    def test_repaired_snapshot_inherits_contract_hashes_but_newer_hashes_win(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            parent=root/'parent'
            parent.mkdir()
            contents=json.dumps({'source_sha256':{'schema.py':'old','catalog.yaml':'same'}})
            (parent/'manifest.json').write_text(contents)
            child={'source_sha256':{'schema.py':'new'},'from_run_dir':str(parent),
                   'source_manifest_sha256':sha(contents)}
            self.assertEqual(train.source_hashes(root,child),{'schema.py':'new','catalog.yaml':'same'})
            (parent/'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'parent manifest changed'):
                train.source_hashes(root,child)

    def test_train_source_manifest_never_selects_a_new_database(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root/'source', root/'target'
            source.mkdir()
            baseline = {'split':'train','run_id':'base','conversation_ids':['c'],
                        'conversations':{'c':{'status':'processed','records':[]}},'extraction':{}}
            (source/'run.json').write_text(json.dumps(baseline))
            files = [Path(train.__file__).with_name(name) for name in
                ('schema.py','extract.md','catalog_types.yaml','catalog_assets.yaml','ontology.yaml')]
            manifest = {'database':'kb_old_experiment_should_not_be_used','input_sha256':{},
                'source_sha256':{str(p.relative_to(pilot.PROJECT_ROOT)):sha(p.read_text()) for p in files}}
            (source/'manifest.json').write_text(json.dumps(manifest))
            args = SimpleNamespace(from_run_dir=source,run_dir=target,k=8,per_conversation=3,
                concurrency=4,batch_size=4,verify='quotes')
            questions=[{'question_id':'q','conversation_id':'c'}]
            with patch.object(pilot,'gold_questions',side_effect=lambda split: questions if split=='train' else []), \
                    patch.object(pilot,'public_questions',return_value={'q':questions[0]}):
                train.prepare(args)
            actual=json.loads((target/'manifest.json').read_text())
            self.assertEqual(actual['database'], 'kb_industrial_qa')
            self.assertEqual(actual['source_database'], manifest['database'])
            self.assertNotIn('backend/knowledge_bases/industrial_qa/experiment.py',actual['source_sha256'])

    def test_rewrite_source_manifest_never_selects_a_new_database(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source,target=root/'source',root/'target'
            source.mkdir()
            (source/'manifest.json').write_text(json.dumps({'database':'kb_old_experiment','run_id':'base'}))
            (source/'run.json').write_text(json.dumps({'run_id':'base'}))
            (source/'question_ids.txt').write_text('q\n')
            rewrite.prepare(SimpleNamespace(from_run_dir=source,run_dir=target))
            actual=json.loads((target/'manifest.json').read_text())
            self.assertEqual(actual['database'],SPEC.database)
            self.assertIs(pilot.SPEC,SPEC)


@unittest.skipUnless(os.getenv('TEST_DATABASE_URL'), 'TEST_DATABASE_URL is not set')
class RunOverwriteDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.schema='test_qa_runs_'+uuid4().hex[:12]
        url=database_url(os.environ['TEST_DATABASE_URL'],SPEC.database)
        self.store=KnowledgeStore.from_url(url,DeterministicFakeEmbedding(size=16),16,schema=self.schema)
        async with self.store.engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{self.schema}"'))
        await self.store.init_schema()
        async with self.store.engine.begin() as conn:
            await conn.execute(text(f'CREATE TABLE "{self.schema}".chunks_full (id text PRIMARY KEY, content text)'))
            await conn.execute(text(f"INSERT INTO \"{self.schema}\".chunks_full VALUES ('manual','unchanged')"))

    async def asyncTearDown(self):
        async with self.store.engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{self.schema}" CASCADE'))
        await self.store.aclose()

    async def add(self, run_id=None, *, supported_by=None):
        meta={'benchmark_run_id':run_id} if run_id else {}
        evidence=await self.store.add_conversation('Expert: inspect the cable.',metadata=meta)
        record=KnowledgeRecord(subject='cable',claim='Inspect the cable.',type=KnowledgeType.RULE,
            evidence_id=evidence.id,context=meta,supported_by=supported_by or [])
        await self.store.add_knowledge([record])
        return record

    def checkpoint(self, record, *, reused=()):
        return {'run_id':record.context['benchmark_run_id'],'reused_run_ids':list(reused),
            'conversations':{'c':{'status':'processed','evidence_id':str(record.evidence_id),
                'records':[record.model_dump(mode='json')]}}}

    async def ids(self, table):
        async with self.store.engine.connect() as conn:
            return set((await conn.execute(text(f'SELECT id FROM "{self.schema}".{table}'))).scalars())

    async def test_new_run_replaces_old_and_unscoped_data_but_preserves_manuals(self):
        await self.add('old')
        await self.add()
        report=await replace_previous_runs(self.store,{'run_id':'new','conversations':{}})
        self.assertEqual(report['removed_records'],2)
        self.assertEqual(report['removed_conversations'],2)
        self.assertEqual(await self.ids('knowledge_items'),set())
        self.assertEqual(await self.ids('raw_conversations'),set())
        self.assertEqual(await self.ids('chunks_full'),{'manual'})

    async def test_resume_and_explicit_reuse_survive_replacement(self):
        old,base,current=await self.add('old'),await self.add('base'),await self.add('current')
        run=self.checkpoint(current,reused=['base'])
        report=await replace_previous_runs(self.store,run)
        self.assertEqual(report['removed_records'],1)
        self.assertEqual(await self.ids('knowledge_items'),{base.id,current.id})
        self.assertEqual(await self.ids('raw_conversations'),{base.evidence_id,current.evidence_id})
        second=await replace_previous_runs(self.store,run)
        self.assertEqual(second['removed_records'],0)

    async def test_stale_checkpoint_rolls_back_without_deleting_current_data(self):
        current=await self.add('current')
        missing=KnowledgeRecord(subject='missing',claim='gone',type=KnowledgeType.RULE,
            evidence_id=uuid4(),context={'benchmark_run_id':'old'})
        with self.assertRaisesRegex(ValueError,'overwritten'):
            await replace_previous_runs(self.store,self.checkpoint(missing))
        self.assertEqual(await self.ids('knowledge_items'),{current.id})

    async def test_changed_checkpoint_and_cross_run_links_are_rejected_atomically(self):
        old=await self.add('old')
        current=await self.add('current',supported_by=[old.id])
        run=self.checkpoint(current)
        with self.assertRaisesRegex(ValueError,'another run'):
            await replace_previous_runs(self.store,run)
        self.assertEqual(await self.ids('knowledge_items'),{current.id,old.id})
        run['conversations']['c']['records'][0]['claim']='tampered'
        with self.assertRaisesRegex(ValueError,'differ'):
            await replace_previous_runs(self.store,run)
        self.assertEqual(await self.ids('knowledge_items'),{current.id,old.id})

    async def test_writer_lock_serializes_runs_and_releases_after_failure(self):
        record=await self.add('current')
        run=self.checkpoint(record)
        entered=asyncio.Event()
        async def second():
            async with overwrite_run(self.store,run):entered.set()
        async with overwrite_run(self.store,run):
            task=asyncio.create_task(second())
            await asyncio.sleep(0.05)
            self.assertFalse(entered.is_set())
        await asyncio.wait_for(task,5)
        self.assertTrue(entered.is_set())
        with self.assertRaises(RuntimeError):
            async with overwrite_run(self.store,run):raise RuntimeError('test failure')
        async with asyncio.timeout(5):
            async with overwrite_run(self.store,run):pass


if __name__=='__main__':
    unittest.main()
