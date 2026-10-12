"""Single-database policy for industrial QA benchmark writes.

Every new ingestion replaces the previous corpus in kb_industrial_qa.
Checkpoint resumes and explicitly reused extractions stay intact; manuals
are shared. Session locks serialize benchmark writers across processes.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy import text

from backend.knowledge_bases.industrial_qa import SPEC


WRITER_LOCK = 914273510


def require_official_database(spec):
    if spec.database != SPEC.database:
        raise ValueError(f"Industrial QA runs must use {SPEC.database}, not {spec.database}")
    return spec


def run_ids(run):
    ids = list(dict.fromkeys([run["run_id"], *run.get("reused_run_ids", [])]))
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("Run IDs must be nonempty strings")
    return ids


async def replace_previous_runs(store, run):
    """Validate the checkpoint before deleting other runs, in one transaction."""
    if store.engine.url.database != SPEC.database:
        raise ValueError(f"Refusing to replace a corpus outside {SPEC.database}")
    knowledge, raw = store._table("knowledge_items"), store._table("raw_conversations")
    allowed = run_ids(run)
    expected = {
        UUID(row["id"]): row
        for conversation in run.get("conversations", {}).values()
        if conversation.get("status") == "processed"
        for row in conversation.get("records", [])
    }
    evidence = [UUID(c["evidence_id"]) for c in run.get("conversations", {}).values()
                if c.get("status") == "processed" and c.get("evidence_id")]
    async with store.engine.begin() as connection:
        await connection.execute(text(f"LOCK TABLE {knowledge}, {raw} IN EXCLUSIVE MODE"))
        if expected:
            rows = (await connection.execute(text(
                f"SELECT id, subject, claim, type, context, evidence_id FROM {knowledge} WHERE id=ANY(:ids)"
            ), {"ids": list(expected)})).mappings().all()
            if {row["id"] for row in rows} != set(expected):
                raise ValueError("Checkpoint records have been overwritten; restore its snapshot or use a fresh run directory")
            for row in rows:
                prior = expected[row["id"]]
                if (str(row["evidence_id"]) != prior["evidence_id"] or
                        any(row[key] != prior[key] for key in ("subject", "claim", "type", "context"))):
                    raise ValueError("Checkpoint records differ from the current corpus")
                if row["context"].get("benchmark_run_id") not in allowed:
                    raise ValueError("Checkpoint records belong to a different run")
        if evidence:
            actual = set((await connection.execute(text(f"SELECT id FROM {raw} WHERE id=ANY(:ids)"),
                                                   {"ids": evidence})).scalars())
            if set(evidence) - actual:
                raise ValueError("Checkpoint evidence has been overwritten")
        # Arrays are application-level links; superseded_by also has an FK.
        # Never silently leave retained records pointing to removed knowledge.
        dangling = await connection.scalar(text(f"""
            WITH kept AS (SELECT * FROM {knowledge} WHERE context->>'benchmark_run_id'=ANY(:runs))
            SELECT count(*) FROM kept k CROSS JOIN LATERAL unnest(
                k.conflict_with || k.supported_by || k.contradicted_by ||
                CASE WHEN k.superseded_by IS NULL THEN ARRAY[]::uuid[] ELSE ARRAY[k.superseded_by] END
            ) link(id) WHERE NOT EXISTS (SELECT 1 FROM kept target WHERE target.id=link.id)
        """), {"runs": allowed})
        if dangling:
            raise ValueError("Retained corpus links to records from another run")
        deleted = await connection.execute(text(
            f"DELETE FROM {knowledge} WHERE COALESCE(context->>'benchmark_run_id', '') <> ALL(:runs)"
        ), {"runs": allowed})
        raw_deleted = await connection.execute(text(f"""
            DELETE FROM {raw} r
            WHERE COALESCE(r.metadata->>'benchmark_run_id', '') <> ALL(:runs)
              AND NOT EXISTS (SELECT 1 FROM {knowledge} k WHERE k.evidence_id=r.id)
        """), {"runs": allowed})
        return {"database": SPEC.database, "retained_run_ids": allowed,
                "removed_records": deleted.rowcount, "removed_conversations": raw_deleted.rowcount}


@asynccontextmanager
async def overwrite_run(store, run):
    """Hold the writer lock through cleanup and extraction/ingestion."""
    if store.engine.url.database != SPEC.database:
        raise ValueError(f"Industrial QA writers must use {SPEC.database}")
    async with store.engine.connect() as lock:
        try:
            await lock.execute(text("SELECT pg_advisory_lock(:key)"), {"key": WRITER_LOCK})
            report = await replace_previous_runs(store, run)
            yield report
        finally:
            await lock.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": WRITER_LOCK})
