"""Industrial maintenance QA over expert conversations and equipment manuals.

The official H1/H3 pipeline is shared by the API, CLI and benchmark adapters.
Manual chunks and extracted knowledge live in the same KB database; isolated
benchmark runs can explicitly select another database without changing policy.
"""

from pathlib import Path

from backend.knowledge_bases.base import ONTOLOGY_FILE, AnswerPipeline, ContextFilter, KnowledgeBaseSpec
from backend.knowledge_bases.industrial_qa.catalog import Catalog
from backend.knowledge_bases.industrial_qa.schema import IndustrialQaExtraction


DIRECTORY = Path(__file__).parent
# Canonical equipment names (catalog_types.yaml + catalog_assets.yaml).
CATALOG = Catalog.from_files()


SPEC = KnowledgeBaseSpec.from_directory(
    DIRECTORY,
    name="industrial_qa",
    description=__doc__,
    answer_pipeline=AnswerPipeline.INDUSTRIAL_QA,
    context_filters=(
        ContextFilter(
            "machine_id",
            "Machine or asset ID named in the question. Unscoped records still match.",
        ),
        ContextFilter(
            "machine_model",
            "Canonical equipment family: MR-JE, FX5 or i4L. Known asset IDs remain eligible even if family metadata disagrees.",
        ),
        ContextFilter("machine_model_specific", "Exact manufacturer model, e.g. MR-JE-70A.", include_unscoped=False),
    ),
    embedding_context_keys=("machine_id", "machine_model", "machine_model_specific", "component_model", "condition"),
    extraction_version="industrial-qa-case-lesson-v3",
    extraction_output_model=IndustrialQaExtraction,
    ontology=CATALOG.ontology(DIRECTORY / ONTOLOGY_FILE),
    record_normalizer=CATALOG.normalize_records,
    knowledge_filter_builder=CATALOG.knowledge_filter,
)
