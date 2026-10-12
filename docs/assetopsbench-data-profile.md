# AssetOpsBench data profile

Profile date: 2026-10-01

## 1. Executive summary

This repository contains an **allowlisted static-reference adapter**, not a full
copy or runnable deployment of IBM AssetOpsBench.

The prepared data has two deliberately separated layers:

| Layer | File | Rows | Used as KB evidence? |
|---|---|---:|---|
| Static reference corpus | `data/assetopsbench/corpus.csv` | 26 documents | Yes |
| Evaluation scenarios | `data/assetopsbench/questions.jsonl` | 467 questions | No |

The 26 corpus documents are produced from every record in 9 allowlisted files
under the pinned IBM companion-repository revision. The 467 questions are
produced from every non-empty JSONL row in 6 allowlisted Hugging Face files.
Question text, grading criteria, hints and labels are kept out of the KB corpus
to prevent evaluation leakage.

## 2. Completeness and scope

| Scope being assessed | Complete? | Explanation |
|---|---|---|
| Each of the 9 downloaded GitHub reference files | Yes | The preparation code reads every CSV row or every object in the top-level JSON list. No sampling or row limit is applied. |
| Each of the 6 downloaded Hugging Face question files | Yes | Every non-empty JSONL row is retained, producing 467 questions. |
| The entire IBM GitHub repository | No | Only 9 files under `src/couchdb/scenarios_data/shared` are allowlisted. Repository code, other scenario assets and runtime components are not mirrored. |
| The entire AssetOpsBench Hugging Face dataset | No | Only 6 configured scenario files are downloaded. Additional FailureSensorIQ QA variants are explicitly excluded. |
| A complete runnable AssetOpsBench environment | No | Sensor time series, live IoT/tool connections, model execution, forecasting/RUL execution, anomaly execution and work-order mutation are absent. |
| Evidence coverage for all 467 questions | Unknown | There is no verified question-to-document mapping or coverage label. Many questions require data and tools outside the 26-document static corpus. |

Pinned source revisions:

- IBM GitHub companion repository: `81265cbbd36d7348ec0aeae4b398c86d332f2fc1`
- Hugging Face dataset: `5e25bb7f2cd37fb68b9a9e1f99d170ca5be7ce17`

The manifest records a SHA-256 hash for every downloaded source and generated
output, so completeness can be verified relative to these exact pinned files.

## 3. Static reference source inventory

| Source type | Raw file | Documents | Content |
|---|---|---:|---|
| `asset_profile` | `iot/asset_profile_sample.json` | 6 | Asset identity, site, type, status, location and installed-sensor lists |
| `asset_catalog` | `catalog/assets.csv` | 1 | One asset-category/catalog entry |
| `sensor_catalog` | `catalog/sensors.csv` | 1 | One sensor catalog entry |
| `failure_mode_catalog` | `catalog/failure_modes.csv` | 1 | One failure-mode catalog entry |
| `failure_code` | `failure_code/failure_code_sample.csv` | 10 | Failure-code definitions |
| `failure_modes` | `fmea/failure_modes_sample.json` | 1 | A synthetic, explicitly non-exhaustive failure-mode list |
| `feature_catalog` | `tsfm/feature_catalog.json` | 2 | One time-series transform and one feature extractor |
| `model_catalog` | `tsfm/model_catalog.json` | 1 | One TinyTimeMixer model catalog entry |
| `work_order` | `work_order/workorders.csv` | 3 | Historical/sample work-order records and labor plans |
| **Total** | **9 files** | **26** | Static reference sample only |

### Asset-profile coverage

| Asset | Site | Asset type | Sensors listed |
|---|---|---|---:|
| `Chiller 6` | `MAIN` | `CHILLER` | 12 |
| `mp_1` | `MAIN` | `COMPRESSOR` | 15 |
| `hyd_1` | `MAIN` | `HYDRAULIC_PUMP` | 18 |
| `Motor_01` | `MAIN` | `MOTOR` | 1 |
| `PUMP3` | `MAIN` | `PUMP` | 0 |
| `AHU2` | `NORTH` | `AHU` | 0 |
| **Total** |  |  | **46 sensor names** |

An empty sensor list means that no sensors are listed in this sample. It is not
proof that the physical asset has no sensors.

## 4. Prepared KB corpus: `corpus.csv`

### Size and completeness

- Rows/documents: **26**
- Columns: **10**
- Unique `doc_id`: **26**
- Unique source types: **9**
- Unique source revision values: **1**
- `document` length: minimum **357**, mean **722.8**, maximum **1,960** characters
- Non-empty scoped context:
  - `asset_id`: 9/26 rows, 7 unique values
  - `asset_type`: 7/26 rows, 7 unique values
  - `site_id`: 9/26 rows, 2 unique values (`MAIN`, `NORTH`)

The sparse context fields are expected: catalog and model entries often have no
asset or site scope. Empty context means “not provided/not applicable”, not an
unknown asset inferred from another record.

### Corpus field dictionary

| Field | Type | Non-empty | Meaning |
|---|---|---:|---|
| `doc_id` | string | 26/26 | Stable adapter-generated ID: `assetopsbench_` plus a hash of source path, source site and source identity. Used for ingestion resume. |
| `source_type` | string | 26/26 | One of the 9 normalized source kinds listed above. |
| `title` | string | 26/26 | Human-readable title in the form `<source_type>: <source identity>`. |
| `asset_id` | string | 9/26 | Exact source `assetnum`, when the source record has one. No alias normalization is applied. |
| `asset_type` | string | 7/26 | Exact source `assettype`, or `asset_class` for the FMEA sample. |
| `site_id` | string | 9/26 | Exact source `siteid`, when provided. |
| `source_path` | string | 26/26 | Path of the originating reference file within the pinned GitHub repository. |
| `source_url` | string | 26/26 | Immutable URL containing the pinned GitHub commit. Multiple documents from the same file share this URL. |
| `source_revision` | string | 26/26 | Pinned Git commit used for the source record. |
| `document` | text | 26/26 | Text ingested into the extraction pipeline: title, sample-scope disclaimer, source path and the complete source record serialized as formatted JSON. |

## 5. Raw static-reference data dictionary

### 5.1 Asset profiles

`iot/asset_profile_sample.json`: 6 records, 16 fields.

| Field | Meaning | Missing/empty |
|---|---|---:|
| `assetnum` | Exact asset identifier | 0/6 |
| `description` | Asset description | 0/6 |
| `siteid` | Site identifier | 0/6 |
| `orgid` | Owning/operating organization identifier | 0/6 |
| `status` | Status recorded in the sample, not necessarily current real-world status | 0/6 |
| `assettype` | Asset class/type | 0/6 |
| `serialnum` | Manufacturer serial number | 6/6 |
| `manufacturer` | Manufacturer | 6/6 |
| `vendor` | Vendor/supplier | 6/6 |
| `priority` | Asset priority | 6/6 |
| `installdate` | Installation date | 5/6 |
| `changedate` | Record change date | 5/6 |
| `location` | Location identifier | 3/6 |
| `parent` | Parent asset identifier | 6/6 |
| `vintage` | Age/vintage category | 5/6 |
| `sensors` | Complete list of sensor names present in this sample record | 2/6 empty lists |

### 5.2 Simple catalogs

| File | Field | Meaning |
|---|---|---|
| Asset catalog | `category` | Broad equipment category |
| Asset catalog | `category_description` | Explanation of the category |
| Asset catalog | `asset` | Catalog asset/equipment name |
| Asset catalog | `description` | Description of that asset type |
| Sensor catalog | `sensor` | Sensor/measurement category name |
| Sensor catalog | `description` | Intended measurement/use description |
| Failure-mode catalog | `category` | Equipment category to which the failure mode applies |
| Failure-mode catalog | `failure_mode` | Failure-mode name |
| Failure-mode catalog | `description` | Failure-mode description |
| Failure-code sample | `code` | Failure-code identifier (`FC001` through the sample list) |
| Failure-code sample | `description` | Human-readable failure symptom/condition |

All fields in these four files are populated. Their very small row counts make
them examples, not exhaustive catalogs.

### 5.3 FMEA failure-mode sample

`fmea/failure_modes_sample.json`: 1 record.

| Field | Meaning |
|---|---|
| `asset_class` | Asset class to which the list applies (`pump`) |
| `failure_modes` | Failure modes listed for the class (`seal leakage`, `impeller wear`) |
| `source` | Provenance statement (`synthetic sample`) |
| `exhaustive` | Whether the list claims completeness; it is explicitly `false` |

### 5.4 Feature catalog

`tsfm/feature_catalog.json`: 2 heterogeneous records. Fields that are meaningful
only to one feature kind are absent from the other record.

| Field | Meaning |
|---|---|
| `feature_id` | Stable feature/transform identifier |
| `name` | Display name |
| `kind` | Entry kind, such as `transform` or `extractor` |
| `description` | Human-readable purpose |
| `modality` | Input modality, such as time series |
| `interface` | Supported transformation interface |
| `class_name` | Class expected from the embedded implementation |
| `invertible` | Whether inverse transformation is supported |
| `code` | Embedded reference source code; data to describe, not code to execute |
| `provenance` | Origin of the feature definition |
| `method` | Feature-engineering method/family |
| `parent_feature_id` | Parent feature in a derivation lineage |
| `generation` | Generation number in that lineage |
| `target_task` | Task for which the feature was designed |
| `target_model` | Optional target model |
| `dataset` | Optional dataset association |
| `output_type` | Type of produced output |
| `columns_added` | Columns added by the transform |
| `columns_dropped` | Columns removed by the transform |
| `validity` | Validation/check flags supplied by the catalog entry |
| `tags` | Search/classification tags |
| `status` | Catalog lifecycle status |
| `version` | Catalog-entry version |
| `created_by` | Creator/provenance label |
| `created_at` | Catalog creation timestamp |
| `extractor_name` | Registered extractor name, for extractor entries |

### 5.5 Model catalog

`tsfm/model_catalog.json`: 1 record.

| Field | Meaning |
|---|---|
| `model_id` | Catalog model identifier |
| `model_checkpoint` | Checkpoint name |
| `model_family` | Model architecture/family |
| `provenance` | Model origin, such as pretrained |
| `base_model_id` | Optional parent/base model ID |
| `task_ids` | Supported task identifiers |
| `context_length` | Number of historical steps consumed by the model |
| `prediction_length` | Forecast horizon in steps |
| `domain` | Intended domain |
| `frequency` | Supported/declared sampling frequency |
| `source` | Artifact source category |
| `artifact_path` | Reference-system local artifact path |
| `hf_repo` | Hugging Face model repository name |
| `description` | Human-readable capability description |
| `trained_on` | Training/pretraining data labels supplied by the catalog |
| `tags` | Search/classification tags |
| `status` | Catalog lifecycle status |
| `version` | Model/catalog version |
| `created_by` | Creator/provenance label |
| `created_at` | Catalog creation timestamp |
| `sktime_class` | Reference Python class path |
| `params` | Model-construction parameters |

This is metadata about a model in the upstream reference system. It does not
mean the model artifact is downloaded or executable by this KB.

### 5.6 Work orders

`work_order/workorders.csv`: 3 records, 34 fields. CSV parsing retains numeric,
date and JSON-looking values as strings; downstream consumers must parse them
explicitly if typed operations are needed.

| Field/group | Meaning |
|---|---|
| `wonum` | Work-order identifier |
| `siteid` | Site identifier |
| `description`, `description_longdescription` | Short and long work descriptions |
| `status` | Recorded work-order status (`WAPPR`, `APPR`, `COMP` in the sample) |
| `wopriority` | Priority value stored as text |
| `worktype` | Work type (`PdM`, `CM`, `PM` in the sample) |
| `assetnum`, `location` | Target asset and location identifiers |
| `failurecode` | Optional associated failure code |
| `parent`, `taskid` | Optional hierarchy/task identifiers |
| `lead`, `reportedby` | Optional lead and reporting identity |
| `reportdate` | Report timestamp |
| `schedstart`, `schedfinish` | Scheduled start/finish timestamps |
| `targstartdate`, `targcompdate` | Target start/completion timestamps |
| `actfinish` | Actual finish timestamp, when supplied |
| `estlabhrs`, `actlabhrs` | Estimated/actual labor hours |
| `estlabcost`, `actlabcost` | Estimated/actual labor cost |
| `estmatcost`, `actmatcost` | Estimated/actual material cost |
| `estservcost`, `actservcost` | Estimated/actual service cost |
| `esttoolcost`, `acttoolcost` | Estimated/actual tool cost |
| `estatapprtotalcost` | Estimated total cost at approval |
| `esttotalcost`, `acttotalcost` | Estimated/actual total cost |
| `wplabor` | Labor plan encoded as a JSON string, containing labor code, craft, hours and start date |

Sparsity highlights:

- `description_longdescription`, `parent`, `taskid`, `lead` and most estimated
  cost fields are empty in all 3 records.
- 2/3 records lack `failurecode`; 2/3 lack actual completion/labor/cost values.
- Only one record has `status=COMP` and an `actfinish`; historical status is not
  evidence of current asset condition.

## 6. Evaluation question dataset: `questions.jsonl`

### Counts by configuration

| Configuration | Questions |
|---|---:|
| `scenarios` | 152 |
| `rule_logic` | 120 |
| `failure_mode_sensor_mapping` | 88 |
| `prognostics_and_health_management` | 75 |
| `hydrolic_pump` | 17 |
| `compressor` | 15 |
| **Total** | **467** |

### Wrapper fields added by the adapter

Every question has all 7 fields below.

| Field | Meaning |
|---|---|
| `question_id` | Namespaced ID `<config>:<upstream id>` to avoid collisions across files |
| `question` | Copy of upstream `text`; the user-facing evaluation prompt |
| `config` | One of the six configurations above |
| `split` | Adapter-assigned value `train` for every row |
| `source_url` | Immutable URL of the pinned Hugging Face source file |
| `source_revision` | Pinned Hugging Face revision |
| `source_record` | Complete unchanged upstream JSON object, including evaluation annotations |

### Fields inside `source_record`

| Field | Used by | Meaning |
|---|---|---|
| `id` | All configs | Upstream question ID; integer in five configs and string in PHM |
| `text` | All configs | Natural-language question |
| `type` | All configs | Upstream task/agent type |
| `category` | All configs | Task category |
| `characteristic_form` | All configs | Expected-answer/grading criteria; evaluation-only, never KB evidence |
| `entity` | All configs | Main asset/entity class; empty in a small number of rule/PHM rows |
| `group` | All except hydraulic-pump | Upstream scenario grouping |
| `deterministic` | Five configs | Whether upstream labels the task deterministic; absent only from compressor questions |
| `note` | Four configs | Optional provenance or annotation note; often empty |
| `hint` | Compressor and hydraulic-pump | Expected multi-agent/tool workflow hint |
| `failure_event` | Compressor and hydraulic-pump | Labeled failure event used in evaluation |
| `labels` | Compressor and hydraulic-pump | Comma-separated scenario labels |
| `asset_component` | Hydraulic-pump only | Component under evaluation |
| `relevant_sensors` | Hydraulic-pump only | Comma-separated sensor names expected to be relevant |
| `question_type` | Failure-mode/sensor mapping only | Mapping-query subtype |

The `characteristic_form`, `hint`, `failure_event`, `labels` and similar fields
are potential gold/evaluation information. They remain in `questions.jsonl` but
are intentionally absent from `corpus.csv`.

## 7. Data-quality and benchmark-readiness findings

1. **Very small evidence corpus.** Twenty-six static documents cannot reproduce
   the full live/tool-based benchmark assumed by many of the 467 questions.
2. **Coverage is not labeled.** No question has a verified supporting-document
   set, answerability flag or gold evidence ID.
3. **Identifier namespaces differ.** For example, asset profile `Chiller 6` is
   not automatically the work-order asset `CHILLER6`; profile `hyd_1` is not
   automatically scenario asset `hp_1`. Joining them would require explicit
   source evidence or a curated alias table.
4. **Sparse fields must remain unknown.** Null or empty values must not be
   interpreted as zero, false or nonexistence.
5. **Mixed and nested representations.** Work-order numerics/dates are CSV
   strings, `wplabor` is JSON encoded inside a string, and catalog fields contain
   arrays, objects and embedded source code.
6. **Potentially inconsistent dates are preserved.** The adapter intentionally
   does not repair or reinterpret contradictory-looking source values.
7. **Sample catalogs are non-exhaustive.** One-row catalogs and the FMEA record's
   explicit `exhaustive=false` prevent general inventory-completeness claims.
8. **No benchmark score is currently valid.** There is no answer-generation
   run, retrieval metric, answer grader, report, or official score artifact for
   this adapter.
