# Configure the dataset root first

Edit `agent_config.json` in the repository root before running either entry point.
`dataset_root` must be the absolute path to an existing, prepared dataset release:

```json
{
  "dataset_root": "E:/SpatialOmni_Dataset"
}
```

Use forward slashes in JSON on Windows, or escape backslashes as `\\`.
Point to the directory containing `dataset.json`, not a directory of raw H5ADs.
The current value points to the development machine's external dataset; each
collaborator must set their own location. Data are not included in this repository.
Keep the release's H5ADs, Parquet catalogs, spatial assets, and existing `gene_index/`
together, preserving their relative layout. Canonical gene alignment requires the
existing identity index; this repository does not build or download it.

Both entry points locate the default config relative to their script, independent
of the terminal's working directory. The priority is `--dataset` override, then
`dataset_root` from `--config` (default: this repository's `agent_config.json`).
An invalid/missing root fails at startup; the resolved path is printed. Gemini
does not choose filesystem roots: the application constructs the loader with it.

## Environment and API key

Python 3.10 or newer is required. From the repository root, in your chosen Python
environment, install the loader and both agents:

```powershell
python -m pip install -e ".[agent,analysis]"
```

The distribution name is `spatial-omics-loader`; it includes `spatial_loader`,
`spatial_data_agent`, and `spatial_analysis_agent`. The analysis extra installs
PyDESeq2, matplotlib, formulaic, and threadpoolctl; the agent extra installs the
Google Gen AI SDK. No R installation is needed for these runtime analyses.

Create a local `secrets` directory and an empty file named `gemini_api_key.txt`:

```powershell
New-Item -ItemType Directory -Path secrets -Force
New-Item -ItemType File -Path secrets/gemini_api_key.txt
```

If the file already exists, edit it rather than recreating it. Paste **only your
Gemini API key**, without quotes, JSON, or an `API_KEY=` prefix. Both agents read
this same file. A different location can be supplied with `--key-file`.

No credential or empty key file is distributed here. `secrets/` is Git-ignored;
do not commit keys or include them in shared archives. This version reads a local
key file; it does not automatically read `GEMINI_API_KEY`. `agent_config.json`
contains ordinary configuration, not secrets. The default model is
`gemini-flash-latest`; use `--model` if another accessible Gemini model is needed.

## Runtime architecture and contracts

```text
Natural-language request
  -> Data agent -> Loader tools -> Exported cohort bundle manifest
  -> Analysis agent -> Selected deterministic analyses -> Result/plan manifests
  -> Downstream consumer reads method-specific gene sets and tested backgrounds
```

| Component | Input | Responsibility | Output |
|---|---|---|---|
| Loader (`spatial_loader`) | Configured dataset root, metadata filters, optional genes | Discover metadata, select observations, read bounded sparse counts, reference/copy images, resolve genes with a frozen index | Catalog summaries, query IDs, per-sample matrix blocks/AnnData objects, or a bundle manifest |
| Data agent (`spatial_data_agent`) | Natural-language request and loader tool responses | Discover exact labels, clarify ambiguous cohorts, query and export requested data | Grounded answer/clarification, selections, artifact manifest paths, tool trace |
| Analysis agent (`spatial_analysis_agent`) | User question and one exported bundle manifest | Plan relevant methods, validate design, run QC/region selection, align genes, aggregate donors, execute statistics | Plan receipt with per-job state; child result manifests, complete statistics, plots and gene sets |
| Entry point (`run_spatial_agent.py`) | One user question, config, execution flags | Keep both agent sessions alive through clarification and hand the exported manifest to analysis | One request directory and `pipeline_result.json` |

### Loader

The prepared release is read-only. Metadata discovery/query uses Parquet catalogs;
it does not load the full expression matrix. `fetch` reads counts in bounded
blocks. Samples retain separate native gene axes. Artifacts use H5AD
`layers/counts` (raw sparse counts); no batch correction or merged normalized
matrix is produced. Image coordinates remain local to their own source section.

```python
from spatial_loader import SpatialStore
from spatial_loader.config import resolve_dataset_path

root = resolve_dataset_path("agent_config.json")
with SpatialStore(root, cache_dir="runtime/query_cache") as store:
    labels = store.describe(filters={"dataset_id": "GSE233208"},
                            fields=["condition", "dominant_cell_type"])
    selection = store.query(filters={"dataset_id": "GSE233208",
                                     "condition": ["AD", "Control"]})
    bundle = store.fetch(selection.query_id, delivery="artifact",
                         output_dir="runtime/manual_export", include_images="paths")
    manifest_path = bundle["manifest_uri"]
```

Use a fresh output directory per export. `delivery="iterator"` provides bounded
blocks; `delivery="memory"` provides per-sample AnnData objects subject to a memory
budget. The frozen gene overlay exposes namespace/mapping coverage and canonical
IDs/symbols; it does not infer mappings using LLM knowledge or alter native counts.

### Data agent

Available tools are `describe`, `query`, `fetch`, `gene_systems`, `resolve_genes`,
and `finish`. Questions about available diseases or dominant types need only
metadata queries. Counts of dominant cell types are **spot/observation counts**,
not independent donor counts or purified cell numbers.

The full workflow requests both comparison groups, all native genes and complete
sections. It leaves cell-type/QC selection to analysis so neighborhood graphs keep
their context. Images are referenced by path unless copies are requested. Gemini
receives questions, metadata summaries and receipts, not expression matrices.

Multiple matching studies trigger clarification; the current workflow expects one
cohort bundle. It does not automatically pool studies or run a cross-study meta-
analysis. The default export guard is 250,000 observations; the standalone data
entry point exposes `--max-export-obs`. Do not silently truncate donors to fit it.

### Analysis agent

The model selects an objective-based job list. `scope=comprehensive` is descriptive
and never automatically adds methods. Each selected job carries a reason.

| Biological objective | Method | Main output |
|---|---|---|
| Overall tissue expression changes | `whole_tissue_deg` | Donor pseudobulk PyDESeq2 results |
| Expression changes in a named cell-type-enriched region | `cell_type_enriched_deg` | Selected-region donor pseudobulk DEG |
| Expression changes around a named type | `spatial_neighborhood_deg` | Anchor-plus-neighbor region donor pseudobulk DEG |
| Changes in spatial gene-expression organization | `spatial_pattern` | Sample Moran's I, donor means, Welch comparison, BH FDR |

Focused questions select only relevant methods. A broad investigation of a disease
and a named cell type's local environment may justify all four complementary
views. Comprehensive expression-only questions do not automatically add spatial
methods. A vague question needs clarification. Explicit exclusions take precedence.
Neighborhood DEG is not cell-cell communication inference. Cell-type-enriched
spots remain mixtures; they do not reconstruct purified-cell expression.

The workflow verifies input hashes, validates species/study/organ and contrast,
resolves donors, aligns eligible canonical genes common to contributing samples,
computes source-feature QC, selects regions, and fits the chosen model. It never
fills absent genes with zeros. Source H5ADs remain unchanged.

Donor identity is scoped by source and dataset. Multiple sections from one donor
aggregate into one replicate. Missing/ambiguous identities are excluded and
recorded by default; sample IDs are not substitutes. Paired/longitudinal models
are not implemented. Reviewed mappings can be passed with `--donor-overrides`;
the CSV schema is in `analysis_config/donor_overrides_template.csv`.

Defaults include counts >=100, detected genes >=200, in-tissue spots, dominant
cell-type weight >=0.6 and margin >=0.15 for enriched regions. A donor must have
at least one selected spot; zero-spot donors are excluded. Retained donors with
fewer than 20 spots are flagged, with their group, ID and actual spot count printed.
Each group needs at least two usable independent donors, and model/feature checks
must still pass. This is an execution minimum, not a guarantee of reliable power.
Spatial scoring additionally needs enough spots and finite donor-level scores.

## Commands to try

Run from the repository root. The discovery commands below invoke Gemini but do
not export counts or execute statistics:

```powershell
python run_data_agent.py --query "List the diseases and observation counts available in human brain data. Do not export data."

python run_data_agent.py --query "For GSE233208, list observation counts for every dominant_cell_type separately in AD and Control. Do not apply confidence thresholds or export data."
```

Start a complete interactive session and enter your own question:

```powershell
python run_spatial_agent.py --execute
```

A focused end-to-end request on real data:

```powershell
python run_spatial_agent.py --query "Analyze differential gene expression in oligodendrocyte-enriched regions between AD and Control in GSE233208." --execute
```

Discovery followed by in-session study selection:

```powershell
python run_spatial_agent.py --query "Analyze differential gene expression in astrocyte-enriched regions between disease and normal controls in human brain data. Discover the available diseases and studies first; ask me to choose if multiple comparisons are available." --execute
```

When prompted, reply, for example, `GSE233208, AD versus Control`. The same session
continues, and that choice reaches analysis. Enter `/quit` or `/exit` to stop.
The agent may also need an exact cell-type label. Statistical failures such as
insufficient donors return evidence; thresholds are not relaxed automatically.

A broad question, without naming internal tools:

```powershell
python run_spatial_agent.py --query "Comprehensively investigate how AD tissue differs from Control in GSE233208, focusing on oligodendrocytes and their local tissue environment." --execute
```

Remove `--execute` to export counts but only plan/preflight the analysis. This still
reads/writes counts during extraction; it is not a metadata-only discovery command.
Add `--non-interactive` for batch callers that should receive clarification as a
receipt rather than wait for input. A one-off root override is also supported:

```powershell
python run_spatial_agent.py --dataset "D:/shared/spatial_dataset" --query "..." --execute
```

Examples are requests, not guarantees that a cohort satisfies all model checks.

## Outputs and downstream integration

Full requests are saved under `runtime/runs/<UUID>/`:

```text
request.json                    Original question and execution settings
conversation.json               Clarification dialogue
turns/                          Each agent turn's receipt
query_cache/                    Local query selections
data_exports/<UUID>/            Per-sample counts/metadata bundle and manifest
data_agent_result.json          Latest data-agent response and tool trace
analysis/<UUID>/                Child analysis results or failure evidence
analysis/plans/<UUID>/          Plan, preflight states and child manifest references
analysis_agent_result.json      Latest analysis-agent response
downstream_gene_lists.json      Successful gene sets, split by job and direction
pipeline_result.json            Overall request entry point
```

Standalone data exports go to `runtime/data_exports/`. The original dataset is not
part of a run. Caches/intermediate exports currently have no automatic expiry.

Analysis statuses include `planned`, `success`, `partial_success`,
`clarification_required`, `partially_planned`, `not_estimable`, and `failed`.
`not_estimable` means a statistical comparison could not be performed, not that
no differences were found. Metadata preflight does not guarantee post-QC power.
Successful execution also does not imply statistically significant findings.

Use manifest paths as the handoff contract rather than passing a large DataFrame
through the LLM. Each successful child has complete result tables, donor/QC audits,
plots, and structured gene sets with IDs, symbols, namespace, effect direction,
p-values/FDR, warnings, and tested-universe references. Multi-job receipts contain
`plan_manifest_uri` and `analysis_manifest_uris`; single-job receipts also retain
`manifest_uri`. DEG and spatial results have different effect semantics and FDR
families, so lists remain separate.

```python
from spatial_analysis_agent import load_analysis_plan, load_plan_gene_lists

plan = load_analysis_plan(plan_manifest_path)
sets = load_plan_gene_lists(plan_manifest_path, "significant_positive")
for item in sets:
    gene_ids = item["gene_set"]["gene_ids"]
    background = item["gene_set"]["background_uri"]
    effect_kind = item["gene_set"]["effect_kind"]
```

Readers verify recorded hashes. Keep referenced child directories available;
manifest references include absolute local paths. Downstream analysis/knowledge
agents can consume these gene sets, but enrichment/pathway/trajectory tools are
not included in this runtime.

## Offline checks and included scope

```powershell
python -m unittest discover -s tests -v
```

These checks use mocks and temporary small files; no API key, production dataset,
or network is required. They cover data tool orchestration, selected-plan execution,
deduplication, blocked/failed jobs, and clarification handoff. They do not replace
scientific validation on real cohorts. The full end-to-end commands above perform
real data operations and invoke Gemini; run them when you intend those actions.

This repository contains runtime loading, export, planning and analysis only.
Raw downloads, archive extraction, metadata preparation, RCTD annotation, release
building, gene-index construction and production data are intentionally external.
