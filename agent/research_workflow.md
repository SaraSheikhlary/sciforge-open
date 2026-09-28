# SciForge Research Workflow (v0.1)

The procedure the orchestrator runs for each investigation. Steps marked **[code]** are deterministic; steps marked **[model]** use the model layer with the system prompt in `system_prompt.md`.

## Inputs

| Input | Required | Description |
|---|---|---|
| `research_question` | Yes | The question, in the user's words |
| `date_range` | No | Publication window, e.g. 2021–2026 |
| `targets` | No | Target organisms, cell types, tissues, or materials |
| `seed_ids` | No | DOIs or PMIDs the user wants included |

## Stages

### 1. Define the question [model]
Restate the question operationally: system or population, variable or intervention, comparator, outcome, scope. List assumptions and ambiguities. If an ambiguity would materially change the answer, surface it to the user before continuing.

### 2. Identify concepts and synonyms [model]
For each key concept, list synonyms, abbreviations, and controlled vocabulary (for example MeSH terms).

### 3. Build and run searches [model → code]
The model proposes query strings; code runs them against each database and logs: query, database, parameters, timestamp, result count. Seed IDs are fetched directly.

### 4. Screen results [model]
Screen titles and abstracts against explicit inclusion and exclusion criteria. Record the reason for each exclusion. Prefer primary research; trace foundational claims back to original studies where they were retrieved.

### 5. Retrieve text [code]
Fetch open-access full text where legitimately available; otherwise use the abstract. Record `access_level` for each source.

### 6. Extract evidence [model]
Create one evidence record per important finding (schema below). Numbers are copied exactly from the text. Unconfirmable fields are `null` or `"not verified"`.

### 7. Compare and assess [model]
Group records into agreeing and conflicting sets. For conflicts, note likely reasons (system, dose, methods, measurement, sample size, definitions). Note methodological limitations.

### 8. Validate [code + model]
1. **[code]** Verify every source exists: resolve DOI via Crossref or PMID via PubMed; confirm title, authors, and year match.
2. **[model]** Check each claim against its source passage; mark `supported`, `partially_supported`, `unsupported`, or `not_verified`.
3. **[code]** Remove unsupported claims from Key Findings or flag them.
4. **[model]** Confirm every statement carries an evidence-category label.
5. **[model]** List known gaps in the search: databases not searched, abstract-only sources, recent unindexed work.

### 9. Identify gaps and hypotheses [model]
List unanswered or weakly answered questions. Propose testable hypotheses, labeled `hypothesis`, each tied to the gaps and evidence records that motivate it.

### 10. Generate report [code + model]
Assemble the ten-section report. The evidence matrix and source list are rendered from structured data. Include a validation summary and run metadata.

## Evidence record schema

| Field | Type | Description |
|---|---|---|
| `record_id` | string | Unique within the investigation |
| `claim` | string | The specific claim, one sentence |
| `source` | string \| null | Short citation, e.g. "Author et al., Journal, Year" |
| `source_url` | string \| null | Where the source was accessed |
| `paper_title` | string \| null | Exact title |
| `authors` | list \| null \| "not verified" | Authors as published |
| `year` | integer \| null \| "not verified" | Publication year |
| `doi` | string \| null | DOI without prefix, or null |
| `evidence_type` | enum | primary_experimental, primary_computational, clinical_trial, systematic_review, meta_analysis, narrative_review, preprint, patent, database, technical_document, other, not verified |
| `experimental_system` | string \| null | Organism, cell type, material, or computational system |
| `methods` | string \| null | Key methods |
| `finding` | string \| null | What the source reports, numbers exactly as stated |
| `limitations` | string \| null | Limitations relevant to this finding |
| `confidence` | object | `level` (high, moderate, low, not verified) and `rationale` |
| `relevance_to_question` | string \| null | How the finding bears on the question |
| `evidence_category` | enum | established, conflicting, inference, hypothesis |
| `access_level` | enum | full_text, abstract_only, not_accessed |
| `verification_status` | enum | supported, partially_supported, unsupported, not_verified |
| `date_retrieved` | date \| null | When the source was accessed |
| `notes` | string \| null | Anything else relevant |

A machine-readable JSON Schema version will be added under `app/` in v0.2.

## Failure handling

Any stage that fails (unreachable database, invalid model output, timeout) is recorded and reported in the Limitations section. The report never hides a failure or fills its gap with unsourced content.
