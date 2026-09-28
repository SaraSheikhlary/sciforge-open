# SciForge Evaluation

How SciForge's outputs are measured. The goal is to know, with evidence, whether SciForge's reports can be trusted, and to catch regressions as it changes.

## What is evaluated

| Dimension | Question | How it is measured |
|---|---|---|
| Citation existence | Does every cited source exist and match its metadata? | Automated: resolve DOI/PMID via Crossref or PubMed; compare title, authors, year |
| Claim support | Does each cited source actually support its claim? | Expert review of a sample of claims; model-assisted pre-screen |
| Evidence labeling | Are established / conflicting / inference / hypothesis labels correct? | Agreement with expert labels |
| Recall of key literature | Did the search find the papers an expert considers essential? | Fraction of expert-designated key papers cited |
| Conflict detection | Were known disagreements in the literature surfaced? | Expert checklist per benchmark question |
| Honesty under thin evidence | Does SciForge say so when evidence is insufficient? | Benchmark questions with deliberately sparse literature |
| Access transparency | Are abstract-only sources labeled? | Automated check against retrieval logs |
| Reproducibility | Does rerunning the logged search return comparable results? | Rerun and compare |

## Benchmark questions

`benchmark_template.csv` defines the format for benchmark questions. Each question should have:

- a clear research question and scope,
- a set of key papers identified by a domain expert, cited by DOI or PMID and verified to exist,
- known points of conflict, if any,
- the expected behavior (for example, "should report insufficient evidence").

Benchmark content must be built only from published, publicly available literature. Do not add unpublished results, private data, or confidential material.

## Scoring (planned)

Per report:
- citation existence rate = verified citations / total citations
- claim support rate = supported claims / reviewed claims
- label agreement = matching labels / reviewed statements
- key-paper recall = key papers cited / key papers listed

Initial targets are listed in `docs/product_spec.md` section 9 and will be revised once baselines exist.

## Error log

Every error found in evaluation (invented citation, unsupported claim, wrong label, missed key paper) is recorded with its cause and fix, so failure patterns can be tracked across versions.
