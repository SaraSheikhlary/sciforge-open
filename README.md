# SciForge

**An open-source research assistant that turns a scientific question into a traceable, evidence-based research brief.**

> Status: **v0.1 — specification stage.** There is no runnable application yet. This repository currently contains the product specification, architecture, agent workflow, and evaluation framework. See [What's next](#whats-next).

## What SciForge is

SciForge is a scientific research intelligence tool. You give it a research question; it searches the scientific literature, extracts evidence from the sources it finds, checks that every claim traces back to a real source, and produces a structured report:

1. **Research question definition** — the question restated precisely, with scope and assumptions
2. **Literature search** — the exact queries, databases, dates, and result counts, so the search can be rerun
3. **Evidence matrix** — one row per finding: claim, source, experimental system, methods, result, limitations, confidence
4. **Citation and source traceability** — every claim linked to a DOI, PMID, or URL that was checked to exist
5. **Conflicting evidence** — where studies disagree, side by side, with likely reasons
6. **Research gaps** — what the literature does not yet answer
7. **Candidate hypotheses** — testable ideas, clearly labeled as hypotheses
8. **Proposed next steps** — concrete literature, computational, or experimental steps

## What problem it solves

Large language models can summarize science fluently, but they also invent citations, blur the line between what a paper showed and what the model guessed, and give answers no one can audit. Manual literature review avoids those problems but is slow.

SciForge is built around one rule: **no claim without a verifiable source.** Every finding is tied to a specific paper, every citation is checked against a bibliographic database before it appears in a report, and every statement is labeled as one of:

- **Established evidence** — directly supported by cited sources
- **Conflicting evidence** — cited sources disagree
- **Inference** — a reasoned conclusion drawn from the evidence, not stated in any source
- **Hypothesis** — a testable proposal not yet supported by direct evidence

When the evidence is thin, SciForge says so instead of filling the gap.

## What the current version does

v0.1 is a design release. It contains:

| Path | Contents |
|---|---|
| `docs/product_spec.md` | What SciForge v0.1 must do, for whom, and how success is measured |
| `docs/architecture.md` | The six-component application architecture |
| `agent/system_prompt.md` | The system prompt that governs the research agent |
| `agent/research_workflow.md` | The step-by-step investigation procedure and evidence record format |
| `evaluation/` | How SciForge's outputs will be evaluated, plus a benchmark template |
| `SECURITY.md` | How secrets are handled and how to report a vulnerability |
| `LICENSE` | MIT License |

No application code exists yet.

## What's next

1. **Minimal pipeline (v0.2):** a command-line tool that takes a question, searches PubMed and Crossref, and verifies citations.
2. **Evidence extraction and reports (v0.3):** model-based evidence extraction with the xAI API, claim-to-source checks, and Markdown reports.
3. **Evaluation and interface (v0.4+):** a benchmark suite, then a simple web interface.

## How to run it (planned)

SciForge is not runnable yet. The intended setup, once v0.2 exists, is:

```bash
git clone https://github.com/SaraSheikhlary/sciforge-open.git
cd sciforge-open
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Secrets come from environment variables, never from files in the repo
export XAI_API_KEY="your-key-here"          # required: model access (https://console.x.ai)
export NCBI_API_KEY="optional"               # optional: higher PubMed rate limits
export SCIFORGE_CONTACT_EMAIL="you@example.org"  # recommended: polite-pool access for Crossref

sciforge investigate "Your research question here"
```

Never commit an API key. See [SECURITY.md](SECURITY.md).

## Limitations to keep in mind

- SciForge can only read what is openly accessible. For paywalled papers it will often have only the abstract, and it labels those findings accordingly.
- Citation verification confirms that a source exists and matches its metadata; claim-to-source checking reduces, but cannot eliminate, the chance that a claim misrepresents its source. Reports are aids to expert judgment, not substitutes for it.

## License

MIT — see [LICENSE](LICENSE).
