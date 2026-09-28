"""SciForge: deterministic literature retrieval and citation verification.

v0.2 searches PubMed and Crossref for a research question, deduplicates the
records, and verifies each record's DOI / PMID against the issuing database.
It does not generate conclusions, extract evidence, or call any language model.
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
