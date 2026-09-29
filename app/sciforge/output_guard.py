"""Guard: prove that given text (e.g. rejected model output) never reaches normal run outputs.

``find_leaks(run_dir, needles)`` byte-searches EVERY file in the run directory
except the opt-in ``debug/`` subdirectory — ``source_texts.json``,
``question.json``, ``evidence.json``, ``model_calls.json`` and any file added
later (e.g. ``report.md``). Each needle is searched as UTF-8 and in its
JSON-escaped form. Used by tests; cheap enough to run after a real run too.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEBUG_DIR_NAME = "debug"


def normal_output_files(run_dir: str | Path) -> list[Path]:
    """All regular files under ``run_dir`` outside ``debug/`` (sorted)."""
    root = Path(run_dir)
    return sorted(p for p in root.rglob("*") if p.is_file() and DEBUG_DIR_NAME not in p.relative_to(root).parts)


def _forms(needle: str) -> set[bytes]:
    return {needle.encode("utf-8"), json.dumps(needle)[1:-1].encode("utf-8"),
            json.dumps(needle, ensure_ascii=False)[1:-1].encode("utf-8")}


def find_leaks(run_dir: str | Path, needles: Iterable[str]) -> list[tuple[str, str]]:
    """``[(relative file path, needle)]`` for every needle found in a normal output file."""
    root = Path(run_dir)
    wanted = [n for n in needles if n]
    leaks: list[tuple[str, str]] = []
    for path in normal_output_files(root):
        data = path.read_bytes()
        for needle in wanted:
            if any(form in data for form in _forms(needle)):
                leaks.append((str(path.relative_to(root)), needle))
    return leaks


def rejected_text_values(rejected_raw: Iterable[dict[str, Any]], *, min_length: int = 12) -> list[str]:
    """Free-text string values of raw rejected items (for leak scanning). Opaque ``rec_`` ids are skipped."""
    out: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str) and len(node) >= min_length and not node.startswith("rec_"):
            out.append(node)

    for entry in rejected_raw:
        walk(entry.get("model_output"))
    return out
