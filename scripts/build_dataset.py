#!/usr/bin/env python3
"""Build the synthetic resume prompt-injection benchmark and write it to disk.

Outputs (under data/benchmark/):
    docs/<doc_id>.json   one file per document (segments + both renderings)
    manifest.csv         one row per document (metadata for quick inspection)
    README.md            a short datasheet for the generated corpus

The corpus is 100% synthetic (no real resumes), so it is safe to commit and
share. Run:  python3 scripts/build_dataset.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from resume_pi.concealment import build_benchmark  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "benchmark")
DOCS_DIR = os.path.join(OUT_DIR, "docs")


def main() -> None:
    os.makedirs(DOCS_DIR, exist_ok=True)
    docs = build_benchmark()

    manifest_rows = []
    for d in docs:
        record = {
            "doc_id": d.doc_id,
            "base_id": d.base_id,
            "injected": d.injected,
            "technique": d.technique,
            "payload_family": d.payload_family,
            "has_hidden_text": d.has_hidden_text(),
            "segments": [asdict(s) for s in d.segments],
            "raw_extract": d.raw_extract(),
            "visible_render": d.visible_render(),
        }
        with open(os.path.join(DOCS_DIR, f"{d.doc_id}.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        manifest_rows.append({
            "doc_id": d.doc_id,
            "base_id": d.base_id,
            "injected": int(d.injected),
            "technique": d.technique,
            "payload_family": d.payload_family,
            "has_hidden_text": int(d.has_hidden_text()),
            "raw_len": len(d.raw_extract()),
            "visible_len": len(d.visible_render()),
        })

    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_inj = sum(r["injected"] for r in manifest_rows)
    n_clean = len(manifest_rows) - n_inj
    datasheet = (
        "# Resume Prompt-Injection Benchmark (synthetic)\n\n"
        f"- Total documents: **{len(manifest_rows)}**\n"
        f"- Injected: **{n_inj}**  (4 resumes x 9 techniques x 2 payload families)\n"
        f"- Clean: **{n_clean}**  (4 resumes x 8 benign variants)\n\n"
        "Reconstructed from the description in Barach (2026), *ResumeShield* "
        "(arXiv:2609.20188). Fully synthetic; no real personal data. Each document "
        "JSON carries its `segments`, the naive `raw_extract`, and the human "
        "`visible_render`. See `manifest.csv` for a flat index.\n"
    )
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(datasheet)

    print(f"Wrote {len(manifest_rows)} documents to {DOCS_DIR}")
    print(f"  injected={n_inj}  clean={n_clean}")
    print(f"Manifest: {os.path.join(OUT_DIR, 'manifest.csv')}")


if __name__ == "__main__":
    main()
