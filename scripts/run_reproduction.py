#!/usr/bin/env python3
"""Run the M2 reproduction experiment and print the before/after tables.

Reproduces, for the anchor paper (Barach 2026, ResumeShield):
  * Attack success rate (ASR) of the naive pipeline vs. the defended pipeline
  * Ablation over the individual defense stages
  * Hidden-prompt detection precision / recall / F1
  * A per-technique ASR breakdown

Writes results/results.json and results/summary.md. Run:
    python3 scripts/run_reproduction.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from resume_pi.concealment import TECHNIQUES, build_benchmark  # noqa: E402
from resume_pi.defense import (  # noqa: E402
    instruction_redact, markup_strip, unicode_normalize,
)
from resume_pi.detect import detect  # noqa: E402
from resume_pi.metrics import attack_success_rate, detection_metrics  # noqa: E402
from resume_pi.payloads import base_resume_text  # noqa: E402
from resume_pi.screener import screen  # noqa: E402

# Pipelines under test (mode name -> Document -> outcome) --------------------
def _naive(doc):
    return screen(doc.raw_extract(), treat_as_data=False)

def _fence_only(doc):
    return screen(doc.raw_extract(), treat_as_data=True)

def _redact_only(doc):
    return screen(instruction_redact(doc.raw_extract()), treat_as_data=False)

def _markup_only(doc):
    return screen(markup_strip(doc).raw_extract(), treat_as_data=False)

def _all_filters_no_fence(doc):
    text = markup_strip(doc).raw_extract()
    text = instruction_redact(unicode_normalize(text))
    return screen(text, treat_as_data=False)

def _full_defense(doc):
    text = markup_strip(doc).raw_extract()
    text = instruction_redact(unicode_normalize(text))
    return screen(text, treat_as_data=True)

PIPELINES = {
    "naive (no defense)": _naive,
    "instruction_redact only": _redact_only,
    "markup_strip only": _markup_only,
    "all 3 filters (no fence)": _all_filters_no_fence,
    "channel separation (fence only)": _fence_only,
    "full defense (filters + fence)": _full_defense,
}

# Original reported numbers (Barach 2026) for side-by-side comparison --------
ORIGINAL = {
    "naive (no defense)": 1.000,
    "instruction_redact only": 0.611,
    "markup_strip only": 0.333,
    "all 3 filters (no fence)": 0.167,
    "channel separation (fence only)": 0.000,
    "full defense (filters + fence)": 0.000,
}
ORIGINAL_DETECTION = {"precision": 1.000, "recall": 0.944, "f1": 0.971}


def baseline(base_id: str) -> str:
    """Honest outcome: screen the legitimate, visible resume only."""
    return screen(base_resume_text(base_id), treat_as_data=False)


def main() -> None:
    docs = build_benchmark()
    injected = [d for d in docs if d.injected]
    clean = [d for d in docs if not d.injected]

    results = {"corpus": {"total": len(docs), "injected": len(injected),
                          "clean": len(clean)},
               "asr": {}, "detection": {}, "per_technique": {}}

    # --- ASR table ---------------------------------------------------------
    print("=" * 74)
    print("ATTACK SUCCESS RATE  (injected cases = %d)" % len(injected))
    print("=" * 74)
    print(f"{'pipeline':<34}{'reproduced':>12}{'original':>12}{'gap':>12}")
    print("-" * 74)
    for name, fn in PIPELINES.items():
        r = attack_success_rate(docs, baseline, fn)
        orig = ORIGINAL[name]
        gap = r["asr"] - orig
        results["asr"][name] = {"reproduced": round(r["asr"], 3),
                                "original": orig, "gap": round(gap, 3),
                                "successes": r["successes"], "total": r["total"]}
        print(f"{name:<34}{r['asr']:>12.3f}{orig:>12.3f}{gap:>+12.3f}")

    # --- Detection table ---------------------------------------------------
    dm = detection_metrics(docs, detect)
    results["detection"] = {
        "reproduced": {k: round(dm[k], 3) for k in ("precision", "recall", "f1")},
        "original": ORIGINAL_DETECTION,
        "confusion": {k: dm[k] for k in ("tp", "fp", "fn", "tn")},
    }
    print("\n" + "=" * 74)
    print("HIDDEN-PROMPT DETECTION")
    print("=" * 74)
    print(f"{'metric':<16}{'reproduced':>14}{'original':>12}{'gap':>12}")
    print("-" * 74)
    for k in ("precision", "recall", "f1"):
        gap = dm[k] - ORIGINAL_DETECTION[k]
        print(f"{k:<16}{dm[k]:>14.3f}{ORIGINAL_DETECTION[k]:>12.3f}{gap:>+12.3f}")
    print(f"\nconfusion: TP={dm['tp']} FP={dm['fp']} FN={dm['fn']} TN={dm['tn']}")

    # --- Per-technique ASR (naive vs full defense) -------------------------
    print("\n" + "=" * 74)
    print("PER-TECHNIQUE ASR   (naive  ->  full defense)")
    print("=" * 74)
    for tech in TECHNIQUES:
        sub = [d for d in docs if d.technique == tech]
        n = attack_success_rate(sub, baseline, _naive)["asr"]
        f = attack_success_rate(sub, baseline, _full_defense)["asr"]
        # which of these evade the detector?
        missed = sum(1 for d in sub if d.injected and not detect(d))
        results["per_technique"][tech] = {"naive_asr": round(n, 3),
                                          "full_defense_asr": round(f, 3),
                                          "detector_missed": missed}
        print(f"{tech:<18}{n:>8.3f}  -> {f:>6.3f}   detector missed: {missed}/{len(sub)}")

    # --- Save --------------------------------------------------------------
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", "results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\nSaved:", out)


if __name__ == "__main__":
    main()
