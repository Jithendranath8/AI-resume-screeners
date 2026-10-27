# Attacking & Defending AI Resume Screeners against Indirect Prompt Injection

A reproducible study of **hidden-text (indirect) prompt injection** in LLM-based resume
screening — how a candidate can smuggle invisible instructions into a résumé to force a
favorable decision, and which defenses actually stop it.

This repository reproduces the core experiment of the anchor paper and packages the whole
pipeline — threat model, benchmark, attacks, defenses, detector, metrics, and figures — into
**one self-contained, runnable notebook**.

> **TL;DR** — A naive screener is **100% compromised** by hidden-text injection across nine
> concealment techniques. Content filters alone stay leaky (17–56% residual attack success),
> but **channel separation (treating résumé text as inert data) drives attack success to 0%**.
> A four-signal detector flags hidden payloads at **precision 1.00 / recall 0.94 / F1 0.97**,
> with one principled blind spot — the *concealment dilemma*.

---

## Anchor paper

> Jay Barach (2026). *ResumeShield: Channel Separation and an Open Benchmark for Indirect
> Prompt Injection in AI Resume Screening.* arXiv:2609.20188.

Chosen because it is exactly on-topic, ships an open synthetic benchmark, is CPU-only and
deterministic (so it reproduces exactly), and names its own extension gap.

## Research question (project thesis)

ResumeShield reaches 0% attack success only against a **deterministic simulated screener** on
short synthetic documents. This project asks whether a *different* defense — **diffing
PDF-extracted text against the visibly-rendered text** to surface hidden content, plus an
**instruction-intent classifier** that redacts flagged spans before ingestion — can protect a
**real LLM screener** (e.g. GPT-4o-mini / Llama-3) on realistic PDF résumés.
**Target:** baseline attack success ~70%+ → **< 5%**. This milestone (**M2**) establishes the
faithful, deterministic baseline and the harness that the real-LLM extension (**M3**) builds on.

---

## Repository contents

| File | What it is |
|---|---|
| **[`M2_Reproduction.ipynb`](M2_Reproduction.ipynb)** | The whole reproduction — narrative, all code (defined inline, explained block by block), live result tables, and three figures rendered inline. Run it top to bottom. |
| **[`M1_Literature_Survey.ipynb`](M1_Literature_Survey.ipynb)** | The literature survey that motivates the project and selects the anchor paper. |
| `requirements.txt` | `pandas` + `matplotlib` (the core logic is pure standard library). |

The reproduction is **self-contained**: no external package to install beyond the two above, no
generated data files on disk, no cross-file imports. The 104-document benchmark is built in
memory and every number/figure is produced live when the notebook runs.

---

## Key results (reproduced vs. original)

**Attack success rate** — the defense ladder, over 72 injected documents. An attack "succeeds"
only when it flips a résumé's honest `REJECT` to `SELECT`.

| Pipeline | Original (paper) | Reproduced |
|---|---:|---:|
| naive (no defense) | 1.000 | **1.000** ✓ |
| instruction_redact only | 0.611 | **0.611** ✓ |
| markup_strip only | 0.333 | 0.556 |
| all 3 filters (no fence) | 0.167 | 0.278 |
| channel separation (fence only) | 0.000 | **0.000** ✓ |
| full defense (filters + fence) | 0.000 | **0.000** ✓ |

**Hidden-prompt detection** — over the full 104-doc corpus (72 injected / 32 clean).

| Metric | Original | Reproduced |
|---|---:|---:|
| Precision | 1.000 | **1.000** ✓ |
| Recall | 0.944 | **0.944** ✓ |
| F1 | 0.971 | **0.971** ✓ |

Confusion matrix: **TP = 68, FP = 0, FN = 4, TN = 32**. The only four misses are evasive
paraphrases left in **plain visible text** — the *concealment dilemma*: to dodge the detector a
payload must stay visible, at which point ordinary human review can catch it.

---

## How it works

**Threat model.** A résumé renders two ways: `raw_extract` (what a naive text extractor ingests,
including hidden content) and `visible_render` (what a human reviewer sees). The gap between them
is the attack surface.

**Benchmark (104 synthetic documents).** 100% synthetic — no real résumés, fully shareable.

- **72 injected** = 4 base résumés × **9 concealment techniques** × **2 payload families**
- **32 clean** = 4 base résumés × 8 benign visible variants

**Nine concealment techniques:** background-colored (white-on-white) text · zero font size ·
hidden element · off-canvas position · markup comment · document metadata · zero-width
characters · homoglyph substitution · plain visible text.
**Two payload families:** `standard` (overt directive) and `evasive` (oblique paraphrase).

**The victim.** A deterministic, rule-based *simulated* screener (as in the anchor paper): it
scores the candidate against a skill rubric and — unless the content is fenced as inert data —
**obeys** any selection directive it reads.

**The defenses.** Three lexical filters — `markup_strip` → `unicode_normalize` →
`instruction_redact` — plus **channel separation (the fence)**, i.e. telling the screener the
résumé text is data, not instructions. The fence is the component that actually closes the attack.

**The detector.** Flags a document if **any** of four signals fire: (a) visibility diff,
(b) zero-width characters, (c) homoglyph anomaly, (d) an overt directive in visible text.

### The reproduction gap (honest)

The two intermediate ablation points reproduce *higher* than the paper (markup_strip 0.556 vs
0.333; all-filters 0.278 vs 0.167) because the paper publishes neither the exact rules of its
filters nor its private corpus. The gap vanishes once channel separation is applied, so it does
**not** affect the headline conclusion — it is a **sensitivity finding**: those intermediate
percentages depend on under-specified implementation details and are not robustly reproducible
from the paper text alone. Full analysis is in the notebook.

---

## Running it

```bash
pip install -r requirements.txt jupyter
jupyter notebook M2_Reproduction.ipynb   # then run all cells, top to bottom
```

- **Deterministic:** no RNG, no training, no seeds — re-runs are byte-identical.
- **Fast & light:** CPU-only; the full experiment completes in well under a second.
- **Self-verifying:** every reproduced number and figure is computed live; only the paper's
  originals are hard-coded, for side-by-side comparison.

---

## Roadmap

- **M1 — Literature survey** ✅ — select and justify the anchor paper. → [`M1_Literature_Survey.ipynb`](M1_Literature_Survey.ipynb)
- **M2 — Reproduce a feasible baseline** ✅ — this notebook. → [`M2_Reproduction.ipynb`](M2_Reproduction.ipynb)
- **M3 — Real-LLM extension** 🔜 — swap the simulated screener for a hosted LLM behind the same
  `screen()` interface; add the visibility-diff + instruction-intent-classifier defense; evaluate
  on **real PDF résumés** and compare head-to-head against channel separation.

---

## Related work

| Paper | Relevance |
|---|---|
| Barach 2026, *ResumeShield* (arXiv:2609.20188) | **Anchor** — channel separation + open résumé-injection benchmark. |
| Greshake et al. 2023 (arXiv:2302.12173) | Foundational indirect prompt injection. |
| Zhang et al. 2026, USENIX (arXiv:2605.28999) | Real-world measurement (~1% of résumés carry injections). |
| Murray et al. 2025, *PhantomLint* (arXiv:2508.17884) | Hidden-prompt detection. |
| Liu et al. 2024, USENIX (arXiv:2310.12815) | Open-Prompt-Injection evaluation harness. |

---

## Notes

- **Context:** semester research project (NLP, Sem 7).
- **Data:** fully synthetic; contains no real personal data and is safe to share.
- **Reproduced numbers** in the notebook are the live output of the code, not estimates.
