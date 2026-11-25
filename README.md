# Attacking and defending real PDF résumé screeners

## Current direction: real PDFs + real LLM + combined defence

The main extension is restored to the original proposal: **a real LLM screening actual PDF résumés, protected by visibility-diff + a transformer injection classifier**. Gemini is now the default hosted option; GPT-4o-mini and local Llama-3 remain available. The earlier TF-IDF study below is a completed pilot, not completion of this extension.

**Start with these self-contained Colab notebooks:**

| Notebook | Purpose |
|---|---|
| [Real PDF LLM Defence](colab/Real_PDF_LLM_Defence_Colab.ipynb) | Upload real PDFs; render/OCR; compare extracted text; score with a transformer detector; test five defence configurations on an actual LLM; export results and show a PDF demo. |
| [Train Transformer Classifier — T4](colab/Train_Transformer_Classifier_T4.ipynb) | Optional DistilBERT fine-tuning on a public injection dataset or your labelled CSV; export weights and a validation-selected threshold for the main notebook. |

### How to run in Colab

1. Download the main notebook and upload it at [Google Colab](https://colab.research.google.com/) using **File → Upload notebook**.
2. The default is `BACKEND='gemini'` with `GEMINI_MODEL='gemini-3.5-flash-lite'`. Add your key as **GEMINI_API_KEY** in Colab Secrets (key icon), then enable notebook access. Do not paste it in chat or notebook cells. Check your project's free-tier eligibility, quota and billing in [AI Studio](https://aistudio.google.com/rate-limit); free access is limited, and a paid project may incur charges. Google documents free-tier content use for product improvement, so use properly anonymized, permitted documents. No GPU is needed for the hosted screener.
   For local Llama-3-8B, set `BACKEND='llama'`, select a **T4 GPU**, obtain Meta model access, and add `HF_TOKEN`. Alternatively set `BACKEND='openai'` and add `OPENAI_API_KEY` for GPT-4o-mini.
3. Upload 4–8 anonymized, consented or licensed real PDF résumés for an initial run. The workflow supports text-extractable English PDFs, not scanned-only documents. Include both qualified and unqualified examples.
4. Fill the downloaded label template against the job rubric, then upload it as `labels.csv`. Do this before looking at LLM responses.
5. Run the protocol, PDF construction, OCR/classifier preparation, and call-count preflight cells; then run the actual experiment. Five configurations are evaluated: naive, channel separation, visibility-only, classifier-only, and combined.
6. Gemini runs in batches of at most 20 new calls by default, with pacing. This is a notebook setting, not a guarantee of Google's daily allowance. A batch/quota pause downloads a checkpoint; rerun the same experiment cell when quota permits. Completed calls are reused. To resume in a fresh runtime, upload the checkpoint through the Files pane and set `RESTORE_ZIP` before setup. Do not switch models mid-experiment; use a new run ID. Finish every batch before computing metrics.
7. Use the inline PDF demo and download the results ZIP before Colab disconnects. Save the executed notebook too. Report results under the actual Gemini model ID; they are not GPT/Llama results.

**Training is optional.** The main notebook defaults to an existing DeBERTa injection detector. To train your own classifier, run the T4 notebook first, download `resume_classifier.zip`, upload it to the main notebook's runtime, and set `CLASSIFIER_ZIP`. The default public training dataset is generic injection text, not a résumé dataset; final PDF evaluation is still necessary.

**Status:** notebook implementation and offline checks are provided. GPU training, Colab OCR integration and real-LLM evaluation have not been executed here. The M4–M6 paper/results for this restored extension remain pending your real run. The current paper and original M3–M6 notebooks document only the earlier pilot. Do not submit their numbers as real-PDF/LLM results.

Controlled attacks are placed on an identical supplementary page added to each PDF, with white, invisible or visible text. Hidden copies must render identically to the clean control while retaining extractable payloads. This is a limited placement experiment—not naturally occurring attacks or a full implementation of every concealment technique. Human labels, model errors, clean utility, detector errors and correlated variants are explicitly accounted for.

## Earlier pilot: lightweight text detection

A small CPU-only NLP course project anchored in **Jay Barach, ResumeShield**, [arXiv:2609.20188v1](https://arxiv.org/html/2609.20188v1).

**Current result:** a normalized character TF-IDF + logistic-regression detector achieved **93.75% recall**, **83.33% precision**, **88.24% F1**, and **18.75% false-positive rate** on the frozen synthetic test set. The prior hypothesis required FPR ≤10%, so it was **not supported**. This is an honest negative result, not an unfinished experiment.

## Start here

| Deliverable | File |
|---|---|
| M1 literature survey, with review corrections | [M1](M1_Literature_Survey.ipynb) |
| M2 partial reconstruction, with interpretation corrections | [M2](M2_Reproduction.ipynb) |
| Freshly executed M2 and asserted benchmark counts | [Verified M2](results/M2_verified.ipynb), [values](results/M2_validation.json) |
| M3 gap, one hypothesis, prediction, plan | [M3 notebook](M3_Research_Gap.ipynb), [frozen protocol](M3_Protocol.json) |
| M4 baseline, experiment, metrics, prediction comparison, ablations | [M4 notebook](M4_Controlled_Experiments.ipynb) |
| M5 actual failures, explanations, limitations, validity threats | [M5 notebook](M5_Failure_Analysis.ipynb) |
| M6 IEEE-style paper | [PDF](paper/M6_Final_Paper.pdf), [IEEEtran source](paper/M6_Final_Paper.tex) |
| Oral defence outline, questions, contribution attribution | [M6 defence](M6_Defence.ipynb) |

## Review of M1 and M2

M1 has the right survey structure and an identifiable anchor. Review notes correct the claim that PhantomLint has no public code, clarify citation/version details, and supersede the larger PDF/LLM proposal.

M2 executes successfully, but is a **partial reconstruction**, not an execution of the original paper's artifact. Its documents, filters and narrower success definition differ. The fence disables instruction obedience by construction. Visibility comes from synthetic flags; the historical zero-width flag is explicitly identified as unrealistic. The simulation cannot establish real-LLM safety or qualified-candidate utility. Review notes are added near the top of both notebooks; older discussion is preserved as historical context.

The attachments supplied in chat were a short opening-cell excerpt and an empty output array, rather than a full notebook execution. Review therefore used the complete workspace notebooks. The read-only attachments were not changed.

## Simple extension

The new contribution is **text detection**, using standard TF-IDF and logistic regression. It uses 80 authored sentences: 32 train, 16 validation, 32 test, with disjoint families. Original, zero-width and homoglyph variants expand these to 96/48/96 rows. Variants stay with their source, and are not independent examples. All text is synthetic.

The primary comparison uses fixed keyword matching versus normalized character features. Ablations remove normalization, switch to word features, and apply normalization to keywords alone. Thresholds are selected on validation only. The test outcome was retained without tuning afterward.

This does not implement PDF extraction, train an LLM, measure real hiring decisions, or demonstrate a reduction in live-model attack success. Those are future extensions. The new experiment's detection metrics must not be compared directly with M2's ASR.

## Run

### Live presentation demo

After installing the dependencies below, run:

```bash
.venv/bin/python demo.py
```

Open **http://127.0.0.1:8501**. The demo uses the existing trained-method configuration; it does not yet implement the proposed sentence-transformer alternative. It recreates the model from training data only and uses the recorded threshold without changing research results. No additional dependencies are needed.

For a three-minute demonstration:

1. Choose **Ordinary experience** and click **Analyze text** to explain the input/output flow.
2. Choose **A directive without known keywords** and compare the keyword and model flags.
3. Choose the **zero-width** example and expand its normalized text to show preprocessing.
4. Show **Known false positive** and **Known miss** to explain actual limitations.
5. Paste or edit a sentence, one sentence per line, and analyze again. These live inputs are demonstrations, not additional controlled-test evidence.

The model score is not a calibrated risk probability. The demo does not rank candidates or process PDFs. Long documents and line-level aggregation are not validated by the small sentence benchmark. Stop the server with Ctrl+C.

Implementation: [demo.py](demo.py), [interface](demo.html). Validate prediction fidelity with `python -m unittest test_demo -v`.

### Research experiments

Python 3.14.0 was used for the recorded run. Create a virtual environment, then:

```bash
python -m pip install -r requirements-experiments.txt
python experiment.py
python build_deliverables.py
python verify_results.py
python verify_notebooks.py M2_Reproduction.ipynb M3_Research_Gap.ipynb M4_Controlled_Experiments.ipynb M5_Failure_Analysis.ipynb
```

In VS Code, select the local virtual environment to open/run the notebooks. Run from this repository's root. The experiment needs no credentials, GPU, API calls or dataset download. Installation requires internet access.

[experiment.py](experiment.py) trains and exports results. [build_deliverables.py](build_deliverables.py) renders reports from saved results without retraining. Rebuilding notebooks clears their cell outputs; execute them afterward. Re-running the experiment overwrites the local result files; preserve them before modifying the protocol or data. The original prediction file is hashed in the run record. This is a local pre-experiment record, not independent preregistration.

## Evidence

- [Authored sentences and fixed splits](data/sentences.json)
- [Expanded dataset](results/dataset.csv)
- [Every test prediction and model score](results/predictions.csv)
- [Metrics, thresholds, validation scores, versions and input hashes](results/run.json)
- [Exact installed environment](results/environment.txt)

The PDF is a two-column IEEE-style reading copy. The official IEEEtran LaTeX source is also supplied, but no LaTeX compiler was available locally. Add your name, institution and course details before submission. Check any course-specific page limit and rehearse the oral defence yourself.
