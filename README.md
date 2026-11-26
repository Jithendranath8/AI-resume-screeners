# Attacking and defending AI resume screeners

An NLP course project (milestones M1 to M6) on indirect prompt injection in resume screening: text hidden in a PDF
that a recruiter never sees but an AI screener reads and obeys.

![The foot of a real resume page as a person sees it, and as a program reads it](figures/overview_two_readings.png)

**What it shows, on a small sample.** A real LLM screener (Llama 3.2 3B Instruct) was steered by text placed in real
resume PDFs in 21 of 24 attacks. Two inexpensive checks in front of the model, a visibility-diff and an injection
classifier, cut that to 3 of 24 without changing any decision on a clean resume. The prediction was recorded before
the first model call and was met. The three attacks that survive are all visible, politely worded instructions.

| Attack success by configuration | Which attacks got through |
|---|---|
| ![Attack success rate](figures/M4_attack_success.png) | ![Ablation](figures/M4_ablation.png) |

The method is also packaged as a working tool, the **resume injection scanner**: any PDF in; a verdict, the attack
type and its place on the page out.

## Start here

| Deliverable | Where |
|---|---|
| M1 Literature survey: five papers, anchor choice | [M1_Literature_Survey.ipynb](M1_Literature_Survey.ipynb) |
| M2 Reproduction of the anchor paper (partial reconstruction) | [M2_Reproduction.ipynb](M2_Reproduction.ipynb), [verified run](results/m2_reproduction/) |
| M3 Research gap, hypothesis, prediction, plan | [M3_Research_Gap.ipynb](M3_Research_Gap.ipynb), [frozen protocol](results/real_pdf/pdf_llama32_3b_01/protocol.json) |
| M4 Controlled experiments and ablations | [M4_Controlled_Experiments.ipynb](M4_Controlled_Experiments.ipynb) |
| M5 Failure analysis, limitations, validity | [M5_Failure_Analysis.ipynb](M5_Failure_Analysis.ipynb) |
| M6 Paper (IEEE conference format, five pages, seven result figures) | [paper/M6_Final_Paper.pdf](paper/M6_Final_Paper.pdf), [LaTeX](paper/M6_Final_Paper.tex) |
| M6 Defence outline, questions, attribution | [M6_Defence.ipynb](M6_Defence.ipynb) |
| Slides, 14 pages | [presentation/AI_Resume_Screeners_Slides.pdf](presentation/AI_Resume_Screeners_Slides.pdf) |
| Result figures, one per notebook result, named by milestone | [figures/](figures/) |
| Resume injection scanner | [scanner/](scanner/), web pages in [web/](web/) |

The anchor paper is J. Barach, *ResumeShield*, [arXiv:2609.20188](https://arxiv.org/abs/2609.20188).

## Repository layout

```
M1 … M6 notebooks      the six milestone deliverables
scanner/               the tool: command line, web server, held-out evaluation, tests
web/                   the two web pages (React); the built copy in web/dist is served by the scanner
experiment/            real-PDF experiment run locally, and classifier fine-tuning
colab/                 the same experiment as Colab notebooks, with its workflow source
report/                generators: notebooks M3 to M6, the paper, the figures
figures/               result figures (PNG and PDF), drawn from the recorded results
paper/                 M6 paper, compiled PDF and IEEEtran source
presentation/          slide deck, PDF and HTML source
pilot/                 the earlier sentence-level pilot, self-contained (code, data, results, demo)
data/                  six real resume PDFs with labels and provenance; classifier training text
results/               recorded outputs: M2 reproduction, real-PDF run, classifier training, scanner evaluation
```

## Run the scanner

```bash
python -m pip install -r requirements.txt        # and Tesseract OCR: brew install tesseract
python scanner/resume_scanner.py resume.pdf                                   # one file
python scanner/resume_scanner.py inbox/ --csv results.csv --json reports.json # a folder
python scanner/scanner_app.py                                                 # web pages at http://127.0.0.1:8502
```

Each file gets one of three verdicts: **attack found**, **needs review** or **no attack found**. The command exits
with 2, 1 or 0 for those, and 3 if a file could not be read, so it can sit in front of a screening pipeline.
Scanning is offline and takes a few seconds per file.

How it decides: every page is rendered at 200 dpi and read by OCR, and each extracted text block is compared with
what OCR saw in the same place. That is the experiment's visibility rule, and `scanner/evaluate_scanner.py regression`
confirms the scanner gives the same flag and score as the recorded run on all 652 text blocks of its 42 documents.
Two classifiers then score every piece of text, including metadata, comments and text inside images: the pretrained
`protectai/deberta-v3-base-prompt-injection-v2` and a DistilBERT fine-tuned in this project. For hidden text the
scanner names the method: same colour as the background, invisible text mode, transparency, microscopic font,
outside the page, covered by a shape, or a switched-off layer.

The fine-tuned weights are not stored in the repository. Create them once, in about 11 minutes on an M2 laptop:
`python experiment/train_classifier_local.py --sources deepset,resume,blocks --out results/classifier_training_02`.
Without them the scanner still runs, says so, and the measured results below do not apply.

The web pages are a Home page that explains the attacks, the method, the results and the limits, and a Scan page
that takes one file or a batch, shows each verdict, and marks the text on the rendered page. Two extras are off
until chosen: a test attack added to a file before scanning, and an AI screener comparison that scores the resume
with and without the flagged text (this one can send text to a model provider). A result can be opened by link,
for example `/scan?sample=r03_software_engineering_analyst_32985311.pdf&attack=white_text`. Design notes are in
[web/DESIGN.md](web/DESIGN.md).

## Results

### Real PDFs and a real LLM (M3 to M5)

Six real resume PDFs (Kaggle Resume Dataset, CC0), 42 controlled documents, five configurations, 210 valid responses
from Llama 3.2 3B Instruct served locally by Ollama. Recorded outputs: `results/real_pdf/pdf_llama32_3b_01/`.

| Configuration | Successful attacks | Attack success rate | Clean decisions kept |
|---|---|---|---|
| No added defence | 21/24 | 87.5% | 100% |
| Channel separation | 9/24 | 37.5% | 83.3% |
| Visibility-diff only | 7/24 | 29.2% | 100% |
| Classifier only | 9/24 | 37.5% | 100% |
| Both checks combined | 3/24 | 12.5% | 100% |

Predicted before the run: a reduction of at least 20 points with at least 95% clean agreement. Actual: 75.0 points
and 100%. An exploratory fine-tuned classifier catches the paraphrase the pretrained one misses; its effect on
attack success has **not** been measured.

### The scanner on resumes it had not seen

![Scanner on held-out resumes](figures/scanner_heldout.png)

| Held-out run | Rules | Clean files judged clean | Hidden text found | Instruction not hidden: called an attack | The same: attack or review |
|---|---|---|---|---|---|
| `test` | 1.0 | 123 of 136 | 300 of 300 | 150 of 150, wording used in development | 150 of 150 |
| `final` | 1.1 | 148 of 160 | 210 of 210 | 57 of 90, new wording | 84 of 90 |
| `confirm` | 1.2, current | 139 of 140 | 140 of 140 | 39 of 60, new wording | 55 of 60 |

Hidden text is the reliable part: across the two runs with new wording, all 350 hidden passages were found, whatever
they said. A plainly visible instruction in unusual wording is the weak part. The rules were changed after the first
two runs, so each version was measured again on fresh files. Summaries are in `results/scanner_eval/`.

### Anchor reproduction (M2) and the earlier pilot

| M2: reproduced against the paper | Pilot: recall and false positives |
|---|---|
| ![M2 reproduction](figures/M2_attack_success.png) | ![Pilot](figures/pilot_detection.png) |

M2 rebuilds the ResumeShield benchmark locally (104 documents). It is a partial reconstruction: two filter-only
rows differ from the paper, and the zero for channel separation is a property of the simulator, not evidence about
a real model. The pilot, a TF-IDF sentence detector, reached 93.75% recall but an 18.75% false-positive rate against
a 10% limit, so its hypothesis was **not supported**. It is kept as a negative result in [pilot/](pilot/).

## Reproduce

```bash
python report/make_figures.py                 # figures/ from the recorded results
python report/build_deliverables.py           # M3 to M6 notebooks and the paper (compiles LaTeX with Tectonic)
python report/verify_notebooks.py M3_Research_Gap.ipynb M4_Controlled_Experiments.ipynb M5_Failure_Analysis.ipynb
python -m unittest discover -s scanner        # scanner tests
python colab/test_workflow.py                 # experiment pipeline checks
python scanner/evaluate_scanner.py regression # the scanner against the recorded experiment
cd pilot && python verify_results.py          # the pilot's recorded hashes and metrics
```

None of these calls an LLM or trains anything. Rebuilding the notebooks clears their outputs, so run
`verify_notebooks.py` afterwards. The paper needs [Tectonic](https://tectonic-typesetting.github.io)
(`brew install tectonic`); without it a reading copy is written with ReportLab.

To repeat the experiment itself, which makes 210 model calls and needs a new run ID:

```bash
ollama pull llama3.2:3b-instruct-q4_K_M && ollama serve
COMPAT_API_KEY=ollama python experiment/run_real_pdf_local.py --set BACKEND=compatible --set RUN_ID=<new id> \
  --set COMPAT_BASE_URL=http://127.0.0.1:11434/v1 --set COMPAT_MODEL=llama3.2:3b-instruct-q4_K_M \
  --set API_MAX_OUTPUT_TOKENS=250 --set API_PAUSE_SECONDS=0
```

The same workflow runs on Google Colab with a hosted model: upload
[colab/Real_PDF_LLM_Defence_Colab.ipynb](colab/Real_PDF_LLM_Defence_Colab.ipynb), add the model key as a Colab secret,
upload four to eight permitted resume PDFs and their labels, and run the cells in order.
[colab/Train_Transformer_Classifier_T4.ipynb](colab/Train_Transformer_Classifier_T4.ipynb) trains the classifier on a T4 GPU.

After changing anything under `web/src`, or after `python scanner/build_home_specimens.py`, rebuild the pages with
`cd web && npm install && npm run build`.

## Limits

- Six resumes from one dataset, four of them eligible; the variants of a resume are correlated. No significance claim.
- A 3-billion-parameter, 4-bit model. Larger screeners may behave differently.
- Two fixed payload wordings on an appended page. Not naturally occurring attacks, and not adaptive ones.
- One annotator. Hosted models could not be used without payment; the two unused attempts are noted in
  `results/real_pdf/unused_attempts/`.
- Instructions are recognised in English only, although hidden text is found in any language.
- A finding is a reason for a person to look at a file, not proof of intent. A flag should never reject a candidate automatically.

## Authors

Akula Jithendranath (230120), Kolli Charan Adithya (230075) and Kammati Adithya (230145).
Department of Computer Science and Artificial Intelligence, Newton School of Technology, Rishihood University.

The names are set once for the paper and the M6 notebook in [report/paper_sections.py](report/paper_sections.py).
