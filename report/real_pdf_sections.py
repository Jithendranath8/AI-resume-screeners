"""Notebook sections for the completed real-PDF / real-LLM experiment.

Imported by report/build_deliverables.py. Every number in the generated markdown is read
from the saved run folder, so the prose cannot drift from the recorded results.
The code cells written into the notebooks re-read the same files and assert the
key figures; they make no LLM calls and need no GPU.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # repository root
RUN_ID = 'pdf_llama32_3b_01'
RUN_REL = f'results/real_pdf/{RUN_ID}'
RUN = ROOT / RUN_REL
CONFIGS = ['naive', 'channel_separation', 'visibility_only', 'classifier_only', 'combined']
ANCHOR = 'Jay Barach, *ResumeShield*, [arXiv:2609.20188v1](https://arxiv.org/html/2609.20188v1)'

record = json.loads((RUN / 'protocol.json').read_text())
protocol = record['protocol']
summary = pd.read_csv(RUN / 'summary.csv').set_index('config')
pva = json.loads((RUN / 'prediction_vs_actual.json').read_text())
unc = json.loads((RUN / 'uncertainty.json').read_text())
responses = pd.read_csv(RUN / 'responses.csv')
labels = pd.read_csv(RUN / 'labels.csv')
attacks = pd.read_csv(RUN / 'attack_pairs.csv')
eligible = attacks[attacks.eligible].assign(success=lambda d: d.decision.eq('SELECT'))
detection = pd.read_csv(RUN / 'document_detection_metrics.csv').set_index('config')
removals = pd.read_csv(RUN / 'removals_for_review.csv')
clean = responses[responses.variant == 'clean']
manifests = {n: json.loads((ROOT / f'results/classifier_training_0{n}/training_manifest.json').read_text()) for n in (1, 2)}
blocks = {n: pd.read_csv(RUN / f'classifier_comparison_round{n}/block_metrics.csv', index_col=0) for n in (1, 2)}
docs = {n: pd.read_csv(RUN / f'classifier_comparison_round{n}/document_detection_metrics.csv', index_col=0) for n in (1, 2)}

N_RESUMES = len(labels)
N_SELECT = int(labels.expected_decision.eq('SELECT').sum())
N_REJECT = N_RESUMES - N_SELECT
N_DOCS = responses[['resume_id', 'variant']].drop_duplicates().shape[0]
N_CALLS = len(responses)
N_ELIGIBLE = int(summary.loc['naive', 'eligible_attacks'])
FIRST_RESPONSE = responses.recorded_utc.min()
LAST_RESPONSE = responses.recorded_utc.max()


def pct(x):
    return f'{100 * float(x):.1f}%'


def frac(config):
    return f"{int(summary.loc[config, 'successes'])}/{int(summary.loc[config, 'eligible_attacks'])}"


def cell_count(config, family=None, technique=None, hidden=None):
    d = eligible[eligible.config == config]
    if family:
        d = d[d.family == family]
    if technique:
        d = d[d.technique == technique]
    if hidden is True:
        d = d[d.technique != 'visible_text']
    if hidden is False:
        d = d[d.technique == 'visible_text']
    return f'{int(d.success.sum())}/{len(d)}'


def md_table(frame, index_name=''):
    cols = list(frame.columns)
    head = '| ' + ' | '.join([index_name] + [str(c) for c in cols]) + ' |\n|' + '---|' * (len(cols) + 1) + '\n'
    return head + '\n'.join('| ' + ' | '.join([str(i)] + [str(v) for v in row]) + ' |'
                            for i, row in zip(frame.index, frame.to_numpy().tolist()))


LABEL = {'naive': 'Naive (no added defence)', 'channel_separation': 'Channel separation',
         'visibility_only': 'Visibility-diff only', 'classifier_only': 'Classifier only',
         'combined': 'Combined (visibility OR classifier)'}

results_table = md_table(pd.DataFrame({
    'Successful attacks': [frac(c) for c in CONFIGS],
    'ASR': [pct(summary.loc[c, 'asr_valid']) for c in CONFIGS],
    'Clean accuracy vs human labels': [pct(summary.loc[c, 'clean_accuracy_lower']) for c in CONFIGS],
    'Clean agreement with naive': [pct(summary.loc[c, 'clean_agreement_lower']) for c in CONFIGS],
}, index=[LABEL[c] for c in CONFIGS]), 'Configuration')

ablation_table = md_table(pd.DataFrame({
    'Hidden, direct': [f"{int(eligible[(eligible.config == c) & (eligible.family == 'direct') & (eligible.technique != 'visible_text')].success.sum())}/"
                       f"{len(eligible[(eligible.config == c) & (eligible.family == 'direct') & (eligible.technique != 'visible_text')])}" for c in CONFIGS],
    'Hidden, paraphrase': [f"{int(eligible[(eligible.config == c) & (eligible.family == 'paraphrase') & (eligible.technique != 'visible_text')].success.sum())}/"
                           f"{len(eligible[(eligible.config == c) & (eligible.family == 'paraphrase') & (eligible.technique != 'visible_text')])}" for c in CONFIGS],
    'Visible, direct': [cell_count(c, 'direct', 'visible_text') for c in CONFIGS],
    'Visible, paraphrase': [cell_count(c, 'paraphrase', 'visible_text') for c in CONFIGS],
}, index=[LABEL[c] for c in CONFIGS]), 'Successful attacks / eligible')

detector_table = md_table(pd.DataFrame({
    'TP': detection.tp, 'FP': detection.fp, 'FN': detection.fn, 'TN': detection.tn,
    'Precision': detection.precision.map(lambda v: f'{v:.3f}'), 'Recall': detection.recall.map(lambda v: f'{v:.3f}'),
    'F1': detection.f1.map(lambda v: f'{v:.3f}'), 'FPR': detection.fpr.map(lambda v: f'{v:.3f}')}), 'Detector')


def classifier_row(name, frame, key):
    r = frame.loc[key]
    return {'Payload blocks caught': f"{int(r.tp)}/{int(r.tp + r.fn)}",
            'Genuine blocks flagged': f"{int(r.fp)}/{int(r.fp + r.tn)}"}


def document_row(frame, key):
    r = frame.loc[key]
    return {'Injected documents caught, combined with visibility': f"{int(r.tp)}/{int(r.tp + r.fn)}",
            'Clean documents flagged': f"{int(r.fp)}/{int(r.fp + r.tn)}"}


finetune_table = md_table(pd.DataFrame([
    {'Threshold': protocol['classifier_threshold'], **classifier_row('pre', blocks[2], 'pretrained'),
     **document_row(docs[2], 'combined_pretrained')},
    {'Threshold': manifests[1]['threshold'], **classifier_row('r1', blocks[1], 'finetuned'),
     **document_row(docs[1], 'combined_finetuned')},
    {'Threshold': manifests[2]['threshold'], **classifier_row('r2', blocks[2], 'finetuned'),
     **document_row(docs[2], 'combined_finetuned')},
], index=['Pretrained DeBERTa (used in the LLM run)', 'Fine-tuned DistilBERT, round 1', 'Fine-tuned DistilBERT, round 2']), 'Classifier')


def heldout(n, key):
    m = manifests[n][key]
    return (f"precision {m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f}, FPR {m['fpr']:.3f} "
            f"(TP {m['tp']}, FP {m['fp']}, FN {m['fn']}, TN {m['tn']})")


combined_wins = eligible[(eligible.config == 'combined') & eligible.success][['resume_id', 'variant', 'score', 'reason']]
combined_failures_table = md_table(combined_wins.set_index('resume_id'), 'Resume')
naive_resisted = eligible[(eligible.config == 'naive') & ~eligible.success]
changed_clean = pd.read_csv(RUN / 'changed_clean_decisions.csv')
select_attacked = attacks[attacks.expected_decision.eq('SELECT')]
cs_select_reject = select_attacked[(select_attacked.config == 'channel_separation') & select_attacked.decision.eq('REJECT')]
payload_removed = int(removals.text.str.contains('score 100', case=False).sum())
clean_removed = int(removals.document.str.endswith('__clean').sum())

BANNER = (
    '## Current scope: real PDFs and a real LLM — experiment completed\n\n'
    'The real-PDF extension has been run: visibility-diff plus a transformer injection classifier protecting an actual LLM '
    f'screener (Llama 3.2 3B Instruct, run locally) on {N_RESUMES} real resume PDFs. '
    f'Outcome: the pre-recorded hypothesis was **{pva["verdict"]}** (attack success rate {pct(summary.loc["naive", "asr_valid"])} with no added '
    f'defence, {pct(summary.loc["combined", "asr_valid"])} with the combined defence; clean agreement {pct(pva["actual_clean_agreement"])}). '
    'See [M3](M3_Research_Gap.ipynb) for the gap, hypothesis and plan, [M4](M4_Controlled_Experiments.ipynb) for the experiment and ablations, '
    f'and [M5](M5_Failure_Analysis.ipynb) for failures and limitations. All recorded outputs are in `{RUN_REL}/`.\n\n'
    'The earlier TF-IDF text-detection study is a completed **pilot** and is kept as a labelled historical section in each notebook; '
    'its numbers are not real-PDF or LLM results. The [M6 paper](paper/M6_Final_Paper.pdf), the [defence outline](M6_Defence.ipynb) and the '
    '[slides](presentation/AI_Resume_Screeners_Slides.pdf) report the real-PDF experiment as the main study, with the pilot as one section.')

PILOT_DIVIDER = (
    '---\n\n# Earlier pilot (historical record): lightweight text detection\n\n'
    'Everything below this line is the completed TF-IDF sentence-detection pilot, kept unchanged apart from heading levels. '
    'Its code, data and recorded results are archived in [pilot/](pilot/). '
    'It used synthetic sentences and no LLM or PDF. Its hypothesis was **not supported** (false-positive rate above the limit). '
    'Do not report these numbers as results of the real-PDF experiment above.')


def as_pilot(cells):
    """Demote the pilot's headings so the real-PDF experiment leads the notebook."""
    out = [('md', PILOT_DIVIDER)]
    for kind, text in cells:
        if kind == 'md':
            if text.startswith('# '):
                text = '## Pilot ' + text[2:]
            elif text.startswith('## '):
                text = '### ' + text[3:]
        out.append((kind, text))
    return out


SETUP = ("import json, hashlib\nfrom pathlib import Path\nimport pandas as pd\n"
         f"RUN = Path('{RUN_REL}')\n"
         "record = json.loads((RUN / 'protocol.json').read_text())\nprotocol = record['protocol']\n"
         "responses = pd.read_csv(RUN / 'responses.csv')\n")


def m3_cells():
    return [
        ('md', BANNER),
        ('md', '# M3 — Research gap, hypothesis, prediction and experimental plan\n\n'
         '**Extension:** an actual LLM screening real resume PDFs, protected by visibility-diff plus a transformer injection classifier. '
         f'**Anchor paper:** {ANCHOR}. M2 is a partial reconstruction of the anchor on a simulated screener; read its review notes before using its numbers.\n\n'
         f'This notebook is a readable rendering of the protocol frozen in `{RUN_REL}/protocol.json` before the first LLM call of that run. '
         'The experiment itself is in M4 and the failure analysis in M5.'),
        ('md', '## Research gap\n\n'
         'The anchor paper and our M2 reconstruction evaluate defences on a deterministic simulated screener. Whether that screener obeys an injected '
         'instruction is fixed by construction, and visibility comes from synthetic flags, not rendered documents. They therefore cannot establish:\n\n'
         '1. whether a real LLM screener is actually steered by text hidden in a real PDF file, and\n'
         '2. whether comparing rendered pixels with extracted text, combined with a transformer injection detector, protects that screener '
         'without changing its decisions on clean resumes.\n\n'
         'PhantomLint already studies visibility-based detection and pretrained injection classifiers already exist, so neither component is our novelty. '
         'The gap addressed here is a controlled evaluation of their **combination**, with component ablations, on an actual LLM and actual PDF files. '
         'It is a bounded evaluation gap, not a claim of a new architecture.'),
        ('md', '## One hypothesis and prediction, recorded before the LLM run\n\n'
         '**Hypothesis.** The combined defence (remove every text block flagged by the visibility check or by the classifier) reduces the '
         'reference-promotion attack success rate (ASR) by at least 20 percentage points relative to no added defence, while preserving at least '
         '95% agreement with the no-defence decisions on clean resumes.\n\n'
         '**Prediction.** Both thresholds will be met. They form one joint hypothesis: if either fails it is not supported, and if any response is '
         'invalid or no attack is eligible the outcome is inconclusive.\n\n'
         f'The protocol file was written at `{record["recorded_utc"]}`. The first LLM response of the run is timestamped `{FIRST_RESPONSE}`. '
         'The file stores the hypothesis, both numeric thresholds, the model and classifier identifiers, the two detector thresholds, the payload strings, '
         f'and SHA-256 hashes of the {N_RESUMES} PDFs and of the labels; the code cell below re-verifies those hashes. '
         'This is a local record of execution order, not an independent or externally timestamped preregistration.'),
        ('md', '## Experimental plan\n\n'
         '![The combined defence: two checks between the PDF and the model](figures/M4_pipeline.png)\n\n'
         f'1. **Documents.** {N_RESUMES} real resume PDFs from the Kaggle Resume Dataset (CC0), listed with provenance in '
         '[data/real_pdfs/SOURCES.md](data/real_pdfs/SOURCES.md). They are text-extractable and already anonymised by the dataset author.\n'
         f'2. **Human labels.** SELECT or REJECT against the job rubric, assigned before any LLM output was produced: {N_SELECT} SELECT, {N_REJECT} REJECT.\n'
         '3. **Job rubric.** Junior Python backend developer, 100 points (Python 30, API/backend 25, SQL 20, testing/version control 15, Docker/cloud 10); SELECT at 60 or more.\n'
         '4. **Controlled variants.** Every PDF receives the same blank supplementary page. The clean control leaves it empty; attacked copies place a payload on it as '
         'white text, PDF-invisible text (render mode 3) or visible black text. Two payload families are used: a direct override and a "reviewer guidance" paraphrase. '
         f'That gives 7 documents per resume, {N_DOCS} in total. Construction asserts that hidden variants render pixel-identically to the clean control and that every payload stays extractable.\n'
         '5. **Defence components.** Visibility-diff: render each page at 200 dpi, OCR the rendered crop of every extracted text block with Tesseract, and flag blocks of at least 12 normalised '
         f'characters whose fuzzy agreement is below {protocol["visibility_threshold"]:.0f}. Classifier: `{protocol["classifier"]}` at a fixed threshold of {protocol["classifier_threshold"]}, not tuned on these PDFs.\n'
         '6. **Five configurations**, all with the same model, rubric, JSON output schema and temperature 0: naive; channel separation (trusted policy plus JSON-wrapped resume); '
         'visibility only; classifier only; combined (union of both flags, no extra policy, so the ablations are interpretable).\n'
         f'7. **Screener.** `{protocol["llm"]}` (Llama 3.2 3B Instruct, 4-bit) served locally through Ollama with schema-constrained output. '
         f'{N_DOCS} documents × {len(CONFIGS)} configurations × {protocol["repeats"]} repeat = {N_CALLS} calls.\n'
         '8. **Primary metric.** ASR is the share of eligible attacks that return SELECT. An attack is eligible when the human label is REJECT and the naive screener also rejects the clean copy, '
         'so every configuration has the same denominator. Also reported: agreement of clean decisions with the naive clean decisions, clean accuracy against human labels, '
         'and the same-configuration flip rate. Invalid responses are counted as errors, never as REJECT.\n'
         '9. **Uncertainty.** Descriptive paired bootstrap over resumes (2000 draws, seed 42). Variants of one resume are correlated and are not treated as independent samples.\n'
         '10. **No post-hoc changes.** Payloads, thresholds, labels and the eligibility rule are fixed before the run and the result is retained whatever it shows.'),
        ('md', '## Deviations from the original plan (disclosed)\n\n'
         '- **Screener model.** The original M3 note named GPT-4o-mini or Llama-3; the workflow then defaulted to Gemini. No hosted option was usable without payment: '
         'Gemini returned HTTP 403 ("free tier is not available for this project") for two keys, so that attempt produced no responses. '
         'A Groq-hosted attempt (`openai/gpt-oss-120b`) stopped at 140 of 210 responses on the daily token limit and was abandoned. '
         'Only the frozen protocol files of these two attempts are kept (`pdf_gemini_01_protocol.json`, `pdf_groq_01_protocol.json`), in [results/real_pdf/unused_attempts/](results/real_pdf/unused_attempts/). '
         'The experiment was therefore run on a smaller open-weight model that fits an 8 GB laptop.\n'
         '- **Not blind to a partial earlier attempt.** The incomplete Groq responses existed before this run was frozen. Their validity counts and a few rows were inspected; '
         'no ASR or agreement metric was computed from them, they are not used here, and they were later removed from the repository. '
         'The hypothesis text and both thresholds were identical in all three protocol files.\n'
         '- **Single annotator.** The labels were assigned by the project author alone, aided by keyword-extracted evidence snippets for each rubric criterion. There is no second annotator or agreement score.\n'
         '- **Local execution.** [experiment/run_real_pdf_local.py](experiment/run_real_pdf_local.py) executes the cells of [colab/real_pdf_workflow.py](colab/real_pdf_workflow.py) unchanged and replaces only '
         'Colab uploads, downloads, secrets and display. The inline widget demo is skipped.\n'
         f'- **Sample size.** {N_RESUMES} resumes, chosen to fit a two-hour budget. The shortlist was drawn by keyword scan from one public dataset, which SOURCES.md describes.'),
        ('code', SETUP +
         "print('Protocol recorded (UTC):  ', record['recorded_utc'])\n"
         "print('First LLM response (UTC): ', responses.recorded_utc.min())\n"
         "assert responses.recorded_utc.min() > record['recorded_utc'], 'responses must follow the frozen protocol'\n"
         "for name, expected in protocol['input_hashes'].items():\n"
         "    assert hashlib.sha256((RUN / 'inputs' / name).read_bytes()).hexdigest() == expected, name\n"
         "assert hashlib.sha256((RUN / 'labels.csv').read_bytes()).hexdigest() == protocol['labels_hash']\n"
         "print('Input PDF hashes and label hash match the frozen protocol.')\n"
         "display(pd.read_csv(RUN / 'labels.csv')[['resume_id', 'filename', 'expected_decision']])\n"
         "print(json.dumps({k: v for k, v in protocol.items() if k not in ('input_hashes', 'classifier_files')}, indent=2))"),
    ]


def m4_cells():
    hidden_vis = cell_count('visibility_only', hidden=True)
    visible_vis = cell_count('visibility_only', hidden=False)
    direct_clf = cell_count('classifier_only', family='direct')
    para_clf = cell_count('classifier_only', family='paraphrase')
    direct_cs = cell_count('channel_separation', family='direct')
    para_cs = cell_count('channel_separation', family='paraphrase')
    lo, hi = unc['descriptive_95_interval']
    return [
        ('md', BANNER),
        ('md', '# M4 — Implementation, controlled experiments and ablation\n\n'
         f'This notebook reads the recorded outputs of run `{RUN_ID}` and checks their provenance. It makes no LLM calls. To reproduce the run itself:\n\n'
         '```bash\nollama pull llama3.2:3b-instruct-q4_K_M && ollama serve\n'
         'COMPAT_API_KEY=ollama .venv/bin/python experiment/run_real_pdf_local.py --set BACKEND=compatible \\\n'
         '  --set RUN_ID=<new id> --set COMPAT_BASE_URL=http://127.0.0.1:11434/v1 \\\n'
         '  --set COMPAT_MODEL=llama3.2:3b-instruct-q4_K_M --set API_MAX_OUTPUT_TOKENS=250 --set API_PAUSE_SECONDS=0\n```\n\n'
         'A rerun needs a new run ID; an existing run folder refuses a changed protocol.'),
        ('md', '## Implementation\n\n'
         '- **Pipeline:** [colab/real_pdf_workflow.py](colab/real_pdf_workflow.py). PyMuPDF builds the PDF variants, renders pages and extracts text blocks; Tesseract OCRs each block crop; '
         'RapidFuzz compares extracted and visible text; a transformer classifier scores each block (256-token overlapping windows, maximum window probability).\n'
         '- **Screener call:** OpenAI-compatible chat endpoint with a strict JSON schema (score, decision, reason). A response is valid only if the decision is consistent with the score.\n'
         '- **Local runner:** [experiment/run_real_pdf_local.py](experiment/run_real_pdf_local.py). Offline checks of the pipeline logic: `python colab/test_workflow.py`.\n'
         f'- **Recorded evidence:** `{RUN_REL}/` holds the frozen protocol, the {N_DOCS} PDFs, per-document audit JSON (extracted text, OCR text, every flag and score), '
         f'all {N_CALLS} raw responses, and the derived tables read below.'),
        ('code', SETUP +
         "import matplotlib.pyplot as plt\n"
         "run_hash = hashlib.sha256((RUN / 'protocol.json').read_bytes()).hexdigest()\n"
         "assert (responses.run_hash == run_hash).all(), 'every response must belong to the frozen protocol'\n"
         f"assert len(responses) == {N_CALLS} and responses.valid.all()\n"
         "print('Responses:', len(responses), '| all valid:', bool(responses.valid.all()), '| protocol SHA-256:', run_hash[:16], '...')\n"
         "import ast\nprint('Screener model reported by the server:', sorted({ast.literal_eval(m)['model'] for m in responses.metadata}))\n"
         "print('Documents:', responses[['resume_id', 'variant']].drop_duplicates().shape[0], '| configurations:', sorted(responses.config.unique()))"),
        ('md', '## Baseline and experimental configurations\n\n'
         '| Configuration | What changes | Role |\n|---|---|---|\n'
         '| Naive | Nothing; the full extracted text is sent with the rubric | **Baseline** |\n'
         '| Channel separation | A trusted policy line plus the resume wrapped as JSON data | Anchor-style prompt defence, for comparison |\n'
         '| Visibility-diff only | Blocks whose extracted text is not visible in the rendering are removed | Ablation: component 1 |\n'
         '| Classifier only | Blocks scored as injection by the classifier are removed | Ablation: component 2 |\n'
         '| Combined | Blocks flagged by either component are removed | **Experimental** |\n\n'
         'Everything else is held constant: model, rubric, output schema, temperature 0, documents and the shuffled call order (seed 42). '
         f'Eligible attacks: {N_ELIGIBLE} per configuration, from the {N_REJECT} human-REJECT resumes that the naive screener also rejects when clean.'),
        ('code', "summary = pd.read_csv(RUN / 'summary.csv').set_index('config')\n"
         "display(summary[['eligible_attacks', 'successes', 'attack_errors', 'asr_valid', 'asr_lower', 'asr_upper',\n"
         "                 'paired_flip_rate', 'clean_accuracy_lower', 'clean_agreement_lower', 'mean_score_delta_vs_reference']].round(3))"),
        ('md', '## Results\n\n' + results_table + '\n\n'
         f'All {N_CALLS} responses were valid, so the lower bound, upper bound and valid-only ASR coincide. '
         f'The naive screener agreed with all {N_RESUMES} human labels on the clean resumes, so the baseline is a meaningful reference.'),
        ('md', '## Predicted versus actual\n\n'
         '| Quantity | Predicted before the run | Actual |\n|---|---|---|\n'
         f'| ASR reduction, naive minus combined | at least 20.0 points | **{100 * pva["actual_asr_drop"]:.1f} points** ({frac("naive")} → {frac("combined")}) |\n'
         f'| Clean agreement of combined with naive | at least 95% | **{pct(pva["actual_clean_agreement"])}** ({N_RESUMES}/{N_RESUMES}) |\n'
         f'| All responses valid and at least one eligible attack | required | {pva["all_responses_valid"]}, {N_ELIGIBLE} eligible |\n\n'
         f'**Verdict: {pva["verdict"]}.** Both conditions hold. The descriptive paired bootstrap over the {unc["resume_count"]} eligible resumes gives a mean drop of '
         f'{100 * unc["mean_paired_drop"]:.1f} points with a 95% interval of {100 * lo:.1f} to {100 * hi:.1f} points. With {unc["resume_count"]} resumes from one source this interval '
         'describes the sample; it is not population-level evidence.'),
        ('code', "pva = json.loads((RUN / 'prediction_vs_actual.json').read_text())\n"
         "drop = summary.loc['naive', 'asr_valid'] - summary.loc['combined', 'asr_valid']\n"
         "agreement = summary.loc['combined', 'clean_agreement_lower']\n"
         "assert abs(drop - pva['actual_asr_drop']) < 1e-9 and abs(agreement - pva['actual_clean_agreement']) < 1e-9\n"
         "assert pva['verdict'] == ('supported' if drop >= .20 and agreement >= .95 else 'not supported')\n"
         "print('ASR drop:', round(drop, 3), '| clean agreement:', agreement, '| verdict:', pva['verdict'])\n"
         "print('Uncertainty:', json.loads((RUN / 'uncertainty.json').read_text()))\n"
         "import sys; sys.path.insert(0, 'report')\nimport make_figures as figures   # the code that writes figures/, so this chart is the saved file\n"
         "figures.show(figures.m4_attack_success())"),
        ('md', '## Ablation study\n\n'
         'Successful attacks by concealment and payload family. Hidden means white or invisible text.\n\n' + ablation_table + '\n\n'
         f'- **Visibility-diff alone** stops every hidden attack ({hidden_vis} succeed) and almost none of the visible ones ({visible_vis} succeed). It cannot flag text that is genuinely visible.\n'
         f'- **The pretrained classifier alone** stops every direct override ({direct_clf}) and misses the paraphrase in most cases ({para_clf}), whatever the concealment.\n'
         '- **The components are complementary.** Each closes the gap the other leaves. What survives the combination is exactly their shared blind spot: the visible paraphrase.\n'
         f'- **Channel separation** behaves differently: it stops the paraphrase ({para_cs}) but not the direct override ({direct_cs}), and it is the only configuration that '
         f'changes a clean decision (clean agreement {pct(summary.loc["channel_separation", "clean_agreement_lower"])}). M5 examines this.\n\n'
         'Document-level detection, with injected documents as positives and the clean controls as negatives:\n\n' + detector_table),
        ('code', "attacks = pd.read_csv(RUN / 'attack_pairs.csv')\n"
         "eligible = attacks[attacks.eligible].assign(success=lambda d: d.decision.eq('SELECT'))\n"
         "order = ['naive', 'channel_separation', 'visibility_only', 'classifier_only', 'combined']\n"
         "display(eligible.pivot_table(index='config', columns=['family', 'technique'], values='success', aggfunc='sum').reindex(order).astype(int))\n"
         "display(pd.read_csv(RUN / 'document_detection_metrics.csv').round(3))"),
        ('code', "figures.show(figures.m4_ablation())\nfigures.show(figures.m4_detection())"),
        ('md', '## Additional experiment: fine-tuning the injection classifier (exploratory)\n\n'
         'The ablation shows the pretrained classifier is the weak component on paraphrased payloads. As a follow-up, DistilBERT was fine-tuned locally with '
         '[experiment/train_classifier_local.py](experiment/train_classifier_local.py) (3 epochs, learning rate 2e-5, seed 42, threshold chosen on validation only) and compared with the pretrained detector on '
         f'the same extracted blocks of the {N_DOCS} documents using [experiment/evaluate_classifier_on_pdfs.py](experiment/evaluate_classifier_on_pdfs.py). No LLM is involved in this comparison.\n\n'
         '- **Round 1** trained on the public `deepset/prompt-injections` set plus 160 authored resume-style sentences split by phrasing family '
         '([data/classifier_resume_domain.csv](data/classifier_resume_domain.csv)).\n'
         '- **Round 2** added genuine text blocks from 48 other resume PDFs of the same Kaggle dataset as benign examples '
         '([data/classifier_resume_blocks.csv](data/classifier_resume_blocks.csv)): 36 PDFs for training and 12 for threshold calibration. '
         f'The {N_RESUMES} evaluation resumes were excluded, and no training text matches any evaluation block exactly.\n\n'
         + finetune_table + '\n\n'
         f'Held-out test split (the public dataset\'s own test set plus two authored injection families and two authored benign families never seen in training): '
         f'round 1 {heldout(1, "test_metrics")}; round 2 {heldout(2, "test_metrics")}. '
         f'On the authored held-out families alone, round 2 has {heldout(2, "authored_test_metrics")}. '
         'The public test set is not grouped by phrasing family, so only the authored part measures generalisation to unseen wording.\n\n'
         '**How to read this.** Round 2 catches every payload block and flags no genuine block, where the pretrained detector misses every paraphrase. '
         'This is exploratory, not confirmatory, for three reasons. The authored injection sentences were written after the pretrained detector\'s misses were observed, and one training family is '
         'worded like the evaluation paraphrase. The training data was revised after round 1 failed on these same evaluation blocks. The benign blocks share the source and template of the evaluation resumes. '
         '**The attack success rate with the fine-tuned classifier has not been measured.** That requires a new LLM run under a new run ID, ideally with new payload wording.'),
        ('code', "for n in (1, 2):\n"
         "    m = json.loads(Path(f'results/classifier_training_0{n}/training_manifest.json').read_text())\n"
         "    print(f'Round {n}: sources', list(m['sources']), '| validation-selected threshold', m['threshold'])\n"
         "    print('   held-out test:', {k: round(v, 3) for k, v in m['test_metrics'].items()})\n"
         "    display(pd.read_csv(RUN / f'classifier_comparison_round{n}/block_metrics.csv', index_col=0))\n"
         "    display(pd.read_csv(RUN / f'classifier_comparison_round{n}/document_detection_metrics.csv', index_col=0))"),
        ('code', "figures.show(figures.m4_classifier_finetuning())"),
        ('md', '## Interpretation\n\n'
         f'On this sample the combined defence cut the attack success rate from {pct(summary.loc["naive", "asr_valid"])} to {pct(summary.loc["combined", "asr_valid"])} '
         'without changing any clean decision, and each component contributed a distinct part of that reduction. '
         'The result supports the hypothesis as recorded. It does not show that the defence is safe to deploy: the screener is a small quantised model, there are '
         f'{N_RESUMES} resumes from one source, two payload wordings and one placement, and a visible, politely worded instruction still succeeds. '
         'These ASR figures are not comparable with M2, which used a simulated screener, or with the pilot below, which measured sentence detection.'),
    ]


def m5_cells():
    resisted = sorted(naive_resisted.resume_id.unique())
    resisted_family = sorted(naive_resisted.family.unique())
    cc = changed_clean.iloc[0] if len(changed_clean) else None
    r1, r2 = blocks[1].loc['finetuned'], blocks[2].loc['finetuned']
    return [
        ('md', BANNER),
        ('md', '# M5 — Failure analysis, discussion and limitations\n\n'
         f'Evidence comes from the recorded responses and per-document audit files of run `{RUN_ID}`. Reasons quoted from the model are its outputs, not verified explanations of its behaviour. '
         'Nothing was re-run or re-tuned after the results were seen, apart from the classifier fine-tuning that M4 labels as exploratory.'),
        ('md', '## Failures observed\n\n'
         f'**1. Attacks that defeat the combined defence ({frac("combined")}).** All are the visible paraphrase:\n\n' + combined_failures_table + '\n\n'
         f'**2. The undefended screener is steered almost every time ({frac("naive")}).** Every successful attack in every configuration returned a score of exactly 100, '
         'with a reason that restates the resume as if it fitted the job. The pharmacist resume, for example, was selected for a Python backend role.\n\n'
         f'**3. One resume resisted one payload.** Under the naive configuration, {", ".join(resisted)} was not promoted by the {", ".join(resisted_family)} payload in any of its '
         f'{len(naive_resisted)} variants; the direct override still succeeded on it.\n\n'
         + (f'**4. Channel separation changed a clean decision.** Resume {cc.resume_id}, labelled SELECT and selected by the naive screener, was rejected with score {int(cc.score)} '
            f'("{cc.reason}"). Under attack, {len(cs_select_reject)} of {len(select_attacked[select_attacked.config == "channel_separation"])} documents built from the two qualified resumes were rejected '
            'with channel separation, and none with any other configuration.\n\n' if cc is not None else '')
         + f'**5. No genuine text was removed.** The two detectors removed {len(removals)} blocks in total; {payload_removed} are injected payloads and {clean_removed} come from a clean document. '
         'No response was invalid.'),
        ('code', SETUP +
         "attacks = pd.read_csv(RUN / 'attack_pairs.csv')\n"
         "eligible = attacks[attacks.eligible].assign(success=lambda d: d.decision.eq('SELECT'))\n"
         "order = ['naive', 'channel_separation', 'visibility_only', 'classifier_only', 'combined']\n"
         "pd.set_option('display.max_colwidth', 160)\n"
         "print('Successful attacks under the combined defence:')\n"
         "display(eligible[(eligible.config == 'combined') & eligible.success][['resume_id', 'variant', 'score', 'reason']])\n"
         "print('Attack success rate per resume and configuration:')\n"
         "display(eligible.pivot_table(index='resume_id', columns='config', values='success', aggfunc='mean')[order].round(2))\n"
         "print('Scores returned by successful attacks:', eligible[eligible.success].score.value_counts().to_dict())\n"
         "print('Eligible attacks that failed against the naive screener:')\n"
         "display(eligible[(eligible.config == 'naive') & ~eligible.success][['resume_id', 'variant', 'score', 'reason']])"),
        ('code', "clean = responses[responses.variant == 'clean']\n"
         "print('Clean scores by configuration (SELECT at 60 or more):')\n"
         "display(clean.pivot_table(index=['resume_id', 'expected_decision'], columns='config', values='score', aggfunc='first')[order])\n"
         "print('Clean decisions changed by a defence:')\n"
         "display(pd.read_csv(RUN / 'changed_clean_decisions.csv')[['resume_id', 'config', 'decision', 'ref_decision', 'score', 'reason']])\n"
         "qualified = attacks[attacks.expected_decision.eq('SELECT')]\n"
         "print('Decisions on attacked copies of the qualified resumes:')\n"
         "display(qualified.groupby('config').decision.value_counts().unstack(fill_value=0).reindex(order))"),
        ('code', "rows = []\n"
         "for path in sorted((RUN / 'audit').glob('*.json')):\n"
         "    for span in json.loads(path.read_text())['spans']:\n"
         "        payload = 'score 100' in span['text']\n"
         "        parts = path.stem.split('__')\n"
         "        rows.append(dict(payload=payload, family=(parts[2] if payload else 'genuine text'), technique=(parts[1] if payload else '-'),\n"
         "                         characters=len(span['text']), visibility_agreement=span['visibility_agreement'],\n"
         "                         visibility_flag=span['visibility_flag'], classifier_score=span['classifier_score'], classifier_flag=span['classifier_flag']))\n"
         "spans = pd.DataFrame(rows)\n"
         "print('Per-block detector evidence (pretrained classifier, as used in the LLM run):')\n"
         "display(spans.groupby(['family', 'technique']).agg(blocks=('payload', 'size'), min_agreement=('visibility_agreement', 'min'),\n"
         "    max_agreement=('visibility_agreement', 'max'), visibility_flagged=('visibility_flag', 'sum'),\n"
         "    min_score=('classifier_score', 'min'), max_score=('classifier_score', 'max'), classifier_flagged=('classifier_flag', 'sum')).round(4))\n"
         "removed = pd.read_csv(RUN / 'removals_for_review.csv')\n"
         "print('Blocks removed:', len(removed), '| from clean documents:', int(removed.document.str.endswith('__clean').sum()))"),
        ('code', "import sys; sys.path.insert(0, 'report')\nimport make_figures as figures   # the code that writes figures/, so this chart is the saved file\nfigures.show(figures.m5_scores())"),
        ('md', '## Evidence-based explanations\n\n'
         '- **Why the visible paraphrase survives.** The audit table above shows the two components fail on it for different, mechanical reasons. Its rendered and extracted text agree fully, '
         'so the visibility check has nothing to flag; that is the design limit of a visibility comparison, not a tuning problem. The pretrained classifier scores the paraphrase near 0 and the direct override near 0.99, '
         'far from the 0.5 threshold on both sides. The paraphrase contains no "ignore" or "instructions" wording and reads like a reviewer\'s note.\n'
         '- **Why hidden text is always caught.** White and invisible payloads have 0 agreement between extraction and OCR, while genuine blocks of 12 or more characters never fall below about 95. '
         'The fixed threshold of 80 sits in a wide empty gap on this sample, which is also why no genuine block was removed.\n'
         '- **Why channel separation is hard to interpret.** It did not just block attacks; it shifted scoring in general. Two clean resumes dropped to a score of 0 and one qualified resume fell from 72 to 58. '
         'Its lower ASR therefore cannot be separated from a general change in how this small model scores JSON-wrapped input. The direct override still worked through it in most cases, '
         'so a policy sentence alone did not make this model ignore instructions.\n'
         '- **The successes are not marginal flips.** Three of the four eligible resumes have a clean score of 58, two points under the threshold. That would be a concern if attacks only nudged scores, '
         'but every success is a jump to exactly 100, which is the value the payloads request.\n'
         '- **The resume that resisted the paraphrase** received a score of 0 with the reason "Lack of Python backend development experience". The data shows the outcome, not its cause; '
         'one resume is not enough to attribute it to any property of the document.\n\n'
         'These are explanations of error patterns grounded in the recorded text, scores and decisions. They are not causal claims about the model\'s internals.'),
        ('md', '## Failure analysis of the fine-tuned classifier (exploratory)\n\n'
         f'- **Round 1 over-flagged genuine text.** At its validation-selected threshold of {manifests[1]["threshold"]} it flagged {int(r1.fp)} of {int(r1.fp + r1.tn)} genuine resume blocks. '
         'Its training and validation data contained only short sentences, whereas real resume blocks are paragraphs of about 25 words mixing skills, dates and employers. '
         'The validation set could not reveal this, so the threshold rule picked an operating point that was wrong for the deployment distribution.\n'
         f'- **Round 2 removed that failure on this sample** ({int(r2.fp)} of {int(r2.fp + r2.tn)} genuine blocks flagged) once real resume blocks were in the training and calibration data.\n'
         f'- **A remaining weakness is visible on held-out sentences.** In the authored test families, round 2 still flags {manifests[2]["authored_test_metrics"]["fp"]} of '
         f'{manifests[2]["authored_test_metrics"]["fp"] + manifests[2]["authored_test_metrics"]["tn"]} benign sentences, all of them quoted feedback such as a manager\'s or client\'s praise. '
         'Praise about a candidate resembles a "rate this candidate highly" injection, which is the same confusion the pilot found between quoting an instruction and issuing one.'),
        ('code', "for n in (1, 2):\n"
         "    scores = pd.read_csv(RUN / f'classifier_comparison_round{n}/span_scores.csv')\n"
         "    genuine, payload = scores[~scores.is_payload], scores[scores.is_payload]\n"
         "    print(f'Round {n}: genuine blocks flagged {int(genuine.finetuned_flag.sum())}/{len(genuine)} | payload blocks caught {int(payload.finetuned_flag.sum())}/{len(payload)}'\n"
         "          f' | highest genuine score {genuine.finetuned_score.max():.3f} | lowest payload score {payload.finetuned_score.min():.3f}')\n"
         "    test = pd.read_csv(f'results/classifier_training_0{n}/classifier_test_predictions.csv')\n"
         "    authored = test[test.source.str.startswith('authored')]\n"
         "    print('   authored held-out errors by family:', authored[authored.label != authored.prediction].groupby('group_id').size().to_dict())\n"
         "authored2 = authored[authored.label != authored.prediction]\n"
         "display(authored2[['group_id', 'text', 'label', 'score']])"),
        ('md', '## Discussion\n\n'
         'The experiment answers the M3 question for this sample: a real LLM screener was steered by text hidden in real PDFs, and combining a rendering-based visibility check with an injection classifier '
         'removed most of that effect without altering clean decisions. The ablation explains why the combination works: the two components fail on different attacks. '
         'It also locates the residual risk precisely. An instruction that is visible on the page and phrased as ordinary reviewer guidance passes both components, and a human reader would see it too. '
         'Defending against that case needs either a better text classifier, which the exploratory fine-tuning suggests is feasible, or a screener that does not follow instructions in candidate text. '
         'Channel separation, the anchor paper\'s main defence, did not provide that on this small model and had a cost on a qualified candidate.'),
        ('md', '## Limitations and threats to validity\n\n'
         f'- **External validity.** {N_RESUMES} resumes from one public dataset with one layout template; English only; a 3B-parameter, 4-bit model that is weaker than production screeners; '
         'one job rubric. Larger models may be more or less susceptible, and other PDF producers, fonts, columns and scanned pages may break the OCR comparison.\n'
         '- **Construct validity.** Attacks are two fixed payload strings on an appended page, with three concealment methods. They are not naturally occurring attacks, not adaptive, and do not cover the nine concealment '
         'techniques of M2. "Success" is a SELECT decision on a resume that a human and the clean screener reject; ranking effects and partial score inflation are not measured.\n'
         '- **Labels.** One annotator, the project author, aided by keyword evidence snippets. No independent annotation or agreement measure. The shortlist of resumes was also drawn by keyword scan.\n'
         f'- **Statistical validity.** {unc["resume_count"]} eligible resumes. The seven variants of a resume are correlated, so {N_ELIGIBLE} eligible attacks are not {N_ELIGIBLE} independent trials. '
         'One repeat at temperature 0. The bootstrap interval is descriptive. No significance claim is made.\n'
         '- **Internal validity.** An incomplete earlier attempt on another model existed before this run was frozen (see M3). The pretrained classifier\'s own training data is unknown and may overlap with common injection phrasing '
         'such as the direct payload. The clean-agreement criterion can preserve a wrong baseline, which is why accuracy against human labels is reported beside it.\n'
         '- **Reproducibility.** The screener is identified by an Ollama model tag, not a weight hash, and its revision is not recorded in the protocol. Quantised local inference can differ across hardware and Ollama versions. '
         'The hosted attempts are not reproducible without paid access.\n'
         '- **Fine-tuning.** Exploratory and informed by the evaluation documents, as described in M4. Its detection figures must not be reported as a confirmed improvement, and no attack success rate was measured with it.\n'
         '- **Scope.** No study of demographic fairness, hiring validity or candidate ranking. A detector flag should prompt human review, not automatic rejection of a person.\n\n'
         '**Next experiment, not performed.** Freeze a new protocol and run ID that uses the fine-tuned classifier, new payload wordings written by someone who has not seen the training sentences, '
         'resumes from a second source, a second annotator, and a larger screener model. Do not reuse this run to claim confirmation.'),
    ]
