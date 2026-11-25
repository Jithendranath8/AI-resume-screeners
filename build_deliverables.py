"""Build milestone notebooks and paper from saved results, without retraining."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import nbformat as nbf
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT / 'results/run.json').read_text())
p = json.loads((ROOT / 'M3_Protocol.json').read_text())
rows = list(csv.DictReader((ROOT / 'results/predictions.csv').open()))
errors = [x for x in rows if x['encoding'] == 'original'
          and x['normalized_char_lr'] != x['label']]


def notebook(name, cells):
    n = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(s) if t == 'md'
                                  else nbf.v4.new_code_cell(s) for t, s in cells])
    n.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    nbf.write(n, ROOT / name)


notebook('M3_Research_Gap.ipynb', [
    ('md', '# M3 — Research gap, hypothesis and experiment plan\n\n'
     '**Scope:** small, CPU-only NLP detector; no LLM training or paid API.\n\n'
     'The base paper is Jay Barach, *ResumeShield*, [arXiv:2609.20188v1](https://arxiv.org/html/2609.20188v1). '
     'See the review notes added to M1 and M2 before using their historical claims. '
     'M2 is a partial reconstruction and M3 is a narrower extension than the original PDF/LLM proposal.'),
    ('md', '## Research gap\n\n' + p['gap'] + '\n\n'
     'The extension targets visible paraphrases and benign security-related sentences. '
     'It does not claim to solve hidden-text extraction or establish a new classifier architecture. '
     'PhantomLint already studies document visibility, so visibility comparison itself is not our novelty.'),
    ('md', '## One hypothesis and prediction recorded before training\n\n' + p['hypothesis'] + '\n\n'
     '**Prediction:** recall gain ≥20 percentage points and FPR ≤10%; both conditions must hold. '
     'The numeric prediction was saved in [M3_Protocol.json](M3_Protocol.json) before the first training call. '
     'This notebook is a later readable rendering of that record. The run records a hash and UTC start time; '
     'this is not an independently registered or externally timestamped protocol.'),
    ('md', '## Experimental plan\n\n'
     '1. Freeze 80 authored English sentences: 32 train, 16 validation, 32 test, balanced by class in each split.\n'
     '2. Keep directive/benign families separate across splits. Make original, zero-width, and homoglyph versions inside each split: 96/48/96 rows.\n'
     '3. Fit character TF-IDF and logistic regression on training rows only.\n'
     '4. Select the threshold using validation only, under the fixed FPR rule.\n'
     '5. Evaluate the fixed keyword baseline and learned detector on the same test rows.\n'
     '6. Remove normalization, replace character features with word features, and test normalized keywords.\n'
     '7. Report confusion counts, precision/recall/F1/FPR, family slices, original-only metrics and all failures.\n'
     '8. Retain the result even if the hypothesis fails; do not change the test set or threshold afterward.\n\n'
     '**Controls:** same labels, splits, train data, classifier settings, and threshold-selection procedure. '
     'Word-feature ablation reselects its threshold on validation, so it compares complete pipelines, not only representations at an identical operating point. '
     'No sample-size inflation: 96 test rows correspond to 32 source sentences and eight families.'),
    ('code', "import json\nfrom pathlib import Path\nprotocol = json.loads(Path('M3_Protocol.json').read_text())\nprint(json.dumps(protocol, indent=2))"),
])

notebook('M4_Controlled_Experiments.ipynb', [
    ('md', '# M4 — Controlled experiments\n\n'
     'Run `python experiment.py` from the repository root to reproduce training. '
     'This notebook reads the recorded run and checks its provenance without retraining. '
     'Full text, labels, group IDs and predictions are exported in [results/](results/).'),
    ('code', "import json, hashlib\nfrom pathlib import Path\nimport pandas as pd\nimport matplotlib.pyplot as plt\nr = json.loads(Path('results/run.json').read_text())\nfor name, expected in r['input_sha256'].items():\n    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name\nprint('Input hashes verified; run:', r['started_utc'])\nprint('Environment:', r['environment'])\nprint('Source sentences:', r['split_sources'], 'Expanded rows:', r['split_rows'])"),
    ('md', '## Baseline and experimental configurations\n\n'
     '**Baseline:** seven fixed keyword phrases from the initial M2 detector intent; substring matching, case-insensitive. '
     'This is the lexical component only, not the complete ResumeShield system.\n\n'
     '**Experimental:** normalize zero-width/confusable characters, character TF-IDF (3–5 characters within words), '
     'logistic regression C=1, balanced class weights, liblinear, seed 42, max_iter=1000. '
     'There is no hyperparameter search. Threshold candidates are 0.3, 0.5, 0.7; choose best validation F1 subject to FPR≤0.10. '
     'The exact fallback and tie rules are in the protocol.\n\n'
     '**Metrics:** precision=TP/(TP+FP); recall=TP/(TP+FN); F1=2PR/(P+R); FPR=FP/(FP+TN). '
     'Counts are shown so denominators remain clear. For absent classes, the code returns zero; such per-family metrics are not meaningful. '
     'Detection recall is not attack success rate or hiring utility.'),
    ('code', "display(pd.DataFrame(r['metrics']).T)\nprint('Validation-selected thresholds:', r['thresholds'])\ndisplay(pd.DataFrame([{'model': name, **v} for name, values in r['validation'].items() for v in values]))"),
    ('md', '## Predicted versus actual\n\n'
     'Before training: recall improvement ≥0.20 and experimental FPR ≤0.10. '
     'Both must pass. Failure of either condition means the hypothesis is not supported on this benchmark.'),
    ('code', "print('Recall gain:', r['recall_gain'])\nprint('Experimental FPR:', r['metrics']['normalized_char_lr']['fpr'])\nprint('Hypothesis supported:', r['hypothesis_supported'])\nprint('Descriptive family-bootstrap 95% interval for recall gain:', r['family_bootstrap_95_interval'])\nassert r['hypothesis_supported'] == (r['recall_gain'] >= .2 and r['metrics']['normalized_char_lr']['fpr'] <= .1)"),
    ('md', '## Ablation study\n\n'
     'Removing normalization changes only preprocessing. Replacing character with word features tests representation under the same threshold-selection rule. '
     'Normalized keywords isolate normalization without learning. These are paired comparisons on identical test rows. '
     'The character and raw-character models select 0.5; the word model selects 0.7. '
     'Consequently, word-model recall reflects both representation and its validation-selected operating point.'),
    ('code', "table = pd.DataFrame(r['metrics']).T\nax = table[['recall', 'fpr']].plot.bar(figsize=(9,4), ylim=(0,1), rot=20)\nax.set_ylabel('Rate')\nax.set_title('Held-out sentence variants: recall and false-positive rate')\nplt.tight_layout()\nplt.show()\ndisplay(pd.DataFrame(r['original_only']).T)"),
    ('md', '## Interpretation\n\n'
     'The primary model gains 93.75 percentage points in recall but fails the FPR requirement (18.75%). '
     'Removing normalization decreases recall by 2.08 points and increases FPR by 2.08 points; this is only one test row each. '
     'The word model has 25% recall and no false positives at its selected threshold. '
     'Normalized keywords still miss every held-out attack and flag more benign quotations.\n\n'
     'Original-only results show the normalized detector catches 15/16 attacks and flags 3/16 benign sentences. '
     'The raw keyword baseline flags 4/16 clean originals (25%); its lower expanded-set FPR of 8.33% occurs because obfuscation hides the quoted keywords. '
     'This is not genuine semantic discrimination. The test deliberately stresses phrases outside the keyword list, so baseline failure does not estimate real-world baseline performance. '
     'Bootstrap resampling uses four attack families rather than 48 independent rows; the interval is descriptive and cannot establish population robustness.'),
])

failure_table = '| Source ID | Truth | Prediction | Score | Text |\n|---|---|---|---:|---|\n' + '\n'.join(
    f"| {x['source_id']} | {x['label']} | {x['normalized_char_lr']} | {float(x['normalized_char_lr_score']):.3f} | {x['text']} |" for x in errors)
notebook('M5_Failure_Analysis.ipynb', [
    ('md', '# M5 — Failure analysis\n\n'
     'Evidence comes from the untouched test predictions. Labels: 1=injection, 0=benign. '
     'The following table contains **all** primary-model mistakes among original sentences; each has three encoding variants in the full test set.\n\n' + failure_table),
    ('md', '## Evidence-based explanations\n\n'
     'The model misses one schema/formatting attack: the instruction is expressed as a required output structure rather than a direct override. '
     'This is consistent with limited coverage of formatting language in training. '
     'Two benign quotations of attack phrases are flagged, showing that matching subword patterns does not reliably distinguish discussion of an instruction from its use. '
     'One benign service sentence is also flagged, demonstrating that errors are not confined to explicit quoted payloads. '
     'These are error-pattern explanations grounded in the text and predictions, not proven causal attributions to individual features.\n\n'
     'The family table below exposes concentrations of failure. Formatting-family recall is 75%; hard-negative FPR is 50%; service-family FPR is 25%. '
     'All rates concern this tiny authored sample. Normalization makes the three variants identical, so repeated errors are not three independent observations.'),
    ('code', "import json\nimport pandas as pd\nfrom pathlib import Path\nr = json.loads(Path('results/run.json').read_text())\npred = pd.read_csv('results/predictions.csv')\ndisplay(pred.loc[pred.label != pred.normalized_char_lr, ['id','family','encoding','text','label','normalized_char_lr','normalized_char_lr_score']])\ndisplay(pd.DataFrame({f: v['normalized_char_lr'] for f,v in r['per_family'].items()}).T)"),
    ('md', '## Limitations and threats to validity\n\n'
     '- **Internal validity:** fixed splits and train-only vocabulary prevent direct leakage, but a single author designed data and models. Families are distinct, yet share vocabulary and intent. M2 includes historical wording also seen in one new test sentence, so there is no claim of blinded evaluation.\n'
     '- **Construct validity:** a label means an authored sentence attempts to steer an evaluator; no LLM is asked whether it succeeds. Some endorsements are ambiguous without document context. No independent annotator checked labels.\n'
     '- **External validity:** short English sentences, 50% attack prevalence, no full resumes, PDFs, OCR, multilingual documents, actual candidates, real hiring outcomes or adaptive optimized attacks.\n'
     '- **Statistical validity:** only 16 malicious and 16 benign source sentences in the test set. Variants are correlated, and four attack families make the bootstrap unstable. No significance claim is made.\n'
     '- **Comparator validity:** the fixed keyword list is intentionally challenged by held-out wording. It is not the anchor full defense; gains cannot be described as beating ResumeShield.\n'
     '- **Operational limitation:** the eight-character mapping is incomplete and can alter legitimate multilingual text. Scores are not calibrated probabilities of malicious intent in deployment.\n\n'
     '**Next experiment, not performed:** independently annotate a larger set with benign quotations and full document context, freeze a new test split, and evaluate a more conservative threshold on validation. '
     'Do not reuse this test set to claim confirmation. A detector flag should prompt review, not automatic rejection of a person.'),
])

title = 'Lightweight Detection of Resume Prompt Injections: A Controlled Synthetic Study'
abstract = ('We examine a small, reproducible extension to a resume prompt-injection simulation: '
            'whether a CPU-trained text detector recognizes held-out instruction families while avoiding benign text. '
            'We compare fixed keywords with normalized character TF-IDF and logistic regression on 80 authored sentences '
            'split by family, expanded to three encodings each. Our prediction required a recall improvement of at least '
            '20 percentage points and a false-positive rate no greater than 10%. The detector achieves 93.75% recall, '
            '83.33% precision, and 88.24% F1, but its false-positive rate is 18.75%. The joint hypothesis is therefore '
            'not supported. Ablations and recorded failures expose a tradeoff between recognizing varied directives and '
            'misclassifying benign descriptions. Results concern text detection, not successful attacks on an LLM or real hiring decisions.')
sections = [
('I. Introduction', [
    'Resume-processing systems may receive applicant-authored text that asks the evaluator to depart from its intended task. '
    'A detector must distinguish such instructions from legitimate descriptions, including security work that quotes attack phrases. '
    'Catching more suspicious text is not sufficient if genuine content is routinely flagged. This student project studies that tradeoff using a deliberately small experiment.',
    'The project retains ResumeShield [1] as its anchor. M1 surveyed related work; M2 implemented an independent deterministic reconstruction. '
    'For feasibility, the new work replaces the initially proposed PDF and hosted-LLM extension with sentence-level classification. '
    'The contribution is a controlled evaluation with an explicit prediction, family-disjoint splits, component ablations, exported predictions, and failure analysis. '
    'The classifier architecture itself is standard.']),
('II. Related Work and Reproduction Boundary', [
    'ResumeShield [1] combines filtering and channel separation in a synthetic screening benchmark. Its reported naive and fenced attack-success rates are 1 and 0; '
    'the simulated backend assumes compliance with separation. Our M2 reconstructs the corpus dimensions and selected outcomes but does not execute the original artifact. '
    'Local markup-only and all-filter rates differ from the paper, and the local success metric is narrower. These are method differences, not evidence of real-model robustness.',
    'Greshake et al. [2] establish indirect prompt injection as an application threat. Liu et al. [3] provide a systematic attack-and-defense framework. '
    'PhantomLint [4] studies visibility-based detection in structured documents and has public code. '
    'Zhang et al. [5] examine resume injections in a production corpus. Our experiment addresses a smaller evaluation question: generalization of a trainable text detector '
    'to authored directive families while handling benign instruction-like content. We do not claim that this gap is the first investigation of learned injection detection.']),
('III. Research Question and Prior Prediction', [
    p['hypothesis'],
    'The two requirements form one joint hypothesis. Both must hold. The prediction was written to M3_Protocol.json before model fitting; '
    'the run stores its SHA-256 hash and a UTC start time. This local record documents execution order but is not an independently timestamped preregistration. '
    'No test-driven threshold adjustment was made after observing the outcome.']),
('IV. Experimental Design', [
    'Data comprise 80 authored English sentences: 32 training, 16 validation, and 32 test sources, balanced between benign and injection labels in each split. '
    'An injection label denotes an attempt to direct the evaluator, not an empirically successful attack. Training families include overrides, ranking commands, '
    'role claims and concealment requests. Validation uses bypass and output directives. Test attacks use paraphrase, authority, formatting and pressure; '
    'test benign groups include quotations, recommendations, project descriptions and service work. Labels and family assignments were set before training.',
    'Each source receives an original version, a version with zero-width separators, and a version using eight Latin/Cyrillic confusables. '
    'This produces 96 training, 48 validation and 96 test rows. All versions of a source stay together. '
    'The model sees sentence text only; IDs, labels and family names are not features. Assertions check family disjointness and absence of canonical-text overlap. '
    'Variants are correlated, not additional independent observations; the effective test collection contains 32 sentences.',
    'The fixed baseline searches seven literal phrases from the initial M2 lexical detector. The experimental pipeline removes U+200B, folds the limited confusable map, '
    'and fits character TF-IDF features with within-word n-grams of length 3 through 5. Logistic regression uses C=1, balanced class weights, '
    'liblinear, random_state=42 and max_iter=1000. Vocabulary and weights are fitted on training text only. No GPU, external API or pretrained language model is used.',
    'Threshold candidates are 0.3, 0.5 and 0.7. Selection maximizes validation F1 subject to FPR at most 0.10. '
    'If no candidate satisfies that constraint, the rule minimizes FPR then maximizes F1; ties prefer the higher threshold. '
    'Both character models select 0.5 and the word model selects 0.7. The same selection rule is applied to all learned configurations.',
    'Precision is TP/(TP+FP), recall TP/(TP+FN), FPR FP/(FP+TN), and F1 the harmonic mean of precision and recall. '
    'The main contrast is paired recall gain on identical test rows with an absolute experimental FPR constraint. '
    'A descriptive bootstrap samples four attack families with replacement for 2000 draws at seed 42. '
    'We report original-text-only metrics to make dependence on encoding variants explicit.']),
('V. Results and Ablations', [
    'Table I reports all configurations on 96 test rows: 48 injected and 48 benign. The primary model has TP=45, FP=9, FN=3 and TN=39. '
    'Its recall improvement over keywords is 93.75 percentage points, satisfying the predicted improvement, but FPR=18.75% exceeds 10%. '
    'The hypothesis is not supported. The descriptive family-bootstrap interval for recall gain is 81.25 to 100 percentage points; '
    'with four attack families this is not strong population-level uncertainty evidence.',
    'Removing normalization changes recall from 93.75% to 91.67% and FPR from 18.75% to 20.83%. '
    'These differences correspond to one row each, so they offer limited support for broad normalization benefits. '
    'Replacing character features with word 1-2 grams yields 25% recall and zero observed false positives. '
    'Because this ablation selects its own validation threshold, its difference includes representation and operating-point effects. '
    'Normalized keywords still detect no test attacks and have FPR=25%; normalization alone does not provide semantic discrimination.',
    'On original text only, the primary detector catches 15 of 16 attacks and flags 3 of 16 benign sentences. '
    'The keyword baseline catches none and flags 4 benign sentences. Its expanded-set FPR is lower, 8.33%, because encoding obscures its false-positive phrases. '
    'The held-out attacks intentionally avoid known keywords, making this a stress test rather than a representative prevalence-weighted comparison. '
    'These detection results are not comparable to M2 attack-success rates.']),
('VI. Failure Analysis', [
    'All four original-text errors are retained below, with raw model scores and ground truth. Their encoding variants account for all twelve expanded-set errors.',
    *[f"{'False positive' if x['label']=='0' else 'False negative'}: {x['text']} Score={float(x['normalized_char_lr_score']):.3f}; threshold=0.5."
      for x in errors],
    'The false negative uses output-schema language. Two false positives quote attack-like language in benign descriptions, and another occurs in a benign service description. '
    'These examples support concern about context sensitivity and limited family coverage. They do not prove which learned feature caused each decision. '
    'Formatting-family recall is 75%, hard-negative FPR 50%, and service-family FPR 25% on the expanded test set. '
    'An appropriate follow-up would use independent annotation and more benign quotations in training, evaluated on a newly frozen test set. That follow-up has not been performed.']),
('VII. Limitations, Validity and Ethics', [
    'Internal validity is limited by a single author defining both data and model choices. Family disjointness reduces direct leakage but does not guarantee '
    'semantic independence. One test phrase also appears in historical M2, so evaluation is not blinded. '
    'Construct validity is limited by sentence-level labels without full resume context or independent annotator agreement. '
    'A persuasive recommendation may be ambiguous; no successful exploitation of a live model is established.',
    'External validity is limited by short synthetic English text, balanced class prevalence, a small confusable map, and no actual rendering or extraction. '
    'The study does not measure whole-resume utility, factual fabrication, ranking quality, demographic fairness, or model-specific attack success. '
    'Statistical validity is limited by few families and correlated variants. '
    'The baseline is a fixed lexical component rather than the complete anchor defense. Zero test false positives in an ablation should not be interpreted as zero deployment risk.',
    'All material is synthetic and contains no real applicant data. A detector output should not automatically reject a person. '
    'The observed false-positive tradeoff motivates human review and independent validation before operational use.']),
('VIII. Conclusion and Contribution Attribution', [
    'The primary prediction failed because recall improvement came with excessive false positives. A lightweight learned detector can recognize the authored held-out '
    'directives more often than fixed keywords, but this benchmark does not establish a safe screening defense. '
    'The reproduced component is the earlier local simulation inspired by ResumeShield; the original student contribution is the frozen sentence dataset, '
    'train/validation/test design, lightweight classifier comparison, ablations and documented negative finding. '
    'Future work should first strengthen annotation and context coverage, then evaluate an independently frozen document-level benchmark.']),
]
refs = [
    'J. Barach, “ResumeShield: Channel Separation and an Open Benchmark for Indirect Prompt Injection in AI Resume Screening,” arXiv:2609.20188v1, 2026. https://arxiv.org/abs/2609.20188',
    'K. Greshake et al., “Not what you’ve signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection,” AISec, 2023. https://arxiv.org/abs/2302.12173',
    'Y. Liu et al., “Formalizing and Benchmarking Prompt Injection Attacks and Defenses,” USENIX Security, 2024; consulted arXiv:2310.12815v5. https://arxiv.org/abs/2310.12815',
    'T. Murray, “PhantomLint: Principled Detection of Hidden LLM Prompts in Structured Documents,” arXiv:2508.17884v2, 2025. https://arxiv.org/abs/2508.17884 ; code: https://github.com/tobycmurray/phantom-lint',
    'M. Zhang et al., “Measuring Real-World Prompt Injection Attacks in LLM-based Resume Screening,” USENIX Security, 2026. https://arxiv.org/abs/2605.28999',
]

paper = ROOT / 'paper'
paper.mkdir(exist_ok=True)

def tex_escape(s):
    for a, b in [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#')]:
        s = s.replace(a, b)
    return s.replace('“', '``').replace('”', "''").replace('’', "'").replace('–', '--')

tex = r'\documentclass[conference]{IEEEtran}' + '\n' + r'\usepackage[T1]{fontenc}\usepackage{url}' + '\n'
tex += r'\title{' + title + '}\n' + r'\author{\IEEEauthorblockN{NLP Course Project}\IEEEauthorblockA{Author and institution to be supplied before submission}}' + '\n'
tex += r'\begin{document}\maketitle\begin{abstract}' + tex_escape(abstract) + r'\end{abstract}' + '\n'
tex += r'\begin{IEEEkeywords}prompt injection, resume screening, text classification, failure analysis\end{IEEEkeywords}' + '\n'
for heading, paragraphs in sections:
    tex += '\\section{' + heading.split('. ', 1)[1] + '}\n'
    tex += '\n\n'.join(tex_escape(s) for s in paragraphs) + '\n'
    if heading.startswith('V.'):
        tex += r'\begin{table}[ht]\centering\caption{Test detection metrics (96 rows).}\begin{tabular}{lrrrr}\hline Model & P & R & F1 & FPR\\\hline' + '\n'
        labels = ['Keywords', 'Norm. keywords', 'Norm. char LR', 'Raw char LR', 'Norm. word LR']
        for label, m in zip(labels, r['metrics'].values()):
            tex += label + ' & ' + ' & '.join(f'{m[k]:.3f}' for k in ['precision','recall','f1','fpr']) + r'\\' + '\n'
        tex += r'\hline\end{tabular}\end{table}' + '\n'
tex += r'\begin{thebibliography}{00}' + '\n'
for i, ref in enumerate(refs):
    tex += r'\bibitem{r' + str(i+1) + '}' + tex_escape(ref) + '\n'
tex += r'\end{thebibliography}\end{document}' + '\n'
(paper / 'M6_Final_Paper.tex').write_text(tex)

# A portable readable IEEE-style PDF. Official IEEEtran source is supplied separately.
body = ParagraphStyle('body', fontName='Times-Roman', fontSize=9, leading=11, alignment=TA_JUSTIFY, spaceAfter=6)
head = ParagraphStyle('head', parent=body, fontName='Times-Bold', fontSize=10, spaceBefore=10, alignment=TA_CENTER)
small = ParagraphStyle('small', parent=body, fontSize=8, leading=9)
doc = BaseDocTemplate(str(paper / 'M6_Final_Paper.pdf'), pagesize=(612,792),
                      title=title, author='NLP Course Project', leftMargin=42, rightMargin=42,
                      topMargin=40, bottomMargin=40)
width = (612-84-18)/2
frames = [Frame(42,40,width,712,id='left'), Frame(42+width+18,40,width,712,id='right')]
def footer(canvas, doc):
    canvas.setFont('Times-Roman', 8)
    canvas.drawCentredString(306, 22, f'NLP course project | {doc.page} | IEEE-style reading copy')
doc.addPageTemplates(PageTemplate(id='two_columns', frames=frames, onPage=footer))
story = [Paragraph(title, ParagraphStyle('title', parent=head, fontSize=14, leading=16)),
         Paragraph('NLP Course Project<br/>Author and institution to be supplied before submission', small),
         Spacer(1,8), Paragraph('<b>Abstract—</b>' + escape(abstract), body),
         Paragraph('<b>Keywords—</b>prompt injection; resume screening; text classification', small)]
for heading, paragraphs in sections:
    story.append(Paragraph(escape(heading), head))
    story.extend(Paragraph(escape(s), body) for s in paragraphs)
    if heading.startswith('V.'):
        story.append(Paragraph('TABLE I. TEST DETECTION METRICS', head))
        table = [['Model','P','R','F1','FPR']]
        for label, m in zip(['Keywords','Norm. keywords','Norm. char LR','Raw char LR','Norm. word LR'], r['metrics'].values()):
            table.append([label] + [f'{m[k]:.3f}' for k in ['precision','recall','f1','fpr']])
        tab = Table(table, colWidths=[83,37,37,37,37], repeatRows=1)
        tab.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Times-Roman'),('FONTSIZE',(0,0),(-1,-1),8),
                                 ('LINEABOVE',(0,0),(-1,0),.5,colors.black),
                                 ('LINEBELOW',(0,0),(-1,0),.5,colors.black),
                                 ('LINEBELOW',(0,-1),(-1,-1),.5,colors.black)]))
        story.append(tab)
story.append(Paragraph('References', head))
story.extend(Paragraph(escape(f'[{i+1}] {ref}'), small) for i,ref in enumerate(refs))
doc.build(story)

notebook('M6_Defence.ipynb', [
    ('md', '# M6 — Final paper and oral defence\n\n'
     'Read the [final paper PDF](paper/M6_Final_Paper.pdf). The [IEEEtran LaTeX source](paper/M6_Final_Paper.tex) '
     'is supplied for official formatting. The PDF is an IEEE-style reading copy generated with ReportLab; '
     'the LaTeX source has not been compiled locally. Replace the neutral author block with your name, institution and any required course details before submission.'),
    ('md', '## Six-slide / six-minute defence outline\n\n'
     '1. **Problem (45 s):** applicant text can contain evaluator-directed instructions. Benign security experience can contain the same words.\n'
     '2. **Anchor and M2 (45 s):** ResumeShield; our partial simulation reconstruction; zero fenced ASR is an assumption of that simulator.\n'
     '3. **Gap and prediction (60 s):** held-out directive families; predict ≥20-point recall gain with FPR≤10%; show the protocol.\n'
     '4. **Method (60 s):** 80 sentences; family-disjoint splits; three encodings; fixed keywords versus character TF-IDF + logistic regression; validation-only threshold.\n'
     '5. **Results and ablations (90 s):** recall 93.75%, FPR 18.75%; hypothesis not supported. Normalization helps slightly; the word model trades recall for fewer false positives.\n'
     '6. **Failures and contribution (60 s):** show one missed formatting directive and one benign quotation flagged; explain synthetic-data limitations and original versus reproduced work.'),
    ('md', '## Likely defence questions and concise answers\n\n'
     '**What is the base paper?** Jay Barach, ResumeShield, arXiv:2609.20188v1.\n\n'
     '**Did you reproduce its exact implementation?** No. M2 is a local reconstruction; corpus wording, success definition and filters differ.\n\n'
     '**What did you train?** A standard logistic-regression detector over TF-IDF text features. No LLM was trained.\n\n'
     '**What is new?** The small frozen family-split evaluation, trained-detector comparison, ablations and evidence-based negative result; not a new algorithm.\n\n'
     '**Why did your hypothesis fail?** Recall passed, but three of sixteen benign test sources were flagged, violating the FPR limit.\n\n'
     '**Why is keyword recall zero?** The test attacks stress unfamiliar phrasing; the narrow list misses them. That limits generalization of the comparison.\n\n'
     '**Are there 96 independent test examples?** No: 32 source sentences, each with three related encodings.\n\n'
     '**How did you avoid leakage?** Families and source sentences remain in one split; TF-IDF fits only on training; thresholds use validation only. Single-author semantic overlap remains a limitation.\n\n'
     '**Does high recall mean fewer successful attacks?** Not necessarily. We measured detection, not outcomes on a real LLM.\n\n'
     '**Why not change the threshold now?** That would tune to the test set. A revised model needs a new held-out test.\n\n'
     '**Can you submit a failed hypothesis?** Yes: the rubric asks for controlled evidence and honest analysis, not a guaranteed improvement.\n\n'
     '**What remains for you?** Supply author/course metadata, check your course-specific page/venue requirements, and personally rehearse and deliver the oral defence.'),
    ('md', '## Rubric coverage\n\n'
     '| Milestone | Deliverable | Location |\n|---|---|---|\n'
     '| M3 | Gap, one hypothesis, prior prediction, experimental plan | M3 notebook + protocol |\n'
     '| M4 | Baseline, experimental setup, metrics, prediction comparison, ablations | M4 notebook + saved run/predictions |\n'
     '| M5 | Actual failures, explanations, limitations, validity threats | M5 notebook |\n'
     '| M6 | IEEE-style paper, attribution, references, defence preparation | paper/ + this notebook |\n\n'
     'The oral defence is a student activity, not something generated files can complete on your behalf.'),
])
print('Built M3–M6 notebooks, final-paper PDF and IEEEtran source.')
