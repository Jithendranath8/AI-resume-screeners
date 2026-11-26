"""Build milestone notebooks and paper from saved results, without retraining.

    .venv/bin/python report/build_deliverables.py

Writes M3–M6 notebooks to the repository root and the paper to paper/.

M3–M5 lead with the completed real-PDF / real-LLM experiment (real_pdf_sections.py) and keep the
earlier TF-IDF pilot (pilot/) as a labelled historical section. The M6 paper and defence notebook
(paper_sections.py) report the real-PDF experiment as the main study, with the pilot as one section.
"""
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

import nbformat as nbf

import paper_sections as ps
import real_pdf_sections as rp
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, Image, PageTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]  # repository root
r = json.loads((ROOT / 'pilot/results/run.json').read_text())
p = json.loads((ROOT / 'pilot/M3_Protocol.json').read_text())
rows = list(csv.DictReader((ROOT / 'pilot/results/predictions.csv').open()))
errors = [x for x in rows if x['encoding'] == 'original'
          and x['normalized_char_lr'] != x['label']]


def notebook(name, cells):
    n = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(s) if t == 'md'
                                  else nbf.v4.new_code_cell(s) for t, s in cells])
    n.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    nbf.write(n, ROOT / name)


notebook('M3_Research_Gap.ipynb', rp.m3_cells() + rp.as_pilot([
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
     'The numeric prediction was saved in [pilot/M3_Protocol.json](pilot/M3_Protocol.json) before the first training call. '
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
    ('code', "import json\nfrom pathlib import Path\nprotocol = json.loads(Path('pilot/M3_Protocol.json').read_text())\nprint(json.dumps(protocol, indent=2))"),
]))

notebook('M4_Controlled_Experiments.ipynb', rp.m4_cells() + rp.as_pilot([
    ('md', '# M4 — Controlled experiments\n\n'
     'Run `python pilot/experiment.py` from the repository root to reproduce training. '
     'This notebook reads the recorded run and checks its provenance without retraining. '
     'Full text, labels, group IDs and predictions are exported in [pilot/results/](pilot/results/).'),
    ('code', "import json, hashlib\nfrom pathlib import Path\nimport pandas as pd\nimport matplotlib.pyplot as plt\nPILOT = Path('pilot')  # the recorded file names are relative to this folder\nr = json.loads((PILOT / 'results/run.json').read_text())\nfor name, expected in r['input_sha256'].items():\n    assert hashlib.sha256((PILOT / name).read_bytes()).hexdigest() == expected, name\nprint('Input hashes verified; run:', r['started_utc'])\nprint('Environment:', r['environment'])\nprint('Source sentences:', r['split_sources'], 'Expanded rows:', r['split_rows'])"),
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
    ('code', "import sys; sys.path.insert(0, 'report')\nimport make_figures as figures   # the code that writes figures/, so this chart is the saved file\nfigures.show(figures.pilot_detection())\ndisplay(pd.DataFrame(r['original_only']).T)"),
    ('md', '## Interpretation\n\n'
     'The primary model gains 93.75 percentage points in recall but fails the FPR requirement (18.75%). '
     'Removing normalization decreases recall by 2.08 points and increases FPR by 2.08 points; this is only one test row each. '
     'The word model has 25% recall and no false positives at its selected threshold. '
     'Normalized keywords still miss every held-out attack and flag more benign quotations.\n\n'
     'Original-only results show the normalized detector catches 15/16 attacks and flags 3/16 benign sentences. '
     'The raw keyword baseline flags 4/16 clean originals (25%); its lower expanded-set FPR of 8.33% occurs because obfuscation hides the quoted keywords. '
     'This is not genuine semantic discrimination. The test deliberately stresses phrases outside the keyword list, so baseline failure does not estimate real-world baseline performance. '
     'Bootstrap resampling uses four attack families rather than 48 independent rows; the interval is descriptive and cannot establish population robustness.'),
]))

failure_table = '| Source ID | Truth | Prediction | Score | Text |\n|---|---|---|---:|---|\n' + '\n'.join(
    f"| {x['source_id']} | {x['label']} | {x['normalized_char_lr']} | {float(x['normalized_char_lr_score']):.3f} | {x['text']} |" for x in errors)
notebook('M5_Failure_Analysis.ipynb', rp.m5_cells() + rp.as_pilot([
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
    ('code', "import json\nimport pandas as pd\nfrom pathlib import Path\nr = json.loads(Path('pilot/results/run.json').read_text())\npred = pd.read_csv('pilot/results/predictions.csv')\ndisplay(pred.loc[pred.label != pred.normalized_char_lr, ['id','family','encoding','text','label','normalized_char_lr','normalized_char_lr_score']])\ndisplay(pd.DataFrame({f: v['normalized_char_lr'] for f,v in r['per_family'].items()}).T)"),
    ('md', '## Limitations and threats to validity\n\n'
     '- **Internal validity:** fixed splits and train-only vocabulary prevent direct leakage, but a single author designed data and models. Families are distinct, yet share vocabulary and intent. M2 includes historical wording also seen in one new test sentence, so there is no claim of blinded evaluation.\n'
     '- **Construct validity:** a label means an authored sentence attempts to steer an evaluator; no LLM is asked whether it succeeds. Some endorsements are ambiguous without document context. No independent annotator checked labels.\n'
     '- **External validity:** short English sentences, 50% attack prevalence, no full resumes, PDFs, OCR, multilingual documents, actual candidates, real hiring outcomes or adaptive optimized attacks.\n'
     '- **Statistical validity:** only 16 malicious and 16 benign source sentences in the test set. Variants are correlated, and four attack families make the bootstrap unstable. No significance claim is made.\n'
     '- **Comparator validity:** the fixed keyword list is intentionally challenged by held-out wording. It is not the anchor full defense; gains cannot be described as beating ResumeShield.\n'
     '- **Operational limitation:** the eight-character mapping is incomplete and can alter legitimate multilingual text. Scores are not calibrated probabilities of malicious intent in deployment.\n\n'
     '**Next experiment, not performed:** independently annotate a larger set with benign quotations and full document context, freeze a new test split, and evaluate a more conservative threshold on validation. '
     'Do not reuse this test set to claim confirmation. A detector flag should prompt review, not automatic rejection of a person.'),
]))

# ---------------------------------------------------------------- M6: paper and defence notebook
# The text lives in paper_sections.py. Here it is rendered twice: as IEEEtran LaTeX, compiled with
# Tectonic when it is installed, and otherwise as a two-column reading copy drawn with ReportLab.
paper = ROOT / 'paper'
paper.mkdir(exist_ok=True)
TEX_CHARS = [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#'), ('$', r'\$'),
             ('é', r"\'e"), ('“', '``'), ('”', "''"), ('’', "'"), ('–', '--'), ('→', r'$\rightarrow$')]


def tex_text(s):
    for a, b in TEX_CHARS:
        s = s.replace(a, b)
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    return re.sub(r'\[(\d+)\]', r'\\cite{r\1}', s)


def tex_inline(s):
    """Backticks mark file names and identifiers; they are set with \\path so they may break across lines."""
    return ''.join(tex_text(x) if i % 2 == 0 else r'\path{' + x + '}' for i, x in enumerate(s.split('`')))


def tex_table(t):
    out = r'\begin{table}[!t]\caption{' + tex_text(t['caption']) + '}\n' + r'\centering\footnotesize\setlength{\tabcolsep}{4pt}' + '\n'
    out += r'\begin{tabular}{' + t['align'] + r'}\hline' + '\n' + ' & '.join(tex_text(str(h)) for h in t['header']) + r'\\\hline' + '\n'
    for row in t['rows']:
        out += ' & '.join(tex_text(str(c)) for c in row) + r'\\' + '\n'
    return out + r'\hline\end{tabular}\end{table}' + '\n'


def tex_figure(f):
    star, width = ('*', r'\textwidth') if f['wide'] else ('', r'\columnwidth')
    return (r'\begin{figure' + star + r'}[!t]\centering' + '\n' + r'\includegraphics[width=' + width + ']{../figures/' + f['figure'] + '.pdf}\n'
            + r'\caption{' + tex_text(f['caption']) + r'}\end{figure' + star + '}\n')


def tex_item(x):
    return tex_inline(x) + '\n' if not isinstance(x, dict) else tex_figure(x) if 'figure' in x else tex_table(x)


missing = [x['figure'] for _, items in ps.SECTIONS for x in items if isinstance(x, dict) and 'figure' in x
           and not (ROOT / 'figures' / (x['figure'] + '.pdf')).exists()]
if missing:
    raise SystemExit('Figures missing: ' + ', '.join(missing) + '. Draw them first with: python report/make_figures.py')

tex = r'\documentclass[conference]{IEEEtran}' + '\n' + r'\usepackage[T1]{fontenc}\usepackage{url}\usepackage{graphicx}' + '\n'
tex += r'\title{' + tex_text(ps.TITLE) + '}\n'
tex += r'\author{' + '\n\\and '.join(
    r'\IEEEauthorblockN{' + tex_text(name) + r'}\IEEEauthorblockA{' + r'\\'.join(tex_text(line) for line in [f'Enrollment No. {number}'] + ps.AUTHOR_BLOCK) + '}'
    for name, number in ps.AUTHORS) + '}\n'
tex += r'\begin{document}\maketitle' + '\n' + r'\begin{abstract}' + tex_inline(ps.ABSTRACT) + r'\end{abstract}' + '\n'
tex += r'\begin{IEEEkeywords}' + tex_text(ps.KEYWORDS) + r'\end{IEEEkeywords}' + '\n'
for heading, items in ps.SECTIONS:
    tex += '\n\\section{' + tex_text(heading) + '}\n'
    tex += '\n'.join(tex_item(x) for x in items)
tex += '\n' + r'\begin{thebibliography}{00}' + '\n'
for i, (ref, url) in enumerate(ps.REFS):
    tex += r'\bibitem{r' + str(i + 1) + '} ' + tex_text(ref) + r' \url{' + url + '}\n'
tex += r'\end{thebibliography}' + '\n' + r'\end{document}' + '\n'
tex.encode('ascii')          # anything outside ASCII needs an entry in TEX_CHARS
(paper / 'M6_Final_Paper.tex').write_text(tex)

compiled = False
if shutil.which('tectonic'):
    run = subprocess.run(['tectonic', str(paper / 'M6_Final_Paper.tex')], capture_output=True, text=True, timeout=600)
    compiled = run.returncode == 0
    if not compiled:
        print('Tectonic failed; writing the ReportLab reading copy instead.\n' + run.stderr[-2000:])

if not compiled:
    def rl_inline(s):
        return ''.join(escape(x) if i % 2 == 0 else '<font face="Courier" size="8">' + escape(x) + '</font>' for i, x in enumerate(s.split('`')))

    body = ParagraphStyle('body', fontName='Times-Roman', fontSize=9, leading=11, alignment=TA_JUSTIFY, spaceAfter=6)
    head = ParagraphStyle('head', parent=body, fontName='Times-Bold', fontSize=10, spaceBefore=10, alignment=TA_CENTER)
    small = ParagraphStyle('small', parent=body, fontSize=8, leading=9)
    doc = BaseDocTemplate(str(paper / 'M6_Final_Paper.pdf'), pagesize=(612, 792), title=ps.TITLE, author=ps.AUTHOR_NAMES,
                          leftMargin=42, rightMargin=42, topMargin=40, bottomMargin=40)
    width = (612 - 84 - 18) / 2
    frames = [Frame(42, 40, width, 712, id='left'), Frame(42 + width + 18, 40, width, 712, id='right')]

    def footer(canvas, doc):
        canvas.setFont('Times-Roman', 8)
        canvas.drawCentredString(306, 22, f'{ps.AUTHOR_NAMES} | {doc.page} | IEEE-style reading copy')

    doc.addPageTemplates(PageTemplate(id='two_columns', frames=frames, onPage=footer))
    story = [Paragraph(escape(ps.TITLE), ParagraphStyle('title', parent=head, fontSize=14, leading=16)),
             Table([[Paragraph('<b>' + escape(name) + '</b><br/>' + '<br/>'.join(escape(line) for line in [f'Enrollment No. {number}'] + ps.AUTHOR_BLOCK),
                               ParagraphStyle('author', parent=small, alignment=TA_CENTER)) for name, number in ps.AUTHORS]], colWidths=[(width - 12) / 3] * 3),
             Spacer(1, 8), Paragraph('<b>Abstract—</b>' + rl_inline(ps.ABSTRACT), body),
             Paragraph('<b>Keywords—</b>' + escape(ps.KEYWORDS), small)]
    roman, tables, figures = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'], 0, 0
    for number, (heading, items) in enumerate(ps.SECTIONS):
        story.append(Paragraph(escape(f'{roman[number]}. {heading}'), head))
        for x in items:
            if not isinstance(x, dict):
                story.append(Paragraph(rl_inline(x), body))
                continue
            if 'figure' in x:
                figures += 1
                picture = Image(str(ROOT / 'figures' / (x['figure'] + '.png')))
                picture.drawHeight, picture.drawWidth = (width - 12) * picture.imageHeight / picture.imageWidth, width - 12
                story.extend([picture, Paragraph(escape(f'Fig. {figures}. {x["caption"]}'), small), Spacer(1, 6)])
                continue
            story.append(Paragraph(escape(f'TABLE {roman[tables]}. {x["caption"]}'), small))
            tables += 1
            first = max(len(str(row[0])) for row in x['rows']) * 4.2 + 8
            tab = Table([x['header']] + [[str(c) for c in row] for row in x['rows']],
                        colWidths=[first] + [(width - 12 - first) / (len(x['header']) - 1)] * (len(x['header']) - 1), repeatRows=1)
            tab.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, -1), 'Times-Roman'), ('FONTSIZE', (0, 0), (-1, -1), 8),
                                     ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                                     ('LINEABOVE', (0, 0), (-1, 0), .5, colors.black), ('LINEBELOW', (0, 0), (-1, 0), .5, colors.black),
                                     ('LINEBELOW', (0, -1), (-1, -1), .5, colors.black)]))
            story.extend([tab, Spacer(1, 6)])
    story.append(Paragraph('References', head))
    story.extend(Paragraph(escape(f'[{i + 1}] {ref} {url}'), small) for i, (ref, url) in enumerate(ps.REFS))
    doc.build(story)

notebook('M6_Defence.ipynb', ps.m6_cells(compiled))
# M1 is hand-written prose; only its scope banner is kept in step with the results.
m1 = nbf.read(ROOT / 'M1_Literature_Survey.ipynb', as_version=4)
for cell in m1.cells:
    if cell.cell_type == 'markdown' and cell.source.startswith('## Current scope'):
        cell.source = rp.BANNER
nbf.write(m1, ROOT / 'M1_Literature_Survey.ipynb')
print('Built M3–M6 notebooks and the IEEEtran source; paper PDF ' + ('compiled with Tectonic' if compiled else 'written as a ReportLab reading copy')
      + '; refreshed the M1 scope banner.')
