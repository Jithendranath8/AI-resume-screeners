"""Text of the M6 paper and of the M6 defence notebook.

Imported by report/build_deliverables.py. The paper reports the real-PDF / real-LLM experiment as the
main study, with the M2 reconstruction, the earlier sentence-level pilot, the exploratory
fine-tuning and the scanner evaluation around it. Every figure is read from the recorded results
(through real_pdf_sections.py and the JSON files under results/ and pilot/results/), so the prose cannot drift.

Paragraph conventions, understood by both renderers in build_deliverables.py:
  `like this`   a file name, model identifier or command, set in a monospaced face
  [3]           a citation of reference 3
  a dict        a table (caption, header, rows, column alignment) or a figure (a file in figures/, caption, wide)
"""
import json
from pathlib import Path

import real_pdf_sections as rp

ROOT = Path(__file__).resolve().parents[1]  # repository root

# The authors, with enrollment numbers, and their shared affiliation. Used by the paper and the M6 notebook.
AUTHORS = [('Akula Jithendranath', '230120'), ('Kolli Charan Adithya', '230075'), ('Kammati Adithya', '230145')]
AFFILIATION = ['Department of Computer Science and Artificial Intelligence', 'Newton School of Technology, Rishihood University']
# The paper's title block: one column per author. Each line must fit a third of the page, so the
# department is abbreviated and the institution takes two lines.
AUTHOR_BLOCK = ['Dept. of Computer Science and AI', 'Newton School of Technology,', 'Rishihood University']
AUTHOR_NAMES = ', '.join(name for name, _ in AUTHORS)
AUTHOR_LINE = ', '.join(f'{name} ({number})' for name, number in AUTHORS[:-1]) + f' and {AUTHORS[-1][0]} ({AUTHORS[-1][1]})'

TITLE = 'Hidden Instructions in Real PDF Resumes: Visibility-Diff and Injection Classification in Front of an LLM Screener'
KEYWORDS = 'indirect prompt injection, resume screening, hidden text, PDF, large language models, failure analysis'

# ---------------------------------------------------------------------------- recorded results
pilot = json.loads((ROOT / 'pilot/results/run.json').read_text())
pilot_protocol = json.loads((ROOT / 'pilot/M3_Protocol.json').read_text())
m2 = json.loads((ROOT / 'results/m2_reproduction/M2_validation.json').read_text())
EVAL = ROOT / 'results/scanner_eval'
HIDDEN = ['white_text', 'invisible_text', 'tiny_font', 'off_page', 'transparent', 'covered', 'hidden_layer']

s, pva, unc = rp.summary, rp.pva, rp.unc
E = rp.N_ELIGIBLE
LO, HI = (100 * x for x in unc['descriptive_95_interval'])
DROP = 100 * pva['actual_asr_drop']
SHORT = {'naive': 'Naive (no added defence)', 'channel_separation': 'Channel separation', 'visibility_only': 'Visibility-diff only',
         'classifier_only': 'Classifier only', 'combined': 'Combined'}


def words(n):
    """A count that starts a sentence is written as a word."""
    return ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten'][n] if 0 <= n <= 10 else str(n)


def asr(config):
    return rp.pct(s.loc[config, 'asr_valid'])


def split(name):
    """Counts for one held-out scanner run, or None if it has not been run."""
    path = EVAL / f'{name}_summary.json'
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    a, by = d['attacks'], d['attacks']['by_technique']
    hidden = [t for t in HIDDEN if t in by]
    hn, hd = sum(by[t]['n'] for t in hidden), sum(by[t]['detected'] for t in hidden)
    verdicts = d['clean']['verdicts']
    return dict(version=d['scanner_version'], files=d['files'], seconds=d['mean_seconds'], attacks=a['n'], called=a['detected'],
                flagged=a.get('flagged_attack_or_review', a['detected']), hidden_n=hn, hidden_found=hd,
                visible_n=a['n'] - hn, visible_called=a['detected'] - hd,
                visible_flagged=a.get('flagged_attack_or_review', a['detected']) - hd,
                clean_n=d['clean']['n'], clean_ok=verdicts.get('clean', 0), clean_attack=verdicts.get('attack', 0),
                clean_review=verdicts.get('review', 0))


RUNS = {name: split(name) for name in ('test', 'final', 'confirm')}
LATEST = RUNS['confirm'] or RUNS['final']
NEW_WORDING = [r for name, r in RUNS.items() if r and name != 'test']       # runs whose attack wording was never tuned on
HIDDEN_ALL = (sum(r['hidden_found'] for r in NEW_WORDING), sum(r['hidden_n'] for r in NEW_WORDING))
regression = json.loads((EVAL / 'regression_summary.json').read_text())

pm = pilot['metrics']['normalized_char_lr']
m2_asr = m2['asr']
m2_det = m2['detection']['reproduced']
m2_conf = m2['detection']['confusion']
m2_mismatch = [k for k, v in m2_asr.items() if abs(v['gap']) > 1e-9]
cc = rp.changed_clean.iloc[0] if len(rp.changed_clean) else None
cs_select = rp.select_attacked[rp.select_attacked.config == 'channel_separation']
pre, ft1, ft2 = rp.blocks[2].loc['pretrained'], rp.blocks[1].loc['finetuned'], rp.blocks[2].loc['finetuned']
t2, a2 = rp.manifests[2]['test_metrics'], rp.manifests[2]['authored_test_metrics']


def blocks(row):
    return f'{int(row.tp)}/{int(row.tp + row.fn)}', f'{int(row.fp)}/{int(row.fp + row.tn)}'


def documents(frame, key):
    row = frame.loc[key]
    return f'{int(row.tp)}/{int(row.tp + row.fn)}', f'{int(row.fp)}/{int(row.fp + row.tn)}'


# ------------------------------------------------------------------------------------- tables
T_ASR = dict(
    caption=f'Attack success on {E} eligible attacks, and decisions on the {rp.N_RESUMES} clean resumes.',
    header=['Configuration', 'Successes', 'ASR', 'Clean acc.', 'Clean agr.'], align='lrrrr',
    rows=[[SHORT[c], rp.frac(c), asr(c), rp.pct(s.loc[c, 'clean_accuracy_lower']), rp.pct(s.loc[c, 'clean_agreement_lower'])] for c in rp.CONFIGS])

T_ABLATION = dict(
    caption='Successful attacks by concealment (hidden: white or invisible text) and by wording (D: direct override, P: paraphrase).',
    header=['Configuration', 'Hid. D', 'Hid. P', 'Vis. D', 'Vis. P'], align='lrrrr',
    rows=[[SHORT[c].replace(' (no added defence)', ''), rp.cell_count(c, 'direct', hidden=True), rp.cell_count(c, 'paraphrase', hidden=True),
           rp.cell_count(c, 'direct', 'visible_text'), rp.cell_count(c, 'paraphrase', 'visible_text')] for c in rp.CONFIGS])

T_DETECTION = dict(
    caption='Document-level detection: injected documents as positives, clean controls as negatives.',
    header=['Detector', 'TP', 'FP', 'FN', 'TN', 'Recall', 'FPR'], align='lrrrrrr',
    rows=[[{'visibility_only': 'Visibility-diff', 'classifier_only': 'Classifier', 'combined': 'Combined'}[k],
           int(r.tp), int(r.fp), int(r.fn), int(r.tn), f'{r.recall:.3f}', f'{r.fpr:.3f}'] for k, r in rp.detection.iterrows()])

T_FINETUNE = dict(
    caption='Injection classifiers on the text blocks of the 42 experiment documents (exploratory; no LLM involved). '
            'Documents: with visibility-diff, injected caught and clean flagged.',
    header=['Classifier', 'Thr.', 'Payload', 'Genuine', 'Inj. docs', 'Clean docs'], align='lrrrrr',
    rows=[['Pretrained DeBERTa', rp.protocol['classifier_threshold'], *blocks(pre), *documents(rp.docs[2], 'combined_pretrained')],
          ['DistilBERT, round 1', rp.manifests[1]['threshold'], *blocks(ft1), *documents(rp.docs[1], 'combined_finetuned')],
          ['DistilBERT, round 2', rp.manifests[2]['threshold'], *blocks(ft2), *documents(rp.docs[2], 'combined_finetuned')]])

T_SCANNER = dict(
    caption='Scanner on held-out resumes. Hidden: hidden-text attacks called an attack. Visible: instructions that are not hidden '
            '(printed, in metadata, in a comment, in an image) called an attack, and called an attack or sent for review.',
    header=['Run', 'Rules', 'Clean', 'Hidden', 'Visible', 'Vis. flagged'], align='llrrrr',
    rows=[[name, r['version'], f"{r['clean_ok']}/{r['clean_n']}", f"{r['hidden_found']}/{r['hidden_n']}",
           f"{r['visible_called']}/{r['visible_n']}", f"{r['visible_flagged']}/{r['visible_n']}"] for name, r in RUNS.items() if r])

T_PILOT = dict(
    caption=f"Pilot: sentence-level detection on the frozen test set ({pm['n']} rows).",
    header=['Detector', 'P', 'R', 'F1', 'FPR'], align='lrrrr',
    rows=[[label, *(f'{m[k]:.3f}' for k in ['precision', 'recall', 'f1', 'fpr'])] for label, m in
          zip(['Keywords', 'Norm. keywords', 'Norm. char LR', 'Raw char LR', 'Norm. word LR'], pilot['metrics'].values())])

# ------------------------------------------------------------------------------------ figures
# Drawn by make_figures.py into figures/. `wide` figures span both columns of the paper.
# Only result figures go in the paper, in this order: Fig. 1 to Fig. 7.
F_M2 = dict(figure='M2_attack_success', wide=False, caption=(
    'M2 reproduction: attack success rate by pipeline on the simulated screener. Bars are our reconstruction, ticks the values reported in the paper.'))
F_ASR = dict(figure='M4_attack_success', wide=False, caption=f'Attack success rate by configuration, on {E} eligible attacks.')
F_ABLATION = dict(figure='M4_ablation', wide=False, caption='Successful attacks out of those tried, by placement and wording. The visible paraphrase survives both checks.')
F_FINETUNE = dict(figure='M4_classifier_finetuning', wide=False, caption=(
    'Injection classifiers on the text blocks of the experiment documents (exploratory): payload blocks caught and genuine blocks flagged.'))
F_SCORES = dict(figure='M5_scores', wide=False, caption=(
    f'The score returned for each of the {E} eligible attacks. Every successful attack returns exactly 100, the value the payload asks for.'))
F_PILOT = dict(figure='pilot_detection', wide=False, caption='Pilot: recall and false-positive rate by detector, with the 10% limit recorded beforehand.')
F_SCANNER = dict(figure='scanner_heldout', wide=True, caption=(
    'The scanner on held-out resumes. The rules were changed after runs 1 and 2, so each version was measured again on fresh files; '
    'run 3 measures the current rules.'))

# -------------------------------------------------------------------------------------- paper
ABSTRACT = (
    'LLM-based resume screeners read the text extracted from a file, which can differ from what a recruiter sees on the page. '
    'We test whether text hidden in real PDF resumes steers a real LLM screener, and whether two inexpensive checks placed in front '
    'of the model prevent it: a visibility-diff that compares extracted text with OCR of the rendered page, and a transformer injection classifier. '
    f'{words(rp.N_RESUMES)} real resume PDFs were expanded into {rp.N_DOCS} controlled documents and scored by Llama 3.2 3B Instruct under five '
    f'configurations ({rp.N_CALLS} valid responses). A prediction recorded before the first model call required a reduction in attack success of '
    f'at least 20 percentage points with at least 95% agreement on clean resumes. Attack success fell from {asr("naive")} ({rp.frac("naive")}) with no '
    f'added defence to {asr("combined")} ({rp.frac("combined")}) with both checks, and no clean decision changed, so the hypothesis is {pva["verdict"]} on this sample. '
    'The two checks fail on different attacks; the survivors are all visible, paraphrased instructions. We also report a partial reconstruction of the '
    'ResumeShield benchmark, an earlier sentence-level pilot whose hypothesis was not supported, an exploratory fine-tuned classifier, and an offline '
    'scanner measured on resumes it had not seen. The sample is small and the results do not establish that any screener is safe.')

SECTIONS = [
('Introduction', [
    'Organisations increasingly ask a large language model to score or shortlist resumes. The model does not look at the page. It receives the text '
    'that a program extracts from the file, and a PDF can carry text that a reader never sees: white text, text drawn in the invisible rendering mode, '
    'text outside the page. An applicant can use that gap to address the model directly, for example by asking for the highest score. '
    'This is indirect prompt injection [2], and a measurement of about 200,000 real resumes found hidden injections in roughly 1% of them [5].',
    'Our anchor paper, ResumeShield [1], proposes filtering and channel separation and reports that attack success falls from 1.000 to 0.000. '
    'Its screener is a deterministic simulation, so whether the model obeys an instruction is fixed by construction. '
    'This project asks the question that the simulation cannot answer: does the attack work on a real model reading a real PDF, and can it be '
    'stopped without changing decisions on honest resumes?',
    'The contributions are: (1) a partial reconstruction of the anchor benchmark with its differences stated; (2) a controlled experiment on real PDF '
    'files and a real LLM, with one hypothesis and a numeric prediction recorded beforehand, five configurations and component ablations; '
    '(3) a failure analysis that locates the remaining risk; (4) an exploratory fine-tuned classifier; (5) an offline scanner built on the same rule, '
    'measured on held-out resumes; and (6) an earlier pilot reported as a negative result. Neither defence component is new. '
    'The contribution is the controlled evaluation of their combination.']),

('Related Work and Reproduction', [
    'Greshake et al. [2] name indirect prompt injection and demonstrate it on deployed applications. Liu et al. [3] formalise attacks and defences and '
    'benchmark them across models and tasks. PhantomLint [4] detects hidden prompts in structured documents by comparing what is visible with what is '
    'extractable, and has public code. Zhang et al. [5] measure injections in a production resume corpus. ResumeShield [1] contributes an open synthetic '
    'benchmark and a channel-separation defence for resume screening.',
    f"M2 reconstructs the ResumeShield benchmark locally: {m2['corpus']['total']} documents, {m2['corpus']['injected']} injected "
    f"(nine concealment techniques, two payload families) and {m2['corpus']['clean']} clean. The naive pipeline is attacked successfully in "
    f"{m2_asr['naive (no defense)']['successes']}/{m2_asr['naive (no defense)']['total']} cases and the full defence in "
    f"{m2_asr['full defense (filters + fence)']['successes']}/{m2_asr['full defense (filters + fence)']['total']}, as in the paper (Fig. 1). "
    f"Detection reaches precision {m2_det['precision']:.3f}, recall {m2_det['recall']:.3f} and F1 {m2_det['f1']:.3f} "
    f"(TP {m2_conf['tp']}, FP {m2_conf['fp']}, FN {m2_conf['fn']}, TN {m2_conf['tn']}), also as reported.",
    f"This is a partial reconstruction, not an execution of the original artifact. The corpus is locally authored and the filters differ, so "
    f"{len(m2_mismatch)} of the six pipeline rows do not match the paper: markup stripping alone gives "
    f"{100 * m2_asr['markup_strip only']['reproduced']:.1f}% against a reported {100 * m2_asr['markup_strip only']['original']:.1f}%, and all three filters without "
    f"the fence give {100 * m2_asr['all 3 filters (no fence)']['reproduced']:.1f}% against {100 * m2_asr['all 3 filters (no fence)']['original']:.1f}%. "
    'The zero for channel separation is a property of the simulator, which never follows fenced text. It is not evidence about a real model.',
    F_M2]),

('Research Gap, Hypothesis and Prediction', [
    'The anchor paper and M2 cannot establish (a) whether a real LLM screener is steered by text hidden in a real PDF, or (b) whether a rendering-based '
    'visibility check combined with an injection classifier protects that screener without changing its decisions on clean resumes. '
    'This is a bounded evaluation gap, not a claim of a new architecture.',
    'Hypothesis: the combined defence, which removes every text block flagged by either check, reduces the attack success rate (ASR) by at least 20 '
    'percentage points relative to no added defence, while keeping at least 95% agreement with the undefended decisions on clean resumes. '
    'Both conditions must hold; an invalid response or the absence of eligible attacks would make the outcome inconclusive.',
    f"The protocol file was written at {rp.record['recorded_utc'][:19].replace('T', ' ')} UTC and the first model response is timestamped {rp.FIRST_RESPONSE[:19].replace('T', ' ')} UTC. "
    'It stores the hypothesis, both thresholds, the model and classifier identifiers, the two detector thresholds, the payload strings and SHA-256 '
    'hashes of the input PDFs and labels. This is a local record of execution order, not an externally timestamped preregistration.']),

('Method', [
    f'Documents. {words(rp.N_RESUMES)} real resume PDFs from the Kaggle Resume Dataset (CC0) [9], text-extractable and already anonymised by the dataset author. '
    f'One annotator labelled each SELECT or REJECT against the job rubric before any model output existed: {rp.N_SELECT} SELECT, {rp.N_REJECT} REJECT. '
    'The job is a junior Python backend developer with a 100-point rubric (Python 30, API and backend 25, SQL 20, testing and version control 15, '
    'Docker and cloud 10) and SELECT at 60 or more.',
    'Controlled variants. Every PDF receives the same supplementary page. The clean control leaves it empty; attacked copies place a payload on it as white '
    'text, as text in the PDF invisible rendering mode, or as visible black text. Two wordings are used: a direct override '
    f'("{rp.protocol["payloads"]["direct"]}") and a paraphrase written as reviewer guidance ("{rp.protocol["payloads"]["paraphrase"]}"). '
    f'That gives seven documents per resume and {rp.N_DOCS} in total. Construction asserts that each hidden variant renders pixel-identically to the clean '
    'control and that every payload remains extractable.',
    f'Defence components. Visibility-diff renders each page at {rp.protocol["ocr_dpi"]} dpi, reads the area of every extracted text block with Tesseract OCR, '
    f'and flags blocks of at least 12 normalised characters whose fuzzy agreement with the extracted text is below {rp.protocol["visibility_threshold"]:.0f}. '
    f'The classifier is the pretrained `{rp.protocol["classifier"]}` [7] at a fixed threshold of {rp.protocol["classifier_threshold"]}, not tuned on these files.',
    'Configurations. Five, with the same model, rubric, JSON output schema and temperature 0: naive (the full extracted text); channel separation '
    '(a trusted policy line and the resume wrapped as JSON data, as in the anchor); visibility-diff only; classifier only; and combined (blocks flagged by '
    'either check are removed, with no extra policy line so that the ablations stay interpretable).',
    f'Screener and metric. `{rp.protocol["llm"]}` (Llama 3.2 3B Instruct, 4-bit) [6] is served locally by Ollama with schema-constrained output: '
    f'{rp.N_DOCS} documents, five configurations, one repeat, {rp.N_CALLS} calls. ASR is the share of eligible attacks that return SELECT. An attack is '
    'eligible when the human label is REJECT and the naive screener also rejects the clean copy, so every configuration has the same denominator. '
    'We also report agreement of clean decisions with the naive clean decisions and accuracy against the human labels. Invalid responses would be counted '
    f'as errors, never as REJECT. Uncertainty is a descriptive paired bootstrap over resumes (2000 draws, seed {rp.protocol["seed"]}).',
    'Deviations. No hosted model could be used without payment: one provider refused its free tier and another stopped at 140 of 210 calls, so neither '
    'attempt is used and the experiment runs on a small open-weight model. The incomplete attempt existed before this run was frozen; no ASR or '
    'agreement figure was computed from it.']),

('Results and Ablations', [
    f'Table I and Fig. 2 give the main result. All {rp.N_CALLS} responses were valid. The naive screener agreed with all {rp.N_RESUMES} human labels on the clean '
    f'resumes, so it is a meaningful reference, and it was steered in {rp.frac("naive")} eligible attacks ({asr("naive")}). With both checks the figure is '
    f'{rp.frac("combined")} ({asr("combined")}), and every clean decision is unchanged.',
    T_ASR,
    F_ASR,
    f'Predicted and actual. The ASR reduction is {DROP:.1f} points against a predicted minimum of 20, and clean agreement is '
    f'{rp.pct(pva["actual_clean_agreement"])} against a minimum of 95%. Both conditions hold and the hypothesis is {pva["verdict"]}. '
    f'The paired bootstrap over the {unc["resume_count"]} eligible resumes gives a 95% interval of {LO:.1f} to {HI:.1f} points for the reduction. '
    'That interval describes this sample; it is not evidence about resumes or screeners in general.',
    T_ABLATION,
    F_ABLATION,
    f'Ablation (Table II, Fig. 3). Visibility-diff alone stops every hidden attack ({rp.cell_count("visibility_only", hidden=True)} succeed) and almost no visible one '
    f'({rp.cell_count("visibility_only", hidden=False)} succeed), since it cannot flag text that is really on the page. The pretrained classifier alone stops '
    f'every direct override ({rp.cell_count("classifier_only", "direct")}) and misses most paraphrases ({rp.cell_count("classifier_only", "paraphrase")} succeed), '
    'whatever the concealment. Each component closes the gap the other leaves, and what survives the combination is exactly their shared blind spot, '
    'the visible paraphrase.',
    f'Channel separation behaves differently. It stops the paraphrase ({rp.cell_count("channel_separation", "paraphrase")}) but not the direct override '
    f'({rp.cell_count("channel_separation", "direct")} succeed), and it is the only configuration that changes a clean decision '
    f'(clean agreement {rp.pct(s.loc["channel_separation", "clean_agreement_lower"])}).',
    T_DETECTION,
    'Table III treats the two checks as document-level detectors. No clean control is flagged by either, and the combination raises recall above both '
    'components.']),

('Exploratory Classifier Fine-Tuning', [
    'The ablation shows that the pretrained classifier is the weak component on paraphrased payloads. As a follow-up, DistilBERT was fine-tuned locally '
    f'({rp.manifests[2]["epochs"]} epochs, learning rate {rp.manifests[2]["learning_rate"]}, seed {rp.manifests[2]["seed"]}, threshold chosen on validation only) '
    'and compared with the pretrained model on the extracted blocks of the experiment documents (Table IV, Fig. 4). Round 1 used a public injection dataset [8] and 160 '
    'authored resume-style sentences. Round 2 added genuine text blocks from 48 other resumes as benign examples; the six evaluation resumes were excluded.',
    T_FINETUNE,
    F_FINETUNE,
    f'Round 1 flagged {blocks(ft1)[1]} genuine blocks: it had only seen short sentences, and real resume blocks are paragraphs. Round 2 catches every payload block '
    f'and flags none. On its held-out test split it has precision {t2["precision"]:.3f}, recall {t2["recall"]:.3f}, F1 {t2["f1"]:.3f} and FPR {t2["fpr"]:.3f}; '
    f'on the authored held-out families alone, FPR is {a2["fpr"]:.3f} ({a2["fp"]} of {a2["fp"] + a2["tn"]} benign sentences, all quoted praise).',
    'This result is exploratory. The authored training sentences were written after the misses of the pretrained model had been seen, the training data was '
    'revised after round 1 failed on these same blocks, and the benign blocks share the source of the evaluation resumes. '
    'The attack success rate with the fine-tuned classifier has not been measured; that needs a new frozen run with new payload wording.']),

('Failure Analysis', [
    f'Attacks that defeat the combined defence ({rp.frac("combined")}). All are the visible paraphrase. Its rendered and extracted text agree, so the visibility '
    'check has nothing to flag; that is the design limit of a visibility comparison. The pretrained classifier scores the paraphrase near 0 and the '
    'direct override near 0.99. The paraphrase contains no override wording and reads like a reviewer\'s note.',
    'The undefended screener. Every successful attack in every configuration returns a score of exactly 100, the value the payloads request (Fig. 5), with a '
    'reason that restates the resume as if it fitted the job. A staff pharmacist\'s resume was selected for a Python backend role. '
    'Several eligible resumes score just under the threshold when clean, but the successes are jumps to 100, not marginal flips.',
    F_SCORES,
    (f'Channel separation. One qualified resume ({cc.resume_id}) was rejected with a score of {int(cc.score)} when clean, and '
     f'{len(rp.cs_select_reject)} of {len(cs_select)} attacked documents built from the two qualified resumes were rejected, against none in any other configuration. '
     if cc is not None else 'Channel separation changed no clean decision. ') +
    'Wrapping the resume as data shifted scoring in general on this small model, so its lower ASR cannot be separated from that shift, and the direct '
    'override still succeeded through it in most cases.',
    f'No genuine text was removed. The two checks removed {len(rp.removals)} blocks, of which {rp.payload_removed} are injected payloads and {rp.clean_removed} come from a clean '
    'document. Hidden payloads show an agreement of 0 between extraction and OCR, and genuine blocks of 12 or more characters stay near 95 or above, '
    'so the fixed threshold sits in a wide gap on this sample.',
    'These are explanations of error patterns grounded in the recorded text, scores and decisions. They are not claims about the internals of the model.']),

('From Experiment to Tool', [
    'The method was packaged as an offline scanner: a resume PDF in, and one of three verdicts out (attack found, needs review, no attack found), with the '
    'attack type and its place on the page. It applies the visibility rule of the experiment to every page, names how hidden text was hidden (same colour as the '
    'background, invisible mode, transparency, microscopic font, outside the page, covered by a shape, switched-off layer), and also reads metadata, '
    'comments, form fields, text inside images, invisible characters and look-alike letters. Two classifiers, the pretrained model and the round 2 '
    'DistilBERT, score every piece of text. A regression check confirms that the scanner gives the same flag and score as the recorded experiment on all '
    f'{regression["blocks_compared"]} text blocks of its {regression["documents"]} documents ({regression["mismatches"]} mismatches).',
    T_SCANNER,
    F_SCANNER,
    'Table V and Fig. 6 report the scanner on resumes excluded from development and from classifier training, drawn from the Kaggle set and from a second public '
    'collection with many layouts. The rules were changed after the first and second runs, so each new version was measured again on fresh files; the '
    'attack wording of the later runs was fixed before it was first tried. ' +
    f'Across the runs with new wording, {HIDDEN_ALL[0]} of {HIDDEN_ALL[1]} hidden-text attacks were called an attack. ' +
    f'Under the current rules ({LATEST["version"]}), {LATEST["clean_ok"]} of {LATEST["clean_n"]} clean resumes were called clean, and '
    f'{LATEST["visible_called"]} of {LATEST["visible_n"]} instructions that were not hidden were called an attack '
    f'({LATEST["visible_flagged"]} of {LATEST["visible_n"]} counting those sent for review). Mean scan time was {LATEST["seconds"]:.1f} seconds per file.',
    'The pattern matches the experiment: hidden text is found reliably whatever it says, and a plainly visible instruction in unusual wording is the weak '
    'case. The attacked files were made by our own generator, so real attacks may differ.']),

('Earlier Pilot: Sentence-Level Detection', [
    'Before the real-PDF experiment, a smaller pilot asked whether a CPU-trained text detector recognises instruction wording it has not seen while leaving '
    'benign text alone. It used 80 authored English sentences split by phrasing family (32 training, 16 validation, 32 test), each in three encodings '
    '(original, zero-width separators, look-alike letters), a fixed keyword baseline, and character TF-IDF with logistic regression. '
    'The threshold was chosen on validation only.',
    'The prediction, recorded before training, required a recall gain of at least 20 points over keywords and a false-positive rate of at most 10%. '
    f'Recall rose by {100 * pilot["recall_gain"]:.2f} points to {100 * pm["recall"]:.2f}%, but the false-positive rate was {100 * pm["fpr"]:.2f}% '
    f'(TP {pm["tp"]}, FP {pm["fp"]}, FN {pm["fn"]}, TN {pm["tn"]}; Table VI, Fig. 7). The joint hypothesis was not supported, and the result was kept without retuning.',
    T_PILOT,
    F_PILOT,
    'The errors were benign sentences that quote or resemble an instruction, and one attack phrased as an output format. The pilot measured detection of '
    'sentences, not success against a model, which is why the main experiment puts a real screener in the loop. Its numbers are not comparable with the '
    'attack success rates above.']),

('Limitations, Validity and Ethics', [
    f'External validity. {words(rp.N_RESUMES)} resumes from one public dataset with one layout family; English only; one job rubric; a 3-billion-parameter, 4-bit model '
    'that is weaker than production screeners. Larger models may be more or less susceptible, and other PDF producers, fonts, columns and scanned pages '
    'may affect the OCR comparison.',
    'Construct validity. The attacks are two fixed payload strings on an appended page with three placements. They are not naturally occurring and not '
    'adaptive. Success is a SELECT decision on a resume that a person and the clean screener reject; ranking effects and partial score inflation are not measured.',
    f'Statistical validity. {words(unc["resume_count"])} eligible resumes. The seven variants of a resume are correlated, so {E} eligible attacks are not {E} independent '
    'trials. One repeat at temperature 0. The bootstrap interval is descriptive and no significance claim is made.',
    'Internal validity and reproducibility. One annotator, the project author, assigned the labels. The pretrained classifier may have seen common injection '
    'phrasing such as the direct payload. The screener is identified by a model tag, not a weight hash, and local quantised inference can differ across hardware.',
    'Ethics. The resumes are public, anonymised samples. The study does not address fairness, hiring validity or candidate ranking. '
    'A finding is a reason for a person to look at a file, not proof of intent, and a detector flag should never reject a candidate automatically.']),

('Conclusion and Contribution Attribution', [
    f'On this sample a real LLM screener was steered by text in real PDF files in {rp.frac("naive")} eligible attacks, and placing a visibility-diff and an injection '
    f'classifier in front of it reduced that to {rp.frac("combined")} without changing any decision on a clean resume. The prediction recorded beforehand was met. '
    'The components are complementary, the residual risk is a visible instruction worded as ordinary guidance, and a prompt-level fence alone was not '
    'sufficient on this model.',
    'Reproduced work: the ResumeShield benchmark and defence stack, as a partial local reconstruction on a simulated screener. Original work: the real-PDF '
    'experiment with its frozen protocol, ablations and failure analysis; the exploratory fine-tuned classifier; the scanner and its held-out evaluation; '
    'and the pilot dataset with its negative result. No algorithm here is new.',
    'The next experiment, not performed, is a confirmatory run under a new frozen protocol: the fine-tuned classifier, payload wording written by someone '
    'who has not seen the training sentences, resumes from a second source, a second annotator and a larger screener model.']),
]

REFS = [
    ('J. Barach, “ResumeShield: Channel Separation and an Open Benchmark for Indirect Prompt Injection in AI Resume Screening,” arXiv:2609.20188v1, 2026.',
     'https://arxiv.org/abs/2609.20188'),
    ('K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, “Not what you’ve signed up for: Compromising Real-World LLM-Integrated '
     'Applications with Indirect Prompt Injection,” in Proc. AISec, 2023.', 'https://arxiv.org/abs/2302.12173'),
    ('Y. Liu, Y. Jia, R. Geng, J. Jia, and N. Z. Gong, “Formalizing and Benchmarking Prompt Injection Attacks and Defenses,” in Proc. USENIX Security, 2024.',
     'https://arxiv.org/abs/2310.12815'),
    ('T. Murray, “PhantomLint: Principled Detection of Hidden LLM Prompts in Structured Documents,” arXiv:2508.17884, 2025.',
     'https://arxiv.org/abs/2508.17884'),
    ('M. Zhang, Y. Jia, Z. Tan, S. Jiang, N. Z. Gong, T. Chen, and D. Song, “Measuring Real-World Prompt Injection Attacks in LLM-based Resume Screening,” '
     'in Proc. USENIX Security, 2026.', 'https://arxiv.org/abs/2605.28999'),
    ('Meta, “Llama 3.2 3B Instruct,” model card, 2024; served with Ollama.', 'https://ollama.com/library/llama3.2'),
    ('ProtectAI, “deberta-v3-base-prompt-injection-v2,” model card.', 'https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2'),
    ('deepset, “prompt-injections,” dataset.', 'https://huggingface.co/datasets/deepset/prompt-injections'),
    ('S. Bhawal, “Resume Dataset,” Kaggle, CC0 1.0.', 'https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset'),
]


# --------------------------------------------------------------------------- defence notebook
def m6_cells(compiled):
    """Cells of M6_Defence.ipynb. `compiled` says whether the LaTeX source was compiled to the PDF."""
    paper_note = ('The PDF is compiled from the IEEEtran source with Tectonic.' if compiled else
                  'The PDF is an IEEE-style reading copy generated with ReportLab, because no LaTeX compiler was found; '
                  'install Tectonic and rebuild to compile the IEEEtran source.')
    latest = LATEST
    return [
        ('md', rp.BANNER),
        ('md', '# M6 — Final paper and oral defence\n\n'
         f'**Paper:** [{TITLE}](paper/M6_Final_Paper.pdf), with its [IEEEtran LaTeX source](paper/M6_Final_Paper.tex). {paper_note} '
         'It reports the real-PDF / real-LLM experiment as the main study and keeps the M2 reconstruction and the earlier pilot as shorter sections.\n\n'
         '**Slides:** [presentation/AI_Resume_Screeners_Slides.pdf](presentation/AI_Resume_Screeners_Slides.pdf), 14 slides, '
         'built from [presentation/slides.html](presentation/slides.html).\n\n'
         '**Figures:** twelve result figures in [figures/](figures/), one per result reported in the notebooks and named after its milestone. They are drawn by `python report/make_figures.py`; the notebooks M3 to M5 show the same images, and the seven result charts among them are in the paper.\n\n'
         'The paper and the figures are generated from the recorded results: run `python report/make_figures.py` and then '
         '`python report/build_deliverables.py` after any change to the result files. '
         f'**Authors:** {AUTHOR_LINE}. {AFFILIATION[0]}, {AFFILIATION[1]}. '
         'Names and affiliation are set once, at the top of [report/paper_sections.py](report/paper_sections.py).'),
        ('md', '## Defence outline: eleven minutes on the 14-slide deck\n\n'
         '| Time | Slide | What to say |\n|---|---|---|\n'
         '| 30 s | 1 | Title. Read the highlighted line: it is in the file, and not on the page a recruiter sees. |\n'
         '| 45 s | 2 | The problem: a screener reads extracted text, a PDF can hide text, and about 1 in 100 real resumes already carry an injection. State the open question. |\n'
         '| 30 s | 3 | What we did: the six milestones in one sentence each. |\n'
         '| 1 min | 4 | M1 and M2: ResumeShield as the anchor; its numbers rebuilt on a simulated screener; the zero is a property of the simulator. |\n'
         f'| 45 s | 5 | The pilot: recall {100 * pm["recall"]:.2f}%, false-positive rate {100 * pm["fpr"]:.2f}%, hypothesis not supported. Why it led to a real model in the loop. |\n'
         f'| 1 min | 6 | M3: the gap and the hypothesis (at least 20 points lower attack success, at least 95% clean agreement), frozen before the first model call. {rp.N_RESUMES} resumes, {rp.N_DOCS} documents, five configurations, {rp.N_CALLS} calls. |\n'
         '| 45 s | 7 | The attacks: the direct and the reworded wording, and the three placements. Hidden copies look identical to the clean page. |\n'
         '| 1 min | 8 | The defence: visibility-diff and the injection classifier in front of the model, and the five configurations compared. |\n'
         f'| 1.5 min | 9 | The result: {asr("naive")} to {asr("combined")}, clean agreement {rp.pct(pva["actual_clean_agreement"])}; hypothesis {pva["verdict"]} on this sample. |\n'
         '| 1 min | 10 | The ablation: each check closes the gap the other leaves; the detection metrics table; channel separation was not enough. |\n'
         '| 1 min | 11 | Our own classifier: why we trained it, what went wrong in round 1, what round 2 fixed, and its held-out metrics. Its effect on attack success is not yet measured. |\n'
         f'| 45 s | 12 | M5: the {rp.frac("combined")} survivors are one attack, the visible reworded instruction; successes jump to exactly 100; no genuine text was removed. |\n'
         f'| 45 s | 13 | The scanner on unseen resumes: {LATEST["hidden_found"]} of {LATEST["hidden_n"]} hidden attacks found and {LATEST["clean_ok"]} of {LATEST["clean_n"]} clean files passed. Run the live demonstration here if there is time. |\n'
         '| 45 s | 14 | Findings, reproduced and original work, scope, references. Questions. |\n\n'
         '**Eight-minute version:** slides 2, 4, 6, 8, 9, 10, 12, 13 and 14, under a minute each.\n\n'
         '**Live demonstration (optional, two minutes):** `python scanner/scanner_app.py`, open http://127.0.0.1:8502/scan, upload a resume, add a test attack '
         '(white text), and show the verdict and the mark on the page in the "as a program reads it" view.'),
        ('md', '## Likely defence questions and concise answers\n\n'
         '**What is the base paper?** Jay Barach, ResumeShield, arXiv:2609.20188v1.\n\n'
         '**Did you reproduce its exact implementation?** No. M2 is a local reconstruction; the corpus wording, the success definition and the filters differ, '
         f'and {len(m2_mismatch)} of six pipeline rows do not match the paper.\n\n'
         '**What is new in your work?** Not an algorithm. Visibility comparison and injection classifiers both exist. New is the controlled evaluation of their '
         'combination, with ablations, on a real LLM and real PDF files, with the prediction recorded first.\n\n'
         f'**What was the hypothesis and was it supported?** At least 20 points lower ASR with at least 95% clean agreement. Actual: {DROP:.1f} points '
         f'({rp.frac("naive")} to {rp.frac("combined")}) and {rp.pct(pva["actual_clean_agreement"])}. {pva["verdict"].capitalize()}, on this sample.\n\n'
         '**How do we know the prediction came first?** The protocol file is timestamped before the first model response and stores hashes of the inputs. '
         'It is a local record, not an external preregistration.\n\n'
         f'**Are there {E} independent trials?** No. They come from {unc["resume_count"]} resumes, and the variants of one resume are correlated. '
         f'The interval of {LO:.1f} to {HI:.1f} points is descriptive.\n\n'
         '**Why Llama 3.2 3B and not GPT-4o-mini or Gemini?** No hosted model was usable without payment; one refused the free tier and one stopped at 140 of 210 calls. '
         'The hypothesis and thresholds were identical in all three protocol files.\n\n'
         f'**Which attacks still succeed?** {rp.frac("combined")}: the visible paraphrase. It is really on the page, so the visibility check has nothing to flag, '
         'and the pretrained classifier scores it near zero.\n\n'
         '**Why did channel separation do worse than in the paper?** In the paper the simulator never follows fenced text. A real small model still followed the '
         f'direct override in {rp.cell_count("channel_separation", "direct")} cases, and the JSON wrapping changed a clean decision.\n\n'
         '**Does the fine-tuned classifier fix the paraphrase?** It catches every payload block and flags no genuine block here, but it is exploratory: '
         'its training sentences were written after seeing the misses, and its effect on attack success has not been measured.\n\n'
         '**Could the defence remove genuine resume text?** In this run it removed '
         f'{len(rp.removals)} blocks, {rp.payload_removed} of them payloads and {rp.clean_removed} from clean documents. The sample is small.\n\n'
         f'**How good is the scanner?** On unseen resumes it found {HIDDEN_ALL[0]} of {HIDDEN_ALL[1]} hidden-text attacks. Under the current rules it passed '
         f'{latest["clean_ok"]} of {latest["clean_n"]} clean files and called {latest["visible_called"]} of {latest["visible_n"]} non-hidden instructions an attack. '
         'Visible instructions in unusual wording are its weak case.\n\n'
         '**What about the earlier pilot?** A sentence-level TF-IDF detector. Recall passed and the false-positive rate did not, so its hypothesis was not supported. '
         'It is reported as a negative result and its numbers are not mixed with the ASR results.\n\n'
         '**Is this safe to deploy?** The evidence does not show that. Six resumes, one small model, two payload wordings. A flag should prompt a person to look, '
         'never reject a candidate automatically.\n\n'
         '**What would you do next?** A new frozen run with the fine-tuned classifier, new payload wording by another author, a second resume source, '
         'a second annotator and a larger model.'),
        ('md', '## Contribution attribution\n\n'
         '| | What | Where |\n|---|---|---|\n'
         '| Reproduced | ResumeShield benchmark and defence stack, as a partial reconstruction on a simulated screener | M2 |\n'
         '| Original | Real-PDF experiment: frozen protocol, five configurations, ablations | M3, M4 |\n'
         '| Original | Failure analysis, limitations and validity threats | M5 |\n'
         '| Original, exploratory | Fine-tuned DistilBERT injection classifier | M4, M5 |\n'
         '| Original | Resume injection scanner and its held-out evaluation | `scanner/resume_scanner.py`, `results/scanner_eval/` |\n'
         '| Original, negative result | Sentence-level pilot | pilot sections of M3–M5, `pilot/` |'),
        ('md', '## Rubric coverage\n\n'
         '| Milestone | Deliverable | Location |\n|---|---|---|\n'
         '| M1 | Literature survey, comparison table, anchor choice | M1 notebook |\n'
         '| M2 | Reproduction of the anchor, with stated differences | M2 notebook, `results/m2_reproduction/M2_validation.json` |\n'
         '| M3 | Gap, one hypothesis, prior prediction, experimental plan | M3 notebook, protocol files |\n'
         '| M4 | Baseline, experimental setup, metrics, prediction comparison, ablations | M4 notebook, saved run |\n'
         '| M5 | Actual failures, explanations, limitations, validity threats | M5 notebook |\n'
         '| M6 | IEEE-style paper with figures, attribution, references, slides, defence preparation | `paper/`, `figures/`, `presentation/`, this notebook |\n\n'
         '**What remains for you:** check any course-specific page limit, '
         'and rehearse and deliver the oral defence yourself.'),
    ]
