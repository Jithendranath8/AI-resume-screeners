"""Draw the result figures into figures/ (PNG for reading, PDF for the paper).

    .venv/bin/python report/make_figures.py

Each figure is one result of a milestone notebook, and its file is named after that notebook. The
notebooks M3 to M5 call the same functions, so the chart a notebook shows is the file saved here:

    import sys; sys.path.insert(0, 'report'); import make_figures as figures
    figures.show(figures.m4_attack_success())

Every number is read from the recorded results through paper_sections.py and real_pdf_sections.py.
Nothing is trained and no model is called; overview_two_readings renders one sample page with PyMuPDF.

Colour: blue is the single data hue; red, amber and green appear only as the scanner's three verdicts
or to mark an attack that succeeded; grey is "nothing found". Every bar carries its own number.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import paper_sections as ps
import real_pdf_sections as rp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
INK, GRAPHITE, LINE, DESK = '#111417', '#59616a', '#dfe3e8', '#eef0f3'
BLUE, BLUE_TINT, BLUE_INK, MARKER = '#0b5fd9', '#d6e6ff', '#0a3a86', '#ffe45c'
ATTACK, REVIEW, CLEAN, MISSED = '#c2261f', '#c8880a', '#2a9463', '#7b8590'
COLUMN, PAGE = 3.5, 7.16          # widths of one column and of the full page of the paper, in inches

installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    'font.family': next(f for f in ['Arial', 'Helvetica', 'DejaVu Sans'] if f in installed),
    'font.size': 8, 'axes.titlesize': 8.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlepad': 7,
    'axes.edgecolor': LINE, 'axes.labelcolor': GRAPHITE, 'axes.linewidth': 0.8, 'text.color': INK,
    'xtick.color': GRAPHITE, 'ytick.color': INK, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
    'pdf.fonttype': 42, 'savefig.dpi': 300, 'figure.dpi': 120,
})
LABEL = {'naive': 'No added defence', 'channel_separation': 'Channel separation', 'visibility_only': 'Visibility-diff only',
         'classifier_only': 'Classifier only', 'combined': 'Both checks combined'}
written = []


def save(fig, name, caption):
    OUT.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{suffix}', bbox_inches='tight', pad_inches=0.04, facecolor='white')
    plt.close(fig)
    written.append((name, caption))
    return name


def show(name):
    """Display a saved figure in a notebook."""
    from IPython.display import Image, display
    wide = name in ('overview_two_readings', 'M4_pipeline', 'scanner_heldout')
    display(Image(filename=str(OUT / f'{name}.png'), width=900 if wide else 520))


def bars(ax, labels, values, total=None, text=None, color=BLUE, xmax=1.0):
    """Thin horizontal bars from one baseline, each labelled with its own value."""
    rows = range(len(labels))[::-1]
    ax.barh(rows, [xmax] * len(labels), height=0.46, color=DESK)
    ax.barh(rows, values, height=0.46, color=color)
    for y, value, i in zip(rows, values, range(len(labels))):
        ax.text(xmax * 1.03, y, text[i] if text else f'{100 * value:.1f}%', va='center', ha='left', fontsize=7.5)
    ax.set_yticks(list(rows), labels)
    ax.set_xlim(0, xmax)
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim(-0.6, len(labels) - 0.4)


# ---------------------------------------------------------------- fig 1: the two readings of one page
def overview_two_readings():
    import io

    from PIL import Image, ImageChops, ImageDraw, ImageFont

    sys.path.insert(0, str(ROOT / 'scanner'))
    import build_home_specimens as home
    import resume_scanner as rs

    clean = home.SAMPLE.read_bytes()
    attacked = clean
    for technique, payload in [('white_text', home.INSTRUCTION), ('invisible_text', home.KEYWORDS)]:
        attacked = rs.add_attack(attacked, technique, payload)
    page = rs.fitz.open(stream=attacked, filetype='pdf')[0]
    before = {tuple(round(v) for v in box) for box in home.text_lines(clean, 1)}
    lines = home.text_lines(attacked, 1)
    words = {tuple(round(v) for v in line['bbox']): ''.join(s['text'] for s in line['spans'])
             for block in page.get_text('dict')['blocks'] for line in block.get('lines', [])}
    scale = 3.0
    image = Image.open(io.BytesIO(page.get_pixmap(matrix=rs.fitz.Matrix(scale, scale)).tobytes('png'))).convert('RGB')
    top, bottom = 560, 716                                    # the last lines of the resume and the space below them, in points
    crop = lambda im: im.crop((0, int(top * scale), im.width, int(bottom * scale)))
    person = crop(image)

    tint = Image.new('RGB', image.size, 'white')
    draw = ImageDraw.Draw(tint)
    for x0, y0, x1, y1 in lines:
        draw.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale], fill='#cbddf9')
    program = ImageChops.multiply(image, tint)
    draw = ImageDraw.Draw(program)
    mono = next((p for p in ['/System/Library/Fonts/Menlo.ttc', '/System/Library/Fonts/Supplemental/Courier New.ttf',
                             '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'] if Path(p).exists()), None)
    font = ImageFont.truetype(mono, int(8.2 * scale)) if mono else ImageFont.load_default()
    for box in lines:
        key = tuple(round(v) for v in box)
        if key in before:
            continue
        x0, y0, x1, y1 = (v * scale for v in box)
        text = words.get(key, '')
        width = draw.textlength(text, font=font) + 6 * scale
        draw.rectangle([x0 - 2 * scale, y0 - 2 * scale, max(x1, x0 + width), y1 + 2 * scale], fill=BLUE_TINT, outline=BLUE, width=int(1.2 * scale))
        if text.startswith('Ignore'):
            draw.rectangle([x0, y0 + scale, x0 + width - 6 * scale, y1 - scale], fill=MARKER)
        draw.text((x0, y0 + 0.6 * scale), text, font=font, fill=BLUE_INK)
    program = crop(program)

    fig, axes = plt.subplots(1, 2, figsize=(PAGE, PAGE / 2 * person.height / person.width + 0.34))
    for ax, im, title in zip(axes, [person, program], ['As a person sees it', 'As a program reads it']):
        ax.imshow(im)
        ax.set_title(title)
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(True), side.set_color(LINE)
    fig.subplots_adjust(wspace=0.05)
    return save(fig, 'overview_two_readings', 'The foot of one real resume page with two test attacks added. Left: the page as it prints. Right: every line of text in '
         'the file is tinted, as a selection would tint it; two lines have nothing printed under them, an instruction to the screener '
         '(highlighted) and a list of keywords.')


# ----------------------------------------------------------------------------- fig 2: the pipeline
def m4_pipeline():
    fig, ax = plt.subplots(figsize=(PAGE, 2.3))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 100), ax.set_ylim(0, 32), ax.axis('off')

    def box(x, y, w, h, title, body, face='white', edge=LINE, colour=INK, sub=GRAPHITE):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.8', fc=face, ec=edge, lw=0.9))
        ax.text(x + 1.4, y + h - 2.4, title, fontsize=7.5, fontweight='bold', color=colour, va='top')
        ax.text(x + 1.4, y + h - 6.2, body, fontsize=6.5, color=sub, va='top', linespacing=1.4)

    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=7, lw=0.9, color=INK, shrinkA=0, shrinkB=0,
                                     linestyle=(0, (3, 2)) if dashed else 'solid'))

    box(0.3, 10, 11.5, 12, 'Resume PDF', 'a real file')
    box(16.5, 19, 21, 12, 'What a program reads', 'PyMuPDF extracts every\ntext block and its position')
    box(16.5, 1, 21, 12, 'What a person sees', 'Page rendered at 200 dpi,\nOCR of each block\'s area')
    box(42.5, 19, 25, 12, 'Injection classifier', 'DeBERTa-v3 scores each block;\n0.5 or more is an instruction', face=MARKER, edge='#c9a900', sub=INK)
    box(42.5, 1, 25, 12, 'Visibility-diff', 'Extracted text against OCR text;\nagreement under 80 is hidden', face=BLUE, edge=BLUE, colour='white', sub='white')
    box(72.5, 10, 12, 12, 'Remove', 'blocks flagged\nby either check')
    box(87.2, 10, 12.5, 12, 'LLM screener', 'score 0–100,\nSELECT/REJECT', face=INK, edge=INK, colour='white', sub='#c9cfd6')
    for a, b in [((11.8, 18), (16.5, 25)), ((11.8, 14), (16.5, 7)), ((37.5, 25), (42.5, 25)), ((37.5, 7), (42.5, 7)),
                 ((67.5, 25), (72.5, 18)), ((67.5, 7), (72.5, 14)), ((84.5, 16), (87.2, 16))]:
        arrow(a, b)
    arrow((37.5, 20.5), (42.5, 11.5), dashed=True)
    return save(fig, 'M4_pipeline', 'M3/M4: the combined defence. Each text block the program extracts is checked twice before the model sees it: against what OCR '
         'reads from the rendered page, and by an injection classifier. A block flagged by either check is removed.')


# ---------------------------------------------------------------------- fig 3: attack success rate
def m4_attack_success():
    s = rp.summary
    fig, ax = plt.subplots(figsize=(COLUMN, 1.65))
    bars(ax, [LABEL[c] for c in rp.CONFIGS], [float(s.loc[c, 'asr_valid']) for c in rp.CONFIGS],
         text=[f"{rp.frac(c)}  {rp.pct(s.loc[c, 'asr_valid'])}" for c in rp.CONFIGS])
    ax.set_title(f'Attacks that changed the decision, of {rp.N_ELIGIBLE} tried')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold' if label.get_text() == LABEL['combined'] else 'normal')
    fig.subplots_adjust(right=0.78)
    return save(fig, 'M4_attack_success', f'M4: attack success rate on {rp.N_ELIGIBLE} eligible attacks against {rp.protocol["llm"]}. Clean decisions were unchanged in every '
         f'configuration except channel separation ({rp.pct(s.loc["channel_separation", "clean_agreement_lower"])} agreement).')


# ------------------------------------------------------------------------------- fig 4: ablation
def m4_ablation():
    kinds = [('Hidden,\ndirect', 'direct', True), ('Hidden,\nparaphrase', 'paraphrase', True), ('Visible,\ndirect', 'direct', False), ('Visible,\nparaphrase', 'paraphrase', False)]
    cells = [[rp.cell_count(c, family, hidden=hidden) for _, family, hidden in kinds] for c in rp.CONFIGS]
    share = [[int(x.split('/')[0]) / int(x.split('/')[1]) for x in row] for row in cells]
    fig, ax = plt.subplots(figsize=(COLUMN, 1.8))
    ramp = LinearSegmentedColormap.from_list('attack', [DESK, '#f0b8b3', ATTACK])
    ax.imshow(share, cmap=ramp, vmin=0, vmax=1, aspect='auto')
    for i, row in enumerate(cells):
        for j, text in enumerate(row):
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color='white' if share[i][j] > 0.55 else (INK if share[i][j] else GRAPHITE))
    ax.set_xticks(range(4), [k[0] for k in kinds]), ax.xaxis.tick_top()
    ax.set_yticks(range(5), [LABEL[c] for c in rp.CONFIGS])
    ax.set_xticks([x - 0.5 for x in range(1, 4)], minor=True), ax.set_yticks([y - 0.5 for y in range(1, 5)], minor=True)
    ax.grid(which='minor', color='white', linewidth=2.5), ax.tick_params(which='minor', length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(axis='x', colors=GRAPHITE)
    return save(fig, 'M4_ablation', 'M4: successful attacks out of those tried, by how the payload was placed and how it was worded. Each check closes the gap the other '
         'leaves; the visible paraphrase survives both.')


# ---------------------------------------------------------------------------- fig 5: classifiers
def m4_classifier_finetuning():
    names = ['Pretrained DeBERTa', 'DistilBERT, round 1', 'DistilBERT, round 2']
    rows = [ps.pre, ps.ft1, ps.ft2]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.3), sharey=True)
    bars(axes[0], names, [r.tp / (r.tp + r.fn) for r in rows], text=[f'{int(r.tp)}/{int(r.tp + r.fn)}' for r in rows])
    bars(axes[1], names, [r.fp / (r.fp + r.tn) for r in rows], text=[f'{int(r.fp)}/{int(r.fp + r.tn)}' for r in rows], color=ATTACK)
    axes[0].set_title('Payload blocks caught'), axes[1].set_title('Genuine blocks flagged')
    fig.subplots_adjust(wspace=0.5, right=0.9)
    return save(fig, 'M4_classifier_finetuning', 'M4: injection classifiers on the text blocks of the 42 experiment documents (exploratory). Round 2 catches every payload block '
         'without flagging genuine resume text; its effect on attack success has not been measured.')


# ------------------------------------------------------------------------- fig 6: scanner held out
def scanner_heldout():
    runs = [(name, r) for name, r in ps.RUNS.items() if r]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 0.4 * len(runs) + 0.8), sharey=True)
    parts = [('Files with an attack added', [('called', 'called an attack', ATTACK), ('review', 'sent for review', REVIEW), ('missed', 'missed', MISSED)],
              lambda r: dict(called=r['called'], review=r['flagged'] - r['called'], missed=r['attacks'] - r['flagged'])),
             ('Clean files', [('clean', 'passed', CLEAN), ('review', 'sent for review', REVIEW), ('attack', 'wrongly called an attack', ATTACK)],
              lambda r: dict(clean=r['clean_ok'], review=r['clean_review'], attack=r['clean_attack']))]
    for ax, (title, segments, counts) in zip(axes, parts):
        for y, (name, r) in zip(range(len(runs))[::-1], runs):
            c = counts(r)
            total, left = sum(c.values()), 0
            for key, _, colour in segments:
                if not c[key]:
                    continue
                width = c[key] / total
                ax.barh(y, width - 0.004, left=left, height=0.5, color=colour)
                if width > 0.07:
                    ax.text(left + width / 2, y, str(c[key]), ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
                else:
                    ax.annotate(str(c[key]), (left + width / 2, y + 0.25), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=7, color=INK)
                left += width
            ax.text(1.015, y, f'of {total}', va='center', fontsize=7.5, color=GRAPHITE)
        ax.set_title(title)
        ax.set_xlim(0, 1), ax.set_xticks([]), ax.spines['bottom'].set_visible(False), ax.set_ylim(-0.55, len(runs) - 0.25)
        handles = [Rectangle((0, 0), 1, 1, color=colour) for _, _, colour in segments]
        ax.legend(handles, [label for _, label, _ in segments], loc='upper left', bbox_to_anchor=(0, -0.02), ncol=3, frameon=False,
                  fontsize=7.5, handlelength=0.9, handleheight=0.9, columnspacing=1.4, handletextpad=0.5, borderaxespad=0)
    axes[0].set_yticks(range(len(runs))[::-1], [f"Run {i + 1}, rules {r['version']}" for i, (_, r) in enumerate(runs)])
    fig.subplots_adjust(wspace=0.16)
    return save(fig, 'scanner_heldout', 'The scanner on resumes it was not developed on. Rules were changed after runs 1 and 2, so each version was measured again on '
         'fresh files; run 3 measures the current rules.')


# ---------------------------------------------------------------------------- fig 7: M2 reproduction
def m2_attack_success():
    names = {'naive (no defense)': 'Naive, no defence', 'instruction_redact only': 'Instruction redaction only', 'markup_strip only': 'Markup strip only',
             'all 3 filters (no fence)': 'All three filters, no fence', 'channel separation (fence only)': 'Channel separation only',
             'full defense (filters + fence)': 'Full defence'}
    asr = ps.m2['asr']
    fig, ax = plt.subplots(figsize=(COLUMN, 1.85))
    bars(ax, [names[k] for k in asr], [v['reproduced'] for v in asr.values()], text=[f"{v['successes']}/{v['total']}" for v in asr.values()])
    rows = range(len(asr))[::-1]
    ax.scatter([v['original'] for v in asr.values()], list(rows), marker='|', s=150, color=INK, linewidths=1.6, zorder=3, label='Reported in the paper')
    ax.barh([-5], [0], color=BLUE, label='Reproduced')
    ax.set_title('Attack success rate on the simulated screener')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.01), ncol=2, frameon=False, fontsize=7.5, handlelength=1, borderaxespad=0)
    fig.subplots_adjust(right=0.86)
    return save(fig, 'M2_attack_success', 'M2: the ResumeShield benchmark rebuilt locally. Bars are our reconstruction, ticks the values reported in the paper; two '
         'filter-only pipelines differ because the corpus and filters are locally authored.')


# ----------------------------------------------------------------------------------- fig 8: pilot
def pilot_detection():
    names = ['Keywords', 'Normalised keywords', 'Character TF-IDF + LR', 'Raw character LR', 'Word TF-IDF + LR']
    metrics = list(ps.pilot['metrics'].values())
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.55), sharey=True)
    bars(axes[0], names, [m['recall'] for m in metrics])
    bars(axes[1], names, [m['fpr'] for m in metrics], color=ATTACK, xmax=0.3)
    axes[1].axvline(0.10, color=INK, lw=0.9, linestyle=(0, (3, 2)))
    axes[1].text(0.107, -0.62, 'limit 10%', fontsize=7, color=INK, ha='left', va='top')
    axes[1].set_ylim(-1.05, len(names) - 0.4)
    axes[0].set_title('Recall'), axes[1].set_title('False-positive rate')
    for label in axes[0].get_yticklabels():
        label.set_fontweight('bold' if label.get_text().startswith('Character') else 'normal')
    fig.subplots_adjust(wspace=0.5, right=0.9)
    return save(fig, 'pilot_detection', 'The earlier sentence-level pilot. The trained detector reached high recall but flagged 18.75% of benign rows, above the '
         'limit recorded beforehand, so its hypothesis was not supported.')


# ------------------------------------------------------------------ M2: detection and techniques
def m2_detection():
    c = ps.m2['detection']['confusion']
    d = ps.m2['detection']['reproduced']
    grid = [[c['tp'], c['fn']], [c['fp'], c['tn']]]
    names = [['caught', 'missed'], ['false alarm', 'passed']]
    fig, ax = plt.subplots(figsize=(COLUMN * 0.86, 2.0))
    ax.imshow([[1, 0.25], [0.02, 0.6]], cmap=LinearSegmentedColormap.from_list('blue', [DESK, '#a9c8f5', BLUE]), vmin=0, vmax=1, aspect='auto')
    for i in range(2):
        for j in range(2):
            dark = (i, j) == (0, 0)
            ax.text(j, i - 0.1, str(grid[i][j]), ha='center', va='center', fontsize=15, fontweight='bold', color='white' if dark else INK)
            ax.text(j, i + 0.24, names[i][j], ha='center', va='center', fontsize=7.5, color='white' if dark else GRAPHITE)
    ax.set_xticks([0, 1], ['Flagged', 'Not flagged']), ax.xaxis.tick_top()
    ax.set_yticks([0, 1], [f"Injected\n({c['tp'] + c['fn']})", f"Clean\n({c['fp'] + c['tn']})"])
    ax.set_xticks([0.5], minor=True), ax.set_yticks([0.5], minor=True), ax.grid(which='minor', color='white', linewidth=3), ax.tick_params(which='minor', length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel(f"precision {d['precision']:.3f}   recall {d['recall']:.3f}   F1 {d['f1']:.3f}", labelpad=8, color=INK)
    ax.set_title('Hidden-prompt detector on the 104 benchmark documents', pad=22)
    return save(fig, 'M2_detection', 'M2: the hidden-prompt detector on the 104 benchmark documents. No clean document is flagged; the four misses are the '
                'plainly visible payloads.')


def m2_by_technique():
    names = {'background_color': 'Background colour', 'zero_font': 'Zero-size font', 'hidden_element': 'Hidden element', 'off_canvas': 'Off canvas',
             'markup_comment': 'Markup comment', 'metadata': 'Metadata', 'zero_width': 'Zero-width characters', 'homoglyph': 'Homoglyphs',
             'plain_visible': 'Plain visible text'}
    per = ps.m2['per_technique']
    caught = [8 - v['detector_missed'] for v in per.values()]
    fig, ax = plt.subplots(figsize=(COLUMN, 2.5))
    bars(ax, [names.get(k, k) for k in per], [n / 8 for n in caught], text=[f'{n}/8' for n in caught])
    ax.set_title('Injected documents the detector caught, of 8 per technique')
    fig.subplots_adjust(right=0.88)
    return save(fig, 'M2_by_technique', 'M2: detection by concealment technique. Every hidden payload is caught; only half of the plainly visible ones are. '
                'Attack success is 100% for every technique against the naive pipeline and 0% against the full defence.')


# ------------------------------------------------------------------ M4: document-level detection
def m4_detection():
    names = {'visibility_only': 'Visibility-diff', 'classifier_only': 'Pretrained classifier', 'combined': 'Both checks combined'}
    d = rp.detection
    fig, ax = plt.subplots(figsize=(COLUMN, 1.3))
    bars(ax, [names[k] for k in d.index], [float(r.recall) for _, r in d.iterrows()],
         text=[f'{int(r.tp)}/{int(r.tp + r.fn)}  {100 * r.recall:.1f}%' for _, r in d.iterrows()])
    ax.set_title('Injected documents detected')
    fig.subplots_adjust(right=0.78)
    return save(fig, 'M4_detection', 'M4: the two checks as document-level detectors, on 36 injected documents. Neither flags any of the 6 clean controls.')


# ---------------------------------------------------------------------- M5: scores under attack
def m5_scores():
    e = rp.eligible
    fig, ax = plt.subplots(figsize=(COLUMN, 2.0))
    rows = {c: y for c, y in zip(rp.CONFIGS, range(len(rp.CONFIGS))[::-1])}
    for config, y in rows.items():
        d = e[e.config == config].sort_values('score')
        for decision, colour in (('REJECT', BLUE), ('SELECT', ATTACK)):
            part = d[d.decision == decision]
            # attacks with the same score are stacked upward so every one of the 24 is visible
            k = part.groupby('score').cumcount()
            side = part.score.map(lambda v: -1 if v > 80 else 1)        # columns grow away from the nearer axis end
            ax.scatter(part.score + side * 2.3 * (k // 6), y - 0.3 + 0.11 * (k % 6), s=9, color=colour, linewidths=0, zorder=3)
    ax.axvline(60, color=INK, lw=0.9, linestyle=(0, (3, 2)))
    ax.text(61.5, len(rows) - 0.42, 'SELECT at 60', fontsize=7, va='bottom')
    ax.set_yticks(list(rows.values()), [LABEL[c] for c in rows])
    ax.set_xlim(-3, 104), ax.set_xticks([0, 20, 40, 60, 80, 100]), ax.set_xlabel('Score returned for an attacked resume')
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.spines['bottom'].set_visible(True), ax.tick_params(axis='x', length=2)
    handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=3.5, color=c) for c in (ATTACK, BLUE)]
    ax.legend(handles, ['attack succeeded', 'attack failed'], loc='upper left', bbox_to_anchor=(0, -0.2), ncol=2, frameon=False, fontsize=7.5, handletextpad=0.2, borderaxespad=0)
    ax.set_title(f'Each of the {rp.N_ELIGIBLE} attacks, by the score the screener returned')
    return save(fig, 'M5_scores', 'M5: the score returned for every eligible attack. A successful attack is not a marginal flip: every one returns exactly 100, '
                'the value the payload asks for.')


# Which notebook each figure belongs to, in the order they are listed in figures/README.md.
FIGURES = [
    ('Overview', [overview_two_readings]),
    ('M2 Reproduction (M2_Reproduction.ipynb)', [m2_attack_success, m2_detection, m2_by_technique]),
    ('M3 and M4 Experiment (M3_Research_Gap.ipynb, M4_Controlled_Experiments.ipynb)', [m4_pipeline, m4_attack_success, m4_ablation, m4_detection, m4_classifier_finetuning]),
    ('M5 Failure analysis (M5_Failure_Analysis.ipynb)', [m5_scores]),
    ('Pilot (pilot sections of M4 and M5)', [pilot_detection]),
    ('Scanner (results/scanner_eval)', [scanner_heldout]),
]

if __name__ == '__main__':
    index = '# Figures\n\nResult figures, one per result reported in the milestone notebooks, drawn by `report/make_figures.py` from the recorded results.\n' \
            'PNG for reading, PDF for the paper. The notebooks M3 to M5 show these same images. Do not edit them by hand; rerun the script instead.\n'
    for heading, functions in FIGURES:
        index += f'\n## {heading}\n'
        for draw in functions:
            start = len(written)
            draw()
            name, caption = written[start]
            index += f'\n### {name}\n\n![{name}]({name}.png)\n\n{caption}\n'
    for stale in OUT.glob('fig[0-9]_*'):
        stale.unlink()
    (OUT / 'README.md').write_text(index)
    print(f'Wrote {len(written)} figures (PNG and PDF) and README.md in {OUT.relative_to(ROOT)}/')
