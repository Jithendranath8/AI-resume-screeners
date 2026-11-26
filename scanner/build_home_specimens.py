"""Build the evidence the web Home page shows, from the real scanner and the recorded results.

    .venv/bin/python scanner/build_home_specimens.py

Writes web/src/data/evidence.json and the page images under web/public/specimens/. Nothing on the Home
page is drawn by hand: each specimen is a sample resume with one test attack added by
resume_scanner.add_attack and then scanned, and the figures come from results/. Run it again after the
scanner or the results change, then rebuild the page with `npm run build` in web/.
"""
import csv
import json
from pathlib import Path

from PIL import Image

import resume_scanner as rs
import scanner_app

ROOT = Path(__file__).resolve().parents[1]  # repository root
OUT_JSON = ROOT / 'web' / 'src' / 'data' / 'evidence.json'
OUT_IMAGES = ROOT / 'web' / 'public' / 'specimens'
SAMPLE = ROOT / 'data' / 'real_pdfs' / 'r03_software_engineering_analyst_32985311.pdf'
RUN = ROOT / 'results' / 'real_pdf' / 'pdf_llama32_3b_01'
WIDTH = 1000
INSTRUCTION = scanner_app.PRESETS['Direct override']
ADDRESSED = scanner_app.PRESETS['Addressed to the AI']
KEYWORDS = scanner_app.PRESETS['Hidden keyword list']
CONFIG_LABELS = {
    'naive': 'No defence', 'channel_separation': 'Resume fenced off as data', 'visibility_only': 'Hidden text removed',
    'classifier_only': 'Instructions removed', 'combined': 'Both removals together',
}


def mark(finding):
    keep = ('title', 'severity', 'technique_label', 'intents', 'text', 'explanation', 'channel', 'page', 'bbox', 'hidden', 'instruction')
    return {k: finding[k] for k in keep}


def text_lines(pdf_bytes, number):
    """The box of every line of text a program extracts from one page: what selecting the whole page would highlight."""
    page = rs.fitz.open(stream=pdf_bytes, filetype='pdf')[number - 1]
    width, height = page.rect.width, page.rect.height
    boxes = []
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            x0, y0, x1, y1 = line['bbox']
            inside = x0 < width and y0 < height and x1 > 0 and y1 > 0
            if inside and ''.join(span['text'] for span in line['spans']).strip():
                boxes.append([round(v, 1) for v in (x0, y0, min(x1, width), y1)])
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def scan(scanner, pdf_bytes, name):
    report = scanner.scan(pdf_bytes, name=name, image_text=True, keep_images=True)
    images = report.pop('_images')
    return report, images


def save(image, name):
    scale = WIDTH / image.width
    image = image.resize((WIDTH, round(image.height * scale)), Image.LANCZOS)
    image.save(OUT_IMAGES / name, format='PNG', optimize=True)
    return image


def main():
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    scanner = rs.Scanner()
    clean_pdf = SAMPLE.read_bytes()

    # The opening page: one real resume carrying four attacks at once.
    pdf = clean_pdf
    for technique, payload in [('white_text', INSTRUCTION), ('invisible_text', KEYWORDS), ('off_page', ADDRESSED), ('tiny_font', INSTRUCTION)]:
        pdf = rs.add_attack(pdf, technique, payload)
    report, images = scan(scanner, pdf, 'hero.pdf')
    page = report['page_sizes'][-1]
    save(images[page['number'] - 1], 'hero.png')
    hero = dict(image='/specimens/hero.png', width=page['width'], height=page['height'], verdict=report['verdict'],
                seconds=report['seconds'], findings=[mark(f) for f in report['findings'] if f['page'] == page['number'] and f['bbox']],
                lines=text_lines(pdf, page['number']))

    # The newest held-out run: `confirm` measures the current rules, `final` the version before.
    summaries = [ROOT / 'results' / 'scanner_eval' / f'{name}_summary.json' for name in ('confirm', 'final')]
    final = json.loads(next(p for p in summaries if p.exists()).read_text())
    specimens = []
    for technique in rs.ATTACK_TECHNIQUES:
        payload = KEYWORDS if technique == 'hidden_layer' else INSTRUCTION
        report, images = scan(scanner, rs.add_attack(clean_pdf, technique, payload), technique + '.pdf')
        finding = next((f for f in report['findings'] if f['bbox']), None) or report['findings'][0]
        entry = dict(key=technique, label=scanner_app.ATTACK_LABELS[technique], verdict=report['verdict']['level'],
                     finding=mark(finding), measured=final['attacks']['by_technique'].get(technique), crop=None)
        if finding['page'] and finding['bbox']:
            size = next(p for p in report['page_sizes'] if p['number'] == finding['page'])
            image = images[finding['page'] - 1]
            unit = image.height / size['height']
            y0, y1 = finding['bbox'][1], finding['bbox'][3]
            top = max(0, min(y0, size['height'] - 30) - 110)
            bottom = min(size['height'], max(y1, top + 150) + 36)
            band = image.crop((0, round(top * unit), image.width, round(bottom * unit)))
            save(band, technique + '.png')
            entry['crop'] = dict(image=f'/specimens/{technique}.png', width=size['width'], height=bottom - top, top=top)
        specimens.append(entry)

    with open(RUN / 'summary.csv') as handle:
        rows = list(csv.DictReader(handle))
    protocol = json.loads((RUN / 'protocol.json').read_text())['protocol']
    with open(RUN / 'attack_pairs.csv') as handle:
        eligible = [r for r in csv.DictReader(handle) if r['eligible'] == 'True']
    trials = [r for r in eligible if r['config'] == 'naive']
    # Successful attacks out of those tried, by how the payload was placed and how it was worded.
    kinds = [('Hidden, direct', 'direct', True), ('Hidden, reworded', 'paraphrase', True),
             ('Visible, direct', 'direct', False), ('Visible, reworded', 'paraphrase', False)]

    def cell(config, family, hidden):
        tried = [r for r in eligible if r['config'] == config and r['family'] == family and (r['technique'] != 'visible_text') == hidden]
        return dict(s=sum(r['decision'] == 'SELECT' for r in tried), n=len(tried))
    experiment = dict(
        model='Llama 3.2 3B Instruct', resumes=len(protocol['input_hashes']), run='pdf_llama32_3b_01',
        rejected_resumes=len({r['resume_id'] for r in trials}), attacks_each=len(protocol['techniques']) * len(protocol['payloads']),
        placements=len(protocol['techniques']), wordings=len(protocol['payloads']),
        verdict=json.loads((RUN / 'prediction_vs_actual.json').read_text())['verdict'],
        configs=[dict(key=r['config'], label=CONFIG_LABELS[r['config']], successes=int(r['successes']), eligible=int(r['eligible_attacks']),
                      clean_agreement=float(r['clean_agreement_lower'])) for r in rows],
        ablation=dict(columns=[k[0] for k in kinds],
                      rows=[dict(key=r['config'], cells=[cell(r['config'], family, hidden) for _, family, hidden in kinds]) for r in rows]))
    attacks, clean = final['attacks'], final['clean']
    hidden = ['white_text', 'invisible_text', 'tiny_font', 'off_page', 'transparent', 'covered', 'hidden_layer']
    evaluation = dict(
        scanner_version=final['scanner_version'], files=final['files'], mean_seconds=final['mean_seconds'],
        attacks=attacks['n'], called_attack=attacks['detected'], flagged=attacks['flagged_attack_or_review'],
        hidden_attacks=sum(attacks['by_technique'][t]['n'] for t in hidden), hidden_found=sum(attacks['by_technique'][t]['detected'] for t in hidden),
        clean=clean['n'], clean_verdicts={level: clean['verdicts'].get(level, 0) for level in ('clean', 'review', 'attack')})
    OUT_JSON.write_text(json.dumps(dict(sample=SAMPLE.name, scanner_version=rs.VERSION, hero=hero, specimens=specimens,
                                        experiment=experiment, evaluation=evaluation, limits=rs.LIMITS), indent=1, ensure_ascii=False))
    print(f'Wrote {OUT_JSON.relative_to(ROOT)} and {len(specimens) + 1} images in {OUT_IMAGES.relative_to(ROOT)}')
    for s in specimens:
        print(f"  {s['key']:15} {s['verdict']:7} {s['finding']['title']}  crop={'yes' if s['crop'] else 'no'}")
    print('  hero:', hero['verdict']['level'], [f['title'] for f in hero['findings']])


if __name__ == '__main__':
    main()
