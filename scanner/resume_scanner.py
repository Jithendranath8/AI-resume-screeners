"""Scan resume PDFs for prompt injection and hidden text, and name the attack type.

    .venv/bin/python scanner/resume_scanner.py resume.pdf                 # one file, verdict on screen
    .venv/bin/python scanner/resume_scanner.py folder/ --csv summary.csv  # every PDF in a folder
    .venv/bin/python scanner/resume_scanner.py a.pdf b.pdf --json out.json

Exit code: 0 no attack found, 1 needs review, 2 attack found, 3 a file could not be scanned.

What is checked, all offline:
  * page text that a program extracts but a reader cannot see, with the hiding method named
    (background-coloured text, invisible rendering mode, transparent text, microscopic font,
    text outside the page, text covered by a shape or image);
  * text that reads as an instruction to the evaluator, visible or hidden, with its intent named;
  * document metadata, comments and form fields, text that exists only inside images,
    invisible characters and look-alike letters, and embedded files.

The visibility rule (render at 200 dpi, OCR each block, fuzzy agreement below 80) and the
classifier inference are the ones used in the recorded experiment (colab/real_pdf_workflow.py).
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pymupdf as fitz
import pytesseract
import torch
from PIL import Image
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]  # repository root
VERSION = '1.2'
OCR_DPI = 200
VISIBILITY_THRESHOLD = 80.0
MIN_VISIBILITY_CHARS = 12
TINY_FONT = 4.0        # points; smaller text cannot be read on a printed or displayed page
PRETRAINED_ID = 'protectai/deberta-v3-base-prompt-injection-v2'
FINETUNED_DIR = ROOT / 'results' / 'classifier_training_02' / 'resume_classifier'
MAX_PAGES = 15
MAX_BYTES = 20_000_000

# Latin look-alikes from Cyrillic and Greek, folded before text is compared or classified.
_LOOKALIKES = dict(zip('аеорсхуіѕјԁһкмтвнАВЕКМНОРСТХУІЅЈοανρικτυΑΒΕΗΙΚΜΝΟΡΤΧΥΖ',
                       'aeopcxyisjdhkmtbhABEKMHOPCTXYISJoavpiktuABEHIKMNOPTXYZ'))
CONFUSABLES = str.maketrans(_LOOKALIKES)

TECHNIQUES = {
    'off_page': 'placed outside the visible page area',
    'invisible_mode': 'drawn in the PDF invisible text mode',
    'transparent': 'made fully transparent',
    'tiny_font': 'set in a microscopic font size',
    'background_colour': 'coloured the same as its background',
    'covered': 'covered by a shape or image drawn on top',
    'hidden_layer': 'placed in a switched-off layer',
    'unidentified': 'not visible in the rendering, method not identified',
}
INTENTS = [
    ('override', 'tells the evaluator to ignore its instructions or rubric',
     r"\b(ignore|disregard|forget|override|bypass|skip|set aside|do not apply|don't apply|cancel(?:led)?|waive[sd]?)\b.{0,60}"
     r"\b(instruction|rubric|criteri|rule|requirement|prompt|guideline|polic|evaluation|scoring|above|previous|prior)"),
    ('decision', 'dictates the score or the hiring decision',
     r"\b(score|rating|grade|marks?|assessment)\b.{0,30}\b(100|one hundred|maximum|highest|perfect|full|top)\b"
     r"|\b(give|award|assign|grant)\b.{0,30}\b(top|highest|maximum|perfect|full|best)\b.{0,12}\b(rating|score|marks?|grade|ranking|assessment)"
     r"|\bpre-?approved\b"
     r"|\b(decision|verdict|outcome|status|result)\b.{0,20}\b(select|hire|accept|approv|pass|qualified)"
     r"|\b(select|hire|shortlist|approve|accept|advance|recommend|rank|rate|mark|grade|treat|consider|classify|register|record)\b.{0,40}"
     r"\b(this|the) (candidate|applicant|profile|resume|person|file)"
     r"|\b(candidate|applicant|profile|resume)\b.{0,60}\b(must|should|is to|has to|deserves?|merits?)\b.{0,50}"
     r"\b(select|hire|shortlist|approv|accept|advanc|highest|top|first|maximum|perfect|outstanding|excellent)"),
    ('impersonation', 'poses as a system, administrator or reviewer message',
     r"\[\s*(system|admin|assistant|moderator|instruction)[^\]]*\]|<<\s*(system|admin)|\{\{\s*system|###\s*instruction"
     r"|\bsystem (message|prompt|override|instruction)"
     r"|\b(reviewer|evaluator|assessor|screener|recruiter|committee|panel|board)\b.{0,14}\b(note|guidance|instruction|directive|memo|comment|resolution)\b"),
    ('addressed_ai', 'speaks directly to an AI or automated screener',
     r"\b(if you are|if this (text|message) is|as an?|dear|attention|to any|message to|note (to|for))\b.{0,24}"
     r"\b(ai|llm|language model|assistant|chatbot|automated|bot|model|gpt|screen(er|ing))\b"
     r"|\b(ai (system|model|assistant|recruiter|screener)|language model|llm|automated (screener|reviewer|evaluator|system))\b\s*[:,]"),
    ('output', 'dictates the exact output to return',
     r"\b(return|output|respond|reply|answer|print|emit)\b.{0,30}\b(only|exactly|json|the following|nothing else)\b"
     r"|\{\s*\"?(score|decision)\"?\s*:"),
]
CATEGORIES = {
    'hidden_injection': ('attack', 'Hidden prompt injection'),
    'visible_injection': ('attack', 'Visible prompt injection'),
    'hidden_text': ('attack', 'Hidden text'),
    'metadata_injection': ('attack', 'Prompt injection in document metadata'),
    'annotation_injection': ('attack', 'Prompt injection in a comment or form field'),
    'image_injection': ('attack', 'Prompt injection inside an image'),
    'possible_instruction': ('review', 'Possible instruction to the evaluator'),
    'possibly_hidden': ('review', 'Text that could not be confirmed as visible'),
    'invisible_characters': ('review', 'Invisible characters inserted in the text'),
    'lookalike_letters': ('review', 'Look-alike letters from another alphabet'),
    'hidden_fragments': ('review', 'Several short hidden fragments'),
    'embedded_file': ('review', 'Embedded file'),
}
LIMITS = [
    'Hidden text is found in any language. Instructions are recognised in English only.',
    'A plainly visible instruction is called an attack only when its wording shows its purpose; unusual wording is sent for review or missed.',
    'Hidden text of fewer than four words is ignored unless a file has three or more such fragments.',
    'Text inside images is checked only on pages that contain images, and depends on OCR quality.',
    'A finding is a reason for a person to look at the file, not proof of intent.',
]


def clean(text):
    """Normal form used for comparing and classifying text: fold compatibility forms and look-alikes,
    drop zero-width and other format characters, lower-case, collapse whitespace."""
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Cf')
    return ' '.join(text.translate(CONFUSABLES).lower().split())


def obfuscation(text):
    """Count characters that change what a program reads without changing what a person sees."""
    # Only characters wedged between letters count: stray ones next to punctuation are common export debris.
    invisible = sum(1 for i in range(1, len(text) - 1) if unicodedata.category(text[i]) == 'Cf' and text[i] != '\u00ad'
                    and text[i - 1].isalnum() and text[i + 1].isalnum())
    mixed = 0
    for word in re.findall(r'[^\W\d_]{3,}', text):
        scripts = {unicodedata.name(ch, '').split(' ')[0] for ch in word}
        if 'LATIN' in scripts and scripts & {'CYRILLIC', 'GREEK'}:
            mixed += 1
    return invisible, mixed


def is_language(text):
    """Classifier scores mean something only for sentences, not for dates, identifiers or number tables."""
    cleaned = re.sub(r'\S+@\S+|https?://\S+|www\.\S+', ' ', clean(text))  # addresses are not sentences
    words = re.findall(r'[a-z]{3,}', cleaned)
    letters = sum(ch.isalpha() for ch in cleaned)
    return len(words) >= 4 and letters >= 0.55 * max(1, len(cleaned.replace(' ', '')))


_FUNCTION_WORDS = frozenset('''the and this that with for you your are not any all please should must will have has from their
    its was were been who which when but our them they there here what into about also only without other each every of is'''.split())


def is_english(text):
    """English prose, judged by its function words. The detectors were trained on English, so on other
    languages and on bare lists of terms their scores carry no information."""
    words = re.findall(r'[^\W\d_]+', clean(text))
    found = [w for w in words if w in _FUNCTION_WORDS]
    return len(set(found)) >= 2 and len(found) >= 0.1 * len(words)


def is_substantial(text):
    """Enough text to carry an instruction or a keyword list, in any script."""
    words = [w for w in re.findall(r'[^\W\d_]+', clean(text)) if len(w) >= 3]
    return len(words) >= 4 or sum(map(len, words)) >= 20


def intents_of(text):
    cleaned = clean(text)
    return [(key, label) for key, label, pattern in INTENTS if re.search(pattern, cleaned)]


class Detectors:
    """The two injection classifiers. Both use class 1 for injection and normalised input."""

    def __init__(self, pretrained=True, finetuned=True):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.models, self.cache, self.notes = {}, {}, []

        def load(name, source, threshold, **kwargs):
            tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
            model = AutoModelForSequenceClassification.from_pretrained(source, **kwargs).cpu().eval()
            self.models[name] = dict(tokenizer=tokenizer, model=model, threshold=float(threshold), source=str(source))

        if pretrained:
            try:
                load('pretrained', PRETRAINED_ID, 0.5, local_files_only=True)
            except Exception:
                load('pretrained', PRETRAINED_ID, 0.5)  # first use: download once
        manifest = FINETUNED_DIR / 'training_manifest.json'
        if finetuned and manifest.exists():
            load('finetuned', FINETUNED_DIR, json.loads(manifest.read_text())['threshold'])
        elif finetuned:
            self.notes.append('The fine-tuned classifier is not installed, so only the pretrained one is used and the measured '
                              'accuracy does not apply. Create it with: python experiment/train_classifier_local.py '
                              '--sources deepset,resume,blocks --out results/classifier_training_02')
        if not self.models:
            raise RuntimeError('No injection classifier could be loaded.')

    def describe(self):
        return [dict(name=name, source=m['source'].replace(str(ROOT) + '/', ''), threshold=m['threshold'])
                for name, m in self.models.items()]

    def score(self, texts):
        """Return one {detector: probability} dict per text; long texts use overlapping 256-token windows."""
        if len(self.cache) > 20000:
            self.cache.clear()
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        for name, m in self.models.items():
            if not todo:
                break
            encoded = m['tokenizer']([clean(t) for t in todo], truncation=True, max_length=256, stride=64,
                                     return_overflowing_tokens=True, padding=True, return_tensors='pt')
            owner = encoded.pop('overflow_to_sample_mapping').tolist()
            best = [0.0] * len(todo)
            for start in range(0, len(owner), 16):
                batch = {k: v[start:start + 16] for k, v in encoded.items()}
                with torch.inference_mode():
                    probabilities = m['model'](**batch).logits.softmax(-1)[:, 1].tolist()
                for index, p in zip(owner[start:start + 16], probabilities):
                    best[index] = max(best[index], p)
            for text, value in zip(todo, best):
                self.cache.setdefault(text, {})[name] = float(value)
        return [dict(self.cache[t]) for t in texts]

    def judge(self, scores, intents, text):
        """strong: a detector fires and the wording shows what the instruction is for. weak: every detector fires
        on English prose but no purpose is recognised. Two rules come from the held-out runs in results/scanner_eval:
        one detector on its own is not reported (it caused 12 of 13 false alarms in `test`), and agreement of the
        detectors counts only on English prose (both fired on Vietnamese text in 8 of 12 false alarms in `final`)."""
        if not is_language(text):
            return None
        hits = [name for name, value in scores.items() if value >= self.models[name]['threshold']]
        if hits and intents:
            return 'strong'
        return 'weak' if hits and len(hits) == len(self.models) and is_english(text) else None


# ---- page inspection -------------------------------------------------------------------------------

def _pixel_box(rect, sx, sy, width, height, pad=5):
    return (max(0, int(rect.x0 * sx) - pad), max(0, int(rect.y0 * sy) - pad),
            min(width, int(rect.x1 * sx) + pad), min(height, int(rect.y1 * sy) + pad))


def _ocr(image, box):
    if box[2] <= box[0] or box[3] <= box[1]:
        return ''
    return pytesseract.image_to_string(image.crop(box), config='--psm 6', lang='eng').strip()


def _spans(page):
    spans = []
    for s in page.get_texttrace():
        text = ''.join(chr(c[0]) for c in s['chars'])
        if text.strip():
            rect = fitz.Rect(s['bbox'])
            colour = tuple(s['color']) if len(s['color']) == 3 else tuple([s['color'][0]] * 3) if len(s['color']) == 1 else (0, 0, 0)
            spans.append(dict(text=text, rect=rect, type=s['type'], opacity=s['opacity'], colour=colour, seqno=s['seqno'],
                              size=min(s['size'], max(rect.height, 0.01)), layer=s.get('layer') or ''))
    return spans


def _group(spans):
    """Join spans that follow each other closely into passages: [(rect, text)]."""
    passages, group = [], []
    for s in list(spans) + [None]:
        if s is not None and (not group or abs(s['rect'].y0 - group[-1]['rect'].y1) < 3 * max(s['size'], 4)):
            group.append(s)
            continue
        if group:
            rect = fitz.Rect(group[0]['rect'])
            for g in group[1:]:
                rect |= g['rect']
            text = ' '.join(g['text'].strip() for g in group)
            if len(clean(text)) >= 3:
                passages.append((rect, text))
        group = [s] if s is not None else []
    return passages


def _overlap(a, b):
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def _off_page(rect, page_rect):
    area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
    if area <= 0:
        return not page_rect.contains(fitz.Point(rect.x0, rect.y0))
    return _overlap(rect, page_rect) / area < 0.3


def _hiding_method(block_rect, spans, bboxlog, image, box, page_rect):
    """Name how a block that OCR cannot see was hidden, or return 'unidentified'."""
    if not spans:
        return 'unidentified'
    weight = sum(len(s['text']) for s in spans) or 1
    share = lambda test: sum(len(s['text']) for s in spans if test(s)) / weight
    if share(lambda s: s['type'] == 3) > 0.5:
        return 'invisible_mode'
    if share(lambda s: s['opacity'] < 0.1) > 0.5:
        return 'transparent'
    if share(lambda s: s['size'] < TINY_FONT) > 0.5:
        return 'tiny_font'
    if box[2] > box[0] and box[3] > box[1]:
        background = np.median(np.asarray(image.crop(box))[::2, ::2].reshape(-1, 3), axis=0) / 255
        if share(lambda s: max(abs(c - b) for c, b in zip(s['colour'], background)) < 0.12) > 0.5:
            return 'background_colour'
    last = max(s['seqno'] for s in spans)
    area = (block_rect.x1 - block_rect.x0) * (block_rect.y1 - block_rect.y0) or 1
    for kind, bbox in bboxlog[last + 1:]:
        if kind in ('fill-path', 'fill-image', 'fill-shade') and _overlap(fitz.Rect(bbox), block_rect) / area > 0.8:
            return 'covered'
    if share(lambda s: bool(s['layer'])) > 0.5:
        return 'hidden_layer'
    return 'unidentified'


def _drawn(image, box, spans):
    """True when ink in the text's own colour is present where the text sits, so a reader can see it."""
    if box[2] <= box[0] or box[3] <= box[1]:
        return False
    pixels = np.asarray(image.crop(box)).reshape(-1, 3) / 255
    background = np.median(pixels, axis=0)
    if spans:
        colour = np.array(max(spans, key=lambda s: len(s['text']))['colour'])
        if np.abs(colour - background).max() <= 0.25:
            return False  # too close to the background to count as visible
        return float((np.abs(pixels - colour).max(axis=1) < 0.3).mean()) >= 0.004
    return float((np.abs(pixels - background).max(axis=1) > 0.3).mean()) >= 0.01


def inspect_page(page, number, image_text=True):
    """Extract every text block of one page with its visibility evidence. Returns (blocks, image, size)."""
    if page.rotation:
        page.remove_rotation()
    pix = page.get_pixmap(dpi=OCR_DPI, alpha=False, colorspace=fitz.csRGB)
    image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
    page_rect = page.rect
    sx, sy = pix.width / page_rect.width, pix.height / page_rect.height
    spans, bboxlog = _spans(page), page.get_bboxlog()
    visible_spans = [s for s in spans if not _off_page(s['rect'], page_rect)]

    raw = [(fitz.Rect(b[:4]), b[4].strip()) for b in page.get_text('blocks', sort=True) if b[6] == 0 and b[4].strip()]
    raw = [(rect, text) for rect, text in raw if not _off_page(rect, page_rect)]
    boxes = [_pixel_box(rect, sx, sy, pix.width, pix.height) for rect, _ in raw]
    with ThreadPoolExecutor(max_workers=4) as pool:
        seen = list(pool.map(lambda box: _ocr(image, box), boxes))

    blocks = []
    for (rect, text), box, ocr in zip(raw, boxes, seen):
        cleaned = clean(text)
        agreement = float(fuzz.ratio(cleaned, clean(ocr)))
        grown = rect + (-1.5, -1.5, 1.5, 1.5)
        mine = [s for s in visible_spans if grown.contains(fitz.Point((s['rect'].x0 + s['rect'].x1) / 2, (s['rect'].y0 + s['rect'].y1) / 2))
                and (len(clean(s['text'])) < 3 or clean(s['text'])[:24] in cleaned)]
        missed = len(cleaned) >= MIN_VISIBILITY_CHARS and agreement < VISIBILITY_THRESHOLD
        # A scanned page with its own OCR layer has invisible-mode text over visible ink; only a large mismatch is hiding.
        if missed and mine and all(s['type'] == 3 for s in mine) and len(clean(ocr)) >= 0.5 * len(cleaned) and agreement >= 50:
            missed = False
        short = 3 <= len(cleaned) < MIN_VISIBILITY_CHARS and agreement < 50 and mine and all(
            s['type'] == 3 or s['opacity'] < 0.1 or s['size'] < TINY_FONT for s in mine)
        technique = _hiding_method(rect, mine, bboxlog, image, box, page_rect) if missed or short else None
        if technique == 'unidentified' and _drawn(image, box, mine):
            missed, technique = False, None  # the text is there in ink; OCR misread a heading, icon or stylised font
        blocks.append(dict(channel='page text', page=number, bbox=[round(v, 1) for v in rect], text=text, ocr=ocr,
                           agreement=agreement, hidden=bool(missed or short), technique=technique))

    # Text outside the page area is extractable by most parsers but can never be seen.
    outside = sorted((s for s in spans if _off_page(s['rect'], page_rect)), key=lambda s: s['seqno'])
    for rect, text in _group(outside):
        blocks.append(dict(channel='page text', page=number, bbox=[round(v, 1) for v in rect], text=text, ocr='',
                           agreement=0.0, hidden=True, technique='off_page'))

    # Text that exists only as pixels: visible to a reader and to an OCR-based screener, absent from the file's text.
    if image_text and (page.get_images() or not raw):
        data = pytesseract.image_to_data(image, config='--psm 3', lang='eng', output_type=pytesseract.Output.DICT)
        paragraphs = {}
        for i, word in enumerate(data['text']):
            if not word.strip() or float(data['conf'][i]) < 40:
                continue
            rect = fitz.Rect(data['left'][i] / sx, data['top'][i] / sy, (data['left'][i] + data['width'][i]) / sx,
                             (data['top'][i] + data['height'][i]) / sy)
            centre = fitz.Point((rect.x0 + rect.x1) / 2, (rect.y0 + rect.y1) / 2)
            if any((r + (-2, -2, 2, 2)).contains(centre) for r, _ in raw):
                continue
            entry = paragraphs.setdefault((data['block_num'][i], data['par_num'][i]), dict(words=[], rect=fitz.Rect(rect)))
            entry['words'].append(word)
            entry['rect'] |= rect
        for entry in paragraphs.values():
            text = ' '.join(entry['words'])
            if len(clean(text)) >= MIN_VISIBILITY_CHARS and len(entry['words']) >= 3:
                blocks.append(dict(channel='image text', page=number, bbox=[round(v, 1) for v in entry['rect']], text=text,
                                   ocr=text, agreement=100.0, hidden=False, technique=None))
    return blocks, image, (page_rect.width, page_rect.height)


# ---- whole document --------------------------------------------------------------------------------

def _layers_off(doc):
    return [xref for xref, layer in (doc.get_ocgs() or {}).items() if not layer['on']]


def _layer_blocks(pdf_bytes):
    """Text that exists only in a switched-off layer. A viewer and this scanner's own renderer skip such text, but
    extractors that ignore layer visibility (pypdf, pdfminer and others) pass it to the screener. Every layer is
    switched on in a copy of the file, and text that appears only then is returned as hidden blocks.
    Returns (blocks, number of layers, number switched off)."""
    with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
        layers = doc.get_ocgs() or {}
        off = _layers_off(doc)
        if not off:
            return [], len(layers), 0

        def keys(page):
            if page.rotation:
                page.remove_rotation()
            return {(clean(s['text']), round(s['rect'].x0), round(s['rect'].y0)): s for s in _spans(page)}

        before = [keys(page) for page in doc]
        for item in doc.layer_ui_configs():
            if not item['on'] and item['type'] in ('checkbox', 'radiobox'):
                doc.set_layer_ui_config(item['number'], action=0)
        if _layers_off(doc):  # layers missing from the viewer's list: rewrite the default configuration instead
            catalog = doc.pdf_catalog()
            kind, value = doc.xref_get_key(catalog, 'OCProperties')
            holder, prefix = (int(value.split()[0]), '') if kind == 'xref' else (catalog, 'OCProperties/')
            kind, value = doc.xref_get_key(holder, prefix + 'D')
            holder, prefix = (int(value.split()[0]), '') if kind == 'xref' else (holder, prefix + 'D/')
            doc.xref_set_key(holder, prefix + 'OFF', '[]')
            doc.xref_set_key(holder, prefix + 'BaseState', '/ON')
            with fitz.open(stream=doc.tobytes(), filetype='pdf') as reopened:
                after = [keys(page) for page in reopened]
        else:
            after = [keys(page) for page in doc]

    blocks = []
    for number, (shown, everything) in enumerate(zip(before, after), 1):
        extra = sorted((s for key, s in everything.items() if key not in shown), key=lambda s: s['seqno'])
        for rect, text in _group(extra):
            blocks.append(dict(channel='page text', page=number, bbox=[round(v, 1) for v in rect], text=text, ocr='',
                               agreement=0.0, hidden=True, technique='hidden_layer'))
    return blocks, len(layers), len(off)


def _other_channels(doc):
    """Text that travels with the file but is not page content."""
    items = []
    seen = set()
    for key, value in (doc.metadata or {}).items():
        if isinstance(value, str) and len(clean(value)) >= MIN_VISIBILITY_CHARS and key not in ('format', 'encryption', 'creationDate', 'modDate'):
            seen.add(clean(value))
            items.append(dict(channel='metadata', page=None, bbox=None, text=value, where=f'metadata field "{key}"'))
    try:
        xml = doc.get_xml_metadata() or ''
    except Exception:
        xml = ''
    for value in re.findall(r'>([^<>]{12,})<', xml):
        if clean(value) not in seen and len(clean(value)) >= MIN_VISIBILITY_CHARS:
            seen.add(clean(value))
            items.append(dict(channel='metadata', page=None, bbox=None, text=value.strip(), where='XMP metadata'))
    for number, page in enumerate(doc, 1):
        for annot in page.annots() or []:
            content = (annot.info or {}).get('content', '') or ''
            if len(clean(content)) >= MIN_VISIBILITY_CHARS:
                items.append(dict(channel='annotation', page=number, bbox=[round(v, 1) for v in annot.rect], text=content,
                                  where=f'{annot.type[1]} comment'))
        for widget in page.widgets() or []:
            value = str(widget.field_value or '')
            if len(clean(value)) >= MIN_VISIBILITY_CHARS:
                items.append(dict(channel='annotation', page=number, bbox=[round(v, 1) for v in widget.rect], text=value,
                                  where=f'form field "{widget.field_name}"'))
    return items


def _explain(block, technique_label, intents):
    parts = []
    if block.get('hidden'):
        parts.append(f"A program reading this file gets this text, but a person looking at page {block['page']} does not: it is {technique_label}.")
    elif block['channel'] == 'metadata':
        parts.append(f"This text sits in the {block['where']}, which readers do not see but automated tools may read.")
    elif block['channel'] == 'annotation':
        parts.append(f"This text is in a {block['where']} on page {block['page']}, outside the resume's own content.")
    elif block['channel'] == 'image text':
        parts.append(f"This text on page {block['page']} exists only as an image, so it reaches screeners that use OCR or vision.")
    else:
        parts.append(f"This text is visible on page {block['page']}.")
    if intents:
        parts.append('It ' + '; '.join(label for _, label in intents) + '.')
    return ' '.join(parts)


class Scanner:
    def __init__(self, detectors=None):
        self.detectors = detectors or Detectors()

    def scan(self, pdf_bytes, name='document.pdf', image_text=True, keep_images=False):
        started = time.time()
        if len(pdf_bytes) > MAX_BYTES:
            raise ValueError('The file is larger than 20 MB.')
        if not pdf_bytes.lstrip()[:5].startswith(b'%PDF'):
            raise ValueError('This is not a PDF file.')
        try:
            doc = fitz.open(stream=pdf_bytes, filetype='pdf')
        except Exception:
            raise ValueError('The PDF could not be opened. It may be damaged.') from None
        with doc:
            if doc.needs_pass:
                raise ValueError('The PDF is password-protected. Remove the password and scan again.')
            if not 0 < len(doc) <= MAX_PAGES:
                raise ValueError(f'The PDF has {len(doc)} pages. Files of 1 to {MAX_PAGES} pages are supported.')
            blocks, images, sizes = [], [], []
            for number, page in enumerate(doc, 1):
                found, image, size = inspect_page(page, number, image_text)
                blocks.extend(found)
                images.append(image)
                sizes.append(size)
            others = _other_channels(doc)
            attachments = list(doc.embfile_names()) if doc.embfile_count() else []
            page_count = len(doc)
        layer_blocks, layer_count, layers_off = _layer_blocks(pdf_bytes)
        blocks.extend(layer_blocks)

        everything = blocks + others
        scores = self.detectors.score([b['text'] for b in everything])
        findings, fragments = [], []
        for index, (block, score) in enumerate(zip(everything, scores)):
            intents = intents_of(block['text'])
            verdict = self.detectors.judge(score, intents, block['text'])
            invisible, mixed = obfuscation(block['text'])
            block.update(id=index, scores={k: round(v, 4) for k, v in score.items()}, instruction=verdict,
                         intents=[label for _, label in intents], invisible_characters=invisible, lookalike_words=mixed)
            category = None
            if block['channel'] == 'page text':
                confirmed = block['hidden'] and block['technique'] != 'unidentified'
                if confirmed and not intents and not is_substantial(block['text']):
                    fragments.append(block)  # a watermark, a spaced-out heading, a redacted number: reported only in bulk
                elif block['hidden'] and (confirmed or verdict == 'strong'):
                    category = 'hidden_injection' if verdict else 'hidden_text'
                elif block['hidden'] and is_language(block['text']):
                    category = 'possibly_hidden'  # not visible, but no hiding method found: broken fonts are ignored
                elif verdict == 'strong':
                    category = 'visible_injection'
                elif verdict == 'weak':
                    category = 'possible_instruction'
            elif verdict:
                strong = {'metadata': 'metadata_injection', 'annotation': 'annotation_injection', 'image text': 'image_injection'}
                category = strong[block['channel']] if verdict == 'strong' else 'possible_instruction'
            block['finding'] = category
            if category:
                technique = block.get('technique')
                label = TECHNIQUES.get(technique)
                title = CATEGORIES[category][1]
                if category == 'hidden_text' and block['text'].count(',') + block['text'].count(';') >= 4:
                    title = 'Hidden keyword list'
                findings.append(dict(id=index, severity=CATEGORIES[category][0], category=category, title=title,
                                     channel=block['channel'], page=block['page'], bbox=block['bbox'], text=block['text'][:2000],
                                     hidden=bool(block.get('hidden')), technique=technique, technique_label=label,
                                     instruction=verdict, intents=block['intents'], scores=block['scores'],
                                     ocr_saw=(block.get('ocr') or '')[:300], agreement=block.get('agreement'),
                                     obfuscated=bool(invisible >= 3 or mixed >= 2),
                                     explanation=_explain(block, label, intents)))
            if invisible >= 3 and category not in ('hidden_injection', 'visible_injection'):
                findings.append(dict(id=index, severity='review', category='invisible_characters', title=CATEGORIES['invisible_characters'][1],
                                     channel=block['channel'], page=block['page'], bbox=block['bbox'], text=block['text'][:2000], hidden=False,
                                     technique=None, technique_label=None, instruction=verdict, intents=[], scores=block['scores'],
                                     ocr_saw='', agreement=block.get('agreement'), obfuscated=True,
                                     explanation=f'{invisible} zero-width or other invisible characters are mixed into this text. '
                                                 'They can be used to slip words past keyword filters.'))
            if mixed >= 2 and category not in ('hidden_injection', 'visible_injection'):
                findings.append(dict(id=index, severity='review', category='lookalike_letters', title=CATEGORIES['lookalike_letters'][1],
                                     channel=block['channel'], page=block['page'], bbox=block['bbox'], text=block['text'][:2000], hidden=False,
                                     technique=None, technique_label=None, instruction=verdict, intents=[], scores=block['scores'],
                                     ocr_saw='', agreement=block.get('agreement'), obfuscated=True,
                                     explanation=f'{mixed} words mix Latin letters with look-alike letters from another alphabet. '
                                                 'The text reads normally but does not match ordinary keyword checks.'))
        if len(fragments) >= 3:
            first = fragments[0]
            findings.append(dict(id=first['id'], severity='review', category='hidden_fragments', title=CATEGORIES['hidden_fragments'][1],
                                 channel='page text', page=first['page'], bbox=first['bbox'], text=' | '.join(b['text'] for b in fragments)[:2000],
                                 hidden=True, technique=first['technique'], technique_label=TECHNIQUES.get(first['technique']),
                                 instruction=None, intents=[], scores={}, ocr_saw='', agreement=None, obfuscated=False,
                                 explanation=f'{len(fragments)} pieces of text in this file are not visible on the page. Each is too short to '
                                             'carry an instruction or a keyword list, but together they are worth a look.'))
        for filename in attachments:
            findings.append(dict(id=None, severity='review', category='embedded_file', title=CATEGORIES['embedded_file'][1], channel='attachment',
                                 page=None, bbox=None, text=filename, hidden=False, technique=None, technique_label=None, instruction=None,
                                 intents=[], scores={}, ocr_saw='', agreement=None, obfuscated=False,
                                 explanation='The PDF carries an attached file, which is unusual for a resume and is not scanned here.'))
        findings.sort(key=lambda f: (f['severity'] != 'attack', f['page'] or 0))

        attacks = [f for f in findings if f['severity'] == 'attack']
        reviews = [f for f in findings if f['severity'] == 'review']
        types = list(dict.fromkeys(f['title'] for f in attacks or reviews))
        if attacks:
            level, title = 'attack', 'Attack found'
            summary = f"{len(attacks)} attack finding{'s' if len(attacks) != 1 else ''}: {', '.join(types).lower()}."
        elif reviews:
            level, title = 'review', 'Needs review'
            summary = f"No confirmed attack, but {len(reviews)} item{'s' if len(reviews) != 1 else ''} should be looked at: {', '.join(types).lower()}."
        else:
            level, title = 'clean', 'No attack found'
            summary = 'Every extracted text block is visible on the page and none reads as an instruction to the evaluator.'
        page_blocks = [b for b in blocks if b['channel'] == 'page text']
        scanned_only = not page_blocks and any(b['channel'] == 'image text' for b in blocks)
        checks = [
            dict(name='Hidden page text', result=sum(b['hidden'] for b in page_blocks),
                 detail=f'{len(page_blocks)} text blocks compared with the rendered pages'
                        + (' (scanned document: no extractable text to compare)' if scanned_only else '')),
            dict(name='Instructions to the evaluator', result=sum(1 for b in everything if b['instruction']),
                 detail=f"{len(everything)} pieces of text scored by {len(self.detectors.models)} classifier{'s' if len(self.detectors.models) != 1 else ''}"),
            dict(name='Metadata, comments and form fields', result=sum(1 for b in others if b['instruction']), detail=f"{len(others)} {'entry' if len(others) == 1 else 'entries'} read"),
            dict(name='Text inside images', result=sum(1 for b in blocks if b['channel'] == 'image text' and b['instruction']),
                 detail=(f"{sum(1 for b in blocks if b['channel'] == 'image text')} image-only passages read by OCR" if image_text else 'not checked')),
            dict(name='Invisible characters and look-alike letters', result=sum(1 for b in everything if b['invisible_characters'] >= 3 or b['lookalike_words'] >= 2),
                 detail='zero-width characters and mixed-alphabet words counted'),
            dict(name='Switched-off layers', result=len(layer_blocks),
                 detail=(f"{layers_off} of {layer_count} layer{'s' if layer_count != 1 else ''} switched off; their text was read" if layers_off
                         else f"{layer_count} layer{'s' if layer_count != 1 else ''}, all switched on" if layer_count else 'this file has no layers')),
            dict(name='Embedded files', result=len(attachments), detail='attachments listed, not opened'),
        ]
        report = dict(tool='resume_scanner', version=VERSION, file=name, sha256=hashlib.sha256(pdf_bytes).hexdigest(), pages=page_count,
                      scanned_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), seconds=round(time.time() - started, 1),
                      verdict=dict(level=level, title=title, summary=summary, attack_types=types if attacks else [], review_types=types if not attacks else []),
                      findings=findings, checks=checks, detectors=self.detectors.describe(), limits=LIMITS + getattr(self.detectors, 'notes', []),
                      page_sizes=[dict(number=i + 1, width=w, height=h) for i, (w, h) in enumerate(sizes)], blocks=everything)
        if keep_images:
            report['_images'] = images
        return report


# ---- test attacks ----------------------------------------------------------------------------------

ATTACK_TECHNIQUES = ['white_text', 'invisible_text', 'tiny_font', 'off_page', 'transparent', 'covered', 'hidden_layer',
                     'visible_text', 'metadata', 'annotation', 'image_text', 'zero_width', 'lookalike']


def _unicode_font():
    for candidate in ('/System/Library/Fonts/Supplemental/Arial.ttf', '/Library/Fonts/Arial Unicode.ttf',
                      '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'C:/Windows/Fonts/arial.ttf'):
        if Path(candidate).exists():
            return candidate
    raise ValueError('This test attack needs a Unicode font, and none was found on this computer.')


def add_attack(pdf_bytes, technique, payload):
    """Return a copy of the PDF carrying `payload` with the named technique. Used for tests and demonstrations."""
    if technique not in ATTACK_TECHNIQUES:
        raise ValueError('Unknown attack technique.')
    with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
        if technique == 'metadata':
            meta = dict(doc.metadata or {})
            meta['keywords'] = payload
            doc.set_metadata(meta)
            return doc.tobytes()
        page = doc[-1]
        bottom = max([b[3] for b in page.get_text('blocks') if b[6] == 0] or [0])
        if page.rect.height - bottom < 120:
            page = doc.new_page(width=595, height=842)
            bottom = 20
        box = fitz.Rect(40, bottom + 20, page.rect.width - 40, bottom + 100)
        if technique == 'annotation':
            page.add_text_annot(fitz.Point(box.x0, box.y0), payload)
        elif technique == 'off_page':
            page.insert_text(fitz.Point(page.rect.width + 60, box.y0 + 12), payload, fontsize=9)
        elif technique == 'image_text':
            with fitz.open() as scratch:
                canvas = scratch.new_page(width=box.width, height=60)
                if canvas.insert_textbox(fitz.Rect(4, 4, box.width - 4, 56), payload, fontsize=11) < 0:
                    raise ValueError('The test instruction is too long for its box. Shorten it.')
                pix = canvas.get_pixmap(dpi=200)
            page.insert_image(fitz.Rect(box.x0, box.y0, box.x1, box.y0 + 60), pixmap=pix)
        else:
            text = payload
            if technique == 'zero_width':
                text = '\u200b'.join(payload)
            elif technique == 'lookalike':
                swap = str.maketrans('aeopcxy', 'аеорсху')
                text = ' '.join(w.translate(swap) if i % 2 == 0 else w for i, w in enumerate(payload.split()))
            options = dict(fontsize=10, fontname='helv')
            if technique in ('zero_width', 'lookalike'):  # the built-in fonts have no glyphs for these characters
                options.update(fontname='unicode', fontfile=_unicode_font())
            if technique == 'white_text':
                options['color'] = (1, 1, 1)
            elif technique == 'invisible_text':
                options['render_mode'] = 3
            elif technique == 'tiny_font':
                options['fontsize'] = 0.6
            elif technique == 'transparent':
                options['fill_opacity'] = 0
            elif technique == 'hidden_layer':
                options['oc'] = doc.add_ocg('Notes', on=False)
            if page.insert_textbox(box, text, **options) < 0:
                raise ValueError('The test instruction is too long for its box. Shorten it.')
            if technique == 'covered':
                page.draw_rect(box + (-2, -2, 2, 2), color=(1, 1, 1), fill=(1, 1, 1))
        return doc.tobytes()


# ---- command line ----------------------------------------------------------------------------------

def public(report, blocks=False):
    """The JSON-serialisable part of a report."""
    return {k: v for k, v in report.items() if not k.startswith('_') and (blocks or k != 'blocks')}


def describe(report):
    mark = {'attack': 'ATTACK FOUND', 'review': 'NEEDS REVIEW', 'clean': 'NO ATTACK FOUND'}[report['verdict']['level']]
    lines = [f"{report['file']}: {mark}  ({report['pages']} page{'s' if report['pages'] != 1 else ''}, {report['seconds']} s)",
             f"  {report['verdict']['summary']}"]
    for number, f in enumerate(report['findings'], 1):
        where = f"page {f['page']}" if f['page'] else f['channel']
        lines.append(f"  {number}. {f['title']} ({where})" + ('' if f['severity'] == 'attack' else '  [review]'))
        if f['technique_label']:
            lines.append(f"     How it is hidden: {f['technique_label']}")
        if f['intents']:
            lines.append(f"     Intent: {'; '.join(f['intents'])}")
        text = ' '.join(f['text'].split())
        lines.append(f"     Text: \"{text[:160]}{'…' if len(text) > 160 else ''}\"")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Scan resume PDFs for prompt injection and hidden text.')
    parser.add_argument('paths', nargs='+', help='PDF files or folders containing PDFs')
    parser.add_argument('--json', help='write full reports to this JSON file')
    parser.add_argument('--csv', help='write a one-line-per-file summary to this CSV file')
    parser.add_argument('--no-image-text', action='store_true', help='skip OCR of text inside images (faster)')
    parser.add_argument('--quiet', action='store_true', help='print one line per file')
    args = parser.parse_args(argv)
    files = []
    for path in map(Path, args.paths):
        files.extend(sorted(path.rglob('*.pdf')) if path.is_dir() else [path])
    if not files:
        parser.error('No PDF files found.')
    scanner = Scanner()
    for note in getattr(scanner.detectors, 'notes', []):
        print('Note:', note, file=sys.stderr)
    reports, worst = [], 0
    for path in files:
        try:
            report = scanner.scan(path.read_bytes(), name=str(path), image_text=not args.no_image_text)
        except (ValueError, OSError) as exc:
            report = dict(file=str(path), error=str(exc))
            worst = max(worst, 3)
            print(f'{path}: COULD NOT BE SCANNED  ({exc})')
        else:
            worst = max(worst, {'clean': 0, 'review': 1, 'attack': 2}[report['verdict']['level']])
            print(describe(report).splitlines()[0] if args.quiet else describe(report))
        reports.append(public(report) if 'error' not in report else report)
    if args.json:
        Path(args.json).write_text(json.dumps(reports if len(reports) > 1 else reports[0], indent=2, ensure_ascii=False))
    if args.csv:
        with open(args.csv, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['file', 'verdict', 'attack_types', 'review_types', 'findings', 'pages', 'seconds', 'error'])
            for r in reports:
                v = r.get('verdict', {})
                writer.writerow([r['file'], v.get('level', 'error'), '; '.join(v.get('attack_types', [])), '; '.join(v.get('review_types', [])),
                                 len(r.get('findings', [])), r.get('pages', ''), r.get('seconds', ''), r.get('error', '')])
    if len(reports) > 1:
        counts = {level: sum(1 for r in reports if r.get('verdict', {}).get('level') == level) for level in ('attack', 'review', 'clean')}
        print(f"\n{len(reports)} files: {counts['attack']} with an attack, {counts['review']} needing review, {counts['clean']} clean, "
              f"{sum(1 for r in reports if 'error' in r)} not scanned.")
    return worst


if __name__ == '__main__':
    sys.exit(main())
