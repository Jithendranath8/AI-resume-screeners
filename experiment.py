"""Run the frozen M3 experiment. CPU only; no hosted models or credentials."""
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
HOMO = dict(zip('aceiopxy', 'асеіорх у'.replace(' ', '')))
REVERSE = str.maketrans({v: k for k, v in HOMO.items()})
KEYWORDS = ['ignore all previous', 'perfect fit', 'rank them first',
            'strongly recommend', 'recommend advancing', 'top candidate',
            'select this candidate']


def normalize(text):
    return text.replace('\u200b', '').translate(REVERSE)


def dataset():
    source = json.loads((ROOT / 'data/sentences.json').read_text())
    rows = []
    for split, families in source.items():
        for family, texts in families.items():
            for i, text in enumerate(texts):
                for encoding, encoded in [
                    ('original', text),
                    ('zero_width', '\u200b'.join(text)),
                    ('homoglyph', text.translate(str.maketrans(HOMO))),
                ]:
                    rows.append(dict(id=f'{family}_{i}_{encoding}',
                                     source_id=f'{family}_{i}', family=family,
                                     split=split, encoding=encoding, text=encoded,
                                     label=int(family.startswith('attack_'))))
    # Family and canonical sentence leakage checks, not just row IDs.
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not {r['family'] for r in rows if r['split'] == a} & {
            r['family'] for r in rows if r['split'] == b}
        assert not {normalize(r['text']).lower() for r in rows if r['split'] == a} & {
            normalize(r['text']).lower() for r in rows if r['split'] == b}
    assert all(normalize(r['text']) == normalize(next(
        x['text'] for x in rows if x['source_id'] == r['source_id']
        and x['encoding'] == 'original')) for r in rows)
    return rows


def metrics(y, pred):
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return dict(n=len(y), tp=tp, fp=fp, fn=fn, tn=tn, precision=precision,
                recall=recall, f1=2*precision*recall/(precision+recall)
                if precision+recall else 0.0, fpr=fp/(fp+tn) if fp+tn else 0.0)


def run():
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
              for p in ['M3_Protocol.json', 'data/sentences.json', 'experiment.py']}
    rows = dataset()
    groups = {s: [r for r in rows if r['split'] == s]
              for s in ['train', 'validation', 'test']}
    train, val, test = (groups[s] for s in ['train', 'validation', 'test'])
    y = [r['label'] for r in test]
    predictions = {'keywords': [int(any(k in r['text'].lower() for k in KEYWORDS))
                                for r in test]}
    # Extra baseline isolates normalization alone from learned classification.
    predictions['normalized_keywords'] = [
        int(any(k in normalize(r['text']).lower() for k in KEYWORDS)) for r in test]
    thresholds, validation, scores = {}, {}, {}
    for name, norm, analyzer, ngrams in [
        ('normalized_char_lr', True, 'char_wb', (3, 5)),
        ('raw_char_lr', False, 'char_wb', (3, 5)),
        ('normalized_word_lr', True, 'word', (1, 2)),
    ]:
        transform = normalize if norm else lambda x: x
        model = make_pipeline(TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams),
                              LogisticRegression(C=1, class_weight='balanced',
                              solver='liblinear', random_state=42, max_iter=1000))
        model.fit([transform(r['text']) for r in train], [r['label'] for r in train])
        val_scores = model.predict_proba([transform(r['text']) for r in val])[:, 1]
        options = [(t, metrics([r['label'] for r in val], val_scores >= t))
                   for t in [0.3, 0.5, 0.7]]
        feasible = [(t, m) for t, m in options if m['fpr'] <= 0.1]
        threshold, _ = max(feasible, key=lambda x: (x[1]['f1'], x[0])) if feasible else max(
            options, key=lambda x: (-x[1]['fpr'], x[1]['f1'], x[0]))
        thresholds[name] = threshold
        validation[name] = [{'threshold': t, **m} for t, m in options]
        scores[name] = model.predict_proba([transform(r['text']) for r in test])[:, 1]
        predictions[name] = (scores[name] >= threshold).astype(int).tolist()
    summary = {name: metrics(y, pred) for name, pred in predictions.items()}
    original = [i for i, r in enumerate(test) if r['encoding'] == 'original']
    original_metrics = {name: metrics([y[i] for i in original], [pred[i] for i in original])
                        for name, pred in predictions.items()}
    per_family = {f: {name: metrics([y[i] for i, r in enumerate(test) if r['family'] == f],
                                  [p[i] for i, r in enumerate(test) if r['family'] == f])
                     for name, p in predictions.items()}
                  for f in sorted({r['family'] for r in test})}
    gains = [m['normalized_char_lr']['recall'] - m['keywords']['recall']
             for f, m in per_family.items() if f.startswith('attack_')]
    rng = np.random.default_rng(42)
    boot = [float(np.mean(rng.choice(gains, len(gains), replace=True))) for _ in range(2000)]
    gain = summary['normalized_char_lr']['recall'] - summary['keywords']['recall']
    protocol = json.loads((ROOT / 'M3_Protocol.json').read_text())
    pred = protocol['prediction']
    result = dict(started_utc=started, input_sha256=hashes,
                  environment=dict(python=platform.python_version(), sklearn=sklearn.__version__,
                                   numpy=np.__version__),
                  split_rows={s: len(v) for s, v in groups.items()},
                  split_sources={s: len({r['source_id'] for r in v}) for s, v in groups.items()},
                  thresholds=thresholds, validation=validation, metrics=summary,
                  original_only=original_metrics, per_family=per_family,
                  recall_gain=gain, family_bootstrap_95_interval=np.quantile(boot, [.025, .975]).tolist(),
                  hypothesis_supported=(gain >= pred['recall_gain_min'] and
                                        summary['normalized_char_lr']['fpr'] <= pred['experimental_fpr_max']))
    (out / 'run.json').write_text(json.dumps(result, indent=2) + '\n')
    with (out / 'predictions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(test[0]) + list(predictions) +
                                [n + '_score' for n in scores])
        writer.writeheader()
        for i, row in enumerate(test):
            writer.writerow({**row, **{n: p[i] for n, p in predictions.items()},
                             **{n + '_score': float(p[i]) for n, p in scores.items()}})
    with (out / 'dataset.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ['split_sources', 'thresholds', 'metrics',
                                           'recall_gain', 'hypothesis_supported']}, indent=2))
    return result


if __name__ == '__main__':
    run()
