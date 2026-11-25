"""Verify saved measurements independently from the experiment's metrics function."""
import csv
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
run = json.loads((root / 'results/run.json').read_text())
rows = list(csv.DictReader((root / 'results/predictions.csv').open()))
for path, sha in run['input_sha256'].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == sha, path
for model, expected in run['metrics'].items():
    tp = sum(r['label'] == '1' and r[model] == '1' for r in rows)
    fp = sum(r['label'] == '0' and r[model] == '1' for r in rows)
    fn = sum(r['label'] == '1' and r[model] == '0' for r in rows)
    tn = sum(r['label'] == '0' and r[model] == '0' for r in rows)
    assert (tp, fp, fn, tn) == tuple(expected[k] for k in ['tp','fp','fn','tn'])
    assert abs(expected['recall'] - tp/(tp+fn)) < 1e-12
    assert abs(expected['fpr'] - fp/(fp+tn)) < 1e-12
assert len(rows) == 96 and len({r['source_id'] for r in rows}) == 32
assert run['hypothesis_supported'] is False
print('PASS: hashes, all five confusion matrices, recall/FPR, source counts and hypothesis verdict')
