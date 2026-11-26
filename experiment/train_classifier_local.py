"""Fine-tune the DistilBERT injection classifier locally (CPU or Apple MPS).

Mirrors colab/train_classifier_t4.py: same base model, seed, epochs, learning rate,
batch size, normalisation, threshold rule and export format. Differences:
no CUDA/FP16, no Colab uploads/downloads, and the training pool can combine the
public deepset/prompt-injections dataset with the authored resume-domain set in
data/classifier_resume_domain.csv (families stay inside one split).

Usage:
    .venv/bin/python experiment/train_classifier_local.py --out results/classifier_training_01
    .venv/bin/python experiment/train_classifier_local.py --out ... --sources deepset   # public data only

Exports <out>/resume_classifier/ (weights, tokenizer, training_manifest.json) and
<out>/resume_classifier.zip for the real-PDF workflow's CLASSIFIER_ZIP setting.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

PROJECT = Path(__file__).resolve().parents[1]  # repository root
BASE_MODEL = 'distilbert/distilbert-base-uncased'
SEED = 42
EPOCHS = 3
MAX_LENGTH = 256
CONFUSABLES = str.maketrans(dict(zip('асеіорху', 'aceiopxy')))


def normalized(text):
    return ' '.join(text.replace('\u200b', '').translate(CONFUSABLES).lower().split())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_deepset():
    source = 'deepset/prompt-injections'
    revision = HfApi().dataset_info(source).sha
    ds = load_dataset(source, revision=revision)
    train_pool = ds['train'].to_pandas()[['text', 'label']].copy()
    heldout = ds['test'].to_pandas()[['text', 'label']].copy()
    train_pool['canonical'] = train_pool.text.map(normalized)
    heldout['canonical'] = heldout.text.map(normalized)
    train_pool = train_pool[~train_pool.canonical.isin(heldout.canonical)].drop_duplicates('canonical')
    heldout = heldout.drop_duplicates('canonical')
    train, validation = train_test_split(train_pool, test_size=.20, random_state=SEED, stratify=train_pool.label)
    frame = pd.concat([train.assign(split='train'), validation.assign(split='validation'),
                       heldout.assign(split='test')], ignore_index=True)
    frame['group_id'] = 'deepset_' + frame.text.map(lambda t: sha(normalized(t).encode()))
    frame['source'] = source
    return frame[['text', 'label', 'split', 'group_id', 'source']], revision


def load_resume_domain():
    path = PROJECT / 'data' / 'classifier_resume_domain.csv'
    frame = pd.read_csv(path, keep_default_na=False)
    assert {'text', 'label', 'split', 'group_id'} <= set(frame.columns)
    frame['source'] = 'authored résumé-domain sentences (data/classifier_resume_domain.csv)'
    return frame, sha(path.read_bytes())


def load_resume_blocks():
    """Genuine text blocks extracted from other Kaggle resume PDFs (never the six evaluation PDFs).

    Label 0 only. train split: benign training paragraphs; validation split: the threshold
    calibration set, so the deployment threshold reflects false positives on real resume blocks.
    """
    path = PROJECT / 'data' / 'classifier_resume_blocks.csv'
    frame = pd.read_csv(path, keep_default_na=False)
    assert {'text', 'label', 'split', 'group_id'} <= set(frame.columns) and set(frame.label) == {0}
    frame['source'] = 'benign blocks from other Kaggle résumé PDFs (data/classifier_resume_blocks.csv)'
    return frame, sha(path.read_bytes())


def classification_metrics(labels, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(labels, pred, labels=[0, 1]).ravel())
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=float(precision), recall=float(recall),
                f1=float(f1), fpr=fp / (fp + tn) if (fp + tn) else float('nan'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(PROJECT / 'results' / 'classifier_training_01'))
    ap.add_argument('--sources', default='deepset,resume,blocks', help='comma list from: deepset, resume, blocks')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    args = ap.parse_args()
    out = Path(args.out)
    assert not (out / 'training_started.json').exists(), 'Training already started here; use a fresh --out.'
    out.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    frames, sources = [], {}
    if 'deepset' in args.sources:
        frame, revision = load_deepset(); frames.append(frame); sources['deepset/prompt-injections'] = revision
    if 'resume' in args.sources:
        frame, digest = load_resume_domain(); frames.append(frame); sources['classifier_resume_domain.csv'] = digest
    if 'blocks' in args.sources:
        frame, digest = load_resume_blocks(); frames.append(frame); sources['classifier_resume_blocks.csv'] = digest
    data = pd.concat(frames, ignore_index=True)
    data['canonical'] = data.text.map(normalized)
    data = data.drop_duplicates('canonical')
    assert set(data.split) == {'train', 'validation', 'test'} and set(data.label) == {0, 1}
    assert data.groupby('canonical').label.nunique().max() == 1, 'Conflicting labels for identical text'
    assert data.groupby('group_id').split.nunique().max() == 1, 'Source/family leakage between splits'
    for split in ['train', 'validation', 'test']:
        group = data[data.split == split]
        assert set(group.label) == {0, 1} and len(group) >= 10
    data.to_csv(out / 'frozen_data.csv', index=False)
    model_rev = HfApi().model_info(BASE_MODEL).sha
    record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, base_model=BASE_MODEL,
                  base_revision=model_rev, sources=sources, data_sha256=sha((out / 'frozen_data.csv').read_bytes()),
                  epochs=args.epochs, learning_rate=2e-5, max_length=MAX_LENGTH, train_batch_size=8,
                  gradient_accumulation_steps=2, device=device,
                  threshold_rule='Choose highest validation F1 with FPR<=0.10 among 0.1,...,0.9; else lowest FPR then highest F1. Tie: higher threshold.',
                  caveat='Exploratory fine-tuning: the authored injection families were written after observing the '
                         'pretrained detector miss paraphrased payloads in run pdf_llama32_3b_01. Not a pre-registered '
                         'classifier; a confirmatory PDF evaluation needs a new run ID and new payload families.')
    (out / 'training_started.json').write_text(json.dumps(record, indent=2))
    print(data.groupby(['split', 'label']).size().unstack(fill_value=0))
    print('authored rows by split:', data[data.source.str.startswith('authored')].groupby('split').size().to_dict())
    print('benign block rows by split:', data[data.source.str.startswith('benign blocks')].groupby('split').size().to_dict())

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=model_rev)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, revision=model_rev, num_labels=2,
        id2label={0: 'BENIGN', 1: 'INJECTION'}, label2id={'BENIGN': 0, 'INJECTION': 1})
    data['token_length'] = data.canonical.map(lambda t: len(tokenizer(t, truncation=False)['input_ids']))
    print('truncated rows per split:', data.assign(tr=data.token_length > MAX_LENGTH).groupby('split').tr.sum().to_dict())
    splits = {}
    for split in ['train', 'validation', 'test']:
        frame = data[data.split == split][['canonical', 'label']].rename(columns={'canonical': 'text'})
        dataset = Dataset.from_pandas(frame, preserve_index=False)
        splits[split] = dataset.map(lambda b: tokenizer(b['text'], truncation=True, max_length=MAX_LENGTH),
                                    batched=True).remove_columns('text')

    def metrics(prediction):
        pred = prediction.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(prediction.label_ids, pred, average='binary', zero_division=0)
        return dict(precision=precision, recall=recall, f1=f1)

    targs = TrainingArguments(output_dir=str(out / 'checkpoints'), learning_rate=2e-5,
                              per_device_train_batch_size=8, per_device_eval_batch_size=16,
                              gradient_accumulation_steps=2, num_train_epochs=args.epochs, weight_decay=.01,
                              eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
                              metric_for_best_model='f1', save_total_limit=1, fp16=False, seed=SEED,
                              data_seed=SEED, report_to='none', logging_steps=20, use_mps_device=(device == 'mps'),
                              dataloader_pin_memory=False)
    trainer = Trainer(model=model, args=targs, train_dataset=splits['train'], eval_dataset=splits['validation'],
                      processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=metrics)
    trainer.train()

    val = trainer.predict(splits['validation'])
    val_scores = torch.tensor(val.predictions).softmax(-1)[:, 1].numpy()
    options = [dict(threshold=float(round(t, 1)), **classification_metrics(val.label_ids, val_scores, t))
               for t in np.arange(.1, 1, .1)]
    feasible = [o for o in options if o['fpr'] <= .10]
    best = (max(feasible, key=lambda o: (o['f1'], o['threshold'])) if feasible
            else max(options, key=lambda o: (-o['fpr'], o['f1'], o['threshold'])))
    threshold = best['threshold']
    test = trainer.predict(splits['test'])
    test_scores = torch.tensor(test.predictions).softmax(-1)[:, 1].numpy()
    test_metrics = classification_metrics(test.label_ids, test_scores, threshold)
    test_rows = data[data.split == 'test'].copy()
    test_rows['score'] = test_scores
    test_rows['prediction'] = (test_scores >= threshold).astype(int)
    test_rows.to_csv(out / 'classifier_test_predictions.csv', index=False)
    authored_test = test_rows[test_rows.source.str.startswith('authored')]
    authored_metrics = classification_metrics(authored_test.label.to_numpy(), authored_test.score.to_numpy(), threshold)
    print('\nvalidation threshold options:'); print(pd.DataFrame(options).round(3).to_string(index=False))
    print('\nselected threshold:', threshold)
    print('test metrics (all sources):', test_metrics)
    print('test metrics (authored resume-domain families only):', authored_metrics)
    print('\nauthored test errors:')
    print(authored_test[authored_test.label != authored_test.prediction][['group_id', 'text', 'score']].to_string(index=False))

    export = out / 'resume_classifier'
    trainer.save_model(str(export)); tokenizer.save_pretrained(export)
    manifest = dict(**record, threshold=threshold, injection_label_id=1, validation_threshold_metrics=options,
                    test_metrics=test_metrics, authored_test_metrics=authored_metrics,
                    training_text_hashes=sorted(set(data[data.split == 'train'].canonical.map(lambda t: sha(t.encode())))))
    (export / 'training_manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'training_manifest.json').write_text(json.dumps(manifest, indent=2))  # copy outside the git-ignored weights folder
    (export / 'classifier_test_metrics.json').write_text(json.dumps(test_metrics, indent=2))
    shutil.rmtree(out / 'checkpoints', ignore_errors=True)
    archive = shutil.make_archive(str(out / 'resume_classifier'), 'zip', out, 'resume_classifier')
    print('\nexported', export, 'and', archive)


if __name__ == '__main__':
    main()
