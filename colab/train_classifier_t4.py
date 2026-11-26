# %% [markdown]
# # Optional T4 training: transformer injection classifier
# Train **DistilBERT**, not TF-IDF, and export it for the real-PDF notebook.
# This is optional: that notebook already offers a pretrained injection detector.
# Start this notebook in a separate **T4 GPU Colab runtime**. It trains only a
# small encoder classifier, not the Llama screener. No API key is required.
#
# Default data: the public `deepset/prompt-injections` dataset. This is a generic
# injection dataset, not real resumes, and its results do not establish resume
# screening performance. Alternatively upload your own independently reviewed
# resume-related sentence CSV with `text,label,split,group_id` (label 0/1;
# split train/validation/test). Keep a source resume and all payload paraphrases
# in one group, and keep this classifier data separate from the final PDF benchmark.
# Model substitution requires a NEW real-PDF run ID and a new pre-run prediction;
# do not replace the model after seeing held-out PDF results and call it confirmatory.

# %%
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'transformers==4.57.1', 'datasets>=3,<5', 'accelerate>=1,<2',
    'scikit-learn', 'pandas', 'sentencepiece', 'huggingface_hub'], check=True)

# %%
import json, hashlib, random, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from google.colab import files
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

USE_CUSTOM_CSV = False
BASE_MODEL = 'distilbert/distilbert-base-uncased'
SEED = 42
EPOCHS = 3
MAX_LENGTH = 256
OUT = Path('/content/classifier_training_01')  # new path for a revised experiment
assert torch.cuda.is_available(), 'Choose a T4 GPU runtime.'
assert not (OUT / 'training_started.json').exists(), 'Training already started here. Use a fresh OUT to preserve the prior run.'
OUT.mkdir(parents=True, exist_ok=True)
set_seed(SEED)
CONFUSABLES = str.maketrans(dict(zip('асеіорху', 'aceiopxy')))
def normalized(text):
    return ' '.join(text.replace('\u200b','').translate(CONFUSABLES).lower().split())
def sha(data):
    return hashlib.sha256(data).hexdigest()

# %% [markdown]
# ## Load and freeze data
# With custom data, upload `classifier_data.csv` below. Both classes must occur in
# every split; exact normalized duplicates across splits and group overlap are
# rejected. Recommended: hundreds of independently authored/reviewed examples,
# including benign quotations, experience descriptions and varied directives.
# Mere copies or name substitutions do not add independent evidence.
#
# The default public dataset has no source-family grouping, so its random
# train/validation split can contain semantically similar examples. Report this
# limitation; do not describe it as family-disjoint evaluation.

# %%
if USE_CUSTOM_CSV:
    upload = files.upload()
    assert 'classifier_data.csv' in upload
    (OUT / 'classifier_data.csv').write_bytes(upload['classifier_data.csv'])
    data = pd.read_csv(OUT / 'classifier_data.csv', keep_default_na=False)
    assert {'text','label','split','group_id'} <= set(data.columns)
    data_source = 'user-labelled classifier_data.csv'
    data_revision = None
else:
    data_source = 'deepset/prompt-injections'
    data_revision = HfApi().dataset_info(data_source).sha
    ds = load_dataset(data_source, revision=data_revision)
    train_pool = ds['train'].to_pandas()[['text','label']].copy()
    heldout = ds['test'].to_pandas()[['text','label']].copy()
    train_pool['canonical'] = train_pool.text.map(normalized)
    heldout['canonical'] = heldout.text.map(normalized)
    # Remove train/test exact overlaps from the training pool before splitting.
    train_pool = train_pool[~train_pool.canonical.isin(heldout.canonical)].drop_duplicates('canonical')
    heldout = heldout.drop_duplicates('canonical')
    train, validation = train_test_split(train_pool, test_size=.20, random_state=SEED, stratify=train_pool.label)
    train = train.assign(split='train'); validation = validation.assign(split='validation')
    heldout = heldout.assign(split='test')
    data = pd.concat([train, validation, heldout], ignore_index=True)
    data['group_id'] = data.text.map(lambda t: sha(normalized(t).encode()))
assert set(data.split) == {'train','validation','test'}
assert set(data.label) == {0,1}
assert data.text.map(lambda x:isinstance(x,str) and bool(x.strip())).all()
data['canonical'] = data.text.map(normalized)
assert data.groupby('canonical').label.nunique().max() == 1, 'Conflicting labels for identical text'
assert data.groupby('canonical').split.nunique().max() == 1, 'Text leakage between splits'
assert data.groupby('group_id').split.nunique().max() == 1, 'Source/family leakage between splits'
for split in ['train','validation','test']:
    group = data[data.split==split]
    assert set(group.label) == {0,1} and len(group) >= 10
data.to_csv(OUT / 'frozen_data.csv', index=False)
MODEL_REV = HfApi().model_info(BASE_MODEL).sha
training_record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), seed=SEED,
    base_model=BASE_MODEL, base_revision=MODEL_REV, source=data_source, source_revision=data_revision,
    data_sha256=sha((OUT / 'frozen_data.csv').read_bytes()), epochs=EPOCHS,
    learning_rate=2e-5, max_length=MAX_LENGTH, train_batch_size=8, gradient_accumulation_steps=2,
    threshold_rule='Choose highest validation F1 with FPR<=0.10 among 0.1,...,0.9; else lowest FPR then highest F1. Tie: higher threshold.',
    prediction='No numeric success claim for this optional classifier training. The PDF hypothesis must be frozen separately before PDF evaluation.')
(OUT / 'training_started.json').write_text(json.dumps(training_record, indent=2))
display(data.groupby(['split','label']).size().unstack(fill_value=0))

# %% [markdown]
# ## Fine-tune on T4
# Three epochs; batch size 8, gradient accumulation 2, FP16, fixed seed 42.
# Best checkpoint selected by validation F1 at threshold 0.5; afterward select
# the deployment threshold using validation only. Test is evaluated once at the
# end. Training time and GPU availability vary; no guaranteed runtime is assumed.
# Long training texts are truncated at 256 tokens; counts are recorded below.

# %%
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=MODEL_REV)
model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, revision=MODEL_REV,
    num_labels=2, id2label={0:'BENIGN',1:'INJECTION'}, label2id={'BENIGN':0,'INJECTION':1})
data['token_length'] = data.canonical.map(lambda t:len(tokenizer(t, truncation=False)['input_ids']))
display(data.assign(truncated=data.token_length>MAX_LENGTH).groupby('split').truncated.sum())
data[['split','token_length']].to_csv(OUT / 'token_lengths.csv', index=False)
splits = {}
for split in ['train','validation','test']:
    frame = data[data.split==split][['canonical','label']].rename(columns={'canonical':'text'})
    dataset = Dataset.from_pandas(frame, preserve_index=False)
    splits[split] = dataset.map(lambda batch:tokenizer(batch['text'], truncation=True, max_length=MAX_LENGTH), batched=True).remove_columns('text')
def metrics(prediction):
    pred = prediction.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(prediction.label_ids, pred, average='binary', zero_division=0)
    return dict(precision=precision, recall=recall, f1=f1)
args = TrainingArguments(output_dir=str(OUT / 'checkpoints'), learning_rate=2e-5,
    per_device_train_batch_size=8, per_device_eval_batch_size=16, gradient_accumulation_steps=2,
    num_train_epochs=EPOCHS, weight_decay=.01, eval_strategy='epoch', save_strategy='epoch',
    load_best_model_at_end=True, metric_for_best_model='f1', save_total_limit=1,
    fp16=True, seed=SEED, data_seed=SEED, report_to='none', logging_steps=20)
trainer = Trainer(model=model, args=args, train_dataset=splits['train'], eval_dataset=splits['validation'],
    processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)
trainer.train()

# %%
def classification_metrics(labels, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(labels, pred, labels=[0,1]).ravel())
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='binary', zero_division=0)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=float(precision), recall=float(recall),
                f1=float(f1), fpr=fp/(fp+tn))
val = trainer.predict(splits['validation'])
val_scores = torch.tensor(val.predictions).softmax(-1)[:,1].numpy()
options = [dict(threshold=float(t), **classification_metrics(val.label_ids,val_scores,t)) for t in np.arange(.1,1,.1)]
feasible = [o for o in options if o['fpr'] <= .10]
best = max(feasible,key=lambda o:(o['f1'],o['threshold'])) if feasible else max(options,key=lambda o:(-o['fpr'],o['f1'],o['threshold']))
threshold = best['threshold']
test = trainer.predict(splits['test'])
test_scores = torch.tensor(test.predictions).softmax(-1)[:,1].numpy()
test_metrics = classification_metrics(test.label_ids, test_scores, threshold)
display(pd.DataFrame(options)); display(test_metrics)
test_rows = data[data.split=='test'].copy()
test_rows['score'] = test_scores; test_rows['prediction'] = (test_scores>=threshold).astype(int)
test_rows.to_csv(OUT / 'classifier_test_predictions.csv', index=False)
display(test_rows[test_rows.label != test_rows.prediction][['text','label','score','prediction']].head(10))

# %% [markdown]
# ## Export to the real-PDF experiment
# Download `resume_classifier.zip`. Upload it through the Files pane of the main
# notebook runtime and set `CLASSIFIER_ZIP='/content/resume_classifier.zip'`.
# The exported model uses normalized text and injection label index 1. Main-notebook
# inference scores overlapping windows and takes their maximum; this can flag more
# long blocks than this sentence-level test, so the PDF evaluation remains necessary.
# The generic-data test result is not a resume test result or an ASR measurement.

# %%
export = OUT / 'resume_classifier'
trainer.save_model(str(export)); tokenizer.save_pretrained(export)
manifest = dict(**training_record, threshold=threshold, injection_label_id=1,
    validation_threshold_metrics=options, test_metrics=test_metrics,
    training_text_hashes=sorted(set(data[data.split=='train'].canonical.map(lambda t:sha(t.encode())))))
(export / 'training_manifest.json').write_text(json.dumps(manifest, indent=2))
(export / 'classifier_test_metrics.json').write_text(json.dumps(test_metrics, indent=2))
files.download(shutil.make_archive('/content/resume_classifier', 'zip', OUT, 'resume_classifier'))
files.download(shutil.make_archive('/content/classifier_training_evidence', 'zip', OUT))

# %% [markdown]
# References: [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased),
# [deepset prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections),
# [Hugging Face text classification](https://huggingface.co/docs/transformers/tasks/sequence_classification).
# This notebook supplies a training procedure; no training results have been invented.
