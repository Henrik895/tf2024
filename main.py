import random
import time
from pathlib import Path

import sacrebleu
import torch
from datasets import Dataset, concatenate_datasets
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          NllbTokenizer, get_scheduler)


# Constants
LANG_EST = "est_Latn"
LANG_VEP = "vep_Latn" # Veps
LANG_MOK = "mdf_Cyrl" # Moksha
LANG_OLO = "olo_Latn" # Livvi Karelian
LANG_MHR = "mhr_Cyrl" # Meadow Mari
LANG_RUS = "rus_Cyrl"

BT_TAG = "¶"


# Global setttings
RANDOM_SEED = 42 # Used only for back-translated data selection (so that runs different runs are comparable)
PARALLEL_CAP = None # Limit to how much parallel data can be included (expects a number)
TAG_SYNTHETIC_SRC = False # Is tagged back-translation used or not
SYNTHETIC_RATIO = 2 # How much synthetic data to use (back-translation) - parallel_examples * ratio
FOLDER_NAME = "training"

BATCH_SIZE = 4
EPOCHS = 5

TEST_SIZE = 0.1

CHECKPOINT = "facebook/nllb-200-distilled-600M"


# Loading model and tokenizer

tokenizer = NllbTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)


# Loading and preparing data

def load_data(path):
  with open(path, "r") as file:
    sents = file.read().splitlines()

  return sents


def prepare_parallel_dataset(source, target, /, nr_examples = None, tag_enabled = False, tag_token = None):
  if tag_enabled and tag_token is None:
    raise ValueError("If tag_enabled is True then tag_token can not be None")

  source_path, source_lang = source["path"], source["lang"]
  target_path, target_lang = target["path"], target["lang"]

  source_sents = load_data(source_path)
  target_sents = load_data(target_path)

  if PARALLEL_CAP is not None:
    nr_examples = PARALLEL_CAP

  if nr_examples is not None:
      combined = list(zip(source_sents, target_sents))

      rng = random.Random(RANDOM_SEED)
      rng.shuffle(combined)

      selection = combined[:nr_examples]
      source_sents, target_sents = zip(*selection)

  if tag_enabled:
    source_sents = [tag_token + sent for sent in source_sents]

  translations = []
  for src_sent, tgt_sent in zip(source_sents, target_sents):
    example = {
        source_lang: src_sent,
        target_lang: tgt_sent,
    }
    translations.append(example)

  data = {
      "id": [i + 1 for i in range(len(translations))],
      "translations": translations,
  }

  return Dataset.from_dict(data)


# Parallel data

# EST-VEP
est_vep = {
    "path": "vep/et-vep.et.normal",
    "lang": LANG_EST,
}

vep_est = {
    "path": "vep/et-vep.vep.normal",
    "lang": LANG_VEP,
}

# EST-MOK
est_mdf = {
    "path": "mdf/et-mdf.et.normal",
    "lang": LANG_EST,
}

mdf_est = {
    "path": "mdf/et-mdf.mdf.normal",
    "lang": LANG_MOK,
}

# EST-OLO
est_olo = {
    "path": "olo/et-olo.et.normal",
    "lang": LANG_EST,
}

olo_est = {
    "path": "olo/et-olo.olo.normal",
    "lang": LANG_OLO,
}

# EST-MHR
est_mhr = {
    "path": "mhr/et-mhr.et.normal",
    "lang": LANG_EST,
}

mhr_est = {
    "path": "mhr/et-mhr.mhr.normal",
    "lang": LANG_MHR,
}

# Preparing datasets
parallel_est_vep = prepare_parallel_dataset(est_vep, vep_est)
parallel_est_mdf = prepare_parallel_dataset(est_mdf, mdf_est)
parallel_est_olo = prepare_parallel_dataset(est_olo, olo_est)
parallel_est_mhr = prepare_parallel_dataset(est_mhr, mhr_est)


# Synthetic data

# Preparing back-translated data
est_vep_examples = len(parallel_est_vep) * SYNTHETIC_RATIO
est_mdf_examples = len(parallel_est_mdf) * SYNTHETIC_RATIO
est_olo_examples = len(parallel_est_olo) * SYNTHETIC_RATIO
est_mhr_examples = len(parallel_est_mhr) * SYNTHETIC_RATIO

# EST-VEP
est_vep_src = {
    "path": "vep/train.est_Latn-vep_Latn.est_Latn",
    "lang": LANG_EST,
}

est_vep_tgt = {
    "path": "vep/train.est_Latn-vep_Latn.vep_Latn",
    "lang": LANG_VEP,
}

vep_est_src = {
    "path": "vep/train.vep_Latn-est_Latn.vep_Latn",
    "lang": LANG_VEP,
}

vep_est_tgt = {
    "path": "vep/train.vep_Latn-est_Latn.est_Latn",
    "lang": LANG_EST,
}

# EST-MOK
est_mdf_src = {
    "path": "mdf/train.est_Latn-mdf_Cyrl.est_Latn",
    "lang": LANG_EST,
}

est_mdf_tgt = {
    "path": "mdf/train.est_Latn-mdf_Cyrl.mdf_Cyrl",
    "lang": LANG_MOK,
}

mdf_est_src = {
    "path": "mdf/train.mdf_Cyrl-est_Latn.mdf_Cyrl",
    "lang": LANG_MOK,
}

mdf_est_tgt = {
    "path": "mdf/train.mdf_Cyrl-est_Latn.est_Latn",
    "lang": LANG_EST,
}

# EST-OLO
est_olo_src = {
    "path": "olo/train.est_Latn-olo_Latn.est_Latn",
    "lang": LANG_EST,
}

est_olo_tgt = {
    "path": "olo/train.est_Latn-olo_Latn.olo_Latn",
    "lang": LANG_OLO,
}

olo_est_src = {
    "path": "olo/train.olo_Latn-est_Latn.olo_Latn",
    "lang": LANG_OLO,
}

olo_est_tgt = {
    "path": "olo/train.olo_Latn-est_Latn.est_Latn",
    "lang": LANG_EST,
}

# EST-MHR
est_mhr_src = {
    "path": "mhr/train.est_Latn-mhr_Cyrl.est_Latn",
    "lang": LANG_EST,
}

est_mhr_tgt = {
    "path": "mhr/train.est_Latn-mhr_Cyrl.mhr_Cyrl",
    "lang": LANG_MHR,
}

mhr_est_src = {
    "path": "mhr/train.mhr_Cyrl-est_Latn.mhr_Cyrl",
    "lang": LANG_MHR,
}

mhr_est_tgt = {
    "path": "mhr/train.mhr_Cyrl-est_Latn.est_Latn",
    "lang": LANG_EST,
}

# Directions are switched on purpose for reason mentioned above
translated_dataset_vep_est = prepare_parallel_dataset(est_vep_tgt, est_vep_src, nr_examples = est_vep_examples, tag_enabled = TAG_SYNTHETIC_SRC, tag_token = BT_TAG)
translated_dataset_est_vep = prepare_parallel_dataset(vep_est_tgt, vep_est_src, nr_examples = est_vep_examples, tag_enabled = TAG_SYNTHETIC_SRC, tag_token = BT_TAG)

translated_dataset_mdf_est = prepare_parallel_dataset(est_mdf_tgt, est_mdf_src, nr_examples = est_mdf_examples, tag_enabled = TAG_SYNTHETIC_SRC, tag_token = BT_TAG)
translated_dataset_est_mdf = prepare_parallel_dataset(mdf_est_tgt, mdf_est_src, nr_examples = est_mdf_examples, tag_enabled = TAG_SYNTHETIC_SRC, tag_token = BT_TAG)

translated_dataset_olo_est = prepare_parallel_dataset(est_olo_tgt, est_olo_src, nr_examples = est_olo_examples, tag_enabled = TAG_SYNTHETIC_SRC, tag_token = BT_TAG)
translated_dataset_est_olo = prepare_parallel_dataset(olo_est_tgt, olo_est_src, nr_examples = est_olo_examples, tag_enabled = TAG_SYNTHETIC_SRC, tag_token = BT_TAG)

translated_dataset_mhr_est = prepare_parallel_dataset(est_mhr_tgt, est_mhr_src, nr_examples = est_mhr_examples, tag_enabled = TAG_SYNTHETIC_SRC, tag_token = BT_TAG)
translated_dataset_est_mhr = prepare_parallel_dataset(mhr_est_tgt, mhr_est_src, nr_examples = est_mhr_examples, tag_enabled = TAG_SYNTHETIC_SRC, tag_token = BT_TAG)


# Add languages to the tokenizer
# Source: https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865
def add_token_to_tokenizer(lang_token, similar_token):
  old_len = len(tokenizer) - int(lang_token in tokenizer.added_tokens_encoder)
  tokenizer.lang_code_to_id[lang_token] = old_len - 1
  tokenizer.id_to_lang_code[old_len - 1] = lang_token
  # always move "mask" to the last position
  tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

  tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
  tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}

  if lang_token not in tokenizer._additional_special_tokens:
      tokenizer._additional_special_tokens.append(lang_token)
  # clear the added lang_token encoder; otherwise a new lang_token may end up there by mistake
  tokenizer.added_tokens_encoder = {}
  tokenizer.added_tokens_decoder = {}

  added_token_id = tokenizer.convert_tokens_to_ids(lang_token)
  similar_lang_id = tokenizer.convert_tokens_to_ids(similar_token)

  model.resize_token_embeddings(len(tokenizer))
  # moving the embedding for "mask" to its new position
  model.model.shared.weight.data[added_token_id + 1] = model.model.shared.weight.data[added_token_id]
  # initializing new language token with a token of a similar language
  model.model.shared.weight.data[added_token_id] = model.model.shared.weight.data[similar_lang_id]


new_tokens = [
    (LANG_VEP, LANG_EST),
    (LANG_OLO, LANG_EST),
    (LANG_MOK, LANG_RUS),
    (LANG_MHR, LANG_RUS),
]

for new, similar in new_tokens:
  add_token_to_tokenizer(new, similar)

# Tokenize datasets

def preprocess(examples, src_lang, tgt_lang):
  tokenizer.src_lang = src_lang
  tokenizer.tgt_lang = tgt_lang
  sources = [example[src_lang] for example in examples["translations"]]
  targets = [example[tgt_lang] for example in examples["translations"]]
  inputs = tokenizer(sources, text_target=targets, max_length=200, truncation=True)
  return inputs


test_size = TEST_SIZE

# EST-VEP
split_est_vep_par = parallel_est_vep.train_test_split(test_size=test_size)
split_est_vep = translated_dataset_vep_est.train_test_split(test_size=test_size)
split_vep_est = translated_dataset_est_vep.train_test_split(test_size=test_size)

est_vep_train = [
  split_est_vep_par["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_VEP}),
  split_est_vep_par["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_VEP, "tgt_lang": LANG_EST}),
  split_est_vep["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_VEP}),
  split_vep_est["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_VEP, "tgt_lang": LANG_EST}),
]

est_vep_test = [
  split_est_vep_par["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_VEP}),
  split_est_vep_par["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_VEP, "tgt_lang": LANG_EST}),
  split_est_vep["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_VEP}),
  split_vep_est["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_VEP, "tgt_lang": LANG_EST}),
]

# EST-MOK
split_est_mdf_par = parallel_est_mdf.train_test_split(test_size=test_size)
split_est_mdf = translated_dataset_mdf_est.train_test_split(test_size=test_size)
split_mdf_est = translated_dataset_est_mdf.train_test_split(test_size=test_size)

est_mdf_train = [
  split_est_mdf_par["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_MOK}),
  split_est_mdf_par["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_MOK, "tgt_lang": LANG_EST}),
  split_est_mdf["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_MOK}),
  split_mdf_est["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_MOK, "tgt_lang": LANG_EST}),
]

est_mdf_test = [
  split_est_mdf_par["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_MOK}),
  split_est_mdf_par["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_MOK, "tgt_lang": LANG_EST}),
  split_est_mdf["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_MOK}),
  split_mdf_est["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_MOK, "tgt_lang": LANG_EST}),
]

# EST-OLO
split_est_olo_par = parallel_est_olo.train_test_split(test_size=test_size)
split_est_olo = translated_dataset_olo_est.train_test_split(test_size=test_size)
split_olo_est = translated_dataset_est_olo.train_test_split(test_size=test_size)

est_olo_train = [
  split_est_olo_par["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_OLO}),
  split_est_olo_par["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_OLO, "tgt_lang": LANG_EST}),
  split_est_olo["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_OLO}),
  split_olo_est["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_OLO, "tgt_lang": LANG_EST}),
]

est_olo_test = [
  split_est_olo_par["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_OLO}),
  split_est_olo_par["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_OLO, "tgt_lang": LANG_EST}),
  split_est_olo["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_OLO}),
  split_olo_est["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_OLO, "tgt_lang": LANG_EST}),
]


# EST-MHR
split_est_mhr_par = parallel_est_mhr.train_test_split(test_size=test_size)
split_est_mhr = translated_dataset_mhr_est.train_test_split(test_size=test_size)
split_mhr_est = translated_dataset_est_mhr.train_test_split(test_size=test_size)

est_mhr_train = [
  split_est_mhr_par["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_MHR}),
  split_est_mhr_par["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_MHR, "tgt_lang": LANG_EST}),
  split_est_mhr["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_MHR}),
  split_mhr_est["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_MHR, "tgt_lang": LANG_EST}),
]

est_mhr_test = [
  split_est_mhr_par["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_MHR}),
  split_est_mhr_par["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_MHR, "tgt_lang": LANG_EST}),
  split_est_mhr["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_MHR}),
  split_mhr_est["test"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_MHR, "tgt_lang": LANG_EST}),
]

# Creating combined train and test datasets

train_datasets = [
  *est_vep_train,
  *est_mdf_train,
  *est_olo_train,
  *est_mhr_train,
]

test_datasets = [
  *est_vep_test,
  *est_mdf_test,
  *est_olo_test,
  *est_mhr_test,
]

def prepare_tokenized_ds(dataset):
  dataset = dataset.remove_columns(['id', 'translations'])
  dataset.set_format('torch')
  dataset.column_names

  return dataset


processed_train_datasets = []
processed_test_datasets = []

for dataset in train_datasets:
  dataset = prepare_tokenized_ds(dataset)
  processed_train_datasets.append(dataset)


for dataset in test_datasets:
  dataset = prepare_tokenized_ds(dataset)
  processed_test_datasets.append(dataset)


tokenized_train = concatenate_datasets(processed_train_datasets)
tokenized_test = concatenate_datasets(processed_test_datasets)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=CHECKPOINT, padding=True)

# Creating the dataloader

train_dataloader = DataLoader(
    tokenized_train, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)

test_dataloader = DataLoader(
    tokenized_test, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)

for batch in train_dataloader:
  print({k: v.shape for k, v in batch.items()})
  break


# Test: can model accept inputs
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

# Fine-tuning the model

optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = EPOCHS
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device

progress_bar = tqdm(range(num_training_steps))

def train(model, dataloader, optimizer, scheduler):
  total_loss = 0
  model.train()
  for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    progress_bar.update(1)

    total_loss += loss.item()

  return total_loss / len(dataloader)


def test(model, dataloader):
  total_loss = 0
  total_bleu = 0
  total_chr = 0
  model.eval()
  for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    predictions = tokenizer.batch_decode(predictions.tolist(), skip_special_tokens=True)
    labels = [[id if id >= 0 else 1 for id in sent] for sent in batch["labels"].tolist()]
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    total_loss += outputs.loss.item()
    total_bleu += sacrebleu.corpus_bleu(predictions, [labels]).score
    total_chr += sacrebleu.corpus_chrf(predictions, [labels], word_order = 2).score

  return total_loss / len(dataloader), total_bleu / len(dataloader), total_chr / len(dataloader)


train_loss_log, dev_loss_log, dev_bleu_log, dev_chr_log, lr_log = [], [], [], [], []
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer, lr_scheduler)
    dev_loss, dev_bleu, dev_chr = test(model, test_dataloader)
    train_loss_log.append(train_loss)
    dev_loss_log.append(dev_loss)
    dev_bleu_log.append(dev_bleu)
    dev_chr_log.append(dev_chr)
    lr_log.append(lr_scheduler.get_lr())


# Store training metrics for later use

now = int(time.time())

path = Path('.') / f"{FOLDER_NAME}-{now}"
path.mkdir(parents=True, exist_ok=True)

def store_metric(items, name, base_path):
  metric_path = base_path / name
  with metric_path.open("w") as f:
    for metric in items:
      f.write(f"{metric}\n")


metrics = [
  (train_loss_log, "train_loss"),
  (dev_loss_log, "dev_loss"),
  (dev_bleu_log, "bleu"),
  (dev_chr_log, "chr"),
  (lr_log, "lr"),
]

for items, name in metrics:
  store_metric(items, name, path)
