from transformers import NllbTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, get_scheduler
from datasets import Dataset, concatenate_datasets
import random
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from tqdm.auto import tqdm
import sacrebleu
from pathlib import Path


# Config
LANG_EST = "est_Latn"
LANG_VEP = "vep_Latn" # Veps
LANG_MOK = "mdf_Cyrl" # Moksha
LANG_OLO = "olo_Latn" # Livvi Karelian
LANG_MHR = "mhr_Cyrl" # Meadow Mari
LANG_RUS = "rus_Cyrl"

# Folder where the translation files for each language are located
BASE_PATH = "data"
PATH_VEP = f"{BASE_PATH}/vep"
PATH_MOK = f"{BASE_PATH}/mdf"
PATH_OLO = f"{BASE_PATH}/olo"
PATH_MHR = f"{BASE_PATH}/mhr"

LANG_TO_PATH = {
    LANG_VEP: PATH_VEP,
    LANG_MOK: PATH_MOK,
    LANG_OLO: PATH_OLO,
    LANG_MHR: PATH_MHR,
}

# Languages that use latn alphabet are initialized with Estonian weights.
# Languages that use cyrl alphabet are initialized with Russian weights.
LANG_TO_SIMILAR = {
    LANG_VEP: LANG_EST,
    LANG_OLO: LANG_EST,
    LANG_MOK: LANG_RUS,
    LANG_MHR: LANG_RUS,
}

# Training and experiment related settings
RANDOM_SEED = 42 # Used to make comparisons between runs more "equal". Initializes random generators, which are used to select examples from translated and monolingual sentences.

BT_ENABLED = False # Back-translation
BT_DATA_RATIO = 3 # How many back-translation examples should be included for each "real" parallel training example.
BT_USE_TAG = False # True for tagged back-translation
BT_TAG = "¶" # Can be really anything, but it should be something that the tokenizer can encode with only one integer

SSL_ENABLED = False # Denoising task (uses monolingual sentences and creates noised source sentences, which the model uses to predict the original denoised sentences)
SSL_DATA_RATIO = 1 # How much monolingual data to include (same as BT_DATA_RATIO)

CHECKPOINT = "facebook/nllb-200-distilled-600M"

EPOCHS = 5
BATCH_SIZE = 4
TEST_SIZE = 0.1 # Used to create training metrics after each epoch

FOLDER_NAME = "training" # Where fine-tuned model and training logs are saved

# Functions

def load_data(path):
    with open(path, "r") as file:
        sents = file.read().splitlines()

    return sents


def prepare_parallel_dataset(source, target, /, nr_examples = None, tag_enabled = False, tag_token = None):
    """
    Takes two language files and merges them for preprocessing.

    Args:
        source: Object with format: {"path": "data/vep/est_Latn-vep_Latn.est_Latn", "lang": "est_Latn"}, which tells the method where to find the 
                source text file and to which language the sentences in this file belong to.

        target: Same as above. Needs to be separately specified because this method is also used to prepare sentences for back-translation and denoising.

        nr_examples: Maximum number of included sentences.

        tag_enabled: If the method is used for preparing back-translation examples, when set to True the sentences in the source side are marking with a
                     the tag_token symbol.

        tag_token: Symbol to use for tagged back-translation.
    Returns:
        Parallel Dataset, with attributes "id" and "translations" with source and target language sentences. 
        {"id": [1], "translations": [{"est_Latn": "Test", "vep_Latn": "test"}]}
    
    """
    if tag_enabled and tag_token is None:
        raise ValueError("If tag_enabled is True then tag_token can not be None")

    source_path, source_lang = source["path"], source["lang"]
    target_path, target_lang = target["path"], target["lang"]

    source_sents = load_data(source_path)
    target_sents = load_data(target_path)

    if nr_examples is not None:
        combined = list(zip(source_sents, target_sents))

        rng = random.Random(RANDOM_SEED) # Remove for totally random selection
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


def noise(input_ids, tok_start, tok_end, rand):
    """
    Replaces ~15% of the tokens in the sequence with random tokens.

    Args:
        input_ids: List of integers (text encoded by the tokenizer)

        tok_start: the lowest acceptable token value

        tok_end: the highest acceptable token value

        rand: Random object that is used for generating random token values
    Returns:
        List of integers, where ~15% of the values are "randomly" sampled from the range [tok_start, tok_end]
    """
    if len(input_ids) < 4: # Nothing to do, only one token (the first and the last token are special tokens, not part of the sentence)
        return input_ids

    for i in range(1, len(input_ids) - 1):
        substitute = rand.random() <= 0.15
        if substitute == False:
            continue

        input_ids[i] = rand.randint(tok_start, tok_end)

    return input_ids


def preprocess(examples, tokenizer, src_lang, tgt_lang, add_noise = False, tok_start = None, tok_end = None, rand = None):
    """
    Preprocesses the text sentences for model training.

    Args:
        examples: Output from prepare_parallel_dataset function: {"id": [1], "translations": [{"est_Latn": "Test", "vep_Latn": "test"}]}.

        tokenizer: NllbTokenizer

        src_lang: The examples "tranlations" consists of source and target side sentences. Specifies, which one is the source.

        tgt_lang: Target side of the sentences.

        add_noise: If set to True, the source side sentences will be modified using the "noise" function.

        tok_start: The lowest allowed noise token value for "noise" function.

        tok_end: The highest allowed noise token value for "noise" function.

        rand: The random generator that is used in the "noise" function.
    Returns:
        Inputs that can be feed into the NllB model. Must remove "id" and "translation" fields beforehand.
    """
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    sources = [example[src_lang] for example in examples["translations"]]
    targets = [example[tgt_lang] for example in examples["translations"]]
    inputs = tokenizer(sources, text_target=targets, max_length=200, truncation=True) # max_length 200 is used by pre-trained NLLB model. Should not be changed.

    if add_noise:
        input_ids = inputs.input_ids
        for i in range(len(input_ids)):
            input_ids[i] = noise(input_ids[i], tok_start, tok_end, rand)

        inputs.input_ids = input_ids

    return inputs



def add_token_to_tokenizer(model, tokenizer, lang_token, similar_token):
    """
    SOURCE: https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865

    Adds the new language support to the tokenizer and adjusts the modele embeddings accordingly.
    The starting value for the new token, lang_token, is copied from similar_token embedding.
    """
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


def prepare_tokenized_ds(dataset):
    dataset = dataset.remove_columns(['id', 'translations'])
    dataset.set_format('torch')
    dataset.column_names

    return dataset


# Classes for working with language files
class LanguageFile:
    """
    Convenience class for holding language files that are used together.
    Is used to create inputs for prepare_parallel_dataset function.
    """
    def __init__(self, folder, src, tgt, extra_path = "") -> None:
        self.folder = folder
        self.src = src
        self.tgt = tgt
        self.extra_path = extra_path

    def get_lang_file(self):
        """ For source side file """
        return {
            "path": str(Path(self.folder) / self.extra_path / f"{self.src}-{self.tgt}.{self.src}"),
            "lang": self.src,
        }
    
    def get_lang_file_opposite(self):
        """ For target side file """
        return {
            "path": str(Path(self.folder) / self.extra_path / f"{self.src}-{self.tgt}.{self.tgt}"),
            "lang": self.tgt,
        }


class LanguageLoader:
    """
    Language file preparation class, which allows to access parallel data (both sides are "real" examples),
    translated data (one side is synthetic) and monolingual data (both sides are in the same language).
    """
    def __init__(self, folder, lang_src, lang_tgt) -> None:
        self.folder = folder
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt

    def get_parallel_data(self):
        parallel = LanguageFile(self.folder, self.lang_src, self.lang_tgt, extra_path="parallel")
        src_tgt = parallel.get_lang_file()
        tgt_src = parallel.get_lang_file_opposite()
        return prepare_parallel_dataset(src_tgt, tgt_src)
    
    def get_translated_data(self, **kwargs):
        src_tgt = LanguageFile(self.folder, self.lang_src, self.lang_tgt)
        src_tgt_src = src_tgt.get_lang_file()
        src_tgt_tgt = src_tgt.get_lang_file_opposite()

        # Direction is changed because translated data, which is synthetic, should be on the source side
        tgt_to_src = prepare_parallel_dataset(src_tgt_tgt, src_tgt_src, **kwargs)

        tgt_src = LanguageFile(self.folder, self.lang_tgt, self.lang_src)
        tgt_src_tgt = tgt_src.get_lang_file()
        tgt_src_src = tgt_src.get_lang_file_opposite()

        # Same reason
        src_to_tgt = prepare_parallel_dataset(tgt_src_src, tgt_src_tgt, **kwargs)

        return tgt_to_src, src_to_tgt
    
    def get_mono_data(self, **kwargs):
        """ 
        Creates two parallel datasets, where the source and target sides of the sentences are in the same language.
        """
        src_tgt_src = LanguageFile(self.folder, self.lang_src, self.lang_tgt).get_lang_file()
        tgt_src_tgt = LanguageFile(self.folder, self.lang_tgt, self.lang_src).get_lang_file()

        src_to_src = prepare_parallel_dataset(src_tgt_src, src_tgt_src, **kwargs)
        tgt_to_tgt = prepare_parallel_dataset(tgt_src_tgt, tgt_src_tgt, **kwargs)

        return src_to_src, tgt_to_tgt


class LanguagePair(LanguageLoader):
   def __init__(self, folder, lang_src, lang_tgt, /, 
                bt_enabled = False, bt_ratio = 1, tag_enabled = False, 
                tag_token = None, ssl_enabled = False, ssl_ratio = 1) -> None:
        super().__init__(folder, lang_src, lang_tgt)

        self.parallel = self.get_parallel_data()

        # Prepare translated datasets (for back-translation)
        if bt_enabled:
            nr_examples = len(self.parallel) * bt_ratio
            self.tgt_to_src, self.src_to_tgt = self.get_translated_data(nr_examples=nr_examples, tag_enabled=tag_enabled, tag_token=tag_token)

        # Prepare datasets for same language translation (denoising)
        if ssl_enabled:
            nr_examples = len(self.parallel) * ssl_ratio
            self.mono_src, self.mono_tgt = self.get_mono_data(nr_examples=nr_examples)


# Fine-tuning
           
tokenizer = NllbTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)

# Language Pairs
kwargs = {
    "bt_enabled": BT_ENABLED,
    "bt_ratio": BT_DATA_RATIO,
    "tag_enabled": BT_USE_TAG,
    "tag_token": BT_TAG,
    "ssl_enabled": SSL_ENABLED,
    "ssl_ratio": SSL_DATA_RATIO,
}

languages = {lang: LanguagePair(path, LANG_EST, lang, **kwargs) for lang, path in LANG_TO_PATH.items()}

# Add new language tokens to the tokenizer and adjust model embeddings
for lang, similar in LANG_TO_SIMILAR.items():
  add_token_to_tokenizer(model, tokenizer, lang, similar)


rng = random.Random(RANDOM_SEED)

# start and end mark the range of "usable" tokens, excluding special and language tokens (such as <s>, est_Latn etc)
tokenizer_start = 4 # "a"
tokenizer_end = tokenizer.vocab_size - len(tokenizer.additional_special_tokens) - 2

train_datasets = []
test_datasets = []

# Every language is currently paired only with Estonian, not each other, so we can use LANG_EST directly
for lang, _ in LANG_TO_PATH.items():
    lang_data = languages[lang]

    parallel_data = lang_data.parallel.train_test_split(test_size=TEST_SIZE)

    # Even-though it is parallel data, the language model learns only in one direction so we must include it in both directions.
    train_datasets.extend(
        [
            parallel_data["train"].map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer, "src_lang": LANG_EST, "tgt_lang": lang}),
            parallel_data["train"].map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer, "src_lang": lang, "tgt_lang": LANG_EST}),
        ]
    )
    
    test_datasets.extend(
        [
            parallel_data["test"].map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer, "src_lang": LANG_EST, "tgt_lang": lang}),
            parallel_data["test"].map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer, "src_lang": lang, "tgt_lang": LANG_EST}),
        ]
    )

    if BT_ENABLED:
        src_to_tgt = lang_data.src_to_tgt.train_test_split(test_size=TEST_SIZE)
        tgt_to_src = lang_data.tgt_to_src.train_test_split(test_size=TEST_SIZE)

        train_datasets.extend(
            [
                src_to_tgt["train"].map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer, "src_lang": LANG_EST, "tgt_lang": lang}),
                tgt_to_src["train"].map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer, "src_lang": lang, "tgt_lang": LANG_EST}),
            ]
        )

        test_datasets.extend(
            [
                src_to_tgt["test"].map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer, "src_lang": LANG_EST, "tgt_lang": lang}),
                tgt_to_src["test"].map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer, "src_lang": lang, "tgt_lang": LANG_EST}),
            ]
        )

    if SSL_ENABLED:
        mono_src = lang_data.mono_src.train_test_split(test_size=TEST_SIZE)
        mono_tgt = lang_data.mono_tgt.train_test_split(test_size=TEST_SIZE)

        args = {
            "tokenizer": tokenizer,
            "add_noise": SSL_ENABLED,
            "tok_start": tokenizer_start,
            "tok_end": tokenizer_end,
            "rand": rng,
        }

        # SSL data is only used for training not validating (there is no point of calculating BLEU and chrFr++ for denoising task)
        train_datasets.extend(
            [
                mono_src["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": LANG_EST, "tgt_lang": LANG_EST, **args}),
                mono_tgt["train"].map(preprocess, batched=True, fn_kwargs={"src_lang": lang, "tgt_lang": lang, **args}),
            ]
        )


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

train_dataloader = DataLoader(
    tokenized_train, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)

test_dataloader = DataLoader(
    tokenized_test, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)


# Training
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

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


def test(model, dataloader, tokenizer):
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


best_path = Path(FOLDER_NAME) / 'best'
best_path.mkdir(parents=True, exist_ok=True)
best_loss = 99999

train_loss_log, dev_loss_log, dev_bleu_log, dev_chr_log, lr_log = [], [], [], [], []
for epoch in range(EPOCHS):
    train_loss = train(model, train_dataloader, optimizer, lr_scheduler)
    dev_loss, dev_bleu, dev_chr = test(model, test_dataloader, tokenizer)
    train_loss_log.append(train_loss)
    dev_loss_log.append(dev_loss)
    dev_bleu_log.append(dev_bleu)
    dev_chr_log.append(dev_chr)
    lr_log.append(lr_scheduler.get_lr())

    if dev_loss < best_loss:
       model.save_pretrained(f"{FOLDER_NAME}/best")
       best_loss = dev_loss

    print(f"Epoch: {epoch + 1}, train_l: {train_loss}, dev_l: {dev_loss}, dev_bleu: {dev_bleu}, dev_chr: {dev_chr}")


last_path = Path(FOLDER_NAME) / 'last'
last_path.mkdir(parents=True, exist_ok=True)

model.save_pretrained(f"{FOLDER_NAME}/last")
tokenizer.save_pretrained(f"{FOLDER_NAME}/last")


# Save metrics
base_metric_path = Path(FOLDER_NAME) / "metrics"
base_metric_path.mkdir(parents=True, exist_ok=True)

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
    store_metric(items, name, base_metric_path)