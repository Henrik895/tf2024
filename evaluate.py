import sys

from transformers import AutoModelForSeq2SeqLM
from transformers import NllbTokenizer

from tqdm import trange
import torch

import sacrebleu

# Constants
LANG_EST = "est_Latn"
LANG_VEP = "vep_Latn" # Veps
LANG_MOK = "mdf_Cyrl" # Moksha
LANG_OLO = "olo_Latn" # Livvi Karelian
LANG_MHR = "mhr_Cyrl" # Meadow Mari
LANG_RUS = "rus_Cyrl"

# Add languages to the tokenizer
# Source: https://cointegrated.medium.com/how-to-fine-tune-a-nllb-200-model-for-translating-a-new-language-a37fc706b865
def add_token_to_tokenizer(lang_token):
    old_len = len(tokenizer) - int(lang_token in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[lang_token] = old_len - 1
    tokenizer.id_to_lang_code[old_len - 1] = lang_token
    # always move "mask" to the last position
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset
    #
    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    #
    if lang_token not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(lang_token)
    # clear the added lang_token encoder; otherwise a new lang_token may end up there by mistake
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}

def read_in_file(file):
  lines = []
  with open(file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def translate(text, model, src_lang='est_Latn', tgt_lang='vep_Latn', max_input_length=256):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_input_length).to(device)
    with torch.no_grad():
        result = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            no_repeat_ngram_size=3,
            num_beams=5,
            max_new_tokens=int(16 + 1.5 * inputs.input_ids.shape[1])
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)

def test_model(flores_one, flores_two, one_lang, two_lang):
    one2two = [translate(flores_one[i], model, one_lang, two_lang) for i in trange(len(flores_one))]
    two2one = [translate(flores_two[i], model, two_lang, one_lang) for i in trange(len(flores_two))]
    #
    print(f"From {one_lang} to {two_lang}")
    print(sacrebleu.corpus_bleu([text[0] for text in one2two], [flores_two]))
    print(sacrebleu.corpus_chrf([text[0] for text in one2two], [flores_two], word_order = 2))
    #
    print(f"From {two_lang} to {one_lang}")
    print(sacrebleu.corpus_bleu([text[0] for text in two2one], [flores_one]))
    print(sacrebleu.corpus_chrf([text[0] for text in two2one], [flores_one], word_order = 2))

tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')

new_tokens = [
    (LANG_VEP, LANG_EST),
    (LANG_OLO, LANG_EST),
    (LANG_MOK, LANG_RUS),
    (LANG_MHR, LANG_RUS),
]

for new, similar in new_tokens:
    add_token_to_tokenizer(new)

model = AutoModelForSeq2SeqLM.from_pretrained(sys.argv[1])

device = torch.device("cuda")
model.to(device)

flores_est = read_in_file("data/FLORES_20langs/flores250.et")
flores_vep = read_in_file("data/FLORES_20langs/flores250.vep")
flores_olo = read_in_file("data/FLORES_20langs/flores250.olo")
flores_mdf = read_in_file("data/FLORES_20langs/flores250.mdf")
flores_mhr = read_in_file("data/FLORES_20langs/flores250.mhr")

test_model(flores_est, flores_vep, "est_Latn", "vep_Latn")

test_model(flores_est, flores_mdf, "est_Latn", "mdf_Cyrl")

test_model(flores_est, flores_olo, "est_Latn", "olo_Latn")

test_model(flores_est, flores_mhr, "est_Latn", "mhr_Cyrl")
