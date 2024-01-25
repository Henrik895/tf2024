# Machine translation for low resource Finno-Ugric languages
## Overview
Transformers 23/24 course project. The goal is to compare tagged back-translation to the regular back-translation. In addition to back-translation, SSL denoising task is also tested.  
The results are measured using BLEU and chrF++.

Included low-resource Finno-Ugric languages:
- Veps (Latin)
- Livvi Karelian (Latin)
- Moksha (Cyrillic)
- Meadow Mari (Cyrillic)

For starting point, the [NLLB-200 600M model](https://huggingface.co/facebook/nllb-200-distilled-600M) was used. This model is pre-trained on Estonian and Russian langauges, which made it suitable for this task. The NLLB Tokenizer was extended with Veps (vep_Latn), Livvi_Karelian (olo_Latn), Moksha (mdf_Cyrl) and (mhr_Cyrl) tokens. The model embeddings corresponding to vep_Latn and olo_Latn were initialized with Estonian (est_Latn) weights and mdf_Cyrl and mhr_Cyrl tokens with Russian (rus_Cyrl) weights.

## Data

To run the fine-tuning code the expectation is that each language has 6 language files in its folder. The file format should be in form `est_Latn-vep_Latn.est_Latn`, where `est_Latn-vep_Latn` from which languages these examples are related to and `.est_Latn` in the end states that this file contains Estonian side of the sentences. Veps examples would be in the file ending with `.vep_Latn`.

The structure of the `vep` folder should be:

```
vep
│   est_Latn-vep_Latn.est_Latn
│   est_Latn-vep_Latn.vep_Latn
|   vep_Latn-est_Latn.vep_Latn
|   vep_Latn-est_Latn.est_Latn   
│
└───parallel
│   │   est_Latn-vep_Latn.est_Latn
│   │   est_Latn-vep_Latn.vep_Latn
│   │
```

⚠️ Parallel folder should include "real" parallel senteneces, which means that neither side is machine translated. The base language folder, which has 4 files, should include translated data pairs where one side is synthetic.

## Fine-tuning

If the data is present, then the fine-tuning should work by just running the `finetune.py` code. The training parameters and other configuration settings can be changed in the file.

## What about the results?

The tagged back-translation and SSL help when evaluated on the test set (90/10), split done on the data before training, but the not on the FLORES benchmark. Most likely cause is the domain set as the sentences in the FLORES benchmark are very different compared to the training data.


