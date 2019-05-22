# RelationalReasoning
Implementations of state-of-the-art relational reasoning algorithms, using PyTorch.

# CLEVR Dataset

- [Preprocessing CLEVR](#preprocessing-clevr)
- [Disclaimers](#disclaimers)

## Preprocessing CLEVR

Before you can train any models, you need to download the
[CLEVR dataset](http://cs.stanford.edu/people/jcjohns/clevr/);
you also need to preprocess the questions ( and optionally extract features for the images, and the programs).

### Step 1: Download the data

Download and unpack the [CLEVR dataset](http://cs.stanford.edu/people/jcjohns/clevr/).

```bash
mkdir DATASETS
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip -O DATASETS/CLEVR_v1.0.zip
unzip DATASETS/CLEVR_v1.0.zip -d DATASETS
```

### Step 2: Preprocess Questions

Preprocess the questions for the CLEVR train, val, and test sets with the following commands:

```bash
python scripts/preprocess_questions.py \
  --input_data_json DATASETS/CLEVR_v1.0/questions/CLEVR_train_questions.json \
  --output_h5_file DATASETS/CLEVR_v1.0/train_questions.h5 \
  --output_vocab_json DATASETS/CLEVR_v1.0/vocab.json

python scripts/preprocess_questions.py \
  --input_data_json DATASETS/CLEVR_v1.0/questions/CLEVR_val_questions.json \
  --output_h5_file DATASETS/CLEVR_v1.0/val_questions.h5 \
  --input_vocab_json DATASETS/CLEVR_v1.0/vocab.json
  
python scripts/preprocess_questions.py \
  --input_data_json DATASETS/CLEVR_v1.0/questions/CLEVR_test_questions.json \
  --output_h5_file DATASETS/CLEVR_v1.0/test_questions.h5 \
  --input_vocab_json DATASETS/CLEVR_v1.0/vocab.json
```

When preprocessing questions, a `vocab.json` file which stores the mapping between
tokens and indices for questions is created once when processing the training set
and then reused (not expanded) to process the validation and test sets.


## Disclaimers

With regards to the benchmarking on [CLEVR dataset](http://cs.stanford.edu/people/jcjohns/clevr/), 
the scripts are heavily inspired on the ones in the dedicated [repo](https://github.com/facebookresearch/clevr-iep).