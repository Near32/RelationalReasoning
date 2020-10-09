#!/bin/bash

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