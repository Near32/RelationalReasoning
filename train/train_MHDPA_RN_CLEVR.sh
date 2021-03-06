#!/bin/bash

python train_CLEVR.py --model=MHDPA-RN   --name=RecurrentShared2Inter64   --epochs=50 --nbrModule=2 --nbrRecurrentSharedLayer=2 --batch-size 192 \
--interactions_dim=64 --units_per_MLP_layer=512  --embedding_size=256 --hidden_size=128 --nbr_RNN_layers=2
