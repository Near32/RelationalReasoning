#!/bin/bash

python train_Sort_of_CLEVR.py --model=MHDPA-RN --name=MH4DPA128_shared2_MLPunits256 --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=2 --withLNGenerator --interactions_dim=128 --dropout_prob=0.5 --units_per_MLP_layer=256 