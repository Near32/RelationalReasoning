#!/bin/bash


python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=2 --withLNGenerator --interactions_dim=128 --dropout_prob=0.5 --units_per_MLP_layer=128  
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=2 --withLNGenerator --interactions_dim=128 --units_per_MLP_layer=128
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=2 --withLNGenerator --interactions_dim=128 --dropout_prob=0.5 --units_per_MLP_layer=256  
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=2 --withLNGenerator --interactions_dim=128 --units_per_MLP_layer=256

python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=3 --withLNGenerator --interactions_dim=128 --dropout_prob=0.5 --units_per_MLP_layer=128  
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=3 --withLNGenerator --interactions_dim=128 --units_per_MLP_layer=128
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=3 --withLNGenerator --interactions_dim=128 --dropout_prob=0.5 --units_per_MLP_layer=256  
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=3 --withLNGenerator --interactions_dim=128 --units_per_MLP_layer=256

