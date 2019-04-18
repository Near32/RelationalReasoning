#!/bin/bash

python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=2 --nbrRecurrentSharedLayer=1 \
--interactions_dim=64 --units_per_MLP_layer=512  &
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=1 \
--interactions_dim=64 --units_per_MLP_layer=512  &
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=6 --nbrRecurrentSharedLayer=1 \
--interactions_dim=64 --units_per_MLP_layer=512  &
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=8 --nbrRecurrentSharedLayer=1 \
--interactions_dim=64 --units_per_MLP_layer=512  &

#python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=2 --nbrRecurrentSharedLayer=2
#python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=2
#python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=6 --nbrRecurrentSharedLayer=2
#python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=8 --nbrRecurrentSharedLayer=2

#python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=2 --nbrRecurrentSharedLayer=3
#python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=3
#python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=6 --nbrRecurrentSharedLayer=3
#python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=8 --nbrRecurrentSharedLayer=3

#python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=2 --nbrRecurrentSharedLayer=4
#python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=4
#python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=6 --nbrRecurrentSharedLayer=4
#python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=8 --nbrRecurrentSharedLayer=4
