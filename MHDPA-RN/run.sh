#!/bin/bash

python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=1 --nbrRecurrentSharedLayer=1
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=2 --nbrRecurrentSharedLayer=1
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=3 --nbrRecurrentSharedLayer=1
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=1

python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=1 --nbrRecurrentSharedLayer=2
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=2 --nbrRecurrentSharedLayer=2
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=3 --nbrRecurrentSharedLayer=2
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=2

python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=1 --nbrRecurrentSharedLayer=3
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=2 --nbrRecurrentSharedLayer=3
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=3 --nbrRecurrentSharedLayer=3
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=3

python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=1 --nbrRecurrentSharedLayer=4
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=2 --nbrRecurrentSharedLayer=4
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=3 --nbrRecurrentSharedLayer=4
python main.py --model=MHDPA-RN      --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=4
