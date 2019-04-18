#!/bin/bash 
# Default value: 256
python main.py --model=SARN      --epochs=40 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=1 --withLeakyReLU --withSoftmaxWeights &
python main.py --model=SARN      --epochs=40 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=1 --withLeakyReLU --withSoftmaxWeights --NoXavierInit &
python main.py --model=SARN      --epochs=40 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=1 --withLeakyReLU --withSoftmaxWeights --withLNGenerator

python main.py --model=SARN      --epochs=40 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=2 --withLeakyReLU --withSoftmaxWeights &
python main.py --model=SARN      --epochs=40 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=2 --withLeakyReLU --withSoftmaxWeights --NoXavierInit &
python main.py --model=SARN      --epochs=40 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=2 --withLeakyReLU --withSoftmaxWeights --withLNGenerator &
