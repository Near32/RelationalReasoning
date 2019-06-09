#!/bin/bash 
# Default value: 256
python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=2 --nbrRecurrentSharedLayers=2 --withLeakyReLU --withSoftmaxWeights &
#python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=2 --nbrRecurrentSharedLayers=2 --withLeakyReLU --withSoftmaxWeights --withModularityPrior &
#python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=2 --nbrRecurrentSharedLayers=2 --withLeakyReLU --withSoftmaxWeights --NoXavierInit --withModularityPrior &
#python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=2 --nbrRecurrentSharedLayers=2 --withLeakyReLU --withSoftmaxWeights --withLNGenerator --withModularityPrior &

python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=2 --nbrRecurrentSharedLayers=3 --withLeakyReLU --withSoftmaxWeights &
#python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=2 --nbrRecurrentSharedLayers=3 --withLeakyReLU --withSoftmaxWeights --withModularityPrior &
#python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=2 --nbrRecurrentSharedLayers=3 --withLeakyReLU --withSoftmaxWeights --NoXavierInit --withModularityPrior &
#python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=2 --nbrRecurrentSharedLayers=3 --withLeakyReLU --withSoftmaxWeights --withLNGenerator --withModularityPrior 

python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=3 --nbrRecurrentSharedLayers=2 --withLeakyReLU --withSoftmaxWeights &
#python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=3 --nbrRecurrentSharedLayers=2 --withLeakyReLU --withSoftmaxWeights --withModularityPrior &
#python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=3 --nbrRecurrentSharedLayers=2 --withLeakyReLU --withSoftmaxWeights --NoXavierInit --withModularityPrior &
#python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=3 --nbrRecurrentSharedLayers=2 --withLeakyReLU --withSoftmaxWeights --withLNGenerator --withModularityPrior &

python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=3 --nbrRecurrentSharedLayers=3 --withLeakyReLU --withSoftmaxWeights &
#python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=3 --nbrRecurrentSharedLayers=3 --withLeakyReLU --withSoftmaxWeights --withModularityPrior &
#python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=3 --nbrRecurrentSharedLayers=3 --withLeakyReLU --withSoftmaxWeights --NoXavierInit --withModularityPrior &
#python main.py --model=SARN      --epochs=25 --units_per_MLP_layer=$1 --conv_dim=24 --batch-size=32 --nbrParallelAttention=3 --nbrRecurrentSharedLayers=3 --withLeakyReLU --withSoftmaxWeights --withLNGenerator --withModularityPrior &
