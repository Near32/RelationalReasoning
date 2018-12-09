Pytorch re-implementation of Relational Module for Relational Reinforcement Learning (RRL) based on Multi-Head Dot-Product Attention (MHDPA) - [Relational Deep Reinforcement Learning](https://arxiv.org/abs/1806.01830). 

Contrary to testing in RL setting like in the paper, here we test on Sort-of-CLEVR in order to compare with other implementation relevant to that benchmark.

## Sort-of-CLEVR

Sort-of-CLEVR is simplified version of [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/).This is composed of 10000 images and 20 questions (10 relational questions and 10 non-relational questions) per each image. 6 colors (red, green, blue, orange, gray, yellow) are assigned to randomly chosen shape (square or circle), and placed in a image.

Non-relational questions are composed of 3 subtypes:

1) Shape of certain colored object
2) Horizontal location of certain colored object : whether it is on the left side of the image or right side of the image
3) Vertical location of certain colored object : whether it is on the upside of the image or downside of the image

Theses questions are "non-relational" because the agent only need to focus on certain object.

Relational questions are composed of 3 subtypes:

1) Shape of the object which is closest to the certain colored object
1) Shape of the object which is furthest to the certain colored object
3) Number of objects which have the same shape with the certain colored object

These questions are "relational" because the agent has to consider the relations between objects.

Questions are encoded into a vector of size of 11 : 6 for one-hot vector for certain color among 6 colors, 2 for one-hot vector of relational/non-relational questions. 3 for one-hot vector of 3 subtypes.

<img src="./data/sample.png" width="256">

I.e., with the sample image shown, we can generate non-relational questions like:

1) What is the shape of the red object? => Circle (even though it does not really look like "circle"...)
2) Is green object placed on the left side of the image? => yes
3) Is orange object placed on the upside of the image? => no

And relational questions:

1) What is the shape of the object closest to the red object? => square
2) What is the shape of the object furthest to the orange object? => circle
3) How many objects have same shape with the blue object? => 3

## Requirements

- Python 2.7
- [numpy](http://www.numpy.org/)
- [pytorch](http://pytorch.org/)
- [opencv](http://opencv.org/)

## Usage

Best settings :
* use LeakyReLU (give better results than ReLU ('withReLU' argument) )
* Adam optimizer
* learning rate : 1e-4 (default)
* batch size : 64 (default)
* number of MHDPA head :: 'nbrModule' : 4
* number of recurrent application :: 'nbrRecurrentSharedLayer' : 2
* layer normalization applied after the key,query, and value generators : [x]
* dimension of the key, query, and value interaction vectors :: 'interactions_dim' : 128 
* number of hidden neurons per MLP layer :: 'units_per_MLP_layer' : 128
* dropout probability :: 'dropout_prob' : 0.0 (default)

Train using :

 	 $ python main.py --model=MHDPA-RN --epochs=20 --nbrModule=4 --nbrRecurrentSharedLayer=2 --withLNGenerator --interactions_dim=128 --units_per_MLP_layer=128 


## Results

| | Relational (MHDPA) Module (20th epoch) | Relational Networks (20th epoch) | CNN + MLP (without RN, 100th epoch) |
| --- | --- | --- | --- |
| Non-relational question | 99% | 99% | 66% |
| Relational question | 88% | 89% | 66% |

You can observe the results in Tensorboard via TensorboardX :

![result](/results/result.png)


## Disclaimer

Inspired/(heavily)based on the work of [@kimhc6028](https://github.com/kimhc6028) on [relational-networks](https://github.com/kimhc6028/relational-networks).