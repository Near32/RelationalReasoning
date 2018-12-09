Pytorch re-implementation of Relational Networks - [A simple neural network module for relational reasoning](https://arxiv.org/pdf/1706.01427.pdf)

Tested on Sort-of-CLEVR.

## Sort-of-CLEVR

Sort-of-CLEVR is a simplified version of [CLEVR](http://cs.stanford.edu/people/jcjohns/clevr/). It is composed of 10000 images and 20 questions (10 relational questions and 10 non-relational questions) for each image. For each image, 6 colors (red, green, blue, orange, gray, yellow) are assigned to randomly chosen shape (square or circle).

Please refer to the links in the disclaimer for even more details.

## Requirements

Tested with :

- Python 3.5
- numpy==1.14.5
- torch==0.4.1
- torchvision==0.1.9
- opencv-python==3.4.1.15

## Usage

 	 $ python main.py 

to train.

## Results

| | Relational Networks (20th epoch, using model 'RN2') | Relational Networks (20th epoch, using model 'RN') | CNN + MLP (without RN, 100th epoch) |
| --- | --- | --- | --- |
| Non-relational questions | 99% | 99% | 66% |
| Relational questions | 89% | 89% | 66% |

Relational networks shows far better results in relational questions and non-relation questions. The main implementation of this repository (RN2) performs on par with the implementation of [@kimhc6028](https://github.com/kimhc6028). Thus, it asserts the (software) reproducibility of the Relational Network architecture.


## Disclaimer

Inspired/Based on the work of [@kimhc6028](https://github.com/kimhc6028) on [relational-networks](https://github.com/kimhc6028/relational-networks).