"""""""""
Pytorch implementation of "A simple neural network module for relational reasoning
Code is based on pytorch/examples/mnist (https://github.com/pytorch/examples/tree/master/mnist)
"""""""""
from __future__ import print_function
import argparse
import os
#import cPickle as pickle
import pickle
import random
import numpy as np

import torch
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from model import SARN, MHDPA_RN, RN, RN2, CNN_MLP


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Network sort-of-CLVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'RN2', 'MHDPA-RN', 'SARN', 'CNN_MLP'], default='SARN', 
                    help='resume from model stored')
parser.add_argument('--dataset', type=str, choices=['CLEVR', 'Sort-of-CLEVR'], default='Sort-of-CLEVR', 
                    help='which dataset to use')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--nbrModule', type=int, default=1,
                    help='number of MHDPA heads to use per recurrent shared layer (default: 1)')
parser.add_argument('--nbrRecurrentSharedLayers', type=int, default=1,
                    help='number of recurrent shared layer application (default: 1)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--withMaxPool',action='store_true',default=False)
parser.add_argument('--withSSM',action='store_true',default=False)
parser.add_argument('--withReLU',action='store_true',default=False)
parser.add_argument('--withLeakyReLU',action='store_true',default=False)
parser.add_argument('--withLNGenerator',action='store_true',default=False)
parser.add_argument('--NoXavierInit',action='store_true',default=False)
parser.add_argument('--withSoftmaxWeights',action='store_true',default=False)
parser.add_argument('--withModularityPrior',action='store_true',default=False)
parser.add_argument('--interactions_dim', type=int, default=0,
                    help='dimension of the entities interaction (default set to 0 --> 4*output_dim)')
parser.add_argument('--units_per_MLP_layer', type=int, default=0,
                    help='nbr units per MLP layer (default set to 0 --> 8*output_dim)')
parser.add_argument('--dropout_prob', type=float, default=0.0,
                    help='dropout probability just before the final FCN (default: 0.0)')
parser.add_argument('--conv_dim', type=int, default=24,
                    help='dimension of the depth channel in the convnet (default: 24)')
parser.add_argument('--nbrParallelAttention', type=int, default=1,
                    help='number of referential entity that are being attended to in parallel by the a_psi heads (default:1)')

# CLEVR dataset : Input data
parser.add_argument('--train_question_h5', default='DATASETS/CLEVR_v1.0/train_questions.h5')
parser.add_argument('--train_images_dir', default='DATASETS/CLEVR_v1.0/images/train')
parser.add_argument('--val_question_h5', default='DATASETS/CLEVR_v1.0/val_questions.h5')
parser.add_argument('--val_images_dir', default='DATASETS/CLEVR_v1.0/image/val')
parser.add_argument('--test_question_h5', default='DATASETS/CLEVR_v1.0/test_questions.h5')
parser.add_argument('--test_images_dir', default='DATASETS/CLEVR_v1.0/image/test')
parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--vocab_json', default='DATASETS/CLEVR_v1.0/vocab.json')
parser.add_argument('--family_split_file', default=None)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=10000, type=int)

    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model=='CNN_MLP': 
  model = CNN_MLP(args)
elif args.model=='RN':
    model = RN(args)
elif args.model=='RN2' :
  model = RN2(args)
elif args.model=='MHDPA-RN' :
  model = MHDPA_RN(args)
else :
    model = SARN(args)

model_dirs = './{}_model'.format(args.dataset)
bs = args.batch_size
input_img = torch.FloatTensor(bs, 3, 75, 75)
input_qst = torch.FloatTensor(bs, 11)
label = torch.LongTensor(bs)

ld = os.path.join(model_dirs,model.name)
writer = SummaryWriter(log_dir=ld)

if args.cuda:
    model.cuda()
    input_img = input_img.cuda()
    input_qst = input_qst.cuda()
    label = label.cuda()

input_img = Variable(input_img)
input_qst = Variable(input_qst)
label = Variable(label)

def tensor_data(data, i):
    img = torch.from_numpy(np.asarray(data[0][bs*i:bs*(i+1)]))
    qst = torch.from_numpy(np.asarray(data[1][bs*i:bs*(i+1)]))
    ans = torch.from_numpy(np.asarray(data[2][bs*i:bs*(i+1)]))

    input_img.data.resize_(img.size()).copy_(img)
    input_qst.data.resize_(qst.size()).copy_(qst)
    label.data.resize_(ans.size()).copy_(ans)


def cvt_data_axis(data):
    img = [e[0] for e in data]
    qst = [e[1] for e in data]
    ans = [e[2] for e in data]
    return (img,qst,ans)

    
def train(epoch, rel, norel):
    model.train()

    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    random.shuffle(rel)
    random.shuffle(norel)

    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    iter_per_epoch = len(rel[0]) // bs

    for batch_idx in range(iter_per_epoch):
        tensor_data(rel, batch_idx)
        accuracy_rel = model.train_(input_img, input_qst, label)

        tensor_data(norel, batch_idx)
        accuracy_norel = model.train_(input_img, input_qst, label)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Relations accuracy: {:.0f}% | Non-relations accuracy: {:.0f}%'.format(epoch, 
                    batch_idx * bs * 2, len(rel[0]) * 2,
                    100. * batch_idx * bs/ len(rel[0]), 
                    accuracy_rel, accuracy_norel
                    )
            )
        
        writer.add_scalar('Train/Relation/Acc', accuracy_rel, epoch*iter_per_epoch+batch_idx)
        writer.add_scalar('Train/Non-Relation/Acc', accuracy_norel, epoch*iter_per_epoch+batch_idx)
                            
            

def test(epoch, rel, norel):
    model.eval()
    if not len(rel[0]) == len(norel[0]):
        print('Not equal length for relation dataset and non-relation dataset.')
        return
    
    rel = cvt_data_axis(rel)
    norel = cvt_data_axis(norel)

    accuracy_rels = []
    accuracy_norels = []
    for batch_idx in range(len(rel[0]) // bs):
        tensor_data(rel, batch_idx)
        accuracy_rels.append(model.test_(input_img, input_qst, label))

        tensor_data(norel, batch_idx)
        accuracy_norels.append(model.test_(input_img, input_qst, label))

    accuracy_rel = sum(accuracy_rels) / len(accuracy_rels)
    accuracy_norel = sum(accuracy_norels) / len(accuracy_norels)
    print('\n Test set: Relation accuracy: {:.0f}% | Non-relation accuracy: {:.0f}%\n'.format(
        accuracy_rel, accuracy_norel))
    writer.add_scalar('Test/Relation/Acc', accuracy_rel, epoch)
    writer.add_scalar('Test/Non-Relation/Acc', accuracy_norel, epoch)
        

    
def load_data():
    print('loading data...')
    dirs = '../DATASETS'
    filename = os.path.join(dirs,'sort-of-clevr.pickle')
    with open(filename, 'rb') as f:
      train_datasets, test_datasets = pickle.load(f)
    rel_train = []
    rel_test = []
    norel_train = []
    norel_test = []
    print('processing data...')

    for img, relations, norelations in train_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_train.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_train.append((img,qst,ans))

    for img, relations, norelations in test_datasets:
        img = np.swapaxes(img,0,2)
        for qst,ans in zip(relations[0], relations[1]):
            rel_test.append((img,qst,ans))
        for qst,ans in zip(norelations[0], norelations[1]):
            norel_test.append((img,qst,ans))
    
    return (rel_train, rel_test, norel_train, norel_test)
    

rel_train, rel_test, norel_train, norel_test = load_data()

try:
    os.makedirs(model_dirs)
except:
    print('directory {} already exists'.format(model_dirs))

if args.resume:
    filename = os.path.join(model_dirs, args.resume)
    if os.path.isfile(filename):
        print('==> loading checkpoint {}'.format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint)
        print('==> loaded checkpoint {}'.format(filename))

for epoch in range(1, args.epochs + 1):
    train(epoch, rel_train, norel_train)
    test(epoch, rel_test, norel_test)
model.save_model(epoch)
