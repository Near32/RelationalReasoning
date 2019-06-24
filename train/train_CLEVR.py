from __future__ import print_function
import argparse
import os
import sys
sys.path.append(os.path.abspath('..'))

import pickle
import random
import numpy as np

import torch
from torch.autograd import Variable

from tensorboardX import SummaryWriter
from models import MHDPA_RN, RN, RN2, CNN_MLP
from utils import CLEVR_DataLoader

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Relational-Reasoning Networks on Sort-of-CLEVR Example')
parser.add_argument('--model', type=str, choices=['RN', 'RN2', 'MHDPA-RN', 'CNN_MLP'], default='MHDPA-RN', 
                    help='model architecture to train')
parser.add_argument('--name', type=str, default='model', 
                    help='name to give to the training model')
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
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=str,
                    help='resume from model stored')
parser.add_argument('--withMaxPool',action='store_true',default=False)
parser.add_argument('--withSSM',action='store_true',default=False)
parser.add_argument('--withReLU',action='store_true',default=False)
parser.add_argument('--withLNGenerator',action='store_true',default=False)
parser.add_argument('--interactions_dim', type=int, default=0,
                    help='dimension of the entities interaction (default set to 0 --> 4*output_dim)')
parser.add_argument('--units_per_MLP_layer', type=int, default=0,
                    help='nbr units per MLP layer (default set to 0 --> 8*output_dim)')
parser.add_argument('--dropout_prob', type=float, default=0.0,
                    help='dropout probability just before the final FCN (default: 0.0)')

parser.add_argument('--embedding_size', type=int, default=0,
                    help='size of the word embedding space in which CLEVR questions are embedded.')
parser.add_argument('--hidden_size', type=int, default=0,
                    help='size of the output of the RNN that encodes CLEVR questions.')
parser.add_argument('--nbr_RNN_layers', type=int, default=0,
                    help='numbers of LSTM RNN layers for the encoder of CLEVR questions.')


parser.add_argument('--dataset_path', type=str, default='../DATASETS/CLEVR_v1.0/')
parser.add_argument('--train_data_path', type=str, default='../DATASETS/CLEVR_v1.0/train_questions.h5')
parser.add_argument('--train_vocab_path', type=str, default='../DATASETS/CLEVR_v1.0/vocab.json')
parser.add_argument('--train_image_path', type=str, default='../DATASETS/CLEVR_v1.0/train_questions.h5.paths.npz')

parser.add_argument('--val_data_path', type=str, default='../DATASETS/CLEVR_v1.0/val_questions.h5')
parser.add_argument('--val_vocab_path', type=str, default='../DATASETS/CLEVR_v1.0/vocab.json')
parser.add_argument('--val_image_path', type=str, default='../DATASETS/CLEVR_v1.0/val_questions.h5.paths.npz')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = dict(args._get_kwargs())



torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def load_data(args):
    print('loading data...')
    train_loader = CLEVR_DataLoader( dataset_path=args.dataset_path,
                                    data_path=args.train_data_path,
                                    vocab_path=args.train_vocab_path,
                                    image_path=args.train_image_path,
                                    batch_size=args.batch_size, shuffle=True)

    val_loader = CLEVR_DataLoader( dataset_path=args.dataset_path,
                                    data_path=args.val_data_path,
                                    vocab_path=args.val_vocab_path,
                                    image_path=args.val_image_path,
                                    batch_size=args.batch_size, shuffle=True)

    return train_loader, val_loader

def train(epoch, loader, val_loader, args):
    loaders = {'train':loader, 'val':val_loader}
    
    for phase in ['train','val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        family_corrects = {}
        family_counts = {}
        iter_per_epoch = len(loaders[phase])

        for batch_idx, (img, qst, qst_lenghts, answer, qst_families) in enumerate(loaders[phase]):
            if args.cuda:
                img = img.cuda()
                qst = qst.cuda()
                answer = answer.cuda()

            pred, output = model.train_clevr(img, qst, qst_lenghts, answer, training=(phase=='train'))
            
            for idx_in_batch, idx_qst_family in enumerate(qst_families.numpy()):
                if not( idx_qst_family in family_corrects):
                    family_corrects[idx_qst_family] = 0
                    family_counts[idx_qst_family] = 0
                family_corrects[idx_qst_family] += int(pred[idx_in_batch] == answer[idx_in_batch])
                family_counts[idx_qst_family] += 1
                
            if batch_idx % args.log_interval == 0:
                print('{} Epoch: {} [{}/{} ({:.0f}%)] :'.format(
                        phase,
                        epoch, 
                        batch_idx, 
                        len(loaders[phase]),
                        100. * batch_idx / len(loaders[phase])
                        )
                )
        
                sum_corrects = sum( list(family_corrects.values()) )
                sum_counts = sum( list(family_counts.values()) ) 
                acc = 100 * sum_corrects / sum_counts
                print('{} :: accuracy: {:.0f}% '.format(phase, acc ) )
                writer.add_scalar('Acc/{}'.format(phase), acc, (epoch-1)*iter_per_epoch+batch_idx)


        sum_corrects = sum( list(family_corrects.values()) )
        sum_counts = sum( list(family_counts.values()) ) 
        acc = 100 * sum_corrects / sum_counts
        print('{} :: final accuracy: {:.0f}% '.format(phase, acc ) )
        writer.add_scalar('Accuracy/{}'.format(phase), acc, (epoch)*iter_per_epoch)    
        
        for family_key in family_corrects:
            family_name = family_key#family_idx2name(family_key)
            acc = 100 * family_corrects[family_key] / family_counts[family_key]
            #print('{} :: accuracy: {:.0f}% '.format(family_key, acc ) )
            #print('{}/{}/Acc {} :::: val {} / count {} :: idx {}'.format(phase, family_name, acc, family_corrects[family_key], family_counts[family_key], epoch*iter_per_epoch+batch_idx) )
            writer.add_scalar('{}/{}/Acc'.format(phase, family_name), acc, (epoch)*iter_per_epoch)
    




train_loader, val_loader = load_data(args)
kwargs['vocab_size'] = train_loader.getVocabSize()
kwargs['answer_vocab_size'] = train_loader.getAnswerVocabSize()

if args.model=='CNN_MLP': 
  model = CNN_MLP(args)
elif args.model=='RN':
    model = RN(args)
elif args.model=='RN2' :
  model = RN2(args)
else :
  model = MHDPA_RN(kwargs)

model_dirs = './results_CLEVR/'+args.name

ld = os.path.join(model_dirs,model.name)
writer = SummaryWriter(logdir=ld)

if args.cuda:
    model.cuda()


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
    train(epoch, train_loader, val_loader, args)
    model.save_model(epoch)
