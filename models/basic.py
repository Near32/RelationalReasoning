import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class ConvInputModel(nn.Module):
    def __init__(self, depth_dim=24):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, depth_dim, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(depth_dim)
        self.conv2 = nn.Conv2d(depth_dim, depth_dim, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(depth_dim)
        self.conv3 = nn.Conv2d(depth_dim, depth_dim, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(depth_dim)
        self.conv4 = nn.Conv2d(depth_dim, depth_dim, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(depth_dim)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x



class ModularityPrioredConvInputModel(nn.Module):
    def __init__(self, depth_dim=24):
        super(ModularityPrioredConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, depth_dim, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(depth_dim)
        self.conv2 = nn.Conv2d(depth_dim, depth_dim, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(depth_dim)
        self.conv3 = nn.Conv2d(depth_dim, depth_dim, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(depth_dim)
        self.conv4 = nn.Conv2d(depth_dim, depth_dim, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(depth_dim)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        #x = F.softmax(x, dim=1)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        #x = F.softmax(x, dim=1)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        #x = F.softmax(x, dim=1)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x_weights = F.softmax(x, dim=1).detach()
        x = x*x_weights
        x = self.batchNorm4(x)
        return x

        
class FCOutputModel(nn.Module):
    def __init__(self, output_dim=10):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x)


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size=256, hidden_size=256, nbr_layers=1, dropout=0, use_cuda=False):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.nbr_layers = nbr_layers
        self.dropout = dropout
        self.batch_first = True
        self.use_cuda = use_cuda

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.LSTM = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.nbr_layers,
                            batch_first=self.batch_first,
                            dropout=self.dropout,
                            bidirectional=True)

        if self.use_cuda:
            self = self.cuda()

    def forward(self, input_seq, input_lengths, init_hidden_state=None):
        # Embedding from word indices to vectors:
        embedded_seq = self.embedding(input_seq)
        # Pack padded batch of sequences:
        packed = nn.utils.rnn.pack_padded_sequence(embedded_seq, input_lengths, batch_first=self.batch_first)
        # Forward pass:
        outputs, hidden = self.LSTM(packed, init_hidden_state)
        # Unpack:
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional outputs: additive approach to the fusion of modalities:
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs[-1] 

class BasicModel(nn.Module):
    def __init__(self, kwargs, name):
        super(BasicModel, self).__init__()
        self.name=name
        self.dirpath = './model'

        # CLEVR:
        if 'vocab_size' in kwargs and kwargs['vocab_size'] != 0 \
        and kwargs['embedding_size'] != 0 \
        and kwargs['hidden_size'] != 0:
            self.vocab_size = kwargs['vocab_size']
            self.encoder = EncoderRNN(vocab_size=kwargs['vocab_size'], 
                                    embedding_size=kwargs['embedding_size'], 
                                    hidden_size=kwargs['hidden_size'],
                                    nbr_layers=kwargs['nbr_RNN_layers'],
                                    use_cuda=kwargs['cuda'])        

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy
        
    def train_clevr(self, input_img, input_qst, qst_lengths, answer, training=True):
        self.optimizer.zero_grad()
        encoded_qst = self.encoder(input_seq=input_qst, input_lengths=qst_lengths)
        output = self(input_img, encoded_qst)
        if training:
            loss = F.nll_loss(output, answer)
            loss.backward()
            self.optimizer.step()
        pred = output.data.max(1)[1]
        return pred, output
    
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy

    def save_model(self, epoch, dirpath=None):
        if dirpath is None:
            dirpath=self.dirpath
        torch.save(self.state_dict(), '{}/epoch_{}_{:02d}.pth'.format(dirpath,self.name, epoch))

class CNN_MLP(BasicModel):
    def __init__(self, kwargs):
        super(CNN_MLP, self).__init__(kwargs, 'CNNMLP')

        qst_dim = 11
        spatialDim = 5
        output_dim = 10
        if 'vocab_size' in kwargs:
            qst_dim = kwargs['hidden_size']
            spatialDim = 6
            output_dim = kwargs['answer_vocab_size']

        self.conv  = ConvInputModel(depth_dim=kwargs['conv_dim'])
        self.fc1   = nn.Linear(spatialDim*spatialDim*kwargs['conv_dim'] + qst_dim, 256)  # question concatenated to all
        self.fcout = FCOutputModel(output_dim=output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=kwargs['lr'])
        
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        
        x_ = torch.cat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)

