import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .basic import BasicModel, ConvInputModel, FCOutputModel

class RN(BasicModel):
    def __init__(self, kwargs):
        super(RN, self).__init__(kwargs, 'RN')
        
        qst_dim = 11
        spatialDim = 5
        output_dim = 10
        if 'vocab_size' in kwargs:
            qst_dim = kwargs['hidden_size']
            spatialDim = 6
            output_dim = kwargs['answer_vocab_size']

        self.conv = ConvInputModel(depth_dim=kwargs['conv_dim'])
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((kwargs['conv_dim']+2)*2+qst_dim, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(kwargs['batch_size'], 2)
        self.coord_oj = torch.FloatTensor(kwargs['batch_size'], 2)
        if kwargs['cuda']:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(kwargs['batch_size'], 25, 2)
        if kwargs['cuda']:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((kwargs['batch_size'], 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))
        

        self.fcout = FCOutputModel(output_dim=output_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=kwargs['lr'])


    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1,25,1)
        qst = torch.unsqueeze(qst, 2)
        #(64x25x1x11)
        
        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat,1) # (64x1x25x26+11)
        # (64x1x25x26)
        x_i = x_i.repeat(1,25,1,1) # (64x25x25x26+11)
        # (64x25x25x26)
        x_j = torch.unsqueeze(x_flat,2) # (64x25x1x26+11)
        # (64x25x1x26)
        x_j = torch.cat([x_j,qst],3)
        # (64x25x1x26+11)
        x_j = x_j.repeat(1,1,25,1) # (64x25x25x26+11)
        # (64x25x25x26+11)
        
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+11)
        # (64x25x25x26+11+26)
        
        # reshape for passing through network
        x_ = x_full.view(mb*d*d*d*d,63)
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        x_g = x_.view(mb,d*d*d*d,256)
        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        
        return self.fcout(x_f)