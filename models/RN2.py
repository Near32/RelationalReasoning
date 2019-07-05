import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .basic import BasicModel, ConvInputModel, FCOutputModel


class RelationModule(nn.Module) :
    def __init__(self, qst_dim=10,output_dim=32,depth_dim=24,use_cuda=True, linearNormalization=False) :
        super(RelationModule,self).__init__()

        self.use_cuda = use_cuda
        self.output_dim = output_dim
        self.depth_dim = depth_dim
        self.qst_dim = qst_dim
        self.linearNormalization = linearNormalization

        #self.g1 = nn.Linear(2*self.depth_dim,256)
        self.g1 = nn.Linear(2*self.depth_dim+self.qst_dim,256)
        #torch.nn.init.xavier_normal_(self.g1.weight)
        self.gln1 = nn.LayerNorm(256)
        self.g2 = nn.Linear(256,256)
        #torch.nn.init.xavier_normal_(self.g2.weight)
        self.gln2 = nn.LayerNorm(256)
        self.g3 = nn.Linear(256,256)
        #torch.nn.init.xavier_normal_(self.g3.weight)
        self.gln3 = nn.LayerNorm(256)
        self.g4 = nn.Linear(256,256)
        #torch.nn.init.xavier_normal_(self.g4.weight)
        
        self.f1 = nn.Linear(256, 256)
        #torch.nn.init.xavier_normal_(self.f1.weight)
        self.f2 = nn.Linear(256,256)
        #torch.nn.init.xavier_normal_(self.f2.weight)
        self.f3 = nn.Linear(256, self.output_dim)
        #torch.nn.init.xavier_normal_(self.f3.weight)
        
        self.fXY = None 
        self.batch = 0

        if self.use_cuda :
            self = self.cuda()

    def addXYfeatures(self,x) :
        batch = x.size(0)
        if self.fXY is None or batch != self.batch :
            xsize = x.size()
            # batch x depth x X x Y
            self.batch = xsize[0]
            self.depth = xsize[1]
            self.sizeX = xsize[2]
            self.sizeY = xsize[3]
            stepX = 2.0/self.sizeX
            stepY = 2.0/self.sizeY

            fx = torch.zeros((self.batch,1,self.sizeX,1))
            fy = torch.zeros((self.batch,1,1,self.sizeY))
            for i in range(self.sizeX) :
                fx[:,:,i,:] = -1+(i+0.5)*stepX
            
            for i in range(self.sizeY) :
                fy[:,:,:,i] = -1+(i+0.5)*stepY
            
            fxy = fx.repeat(1,1,1,self.sizeY)
            fyx = fy.repeat(1,1,self.sizeX,1)
            fXY = torch.cat( [fxy,fyx], dim=1)
            self.fXY = Variable(fXY)
            
            if self.use_cuda : self.fXY = self.fXY.cuda()
        
        out = torch.cat( [x,self.fXY], dim=1)

        return out 

    def applyG(self,oinput) :
        gout = F.relu( self.g1(oinput) )
        gout = self.gln1(gout)
        gout = F.relu( self.g2(gout) )
        gout = self.gln2(gout)
        gout = F.relu( self.g3(gout) )
        gout = self.gln3(gout)
        gout = self.g4(gout)

        return gout 

    def applyGNoLN(self,oinput) :
        gout = F.relu( self.g1(oinput) )
        #gout = self.gln1(gout)
        gout = F.relu( self.g2(gout) )
        #gout = self.gln2(gout)
        gout = F.relu( self.g3(gout) )
        #gout = self.gln3(gout)
        gout = self.g4(gout)

        return gout 

    def applyF(self,x) :
        #fout = F.relu( F.dropout( self.f1(x), p=0.5) )
        fout = F.relu( self.f1(x) )
        fout = F.relu( F.dropout( self.f2(fout), p=0.5) )
        #fout = F.relu( self.f2(fout) )
        fout = self.f3(fout)

        return fout

    def forward(self,x,batchsize=32,featuresize=25) :

        if self.linearNormalization :
            gout = self.applyG(x)
        else :
            gout = self.applyGNoLN(x)

        x_g = gout.view(batchsize,featuresize*featuresize,256)
        sumgout = x_g.sum(1).squeeze()
        
        foutput = self.applyF(sumgout)

        return foutput


class RN2(BasicModel):
    def __init__(self, kwargs):
        path = 'RN2'
        if kwargs['NoLayerNormalization'] :
            path += '+NoLN'
        super(RN2, self).__init__(kwargs, path)
        
        self.conv = ConvInputModel()
        
        ##(number of filters per object+2 depth for the coordinates of object)*2+question vector
        qst_dim = 11
        spatialDim = 5
        output_dim = 10
        if 'vocab_size' in kwargs:
            qst_dim = kwargs['hidden_size']
            spatialDim = 6
            output_dim = kwargs['answer_vocab_size']

        self.relationModule = RelationModule(output_dim=output_dim,
            depth_dim=(kwargs['conv_dim']+2),
            qst_dim=qst_dim,
            use_cuda=kwargs['cuda'],
            linearNormalization=not(kwargs['NoLayerNormalization']) )
        
        self.optimizer = optim.Adam(self.parameters(), lr=kwargs['lr'])


    def forward(self, img, qst):
        begin = time.time()
        
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        
        # add coordinates
        augx = self.relationModule.addXYfeatures(x)
        augxsize = augx.size()
        batchsize = augxsize[0]
        depthsize = augxsize[1]
        dsize = augxsize[2]
        featuresize = dsize*dsize

        augx_flat = augx.view(batchsize,depthsize,-1).permute(0,2,1)
        #(batch x feature x depth)

        # add question everywhere
        # ( batch x 11 )
        qst = torch.unsqueeze(qst, 1)
        # ( batch x 1 x 11 )
        qst = qst.repeat(1,featuresize,1)
        # ( batch x featuresize x 11 )
        qst = torch.unsqueeze(qst, 2)
        # ( batch x featuresize x 1 x 11 )
        
        # (64x25x25x2*26+11)
        oi = augx_flat.unsqueeze(1)
        #(batch x 1 x feature x depth)
        oi = oi.repeat(1,featuresize,1,1)
        #(batch x feature x feature x depth)
        oj = augx_flat.unsqueeze(2)
        #(batch x feature x 1 x depth)
        ojqst = torch.cat([oj,qst],3)
        #(batch x feature x 1 x depth+11)
        oj = ojqst.repeat(1,1,featuresize,1)
        #(batch x feature x feature x depth+11)
        
        augx_full = torch.cat([oi,oj],3) 
        # ( batch x featuresize x featuresize x 2*depth+11)
        augx_full_flat = augx_full.view( batchsize*featuresize*featuresize, -1) 
        self.output = self.relationModule(augx_full_flat, batchsize=batchsize,featuresize=featuresize)

        elt = time.time() - begin 
        print('ELT forward RN2 : {} seconds.'.format(elt), end="\r")
        
        return F.log_softmax(self.output,dim=1)

