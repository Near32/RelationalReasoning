import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .basic import BasicModel, ConvInputModel, FCOutputModel, ModularityPrioredConvInputModel


class SequentialAttentionRelationModule(nn.Module) :
    def __init__(self, qst_dim=10,output_dim=32,depth_dim=24,use_cuda=True, kwargs=None) :
        super(SequentialAttentionRelationModule,self).__init__()

        self.use_cuda = use_cuda
        self.output_dim = output_dim
        self.depth_dim = depth_dim
        self.qst_dim = qst_dim
        self.kwargs = kwargs
        self.linearNormalization = self.kwargs['withLNGenerator']

        self.nonlinearity = nn.ReLU()
        if kwargs['withLeakyReLU'] :
            self.nonlinearity = nn.LeakyReLU()

        self.apsi1 = nn.Linear(self.depth_dim+self.qst_dim,self.kwargs['units_per_MLP_layer'])
        self.apsiln1 = nn.LayerNorm(self.kwargs['units_per_MLP_layer'])
        self.apsi2 = nn.Linear(self.kwargs['units_per_MLP_layer'],self.kwargs['units_per_MLP_layer'])
        self.apsiln2 = nn.LayerNorm(self.kwargs['units_per_MLP_layer'])
        self.apsi3 = nn.Linear(self.kwargs['units_per_MLP_layer'],1)
        
        if not(self.kwargs['NoXavierInit']) :
            torch.nn.init.xavier_normal_(self.apsi1.weight)
            torch.nn.init.xavier_normal_(self.apsi2.weight)
            torch.nn.init.xavier_normal_(self.apsi3.weight)    
            
        self.g1 = nn.Linear(2*(self.depth_dim+self.qst_dim),self.kwargs['units_per_MLP_layer'])
        self.gln1 = nn.LayerNorm(self.kwargs['units_per_MLP_layer'])
        self.g2 = nn.Linear(self.kwargs['units_per_MLP_layer'],self.kwargs['units_per_MLP_layer'])
        self.gln2 = nn.LayerNorm(self.kwargs['units_per_MLP_layer'])
        self.g3 = nn.Linear(self.kwargs['units_per_MLP_layer'],self.kwargs['units_per_MLP_layer'])
        self.gln3 = nn.LayerNorm(self.kwargs['units_per_MLP_layer'])
        
        if not(self.kwargs['NoXavierInit']) :
            torch.nn.init.xavier_normal_(self.g1.weight)
            torch.nn.init.xavier_normal_(self.g2.weight)
            torch.nn.init.xavier_normal_(self.g3.weight)
                
        self.f1 = nn.Linear(self.kwargs['units_per_MLP_layer'], self.kwargs['units_per_MLP_layer'])
        self.f2 = nn.Linear(self.kwargs['units_per_MLP_layer'],self.kwargs['units_per_MLP_layer'])
        self.f3 = nn.Linear(self.kwargs['units_per_MLP_layer'], self.output_dim)
        
        if not(self.kwargs['NoXavierInit']) :
            torch.nn.init.xavier_normal_(self.f1.weight)
            torch.nn.init.xavier_normal_(self.f2.weight)
            torch.nn.init.xavier_normal_(self.f3.weight)
            
        
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

            #fx = -1*torch.ones((batch,1,sizeX,1),dtype=torch.float)
            fx = torch.zeros((self.batch,1,self.sizeX,1))
            #fy = -1*torch.ones((batch,1,1,sizeY),dtype=torch.float)
            fy = torch.zeros((self.batch,1,1,self.sizeY))
            for i in range(self.sizeX) :
                #fx[:,:,i,:] = (i+0.5)*stepX/2.
                fx[:,:,i,:] = -1+(i+0.5)*stepX
            
            for i in range(self.sizeY) :
                #fy[:,:,:,i] = (i+0.5)*stepY/2.
                fy[:,:,:,i] = -1+(i+0.5)*stepY
            
            fxy = fx.repeat(1,1,1,self.sizeY)#torch.cat( [fx]*self.sizeY, dim=3)
            fyx = fy.repeat(1,1,self.sizeX,1)#torch.cat( [fy]*self.sizeX, dim=2)
            fXY = torch.cat( [fxy,fyx], dim=1)
            self.fXY = Variable(fXY)
            
            if self.use_cuda : self.fXY = self.fXY.cuda()
        
        out = torch.cat( [x,self.fXY], dim=1)

        #out = out.view((self.batch,self.depth+2,-1))

        return out 

    def applyAPsi(self,oinput) :
        aout = self.nonlinearity( self.apsi1(oinput) )
        aout = self.apsiln1(aout)
        aout = self.nonlinearity( self.apsi2(aout) )
        aout = self.apsiln2(aout)
        aout = self.nonlinearity( self.apsi3(aout) )

        return aout 

    def applyG(self,oinput) :
        gout = self.nonlinearity( self.g1(oinput) )
        gout = self.gln1(gout)
        gout = self.nonlinearity( self.g2(gout) )
        gout = self.gln2(gout)
        gout = self.nonlinearity( self.g3(gout) )
        gout = self.gln3(gout)
        
        return gout 

    def applyGNoLN(self,oinput) :
        gout = self.nonlinearity( self.g1(oinput) )
        gout = self.nonlinearity( self.g2(gout) )
        gout = self.nonlinearity( self.g3(gout) )
        
        return gout 

    def applyF(self,x) :
        fout = self.nonlinearity( self.f1(x) )
        fout = self.nonlinearity( F.dropout( self.f2(fout), p=0.5) )
        fout = self.nonlinearity(fout)

        return fout

    def forward(self,ojqst, batchsize=32,featuresize=25) :
        #(batch*feature x depth+qstsize)
        weights = self.applyAPsi(ojqst).view(batchsize,-1).unsqueeze(2)
        softmax_weights = F.softmax( weights, dim=1)
        #(batch x feature x 1 )
        if self.kwargs['withSoftmaxWeights'] :
            oi = torch.sum( softmax_weights * ojqst.view(batchsize,featuresize,-1), dim=1).unsqueeze(1)
        else :
            oi = torch.sum( weights * ojqst.view(batchsize,featuresize,-1), dim=1).unsqueeze(1)
        #(batch x 1 x depth+qstsize )
        oi = oi.repeat(1,featuresize,1)
        #(batch x feature x depth+qstsize )
        oj = ojqst.view(batchsize,featuresize,-1)
        #(batch x feature x depth+qstsize )
        x = torch.cat( [oi,oj],dim=2)
        #(batch x feature x 2*(depth+qstsize) )
        x = x.view( batchsize*featuresize, -1)
        #(batch*feature x 2*(depth+qstsize) )
        
        if self.linearNormalization :
            gout = self.applyG(x)
        else :
            gout = self.applyGNoLN(x)

        x_g = gout.view(batchsize,featuresize,-1)
        #(batch x feature x kwargs['units_per_MLP_layer'] )
        
        sumgout = x_g.sum(1).squeeze()
        #(batch x kwargs['units_per_MLP_layer'] )
        
        foutput = self.applyF(sumgout)

        return foutput


class ParallelSequentialAttentionRelationModule(nn.Module) :
    def __init__(self, qst_dim=10,output_dim=32,depth_dim=24,use_cuda=True, kwargs=None) :
        super(ParallelSequentialAttentionRelationModule,self).__init__()

        self.use_cuda = use_cuda
        self.output_dim = output_dim
        self.depth_dim = depth_dim
        self.qst_dim = qst_dim
        self.nbrParallelAttention = kwargs['nbrParallelAttention']
        self.kwargs = kwargs
        self.linearNormalization = self.kwargs['withLNGenerator']

        self.nonlinearity = nn.ReLU()
        if kwargs['withLeakyReLU'] :
            self.nonlinearity = nn.LeakyReLU()

        self.apsi = nn.ModuleList()
        for i in range(self.nbrParallelAttention) :
            self.apsi.append(  
                nn.Sequential( 
                    nn.Linear(self.depth_dim+self.qst_dim,self.kwargs['units_per_MLP_layer']),
                    nn.LayerNorm(self.kwargs['units_per_MLP_layer']),
                    self.nonlinearity,
                    nn.Linear(self.kwargs['units_per_MLP_layer'],self.kwargs['units_per_MLP_layer']),
                    nn.LayerNorm(self.kwargs['units_per_MLP_layer']),
                    self.nonlinearity,
                    nn.Linear(self.kwargs['units_per_MLP_layer'],1)
                    )
                )
            if not(self.kwargs['NoXavierInit']) :
                torch.nn.init.xavier_normal_(self.apsi[-1][0].weight)
                torch.nn.init.xavier_normal_(self.apsi[-1][3].weight)
                torch.nn.init.xavier_normal_(self.apsi[-1][6].weight)    
            
        self.g1 = nn.Linear((self.nbrParallelAttention+1)*(self.depth_dim+self.qst_dim),self.kwargs['units_per_MLP_layer'])
        self.gln1 = nn.LayerNorm(self.kwargs['units_per_MLP_layer'])
        self.g2 = nn.Linear(self.kwargs['units_per_MLP_layer'],self.kwargs['units_per_MLP_layer'])
        self.gln2 = nn.LayerNorm(self.kwargs['units_per_MLP_layer'])
        self.g3 = nn.Linear(self.kwargs['units_per_MLP_layer'],self.kwargs['units_per_MLP_layer'])
        self.gln3 = nn.LayerNorm(self.kwargs['units_per_MLP_layer'])
        
        if not(self.kwargs['NoXavierInit']) :
            torch.nn.init.xavier_normal_(self.g1.weight)
            torch.nn.init.xavier_normal_(self.g2.weight)
            torch.nn.init.xavier_normal_(self.g3.weight)
                
        self.f1 = nn.Linear(self.kwargs['units_per_MLP_layer'], self.kwargs['units_per_MLP_layer'])
        self.f2 = nn.Linear(self.kwargs['units_per_MLP_layer'],self.kwargs['units_per_MLP_layer'])
        self.f3 = nn.Linear(self.kwargs['units_per_MLP_layer'], self.output_dim)
        
        if not(self.kwargs['NoXavierInit']) :
            torch.nn.init.xavier_normal_(self.f1.weight)
            torch.nn.init.xavier_normal_(self.f2.weight)
            torch.nn.init.xavier_normal_(self.f3.weight)
            
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

            #fx = -1*torch.ones((batch,1,sizeX,1),dtype=torch.float)
            fx = torch.zeros((self.batch,1,self.sizeX,1))
            #fy = -1*torch.ones((batch,1,1,sizeY),dtype=torch.float)
            fy = torch.zeros((self.batch,1,1,self.sizeY))
            for i in range(self.sizeX) :
                #fx[:,:,i,:] = (i+0.5)*stepX/2.
                fx[:,:,i,:] = -1+(i+0.5)*stepX
            
            for i in range(self.sizeY) :
                #fy[:,:,:,i] = (i+0.5)*stepY/2.
                fy[:,:,:,i] = -1+(i+0.5)*stepY
            
            fxy = fx.repeat(1,1,1,self.sizeY)#torch.cat( [fx]*self.sizeY, dim=3)
            fyx = fy.repeat(1,1,self.sizeX,1)#torch.cat( [fy]*self.sizeX, dim=2)
            fXY = torch.cat( [fxy,fyx], dim=1)
            self.fXY = Variable(fXY)
            
            if self.use_cuda : self.fXY = self.fXY.cuda()
        
        out = torch.cat( [x,self.fXY], dim=1)

        #out = out.view((self.batch,self.depth+2,-1))

        return out 

    def applyAPsi(self,oinput, i=0) :
        aout = self.apsi[i](oinput)
        return aout 

    def applyG(self,oinput) :
        gout = self.nonlinearity( self.g1(oinput) )
        gout = self.gln1(gout)
        gout = self.nonlinearity( self.g2(gout) )
        gout = self.gln2(gout)
        gout = self.nonlinearity( self.g3(gout) )
        gout = self.gln3(gout)
        
        return gout 

    def applyGNoLN(self,oinput) :
        gout = self.nonlinearity( self.g1(oinput) )
        gout = self.nonlinearity( self.g2(gout) )
        gout = self.nonlinearity( self.g3(gout) )
        
        return gout 

    def applyF(self,x) :
        fout = self.nonlinearity( self.f1(x) )
        fout = self.nonlinearity( F.dropout( self.f2(fout), p=0.5) )
        fout = self.nonlinearity(fout)

        return fout

    def forward(self,ojqst, batchsize=32,featuresize=25) :
        #(batch*feature x depth+qstsize)
        weights = []
        softmax_weights = []
        ois = []
        for i in range(self.nbrParallelAttention) :
            weights.append( self.applyAPsi(ojqst, i).view(batchsize,-1).unsqueeze(2) )
            softmax_weights.append( F.softmax( weights[-1], dim=1) )
            #(batch x feature x 1 )
        
            if self.kwargs['withSoftmaxWeights'] :
                ois.append( torch.sum( softmax_weights[-1] * ojqst.view(batchsize,featuresize,-1), dim=1).unsqueeze(1) )
            else :
                ois.append( torch.sum( weights[-1] * ojqst.view(batchsize,featuresize,-1), dim=1).unsqueeze(1) )
            #(batch x 1 x depth+qstsize )
        
        ois = torch.cat(ois, dim=1)
        #(batch x nbrParallelAttention x depth+qstsize )

        ois = ois.view( batchsize, -1).unsqueeze(1)
        #(batch x 1 x nbrParallelAttention*(depth+qstsize) )

        ois = ois.repeat(1,featuresize,1)
        #(batch x feature x nbrParallelAttention*(depth+qstsize) )
        
        oj = ojqst.view(batchsize,featuresize,-1)
        #(batch x feature x depth+qstsize )
        
        x = torch.cat( [ois,oj],dim=2)
        #(batch x feature x (nbrParallelAttention+1)*(depth+qstsize) )
        
        x = x.view( batchsize*featuresize, -1)
        #(batch*feature x (nbrParallelAttention+1)*(depth+qstsize) )
        
        if self.linearNormalization :
            gout = self.applyG(x)
        else :
            gout = self.applyGNoLN(x)

        x_g = gout.view(batchsize,featuresize,-1)
        #(batch x feature x kwargs['units_per_MLP_layer'] )
        
        sumgout = x_g.sum(1).squeeze()
        #(batch x kwargs['units_per_MLP_layer'] )
        
        foutput = self.applyF(sumgout)

        return foutput



class SARN(BasicModel):
    def __init__(self, kwargs):
        path = 'SARN'
        if kwargs['nbrParallelAttention'] > 1 :
            path = 'P{}SARN'.format(kwargs['nbrParallelAttention'])
        if kwargs['withModularityPrior']:
            path = 'ModularityPriored'+path
        if kwargs['NoXavierInit'] :
            path += '+NoXavierInit'
        if not(kwargs['withLNGenerator']) :
            path += '+NoLN'
        if kwargs['units_per_MLP_layer']!=0:
            path += '+MLP{}'.format(kwargs['units_per_MLP_layer'])
        if kwargs['withLeakyReLU'] :
            path += '+LeakyReLU'
        path += '+Conv{}'.format(kwargs['conv_dim'])
        path += '+Batch{}'.format(kwargs['batch_size'])
        if kwargs['withSoftmaxWeights'] :
            path += '+withSoftmaxWeights'

        super(SARN, self).__init__(kwargs, path)
        
        if kwargs['withModularityPrior']:
            self.conv = ModularityPrioredConvInputModel(depth_dim=kwargs['conv_dim'])
        else :
            self.conv = ConvInputModel(depth_dim=kwargs['conv_dim'])
        
        ##(number of filters per object+2 depth for the coordinates of object)*2+question vector
        qst_dim = 11
        spatialDim = 5
        output_dim = 10
        if 'vocab_size' in kwargs:
            qst_dim = kwargs['hidden_size']
            spatialDim = 6
            output_dim = kwargs['answer_vocab_size']

        '''
        self.relationModule = SequentialAttentionRelationModule(
            output_dim=output_dim,
            depth_dim=kwargs['conv_dim']+2,
            qst_dim=qst_dim,
            use_cuda=True,
            kwargs=kwargs 
            )
        '''
        self.relationModule = ParallelSequentialAttentionRelationModule(
            output_dim=output_dim,
            depth_dim=kwargs['conv_dim']+2,
            qst_dim=qst_dim,
            use_cuda=kwargs['cuda'], 
            kwargs=kwargs )
        
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
        ojqst = ojqst.squeeze().view( batchsize*featuresize, -1)
        #(batch*feature x depth+11)
        self.output = self.relationModule( ojqst, batchsize=batchsize,featuresize=featuresize)

        elt = time.time() - begin 
        print('ELT forward {} : {} seconds.'.format(self.name,elt), end="\r")
        
        return F.log_softmax(self.output,dim=1)

