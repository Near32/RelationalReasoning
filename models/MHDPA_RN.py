import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from .basic import BasicModel, ConvInputModel, FCOutputModel


class MHDPARelationModule(nn.Module) :
    def __init__(self,output_dim=32, 
                    qst_dim=11, 
                    depth_dim=24,
                    interactions_dim=64, 
                    hidden_size=256, 
                    use_cuda=True, 
                    withLNGenerator=False,
                    withRecurrentQKVgenerators=False,
                    nbr_rnn_layers_recurrentQKVgenerators=None) :
        super(MHDPARelationModule,self).__init__()

        self.use_cuda = use_cuda
        self.output_dim = output_dim
        self.qst_dim = qst_dim
        self.depth_dim = depth_dim
        self.interactions_dim = interactions_dim
        self.hidden_size = hidden_size
        self.fXY = None 
        self.batch = None 
        self.withLNGenerator = withLNGenerator
        self.input_size = self.depth_dim+2+self.qst_dim
        self.withRecurrentQKVgenerators = withRecurrentQKVgenerators
        self.nbr_rnn_layers_recurrentQKVgenerators = nbr_rnn_layers_recurrentQKVgenerators
        self.dropout = 0.0

        if not(self.withLNGenerator):
            if not(self.withRecurrentQKVgenerators):
                self.queryGenerator = nn.Linear(self.depth_dim+2+self.qst_dim,self.interactions_dim,bias=False)
                self.keyGenerator = nn.Linear(self.depth_dim+2+self.qst_dim,self.interactions_dim,bias=False)
                self.valueGenerator = nn.Linear(self.depth_dim+2+self.qst_dim,self.interactions_dim,bias=False)
            else:
                assert(isinstance(int,nbr_rnn_layers_recurrentQKVgenerators))
                self.queryGenerator = nn.LSTM(input_size=self.input_size,
                    hidden_size=self.interactions_dim,
                    num_layers=self.nbr_rnn_layers_recurrentQKVgenerators,
                    batch_first=True,
                    dropout=self.dropout,
                    bidirectional=False,
                    bias=False)
                self.keyGenerator = nn.LSTM(input_size=self.input_size,
                    hidden_size=self.interactions_dim,
                    num_layers=self.nbr_rnn_layers_recurrentQKVgenerators,
                    batch_first=True,
                    dropout=self.dropout,
                    bidirectional=False,
                    bias=False)
                self.valueGenerator = nn.LSTM(input_size=self.input_size,
                    hidden_size=self.interactions_dim,
                    num_layers=self.nbr_rnn_layers_recurrentQKVgenerators,
                    batch_first=True,
                    dropout=self.dropout,
                    bidirectional=False,
                    bias=False)
                
        else :
            if not(self.withRecurrentQKVgenerators):
                self.queryGenerator = nn.Sequential( 
                    nn.Linear(self.depth_dim+2+self.qst_dim,self.interactions_dim,bias=False),
                    nn.LayerNorm(self.interactions_dim,elementwise_affine=False)
                    )
                self.keyGenerator = nn.Sequential( 
                    nn.Linear(self.depth_dim+2+self.qst_dim,self.interactions_dim,bias=False),
                    nn.LayerNorm(self.interactions_dim,elementwise_affine=False)
                    )
                self.valueGenerator = nn.Sequential( 
                    nn.Linear(self.depth_dim+2+self.qst_dim,self.interactions_dim,bias=False),
                    nn.LayerNorm(self.interactions_dim,elementwise_affine=False)
                    )
            else:
                assert(isinstance(int,nbr_rnn_layers_recurrentQKVgenerators))
                self.queryGenerator = nn.LSTM(input_size=self.input_size,
                    hidden_size=self.interactions_dim,
                    num_layers=self.nbr_rnn_layers_recurrentQKVgenerators,
                    batch_first=True,
                    dropout=self.dropout,
                    bidirectional=False,
                    bias=False)
                self.query_rnn_states = None
                self.queryGenerator_layerNorm = nn.LayerNorm(self.interactions_dim,elementwise_affine=False)
                self.keyGenerator = nn.LSTM(input_size=self.input_size,
                    hidden_size=self.interactions_dim,
                    num_layers=self.nbr_rnn_layers_recurrentQKVgenerators,
                    batch_first=True,
                    dropout=self.dropout,
                    bidirectional=False,
                    bias=False)
                self.key_rnn_states = None
                self.keyGenerator_layerNorm = nn.LayerNorm(self.interactions_dim,elementwise_affine=False)
                self.valueGenerator = nn.LSTM(input_size=self.input_size,
                    hidden_size=self.interactions_dim,
                    num_layers=self.nbr_rnn_layers_recurrentQKVgenerators,
                    batch_first=True,
                    dropout=self.dropout,
                    bidirectional=False,
                    bias=False)
                self.value_rnn_states = None
                self.valueGenerator_layerNorm = nn.LayerNorm(self.interactions_dim,elementwise_affine=False)

        self.f0 = nn.Linear(self.interactions_dim,self.hidden_size)
        #torch.nn.init.xavier_normal_(self.f0.weight)
        self.f1 = nn.Linear(self.hidden_size, self.hidden_size)
        #torch.nn.init.xavier_normal_(self.f1.weight)
        self.f2 = nn.Linear(self.hidden_size,self.hidden_size)
        #torch.nn.init.xavier_normal_(self.f2.weight)
        self.f3 = nn.Linear(self.hidden_size, self.output_dim)
        #torch.nn.init.xavier_normal_(self.f3.weight)
        
        if self.use_cuda :
            self = self.cuda()

    def addXYfeatures(self,x,outputFsizes=False) :
        xsize = x.size()
        batch = xsize[0]
        if self.batch != batch or self.fXY is None :
            # batch x depth x X x Y
            self.batch = xsize[0]
            self.depth = xsize[1]
            self.sizeX = xsize[2]
            self.sizeY = xsize[3]
            stepX = 2.0/self.sizeX
            stepY = 2.0/self.sizeY

            fx = torch.zeros((self.batch,1,self.sizeX,1))
            fy = torch.zeros((self.batch,1,1,self.sizeY))
            vx = -1+0.5*stepX
            for i in range(self.sizeX) :
                fx[:,:,i,:] = vx
                vx += stepX
            vy = -1+0.5*stepY
            for i in range(self.sizeY) :
                fy[:,:,:,i] = vy
                vy += stepY
            fxy = fx.repeat( 1,1,1,self.sizeY)
            fyx = fy.repeat( 1,1,self.sizeX,1)
            fXY = torch.cat( [fxy,fyx], dim=1)
            fXY = Variable(fXY)
            if self.use_cuda : fXY = fXY.cuda()
            self.fXY = fXY 

        out = torch.cat( [x,self.fXY], dim=1)
        out = out.view((self.batch,self.depth+2,-1))

        if outputFsizes :
            return out, self.sizeX, self.sizeY

        return out 

    def applyF(self,x) :
        #fout = F.relu( F.dropout( self.f1(x), p=0.5) )
        fout = F.relu( self.f0(x) )
        fout = F.relu( self.f1(fout) )
        fout = F.relu( F.dropout( self.f2(fout), p=0.5) )
        fout = self.f3(fout)#*1e-5

        return fout
    
    def applyMHDPA(self,x, usef=False, reset_hidden_states=False) :
        # input : b x d+2+qst_dim x f
        batchsize = x.size()[0]
        depth_dim = x.size()[1]
        updated_entities = []
        for b in range(batchsize) :
            # depth_dim x featuremap_dim^2 : stack of column entity : d x f   
            xb = x[b].transpose(0,1)
            #  f x d   
            if not(self.withRecurrentQKVgenerators):
                queryb = self.queryGenerator( xb )
                keyb = self.keyGenerator( xb )
                valueb = self.valueGenerator( xb )
            else:
                if reset_hidden_states:
                    self.query_rnn_states=None
                    self.key_rnn_states=None
                    self.value_rnn_states=None
                queryb, self.query_rnn_states = self.queryGenerator( xb, self.query_rnn_states )
                queryb = self.queryGenerator_layerNorm(queryb)
                keyb, self.key_rnn_states = self.keyGenerator( xb, self.key_rnn_states )
                keyb = self.keyGenerator_layerNorm(keyb)
                valueb, self.value_rnn_states = self.valueGenerator( xb, self.value_rnn_states )
                valueb = self.valueGenerator_layerNorm(valueb)
            # f x interactions_dim = d x i

            att_weights = F.softmax( torch.matmul(queryb, keyb.transpose(0,1) ) / np.sqrt(self.interactions_dim), dim=0 )
            # f x i * i x f 
            intb = torch.matmul( att_weights, valueb)
            # f x f * f x i = f x i 

            # apply f in parallel to each entity-feature : was used with the previous version of the RN...
            # the MHDPARelational network does not make use of this f function.
            if usef :
                upd_entb = self.applyF(intb)
                # f x output_dim
                sum_upd_entb = torch.sum( upd_entb, dim=0).unsqueeze(0)
                # 1 x output_dim
            else :
                upd_entb = intb
                # f x i
                sum_upd_entb = upd_entb.unsqueeze(0)
                # 1 x f x i

            updated_entities.append(sum_upd_entb)

        updated_entities = torch.cat( updated_entities, dim=0)
        # either b x output_dim or b x f x i

        return updated_entities 

    def applyMHDPA_batched(self,x,usef=False, reset_hidden_states=False) :
        # input : b x d+2+qst_dim x f
        batchsize = x.size()[0]
        depth_dim = x.size()[1]
        featuresize = x.size()[2]
        updated_entities = []
        
        xb = x.transpose(1,2).contiguous()
        # batch x depth_dim x featuremap_dim^2 : stack of column entity : d x f   
        #  b x f x d   

        augx_full_flat = xb.view( batchsize*featuresize, -1) 
        # ( batch*featuresize x depth )
        if not(self.withRecurrentQKVgenerators):
            queryb = self.queryGenerator( augx_full_flat )
            keyb = self.keyGenerator( augx_full_flat )
            valueb = self.valueGenerator( augx_full_flat )
        else:
            if reset_hidden_states:
                self.query_rnn_states=None
                self.key_rnn_states=None
                self.value_rnn_states=None
            queryb, self.query_rnn_states = self.queryGenerator( augx_full_flat, self.query_rnn_states )
            queryb = self.queryGenerator_layerNorm(queryb)
            keyb, self.key_rnn_states = self.keyGenerator( augx_full_flat, self.key_rnn_states )
            keyb = self.keyGenerator_layerNorm(keyb)
            valueb, self.value_rnn_states = self.valueGenerator( augx_full_flat, self.value_rnn_states )
            valueb = self.valueGenerator_layerNorm(valueb)
        # b*f x interactions_dim = b*f x i

        weights = F.softmax( torch.matmul(queryb, keyb.transpose(0,1) ), dim=1 )
        # bf x i * i x bf
        intb = torch.matmul( weights, valueb) / self.interactions_dim
        # bf x bf * bf x i = bf x i 

        # apply f in parallel to each pixel-feature :
        if usef :
            # bf x i
            upd_entb = self.applyF(intb)
            # bf x output_dim
            x_g = upd_entb.view(batchsize,featuresize,-1)
            sum_upd_entb = torch.sum( x_g, dim=1).unsqueeze(1)
            # b  x output_dim
        else :
            upd_entb = intb
            # bf x i
            x_g = upd_entb.view(batchsize,featuresize,-1)
            sumgout = x_g.sum(1).squeeze()
            sum_upd_entb = upd_entb.view( (batchsize,-1,self.interactions_dim) )
            # b x i

        updated_entities = sum_upd_entb
        
        
        return updated_entities 
    

    def forward(self,x) :
        #begin = time.time()

        augx = self.addXYfeatures(x)
        foutput = self.applyMHDPA(augx)
        
        #elt = time.time() - begin 
        #print('ELT SYSTEM :{} seconds.'.format(elt))
        
        return foutput 
    
    def save(self,path) :
        wts = self.state_dict()
        rnpath = path + 'MHDPARelationModule.weights'
        torch.save( wts, rnpath )
        print('MHDPARelationModule saved at : {}'.format(rnpath) )


    def load(self,path) :
        rnpath = path + 'MHDPARelationModule.weights'
        self.load_state_dict( torch.load( rnpath ) )
        print('MHDPARelationModule loaded from : {}'.format(rnpath) )



class MHDPARelationNetwork(nn.Module) :
    def __init__(self,output_dim=32,
                 depth_dim=24,
                 qst_dim=11, 
                 nbrModule=3,
                 nbrRecurrentSharedLayers=1,
                 use_cuda=True,
                 spatialDim=7,
                 withMaxPool=False, 
                 withLNGenerator=False,
                 withRecurrentQKVgenerators=False,
                 kwargs=None) :
        super(MHDPARelationNetwork,self).__init__()

        self.withMaxPool = withMaxPool
        self.withLNGenerator = withLNGenerator
        self.kwargs = kwargs 

        self.use_cuda = use_cuda
        self.spatialDim = spatialDim
        self.output_dim = output_dim
        self.depth_dim = depth_dim
        self.qst_dim = qst_dim

        self.nbrModule = nbrModule
        self.nbrRecurrentSharedLayers = nbrRecurrentSharedLayers
        self.withRecurrentQKVgenerators = withRecurrentQKVgenerators
        self.nbr_rnn_layers_recurrentQKVgenerators = self.kwargs['nbr_rnn_layers_recurrentQKVgenerators']
        
        self.units_per_MLP_layer = self.output_dim*8#384
        if kwargs['units_per_MLP_layer'] != 0:
            self.units_per_MLP_layer = kwargs['units_per_MLP_layer'] 
        self.interactions_dim = kwargs['interactions_dim'] 
        if self.interactions_dim==0 :
            self.interactions_dim = self.output_dim*4#32
        self.trans = None 
        self.use_bias = False 

        self.MHDPAmodules = nn.ModuleList()
        for i in range(self.nbrModule) :
            self.MHDPAmodules.append( 
                MHDPARelationModule(output_dim=self.output_dim,
                                    qst_dim=self.qst_dim,
                                    depth_dim=self.depth_dim,
                                    interactions_dim=self.interactions_dim,
                                    withLNGenerator=self.withLNGenerator,
                                    use_cuda=self.use_cuda,
                                    withRecurrentQKVgenerators=self.withRecurrentQKVgenerators,
                                    nbr_rnn_layers_recurrentQKVgenerators=self.nbr_rnn_layers_recurrentQKVgenerators
                                    ) 
                )

        self.nonLinearModule = nn.LeakyReLU
        if self.kwargs['withReLU'] :
            self.nonLinearModule = nn.ReLU 

        # F function :
        """
        self.finalParallelLayer = nn.Sequential( nn.Linear(self.nbrModule*self.interactions_dim,self.depth_dim+2+self.qst_dim,bias=self.use_bias)                                              
                                                )
        """
        """
        self.finalParallelLayer = nn.Sequential( nn.Linear(self.nbrModule*self.interactions_dim,self.units_per_MLP_layer,bias=self.use_bias),
                                        self.nonLinearModule(),
                                        nn.Linear(self.units_per_MLP_layer,self.depth_dim+2+self.qst_dim,bias=self.use_bias)                                              
                                                )
        """
        self.finalParallelLayer = nn.Sequential( nn.Linear(self.nbrModule*self.interactions_dim,self.units_per_MLP_layer,bias=self.use_bias),
                                        self.nonLinearModule(),
                                        nn.Linear(self.units_per_MLP_layer,self.units_per_MLP_layer,bias=self.use_bias),
                                        self.nonLinearModule(),
                                        nn.Linear(self.units_per_MLP_layer,self.depth_dim+2+self.qst_dim,bias=self.use_bias)                                              
                                                )
        """
        self.finalParallelLayer = nn.Sequential( nn.Linear(self.nbrModule*self.interactions_dim,self.units_per_MLP_layer,bias=self.use_bias),
                                        self.nonLinearModule(),
                                        nn.Linear(self.units_per_MLP_layer,self.units_per_MLP_layer,bias=self.use_bias),
                                        self.nonLinearModule(),
                                        nn.Linear(self.units_per_MLP_layer,self.units_per_MLP_layer,bias=self.use_bias),
                                        self.nonLinearModule(),
                                        nn.Linear(self.units_per_MLP_layer,self.depth_dim+2+self.qst_dim,bias=self.use_bias)                                              
                                                )
        """
        # Layer Normalization at the spatial level :
        self.AttentionalBlockFinalLayer = nn.LayerNorm(int(self.spatialDim*self.spatialDim) )
        
        if self.withMaxPool :
            self.finalLayer_input_dim = int(self.depth_dim+2+self.qst_dim)
        else :
            self.finalLayer_input_dim = int( (self.depth_dim+2+self.qst_dim) * self.spatialDim*self.spatialDim )

        self.finalLayer = nn.Sequential( nn.Linear(self.finalLayer_input_dim,self.units_per_MLP_layer,bias=self.use_bias),
                                        self.nonLinearModule(),
                                        nn.Linear(self.units_per_MLP_layer,self.output_dim,bias=self.use_bias))

        
        
    def applyMHDPA(self, x, i, reset_hidden_states=False) :
        # input : b x d+2+qst_dim x f
        #output = self.MHDPAmodules[i].applyMHDPA(x,usef=False)
        output = self.MHDPAmodules[i].applyMHDPA_batched(x,
                                                        usef=False, 
                                                        reset_hidden_states=reset_hidden_states)
        # batch x f x i or batch x output_dim
        return output 

    def applyF(self, x) :
        outf = self.finalParallelLayer(x)
        return outf 

    def forwardAttentionalBlocks(self, augx, reset_hidden_states=False) :
        #begin = time.time()
        # input : b x d+2+qst_dim x f
        MHDPAouts = []
        for i in range(self.nbrModule) :
            MHDPAouts.append( self.applyMHDPA(augx,
                                              i, 
                                              reset_hidden_states=reset_hidden_states) )
            # head x [ batch x f x i ]
        concatOverHeads = torch.cat( MHDPAouts, dim=2)
        # [ batch x f x (head X i) ]
        
        updated_entities = []
        for b in range(self.batchsize) :
            xb = concatOverHeads[b]
            #  f x (head X i)   
            augxb = augx[b]
            #  d+2+qst_dim x f   
            upd_entb = self.applyF(xb).transpose(0,1)
            #  d+2+qst_dim x f 
            updated_entities.append((upd_entb+augxb).unsqueeze(0))
            # batch++ x d+2 x f
        updated_entities =torch.cat( updated_entities, dim=0)

        # Apply layer normalization :
        updated_entities = self.AttentionalBlockFinalLayer(updated_entities)

        #elt = time.time() - begin 
        #print('ELT SYSTEM :{} seconds.'.format(elt))
        return updated_entities

    def forward(self,x,qst=None,dummy=False) :
        #begin = time.time()

        self.batchsize = x.size()[0]

        # add coordinate channels :
        augx, self.sizeX, self.sizeY = self.MHDPAmodules[0].addXYfeatures(x,outputFsizes=True)
        featuresize = self.sizeX*self.sizeY
        # batch x d+2 x f(=featuremap_dim^2)
        
        # add questions:
        if qst is not None :
            # ( batch x qst_dim )
            qst = torch.unsqueeze(qst, 2)
            # ( batch x qst_dim x 1 )
            qst = qst.repeat(1,1,featuresize)
            # ( batch x qst_dim x featuresize  )
            
            # augx : batch x d+2 x f(=featuremap_dim^2)
            augx_full = torch.cat([augx,qst],1) 
            # ( batch x d+2+11 x featuresize )
            #augx_full_flat = augx_full.view( batchsize*featuresize*featuresize, -1) 
            

        self.outputRec = [augx_full]
        for i in range(self.nbrRecurrentSharedLayers) :
            # input/output : b x d+2+qst_dim x f
            self.outputRec.append( 
                self.forwardAttentionalBlocks(self.outputRec[i],
                                              reset_hidden_states=(i==0)
                                              ) 
                )
            
        intermediateOutput = self.outputRec[-1].view( (self.batchsize, self.depth_dim+2+self.qst_dim, self.sizeX,self.sizeY))
        # batch x d+2 x sizeX x sizeX=sizeY

        # Max pooling over the dimensional maps :
        if self.withMaxPool :
            maxPooledOutput = F.max_pool2d(intermediateOutput,kernel_size=self.sizeX,stride=1,padding=0)
            intermediateOutput = maxPooledOutput.view( (self.batchsize, self.depth_dim+2+self.qst_dim))
            # batch x d+2

        # Flattening :
        if not(self.withMaxPool) :
            intermediateOutput = intermediateOutput.view( (self.batchsize, -1) )    

        if self.kwargs['dropout_prob'] != 0.0 :
            intermediateOutput = F.dropout(intermediateOutput, p=self.kwargs['dropout_prob'])

        foutput = self.finalLayer(intermediateOutput)

        #elt = time.time() - begin 
        #print('ELT SYSTEM :{:0.4f} seconds // {:0.4f} Hz.'.format(elt,1.0/elt),end='\r')
        
        return foutput 


class MHDPA_RN(BasicModel):
    def __init__(self, kwargs):
        path = 'MHDPA{}-RN{}'.format(kwargs['nbrModule'], kwargs['nbrRecurrentSharedLayers'])
        if kwargs['withRecurrentQKVgenerators']:
            path = 'RNN{}-'.format(kwargs['nbr_rnn_layers_recurrentQKVgenerators'])+path
        if kwargs['withMaxPool']:
            path += '+MaxPool'
        if kwargs['withSSM']:
            path += '+SSM'
        if kwargs['withLNGenerator'] :
            path += '+LNGen'
        if kwargs['withReLU'] :
            path += '+ReLU'
        if kwargs['interactions_dim']!=0:
            path += '+InterDim{}'.format(kwargs['interactions_dim'])
        if kwargs['units_per_MLP_layer']!=0:
            path += '+MLP{}'.format(kwargs['units_per_MLP_layer'])
        if kwargs['dropout_prob']!=0.0 :
            path += '+DropOut{}'.format(kwargs['dropout_prob'])
        path += '+Conv{}'.format(kwargs['conv_dim'])
        path += '+Batch{}'.format(kwargs['batch_size'])
        
        super(MHDPA_RN, self).__init__(kwargs, path)
        
        self.kwargs = kwargs
        self.conv = ConvInputModel(depth_dim=kwargs['conv_dim'])
        
        ##(number of filters per object+2 depth for the coordinates of object)*2+question vector
        qst_dim = 11
        spatialDim = 5
        output_dim = 10
        if 'vocab_size' in kwargs:
            qst_dim = kwargs['hidden_size']
            spatialDim = 6
            output_dim = kwargs['answer_vocab_size']

        self.relationModule = MHDPARelationNetwork(output_dim=output_dim,
                                                    depth_dim=kwargs['conv_dim'],
                                                    qst_dim=qst_dim,
                                                    nbrModule=kwargs['nbrModule'], 
                                                    nbrRecurrentSharedLayers=kwargs['nbrRecurrentSharedLayers'], 
                                                    spatialDim=spatialDim, 
                                                    use_cuda=kwargs['cuda'],
                                                    withMaxPool=kwargs['withMaxPool'], 
                                                    withLNGenerator=kwargs['withLNGenerator'], 
                                                    withRecurrentQKVgenerators=kwargs['withRecurrentQKVgenerators'],
                                                    kwargs=kwargs 
                                                    )
        
        for name,p in self.named_parameters() :
            print(name)

        self.optimizer = optim.Adam(self.parameters(), lr=kwargs['lr'])


    def forward(self, img, qst):
        begin = time.time()
        
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        xsize = x.size()
        batchsize = xsize[0]
        depthsize = xsize[1]
        dsize = xsize[2]
        featuresize = dsize*dsize

        if self.kwargs['withSSM'] :
            x = x.view(batchsize,depthsize,featuresize)
            x = F.softmax(x,dim=2)
            x = x.view(batchsize,depthsize,dsize,dsize)

        self.output = self.relationModule(x,qst=qst)
        output = F.log_softmax(self.output,dim=1) 
        
        elt = time.time() - begin 
        print('ELT forward MHDPA-RN : {} seconds.'.format(elt), end="\r")
        
        return output

