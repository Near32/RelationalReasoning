import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        
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

  
class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

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

    def save_model(self, epoch):
        torch.save(self.state_dict(), '{}/epoch_{}_{:02d}.pth'.format(self.dirpath,self.name, epoch))


class RN(BasicModel):
    def __init__(self, kwargs):
        super(RN, self).__init__(kwargs, 'RN')
        
        self.conv = ConvInputModel()
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((24+2)*2+11, 256)

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
        print(self.coord_tensor[0])
        

        self.fcout = FCOutputModel()
        
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
            
            print(self.fXY[0])
            if self.use_cuda : self.fXY = self.fXY.cuda()
        
        out = torch.cat( [x,self.fXY], dim=1)

        #out = out.view((self.batch,self.depth+2,-1))

        return out 

    def addXYfeatures_(self,x)  :
        batch = x.size(0)
        if self.fXY is None or batch != self.batch :
            xsize = x.size()
            # batch x depth x X x Y
            self.batch = xsize[0]
            self.depth = xsize[1]
            self.sizeX = xsize[2]
            self.sizeY = xsize[3]
            
            self.coord_oi = torch.FloatTensor(self.batch, 2)
            self.coord_oj = torch.FloatTensor(self.batch, 2)
            if self.use_cuda:
                self.coord_oi = self.coord_oi.cuda()
                self.coord_oj = self.coord_oj.cuda()
            self.coord_oi = Variable(self.coord_oi)
            self.coord_oj = Variable(self.coord_oj)

            # prepare coord tensor
            def cvt_coord(i):
                return [(i/self.sizeX-2)/2., (i%self.sizeY-2)/2.]
            
            self.fXY = torch.FloatTensor(self.batch, self.sizeX*self.sizeY, 2)
            if self.use_cuda:
                self.fXY = self.fXY.cuda()
            self.fXY = Variable(self.fXY)
            np_coord_tensor = np.zeros((self.batch, self.sizeX*self.sizeY, 2))
            for i in range(self.sizeX*self.sizeY):
                np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
            self.fXY.data.copy_(torch.from_numpy(np_coord_tensor))
            self.fXY = self.fXY.view( self.batch, 2, self.sizeX,self.sizeY)

        #print(x.size(),self.fXY.size())

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
        fout = self.f3(fout)#*1e-5

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
    def __init__(self, args):
        path = 'RN2'
        if kwargs['NoLayerNormalization'] :
            path += '+NoLN'
        super(RN2, self).__init__(kwargs, path)
        
        self.conv = ConvInputModel()
        
        ##(number of filters per object+2 depth for the coordinates of object)*2+question vector
        self.relationModule = RelationModule(output_dim=10,depth_dim=(24+2),qst_dim=11,use_cuda=True,linearNormalization=not(kwargs['NoLayerNormalization']) )
        
        #for name,p in self.named_parameters() :
        #   print(name)

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
        #print(augx_flat[0,:,depthsize-2:].view(-1,2))
        #raise
        
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





class MHDPARelationModule(nn.Module) :
    def __init__(self,output_dim=32, qst_dim=11, depth_dim=24,interactions_dim=64, hidden_size=256, use_cuda=True, withLNGenerator=False) :
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

        if not(self.withLNGenerator):
            self.queryGenerator = nn.Linear(self.depth_dim+2+self.qst_dim,self.interactions_dim,bias=False)
            self.keyGenerator = nn.Linear(self.depth_dim+2+self.qst_dim,self.interactions_dim,bias=False)
            self.valueGenerator = nn.Linear(self.depth_dim+2+self.qst_dim,self.interactions_dim,bias=False)
        else :
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
    
    def applyMHDPA(self,x,usef=False) :
        # input : b x d+2+qst_dim x f
        batchsize = x.size()[0]
        depth_dim = x.size()[1]
        updated_entities = []
        for b in range(batchsize) :
            # depth_dim x featuremap_dim^2 : stack of column entity : d x f   
            xb = x[b].transpose(0,1)
            #  f x d   
            queryb = self.queryGenerator( xb )
            keyb = self.keyGenerator( xb )
            valueb = self.valueGenerator( xb )
            # f x interactions_dim = d x i

            att_weights = F.softmax( torch.matmul(queryb, keyb.transpose(0,1) ) / np.sqrt(self.interactions_dim), dim=0 )
            #print('b{} :'.format(b),att_weights)
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

    def applyMHDPA_batched(self,x,usef=False) :
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
        queryb = self.queryGenerator( augx_full_flat )
        keyb = self.keyGenerator( augx_full_flat )
        valueb = self.valueGenerator( augx_full_flat )
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
    def __init__(self,output_dim=32,depth_dim=24,qst_dim=11, nbrModule=3,nbrRecurrentSharedLayers=1,use_cuda=True,spatialDim=7,withMaxPool=False, withLNGenerator=False,kwargs=None) :
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
            self.MHDPAmodules.append( MHDPARelationModule(output_dim=self.output_dim,
                qst_dim=self.qst_dim,
                depth_dim=self.depth_dim,
                interactions_dim=self.interactions_dim,
                withLNGenerator=self.withLNGenerator,
                use_cuda=self.use_cuda) )

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

        
        
    def applyMHDPA(self,x,i) :
        # input : b x d+2+qst_dim x f
        #output = self.MHDPAmodules[i].applyMHDPA(x,usef=False)
        output = self.MHDPAmodules[i].applyMHDPA_batched(x,usef=False)
        # batch x f x i or batch x output_dim
        return output 

    def applyF(self,x) :
        outf = self.finalParallelLayer(x)
        return outf 

    def forwardAttentionalBlocks(self,augx) :
        #begin = time.time()
        # input : b x d+2+qst_dim x f
        MHDPAouts = []
        for i in range(self.nbrModule) :
            MHDPAouts.append( self.applyMHDPA(augx,i) )
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
            self.outputRec.append( self.forwardAttentionalBlocks(self.outputRec[i]) )
            
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
        #path = '1MHDPA{}-RN{}'.format(args.nbrModule,args.nbrRecurrentSharedLayers)
        #path = '2MHDPA{}-RN{}'.format(args.nbrModule,args.nbrRecurrentSharedLayers)
        #path = 'ResidualMHDPA{}-RN{}'.format(args.nbrModule,args.nbrRecurrentSharedLayers)
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

        super(MHDPA_RN, self).__init__(kwargs, path)
        
        self.kwargs = kwargs
        self.conv = ConvInputModel()
        
        ##(number of filters per object+2 depth for the coordinates of object)*2+question vector
        qst_dim = 11
        spatialDim = 5
        output_dim = 10
        if 'vocab_size' in kwargs:
            qst_dim = kwargs['hidden_size']
            spatialDim = 6
            output_dim = kwargs['answer_vocab_size']

        self.relationModule = MHDPARelationNetwork(output_dim=output_dim,
                                                    depth_dim=24,
                                                    qst_dim=qst_dim,
                                                    nbrModule=kwargs['nbrModule'], 
                                                    nbrRecurrentSharedLayers=kwargs['nbrRecurrentSharedLayers'], 
                                                    spatialDim=spatialDim, 
                                                    use_cuda=True,
                                                    withMaxPool=kwargs['withMaxPool'], 
                                                    withLNGenerator=kwargs['withLNGenerator'], 
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

        
        elt = time.time() - begin 
        print('ELT forward RN2 : {} seconds.'.format(elt), end="\r")
        
        return F.log_softmax(self.output,dim=1)



class CNN_MLP(BasicModel):
    def __init__(self, kwargs):
        super(CNN_MLP, self).__init__(kwargs, 'CNNMLP')

        self.conv  = ConvInputModel()
        self.fc1   = nn.Linear(5*5*24 + 11, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=kwargs['lr'])
        #print([ a for a in self.parameters() ] )
  
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        
        x_ = torch.cat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)

