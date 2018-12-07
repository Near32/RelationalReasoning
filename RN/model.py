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

  

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

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
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        
        self.conv = ConvInputModel()
        
        ##(number of filters per object+coordinate of object)*2+question vector
        self.g_fc1 = nn.Linear((24+2)*2+11, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))
        print(self.coord_tensor[0])
        

        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


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

        return F.log_softmax(fout)

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
        if args.NoLayerNormalization :
            path += '+NoLN'
        super(RN2, self).__init__(args, path)
        
        self.conv = ConvInputModel()
        
        ##(number of filters per object+2 depth for the coordinates of object)*2+question vector
        self.relationModule = RelationModule(output_dim=10,depth_dim=(24+2),qst_dim=11,use_cuda=True,linearNormalization=not(args.NoLayerNormalization) )
        
        #for name,p in self.named_parameters() :
        #   print(name)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


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
        
        return self.output

class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel()
        self.fc1   = nn.Linear(5*5*24 + 11, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        #print([ a for a in self.parameters() ] )
  
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        
        x_ = torch.cat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)

