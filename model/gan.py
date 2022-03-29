from torch import nn
import torch.nn.functional as F
import torch
import config
from torchvision.ops import roi_pool
import pdb

class Generator(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        feats = 64
        self.main = nn.Sequential(
                    nn.ConvTranspose2d(in_features, feats * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(feats * 8),
                    nn.ReLU(True),
                    # size --> feats*8 X 4 X 4
                    nn.ConvTranspose2d(feats * 8, feats * 4, 2, 2, 1, bias=False),
                    nn.BatchNorm2d(feats * 4),
                    nn.ReLU(True),
                    # size --> feats*8 X 6 X 6
                    nn.ConvTranspose2d(feats * 4, feats * 2, 2, 2, 0, bias=False),
                    nn.BatchNorm2d(feats * 2),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(feats * 2, feats , 2, 2, 0, bias=False),
                    nn.BatchNorm2d(feats ),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(feats , 3 , 2, 2, 0, bias=False),
                    nn.Tanh()

                    )
        
    def forward(self,input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        feats = 64
        self.main = nn.Sequential(
                    #input --> bs X 3 X 48 X 48 
                    nn.Conv2d(3,feats,2,2,bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    #state size --> bs X 64 X 13 X 13
                    nn.Conv2d(feats,feats*2,2,2,bias=False),
                    nn.BatchNorm2d(feats*2),
                    nn.LeakyReLU(0.2, inplace=True), 

                    nn.Conv2d(feats*2,feats*2,2,stride=2,bias=False),
                    nn.BatchNorm2d(feats*2),
                    nn.LeakyReLU(0.2, inplace=True),
                    
                    nn.Conv2d(feats*2,feats*4,2,bias=False),
                    nn.BatchNorm2d(feats*4),
                    nn.LeakyReLU(0.2, inplace=True),
                    #state  
                    nn.Conv2d(feats*4,feats*8,2,stride=2,bias=False),
                    nn.BatchNorm2d(feats*8),
                    nn.LeakyReLU(0.2, inplace=True),
                )
        self.linear = nn.Linear(feats*8*4,1)

    def forward(self,input):
        bs = input.shape[0]
        out = self.main(input)
        out = out.view(bs,-1)
        #out = torch.sigmoid(self.linear(out))
        return out

class ExtractTrueCrops(nn.Module):
    def __init__(self,cdim=48):
        super().__init__()
                
        inds = config.INDS
        self.center_left = [config.AU_CENTERS_LEFT[i] for i in inds]
        self.center_right = [config.AU_CENTERS_RIGHT[i] for i in inds]
        self.location_scale = [config.LOCATION_SCALE[i] for i in inds]
        self.crop_dim = cdim
        self.ncenters = len(self.center_left)

    def forward(self,imgtensors, landmarks):
        batch,_,h,w=imgtensors.shape
        left,right = self.au_centers(landmarks.type(torch.float32))
        left[:,0].clamp_(0,w-1);left[:,1].clamp_(0,h-1)
        right[:,0].clamp_(0,w-1);right[:,1].clamp_(0,h-1)
        ph = pw = self.crop_dim/2
        leftx1 = left[:,0,:] - pw; lefty1 = left[:,1,:] - ph
        leftx2 = left[:,0,:] + pw; lefty2 = left[:,1,:] + ph
        rightx1 = right[:,0,:] - pw; righty1 = right[:,1,:] - ph
        rightx2 = right[:,0,:] + pw; righty2 = right[:,1,:] + ph
        pdb.set_trace()
        inds = torch.repeat_interleave(torch.arange(batch),self.ncenters).view(batch,self.ncenters)
        left_boxes = torch.stack((inds,leftx1,lefty1,leftx2,lefty2),dim=-1).reshape(batch*self.ncenters,5)
        right_boxes = torch.stack((inds,rightx1,righty1,rightx2,righty2),dim=-1).reshape(batch*self.ncenters,5)
        lboxs = roi_pool(imgtensors,left_boxes,output_size=(self.crop_dim,self.crop_dim))
        rboxs = roi_pool(imgtensors,right_boxes,output_size=(self.crop_dim,self.crop_dim))
        return inds,lboxs,rboxs
 
    def au_centers(self,landmarks):
        #pdb.set_trace()
        ruler = abs(landmarks[:,0, 22] - landmarks[:,0, 25]) #batch
        scales = ruler[:,None] * torch.tensor(self.location_scale,device = landmarks.device) #batch X n_loc
        centers_left_x = landmarks[:,0,self.center_left]; centers_right_x = landmarks[:,0,self.center_right] #batchXn_cent
        centers_left_y = landmarks[:,1,self.center_left] + scales; centers_right_y = landmarks[:,1,self.center_right] + scales 
        return torch.stack((centers_left_x,centers_left_y),dim=1).round(),torch.stack((centers_right_x,centers_right_y),dim=1).round()



class AuxClassifier():
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(

            
        )

class Adversarial(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = Generator()
        self.disc = Discriminator()
        self.crop_pool = ExtractTrueCrops()
        self.aclf = AuxClassifier()

    def forward(self,imgtensors,data,):
        return 0


