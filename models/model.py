from sklearn.feature_extraction import img_to_graph
import torch
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import FeaturePyramidNetwork,roi_pool, roi_align
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torch import feature_alpha_dropout, nn, sigmoid
from models.head import TwoMLPHead,HeadModule
import config
import pdb


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneFPN(nn.Module):
    def __init__(self,name, returned_layers=None,use_fpn=False):
        super().__init__()
        self.use_fpn = use_fpn
        if returned_layers == None:
            returned_layers = [1, 2, 3]
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        backbone = getattr(torchvision.models,name)(pretrained = True, norm_layer= FrozenBatchNorm2d)
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        if use_fpn:
            in_channels_stage2 = backbone.inplanes // 8
            in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
            out_channels = backbone.inplanes //2
            extra_blocks = LastLevelMaxPool()
            self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list,out_channels=out_channels,extra_blocks=extra_blocks)

    def forward(self,x):
        x = self.body(x)
        if self.use_fpn:
            x = self.fpn(x)
        return x['0']



class FeatureExtraction(nn.Module):
    def __init__(self,crop_dim=48,roisize=7):
        super().__init__()
        self.center_left = config.AU_CENTERS_LEFT
        self.center_right = config.AU_CENTERS_RIGHT
        self.location_scale = config.LOCATION_SCALE
        self.crop_dim = crop_dim #wrt to original image scale not the feature scale
        self.roisize = roisize
        self.ncenters = len(self.center_left)

        
    def forward(self,features, landmarks):
        batch,_,h,w=features.shape
        left,right = self.au_centers(landmarks)
        spatial_scale = min(h,w)/config.IMG_SIZE
        # left = (left*spatial_scale).round()
        # right = (right*spatial_scale).round()
        
        left[:,0].clamp_(0,w-1);left[:,1].clamp_(0,h-1)
        right[:,0].clamp_(0,w-1);right[:,1].clamp_(0,h-1)
        ph = pw = self.crop_dim//2
        
        leftx1 = left[:,0,:] - pw; lefty1 = left[:,1,:] - ph
        leftx2 = left[:,0,:] + pw+1; lefty2 = left[:,1,:] + ph+1
        rightx1 = right[:,0,:] - pw; righty1 = right[:,1,:] - ph
        rightx2 = right[:,0,:] + pw+1; righty2 = right[:,1,:] + ph+1
   
        inds = torch.repeat_interleave(torch.arange(batch,device=landmarks.device),self.ncenters).view(batch,self.ncenters)
        left_boxes = torch.stack((inds,leftx1,lefty1,leftx2,lefty2),dim=-1).reshape(batch*self.ncenters,5)
        right_boxes = torch.stack((inds,rightx1,righty1,rightx2,righty2),dim=-1).reshape(batch*self.ncenters,5)
        # lboxs = roi_pool(imgtensors,left_boxes,output_size=(self.crop_dim,self.crop_dim))
        # rboxs = roi_pool(imgtensors,right_boxes,output_size=(self.crop_dim,self.crop_dim))
        lboxs = roi_align(features, left_boxes, self.roisize, spatial_scale )
        rboxs = roi_align(features, right_boxes, self.roisize, spatial_scale )
        # lboxs = lboxs.view(batch,self.ncenters,-1)
        # rboxs = rboxs.view(batch,self.ncenters,-1)
        return lboxs,rboxs
 
    def au_centers(self,landmarks):
        ruler = abs(landmarks[:,0, 22] - landmarks[:,0, 25]) #batch
        scales = ruler[:,None] * torch.tensor(self.location_scale,device = landmarks.device) #batch X n_loc
        centers_left_x = landmarks[:,0,self.center_left]; centers_right_x = landmarks[:,0,self.center_right] #batchXn_cent
        centers_left_y = landmarks[:,1,self.center_left] + scales; centers_right_y = landmarks[:,1,self.center_right] + scales 
        return torch.stack((centers_left_x,centers_left_y),dim=1).round(),torch.stack((centers_right_x,centers_right_y),dim=1).round()





class AUdetector(nn.Module):
    def __init__(self,backbone, roipooler, head,classifier):
        super().__init__()

        self.backbone = backbone
        self.roipooler = roipooler
        self.head = head
        self.classifier = classifier

        # _,centers = config.get_inds()
        # _,self.counts = torch.unique(torch.tensor(centers),return_counts=True)
        # self.fcn = []
        # for j in self.counts:     
        #     self.fcn.extend([nn.Linear(representation_size,j.item())])
        # self.fcn = torch.nn.ModuleList(self.fcn)

    def forward(self,img_tensors,landmarks):
        """
            tensors: n_batch X n_center X hidden_dim
        """
        features = self.backbone(img_tensors)
        lrois,rrois = self.roipooler(features,landmarks)
        left = self.head(lrois);right = self.head(rrois)
        left = self.classifier(left);right = self.classifier(right)
        # bs = roi_feats.shape[0]
        # out_tensor = torch.cat([f(roi_feats[:,i,:]).view(bs,-1) for i,f in enumerate(self.fcn)],dim=1)
        return (left,right)

class Criterion(nn.Module):
    def __init__(self,device=None,au_weights=None,loss_weight_dict=None):
        super().__init__()
        self.au_weights = au_weights
        self.loss_weight_dict = loss_weight_dict
        self.au_loss = nn.BCEWithLogitsLoss(pos_weight = au_weights,reduction='none')
        self.label_skeleton = torch.tensor(config.LABEL_SKELETON,device = device)

    def forward(self,preds,aus_gt):
        #assert aus_gt.shape[-1]==preds.shape[1]
        #b,n_au = aus_gt.shape

        # au_label = torch.zeros(b,self.ncenters,n_au,device=aus_gt.device)
        # nz_inds = torch.nonzero(aus_gt)
        # au_label[nz_inds[:,0],self.center_ids[nz_inds[:,0],nz_inds[:,1]],nz_inds[:,1]]=1
        # au_label = au_label.view(b*self.ncenters,n_au)
        return self.sigmoid_loss(preds,aus_gt)
        
    def sigmoid_loss(self,preds,aus):
        #generate au_labels for loss
        b,n_au = aus.shape
        au_label = aus.unsqueeze(1)*self.label_skeleton      
        au_label = au_label.view(-1,n_au)     
        left,right = preds
        left_loss = self.au_loss(left,au_label)  
        right_loss = self.au_loss(right,au_label)
        return left_loss+right_loss

    

def build(args):
    bbonefpn = BackboneFPN(name= args.backbone,use_fpn=args.fpn)
    roi_pooler = FeatureExtraction(crop_dim=args.crop_dim,roisize=args.roi_size)
    roi_head = HeadModule()
    cls = TwoMLPHead(roi_size = args.roi_size,in_channels=2048,au_classes=12)
    detector = AUdetector(bbonefpn,roi_pooler,roi_head,cls)
    au_weights = torch.tensor(args.weights,device=args.device)
    loss_criterion = Criterion(device=args.device,au_weights=au_weights)
    return detector,loss_criterion
