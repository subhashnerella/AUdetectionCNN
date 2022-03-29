import torch
from torch import nn
import torch.nn.functional as F
from model.backbone import Backbone,FeatureExtraction,Interpolate
from model.position_encodings import PositionEmbeddingSine
from model.transformer import Transformer
import config 
import pdb



class AUdetection(nn.Module):
    
    def __init__(self, backbone, transformer):
        super().__init__()
        self.transformer = transformer
        self.backbone = backbone
        hidden_dim = self.transformer.d_model
        #self.projection = nn.Linear(j,hidden_dim)
        self.classifier = Classifier(hidden_dim)
        self.pos = PositionEmbeddingSine()

    def forward(self,images_tensor,data):
        landmarks = data['landmarks']
        enc, dec = self.backbone(images_tensor,landmarks)
        pos_embed = self.pos(enc.shape,enc.device)
        enc_mask, dec_mask = data['left_center_mask'],data['right_center_mask']
        hidden_state,masks = self.transformer(enc,dec,pos_embed,enc_mask,dec_mask) #n_batch X n_cent X n_hidden
        output,masks = self.classifier(hidden_state,masks)
        return output,masks


class Classifier(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        _,centers = config.get_inds()
        _,self.counts = torch.unique(torch.tensor(centers),return_counts=True)
        self.fcn = []
        for j in self.counts:     
            self.fcn.extend([nn.Linear(hidden_dim,2*j.item())])
        self.fcn = torch.nn.ModuleList(self.fcn)

    def forward(self,tensors,masks):
        """
            tensors: n_batch X n_center X hidden_dim
        """
        bs = tensors.shape[0]
        out_tensor = torch.cat([f(tensors[:,i,:]).view(bs,-1,2) for i,f in enumerate(self.fcn)],dim=1)
        out_tensor = F.log_softmax(out_tensor, dim=2)
        au_masks = torch.repeat_interleave(masks,self.counts.to(masks.device),dim=1)
        return out_tensor,au_masks

def reduce(bool_tensor):
    #out = torch.tensor(bool_tensor.shape[0],device=bool_tensor.device)
    out =torch.tensor(0,device=bool_tensor.device)
    for col in bool_tensor.T:
        out = torch.bitwise_or(out,col)
    return out.type(torch.bool)

class Criterion(nn.Module):

    def __init__(self,losses,au_weights=None,loss_weight_dict=None):
        
        super().__init__()
        self.au_weights = au_weights
        self.losses = losses
        self.loss_weight_dict = loss_weight_dict

      
    def forward(self,output, data, masks):
        """
        output: n_batch  X 2 X n_au
        aus_gt: n_batch X n_au
        #TODO: weighted loss
        """
        #aus_gt = torch.cat([v['aus'] for k,v in data.items()],axis=0) #n_batch X n_aus
        #pdb.set_trace()
        
        # inds = reduce(masks)
        # def modify_grad(x,inds):
        #     x[inds] = 0 
        #     return x
        # #pdb.set_trace()
        # #output.register_hook(lambda x: modify_grad(x,ignore))
        # output.register_hook(lambda x: print(x))
        #output=output[inds];aus_gt=aus_gt[inds];masks=masks[inds]
        aus_gt = data['aus']
        assert aus_gt.shape[-1]==output.shape[1]

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss,output,aus_gt,masks))
        return losses

    def softmax_loss(self,output,aus_gt,masks):
        loss = torch.empty(0,device=output.device,requires_grad=True)
        for i in range(output.shape[1]):
            pred = output[:,i,:]
            gt = aus_gt[:,i]
            gt = gt[masks[:,i]]
            pred = pred[masks[:,i]]
            if len(gt)==0  or len(pred) == 0:
                continue
            t_loss = F.nll_loss(pred,gt)
            if self.au_weights is not None:
                t_loss = t_loss * self.au_weights[i]
            t_loss = torch.unsqueeze(t_loss, 0)
            loss = torch.cat((loss, t_loss), 0)
        losses = {'softmax_loss':loss.mean()}
        return losses


    def dice_loss(self,output,aus_gt,masks):
        loss = torch.empty(0,device=output.device)
        for i in range(output.shape[1]):
            pred = output[:,i,1].exp()
            gt = aus_gt[:,i]
            gt = gt[masks[:,i]]
            pred = pred[masks[:,i]]
            if len(gt)==0  or len(pred) == 0:
                continue
            t_loss = dice_coef_loss(pred,gt)
            if self.au_weights is not None:
                t_loss = t_loss * self.au_weights[i]
            t_loss = torch.unsqueeze(t_loss, 0)
            loss = torch.cat((loss, t_loss), 0)
        losses = {'dice_loss':loss.mean()}
        return losses

    def get_loss(self, loss, output, aus_gt,masks):
        loss_map = {
                    'softmax': self.softmax_loss,
                    'dice': self.dice_loss,
                    }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](output, aus_gt,masks)  




def dice_coef_loss(pred, target, smooth = 1):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth)) / iflat.size(0)
    


def build(args):
    train_backbone = args.lr_backbone > 0
    feat_extractor = FeatureExtraction()
    interpolate = Interpolate(size=(16,16),mode = 'bilinear')
    backbone =  Backbone(args.backbone, train_backbone,feat_extractor,interpolate,embedding_dim=args.hidden_dim)
    #backbone = Joiner(bbone,feat_extracts,position_encoding)    
    transformer = Transformer(d_model=args.hidden_dim,
                              dropout=args.dropout,
                              nhead=args.nheads,
                              dim_feedforward=args.dim_feedforward,
                              num_encoder_layers=args.enc_layers,
                              num_decoder_layers=args.dec_layers,)
    model = AUdetection(backbone,transformer)
    losses = ['softmax']#, 'dice']
    loss_weight_dict = {'softmax_loss': 1, 'dice_loss': 1}
    criterion = Criterion(losses, au_weights= args.weights,loss_weight_dict=loss_weight_dict)  
    return model,criterion
        
        
        
        
        