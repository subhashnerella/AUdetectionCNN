import numpy as np
from PIL import Image
import config
import torch
import random
import torchvision.transforms.functional as F
from torchvision import transforms as T
import math
import pdb
import cv2


class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, landmarks):
        for t in self.transforms:
            image, landmarks = t(image, landmarks)
        return image, landmarks


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image,data):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image,data

class RandomSelect():
    def __init__(self, transforms,p=0.5):
        self.transforms = list(transforms)
        self.p = p
    def __call__(self, img, data):
        if random.random()>self.p:
            return random.choice(self.transforms)(img,data)
        return img,data


def hflip(img,data):
    flipped_image = F.hflip(img)
    w = img.shape[1]
    landmarks = data['landmarks']
    landmarks[1] = landmarks[1][config.REFLECT_ORDER]
    landmarks[0] = w-1-landmarks[0]
    landmarks[0] = landmarks[0][config.REFLECT_ORDER] 
    data['landmarks'] = landmarks
    return flipped_image ,data
    
    
class RandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, data):
        if random.random() < self.p:
            return hflip(img, data)
        return img, data



class ToTensor(object):
    def __call__(self, img, data):

        data_tensor={}
        for k, v in data.items():
            if k == 'id':
                data_tensor[k] = torch.tensor(int(v),dtype=torch.int)
            else:
                data_tensor[k] = torch.tensor(v,dtype= torch.float)
        return F.to_tensor(img), data_tensor
    


def resized_crop(img, data, params):
    cr_image = F.resized_crop(img, *params)
    i, j, h, w,size = params
    landmarks = data['landmarks']
    start = torch.tensor([j,i])[:,None]
    inds = torch.where(landmarks>start,1,0)
    inds = torch.bitwise_and(inds[0],inds[1])
    inds = torch.nonzero(inds).squeeze_()
    landmarks = landmarks[:,inds]
    landmarks.sub_(start)
    landmarks.mul_(torch.tensor([size/h,size/w])[:,None]).round_()
    data['landamrks'] = landmarks
    return cr_image, data

    
class RandomResizedCrop():
    def __init__(self,size,scale=(0.5,1.0),ratio=(1,1)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
    def __call__(self,img,data):
        params = T.RandomResizedCrop.get_params(img,self.scale,self.ratio)
        params = params+(self.size,)
        return crop(img,data,params)


class FaceAlign():
    def __init__(self,img_size=config.IMG_SIZE,enlarge = 2.9, mcManager=None):
        self.size = img_size
        self.enlarge = enlarge
        self.mcManager = mcManager

    def __call__(self, img, data):
        id = data['id']
        if self.mcManager is not None and id  in self.mcManager:
            result = self.mcManager.get(id)
            landmarks = np.array(result['landmarks'])
            mat = np.array(result['affine_mat'])
            aligned_img = cv2.warpAffine(img, mat[0:2, :], (self.size, self.size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))
        else:  
            landmarks = data['landmarks']
            left_eye_x  = landmarks[0,19:25].sum()/6.0
            left_eye_y  = landmarks[1,19:25].sum()/6.0
            right_eye_x = landmarks[0,25:31].sum()/6.0
            right_eye_y = landmarks[1,25:31].sum()/6.0

            dx = right_eye_x-left_eye_x
            dy = right_eye_y-left_eye_y

            l = math.sqrt(dx * dx + dy * dy)
            sinVal = dy / l
            cosVal = dx / l
            mat1 = np.mat([[cosVal, sinVal, 0], [-sinVal, cosVal, 0], [0, 0, 1]])
            mat2 = np.mat([[left_eye_x,left_eye_y,1],
                        [right_eye_x,right_eye_y,1],
                        [landmarks[0,13],landmarks[1,13],1],
                        [landmarks[0,31],landmarks[1,31],1],
                        [landmarks[0,37],landmarks[1,37],1]])

            mat2 = (mat1 * mat2.T).T
            cx = float((max(mat2[:, 0]) + min(mat2[:, 0]))) * 0.5
            cy = float((max(mat2[:, 1]) + min(mat2[:, 1]))) * 0.5
            if (float(max(mat2[:, 0]) - min(mat2[:, 0])) > float(max(mat2[:, 1]) - min(mat2[:, 1]))):
                halfSize = 0.5 * self.enlarge * float((max(mat2[:, 0]) - min(mat2[:, 0])))
            else:
                halfSize = 0.5 * self.enlarge * float((max(mat2[:, 1]) - min(mat2[:, 1])))

            scale = (self.size - 1) / 2.0 / halfSize
            mat3 = np.mat([[scale, 0, scale * (halfSize - cx)], [0, scale, scale * (halfSize - cy)], [0, 0, 1]])
            mat = mat3 * mat1

            aligned_img = cv2.warpAffine(img, mat[0:2, :], (self.size, self.size), cv2.INTER_LINEAR, borderValue=(128, 128, 128))
            land_3d = np.ones((landmarks.shape[0]+1,landmarks.shape[1]))
            land_3d[:-1,:] = landmarks
            landmarks = (mat*land_3d)[:-1]
            landmarks = landmarks.round()
            
            if self.mcManager is not None:
                try:
                    save_dict={'landmarks':landmarks,'affine_mat':mat}
                    self.mcManager.set(id,save_dict)
                except Exception:
                    pass


        data['landmarks'] = landmarks
        return cv2.cvtColor(aligned_img,cv2.COLOR_BGR2RGB), data



def crop(img,params):
    crop = F.crop(img,*params)
    i, j, h, w = params
    mask = torch.zeros(img.shape[1:],dtype = torch.bool)
    mask[i:i+h,j:j+w] = 1
    crop_img = torch.zeros(img.shape)
    crop_img[:,i:i+h,j:j+w] =crop
    return crop_img,mask


class RandomCrop():
    #TODO: crop regions correspond to au presence in a given image
    def __init__(self,size = 176):
        self.size = (size,size)
        inds = config.INDS
        self.center_left = [config.AU_CENTERS_LEFT[i] for i in inds]
        self.center_right = [config.AU_CENTERS_RIGHT[i] for i in inds]
        self.location_scale = [config.LOCATION_SCALE[i] for i in inds]
    
    def __call__(self,img,data):
        params = T.RandomCrop.get_params(img, self.size)
        crop_img,mask = crop(img,params)
        landmarks = data['landmarks']
        data['left_center_mask'],data['right_center_mask'] = self.getCenterMasks(mask,landmarks)
        return crop_img,data
        

    def getCenterMasks(self,mask,landmarks):
        ruler = abs(landmarks[0,22]-landmarks[0,25])
        scale = (ruler*torch.tensor(self.location_scale)).round()
        centers_left_x = landmarks[0,self.center_left]; centers_right_x = landmarks[0,self.center_right] #batchXn_cent
        centers_left_y = landmarks[1,self.center_left] + scale; centers_right_y = landmarks[1,self.center_right] + scale
        left = ~mask[centers_left_y.long(),centers_left_x.long()] 
        right = ~mask[centers_right_y.long(),centers_right_x.long()]
        return left,right


class RandomOccludeHalf():
    def __init__(self):
        inds = config.INDS
        self.center_left = [config.AU_CENTERS_LEFT[i] for i in inds]
        self.center_right = [config.AU_CENTERS_RIGHT[i] for i in inds]
        self.location_scale = [config.LOCATION_SCALE[i] for i in inds]
    
    def __call__(self,img,data):
        pad =0
        landmarks = data['landmarks']
        lip_center = landmarks[:,34]
        mask = torch.ones(img.shape[1:],dtype = torch.bool)
        half = int(lip_center[0].item()+pad)
        mask[:, half:] = 0
        if random.random()>0.5:
            mask = mask.flip((-1))
        img = img * mask
        data['left_center_mask'],data['right_center_mask'] = self.getCenterMasks(mask,landmarks)
        return img,data

    def getCenterMasks(self,mask,landmarks):
        ruler = abs(landmarks[0,22]-landmarks[0,25])
        scale = (ruler*torch.tensor(self.location_scale)).round()
        centers_left_x = landmarks[0,self.center_left]; centers_right_x = landmarks[0,self.center_right] #batchXn_cent
        centers_left_y = landmarks[1,self.center_left] + scale; centers_right_y = landmarks[1,self.center_right] + scale
        left = ~mask[centers_left_y.long(),centers_left_x.long()] 
        right = ~mask[centers_right_y.long(),centers_right_x.long()]
        return left,right


def erase(img,params):
    erase = F.erase(img,*params)
    i,j,h,w,_ = params
    mask = torch.ones(img.shape[1:],dtype=torch.bool)
    mask[i:i+h,j:j+w] = 0
    return erase,mask

class RandomErase():
    def __init__(self, scale=(0.1, 0.25), ratio=(0.75, 1.75), value=[0]):
        self.scale = scale
        self.ratio = ratio
        self.value = value
        inds = config.INDS
        self.center_left = [config.AU_CENTERS_LEFT[i] for i in inds]
        self.center_right = [config.AU_CENTERS_RIGHT[i] for i in inds]
        self.location_scale = [config.LOCATION_SCALE[i] for i in inds]
    
    def __call__(self,img,data):
        landmarks = data['landmarks']
        params = T.RandomErasing.get_params(img,scale = self.scale,ratio = self.ratio,value=self.value)
        erased_img,mask = erase(img,params)
        data['left_center_mask'],data['right_center_mask'] = self.getCenterMasks(mask,landmarks)
        return erased_img,data           


    def getCenterMasks(self,mask,landmarks):
        ruler = abs(landmarks[0,22]-landmarks[0,25])
        scale = (ruler*torch.tensor(self.location_scale)).round()
        centers_left_x = landmarks[0,self.center_left]; centers_right_x = landmarks[0,self.center_right] #batchXn_cent
        centers_left_y = landmarks[1,self.center_left] + scale; centers_right_y = landmarks[1,self.center_right] + scale
        left = ~mask[centers_left_y.long(),centers_left_x.long()] 
        right = ~mask[centers_right_y.long(),centers_right_x.long()]
        return left,right


        