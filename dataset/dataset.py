import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import dataset.transform as T
import config
import cv2
import torch
import random
import os
import pandas as pd


class BP4D_Dataset(Dataset):
    def __init__(self,file,mcManager=None, transform=None):
        self.df = pd.read_csv(file)
        #self.df = df.loc[df.participant.isin(fold)].reset_index(drop=True)
        #self.df = self.df.iloc[:1000]
        self._length = len(self.df) 
        self.transform = transform
        self.mcManager = mcManager
        #print(split,self._length)

    def __len__(self):
        return self._length
    
    def __getitem__(self,idx):
        data_dict = {}
        data = self.df.iloc[idx] #BP4D/data/F003/T6/0120.png
        img_path = data['path']
        #print(img_path)
        aus = data[config.BP4D_AU]
        image = cv2.imread(config.DATA_ROOT_PATH+img_path)
        participant,task,iid = img_path.split('/')[2:]
        iid = iid.split('.')[0]
        landmarks = np.load(config.LANDMARK_PATH+f'{participant}_{task}_{int(iid)-1}.npy')
#         if landmarks.shape != (2,49):
#             os.remove(config.LANDMARK_PATH+f'{participant}_{task}_{int(iid)-1}.npy')
#             self.df = self.df.drop(idx).reset_index(drop=True)
#             self._length-=1
#             idx = random.randint(0,idx) if idx >= self._length else idx
#             return self.__getitem__(idx)
        data_dict['landmarks'] = np.array(landmarks,dtype=np.int)
        # data_dict['left_center_mask'] = np.zeros(len(config.INDS),dtype=np.bool)
        # data_dict['right_center_mask'] = np.zeros(len(config.INDS),dtype=np.bool)
        data_dict['aus'] = np.array(aus,dtype=np.int) 
        data_dict['id'] = gen_id(img_path)
        if self.transform is not None:
            image,data_dict = self.transform(image,data_dict)
        return image, data_dict


def gen_id(path):
    participant,task,iid = path.split('/')[2:]
    gen_dict = {'M':'1','F':2}
    g, participant = participant[0],participant[1:]
    iid = iid.split('.')[0]
    return f"{gen_dict[g]}{int(participant):02d}{int(task[-1])}{int(iid):04d}"

def deconstruct_id(pid):
    pid = str(pid)
    g, participant, task, iid = pid[0],pid[1:3],pid[3],pid[4:]
    gen_dict = {'1':'M','2':'F'}
    g = gen_dict[g]
    return f"BP4D/data/{g}{int(participant):03d}/T{task}/{iid}.jpg"
