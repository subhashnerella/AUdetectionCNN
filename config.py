import numpy as np
import pandas as pd

NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYTZkMWZkYi0yMzRmLTQ1YjktOGM0Ni02ZGYyYmYyMmQwZDMifQ=="

DATA_ROOT_PATH = '/data/datasets/users/subhash/'
LANDMARK_PATH = '/data/datasets/users/subhash/BP4D/landmarks/'

IMG_SIZE=256
label_dict =  { 1:'Inner brow raiser',
                2:'Outer brow raiser', 
                4:'Brow lowerer',
                6:'Cheek raiser', 
                7:'Lid tightener Eye',
                9:'Nose wrinkler', 
                10:'Upper lip raiser',
                12:'Lip corner puller', 
                14:'Dimpler', 
                15:'Lip corner depressor', 
                17:'Chin raiser', 
                23:'Lip tightener',
                24:'Lip pressor', 
                25:'Lips part',
                26:'Jaw drop'
            }

AUs = ['1', '2', '4', '6', '7', '9', '10', '12', '14', '15', '17', '23', '24', '25', '26']
CENTER_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9, 9, 9, 8 ]

LABEL_SKELETON = np.zeros((np.amax(CENTER_IDS)+1, len(AUs)))
for i, center in enumerate(CENTER_IDS):
    LABEL_SKELETON[center,i] = 1
                           
                           

# LOCATION_SCALE = [-1/2,-1/3,1/3,1,0,-1/2,0,0,1/2,0,1/2]
# AU_CENTERS_LEFT = [4,1,2,24,21,15,43,31,41,34,41]
# AU_CENTERS_RIGHT = [5,8,7,29,26,17,45,37,39,40,39]

LOCATION_SCALE = [-1/2,-1/3,1/3,1,0,-1/2,0,0,1/2,0]
AU_CENTERS_LEFT = [4,1,2,24,21,15,43,31,41,34]
AU_CENTERS_RIGHT = [5,8,7,29,26,17,45,37,39,40]

REFLECT_ORDER = [ 9,  8,  7,  6,  5,  4,  3,  2,  1,  0, 10, 11, 12, 13, 18, 17, 16, 15, 14, 28, 27, 26, 25, 30, 29, 22, 21, 20, 19, 24, 23, 37, 36, 35, 34, 33, 32, 31, 42, 41, 40, 39, 38, 45, 44, 43, 48, 47, 46]

NEIGHBORHOOD = {
                0:[1,2],
                1:[0,2],
                2:[0,1,4],
                3:[7],
                4:[2],
                5:[],
                6:[7,9],
                7:[3,6],
                8:[],
                9:[6]
            }


    
NEIGHBOR_AU_ARRAY= np.zeros((np.amax(CENTER_IDS)+1, len(AUs)))
for i,(au,center) in enumerate(zip(AUs,CENTER_IDS)):
    ncs = NEIGHBORHOOD[center]
    NEIGHBOR_AU_ARRAY[ncs,i] = 1
    
    
    
#BP4D_Dataset
BP4D_AU = ['1', '2', '4', '6', '7', '10', '12', '14', '15', '17', '23', '24']

def get_inds(dataset = 'BP4D',neighbors=True):
    dataset = dataset+'_AU'
    inds = [i for i,e in enumerate(AUs) if e in eval(dataset)]    
    centers = [CENTER_IDS[i] for i in inds]
    global LABEL_SKELETON
    if neighbors:
        LABEL_SKELETON += NEIGHBOR_AU_ARRAY
    drop = list(set(np.arange(len(AUs))) - set(inds))
    LABEL_SKELETON = np.delete(LABEL_SKELETON,drop,axis=1)
    drop = list(set(CENTER_IDS)-set(centers))
    LABEL_SKELETON = np.delete(LABEL_SKELETON,drop,axis=0)
    global LOCATION_SCALE, AU_CENTERS_LEFT, AU_CENTERS_RIGHT
    drop[::-1].sort()
    for c in drop:
        del LOCATION_SCALE[c]
        del AU_CENTERS_LEFT[c]
        del AU_CENTERS_RIGHT[c]

get_inds()

BP4D_FOLD1 = ['F001', 'F002', 'F008', 'F009', 'F010', 'F016', 'F018', 'F023', 'M001', 'M004', 'M007', 'M008', 'M012', 'M014']

BP4D_FOLD2 = ['F003', 'F005', 'F011', 'F013', 'F020', 'F022', 'M002', 'M005', 'M010', 'M011', 'M013', 'M016', 'M017', 'M018']

BP4D_FOLD3 = ['F004', 'F006', 'F007', 'F012', 'F014', 'F015', 'F017', 'F019', 'F021', 'M003', 'M006', 'M009', 'M015']





def get_weights(file,dataset='BP4D'):
    dataset = eval(dataset+'_AU')
    data_df = pd.read_csv(file)[dataset]
    n_class_totals = data_df.sum(axis=0).to_numpy()
    weights = (len(data_df)-n_class_totals)/n_class_totals
    return weights

