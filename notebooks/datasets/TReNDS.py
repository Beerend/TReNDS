import os
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
# from monai.transforms import RandAffined
import torch
from torch.utils.data import Dataset

class TReNDSDataset(Dataset):
    
    def __init__(self, root, mode='train', n_splits=5, fold=0, rand_affine=False):
        self.root  = root
        self.mode  = mode
        self.splts = n_splits
        self.fold  = fold
        self.rand  = rand_affine
        
        # Read samples (Id and age)
        data = pd.read_csv('{}/train_scores.csv'.format(self.root), usecols=[0, 1]).dropna()
        ids  = list(data.Id)
        lbls = list(data.age)
        
        # Create dict of samples
        self.all_samples = []
        for i in range(len(ids)):
            filename = os.path.join('{}/fMRI_train_npy/{}.npy'.format(self.root, ids[i]))
            self.all_samples.append([filename, lbls[i], str(ids[i])])
            
        # Split in train and test set according to fold number
        kf = KFold(n_splits=self.splts, shuffle=True, random_state=0)
        for i, [train_index, test_index] in enumerate(kf.split(self.all_samples)):
            if i==self.fold:
                self.train_index = train_index
                self.test_index  = test_index
                
        print('Loaded dataset with %d train and %d test samples in fold %d.'%(len(train_index), len(test_index), self.fold))
        
    def __getitem__(self, idx):
        if self.mode=='train':
            filename, lbl, idx = self.all_samples[self.train_index[idx]]
            
        elif self.mode=='test':
            filename, lbl, idx = self.all_samples[self.test_index[idx]]
            
        # Load the 4-dimensional fMRI image
        img = np.load(filename).astype(np.float32)
        img = img.transpose((3,2,1,0)) # 53 (temporal), 52 (axial), 63 (medial), 53 (lateral)
#         img = np.array([img[26,:,:,:]])
        
        # Randomly affine the image (for generalisation purposes during training)
#         if self.rand:
#             img_dict = {'img':img}
#             rand_affine = RandAffined(keys=['img'], mode=('bilinear', 'nearest'), prob=.5, spatial_size=(52, 63, 53),
#                                      translate_range=(5, 5, 5), rotate_range=(.15, .15, .15), padding_mode='border')
#             img_dict = rand_affine(img_dict)
#             img = img_dict['img']
                
        return torch.from_numpy(img), torch.tensor([lbl])
    
    def __len__(self):
        if self.mode=='train':
            return len(self.train_index)
        elif self.mode=='test':
            return len(self.test_index)