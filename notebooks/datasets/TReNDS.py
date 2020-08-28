import os
import h5py
import numpy as np
import nilearn as nl
import pandas as pd
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset

class TReNDSDataset(Dataset):
    
    def __init__(self, root, mode, n_splits=5, fold=0, preprocess=True, norm=True, temp_mean=False):
        self.root  = os.path.join(root, 'TReNDS')
        self.mode  = mode
        self.splts = n_splits
        self.fold  = fold
        self.prepr = preprocess
        self.norm  = norm
        self.mean  = temp_mean
        
        # Read samples (Id and age)
        data = pd.read_csv('{}/train_scores.csv'.format(self.root), usecols=[0, 1]).dropna()
        ids  = list(data.Id)
        lbls = list(data.age)
        
        # Create dict of samples
        self.all_samples = []
        for i in range(len(ids)):
            filename = '{}/fMRI_train_npy/{}.npy'.format(self.root, ids[i])
            self.all_samples.append([filename, lbls[i], str(ids[i])])
            
        # Split in train and test set according to fold number
        kf = KFold(n_splits=self.splts, shuffle=True, random_state=1)
        for i, [train_index, test_index] in enumerate(kf.split(self.all_samples)):
            if i==self.fold:
                if self.mode=='train':
                    self.samples_index = train_index
                elif self.mode=='test':
                    self.samples_index = test_index
                
        print('Loaded dataset with %d %s samples in fold %d.'%(len(self.samples_index), self.mode, self.fold))
        
        # If pre-processing, calculate fMRI mask with Nilearn
        if self.prepr:
            filename = os.path.join(self.root, 'fMRI_mask.nii')
            self.mask = nl.image.load_img(filename)
        
    def __getitem__(self, idx):
        filename, lbl, idx = self.all_samples[self.samples_index[idx]]
            
        # Load the 4-dimensional fMRI image
        img = np.load(filename).astype(np.float32)
        
        # Average results over temporal dimension
        if self.mean:
            img = np.mean(img, axis=3, keepdims=True)
        
        # Pre-processing by applying a mask, Gaussian smoothing (3mm) and signal cleaning
        if self.prepr:
            img = nl.image.new_img_like(self.mask, img, affine=self.mask.affine, copy_header=True)
            img = nl.image.smooth_img(img, 3)
            img = nl.image.clean_img(img, detrend=True, standardize='zscore')
            img = nl.image.get_data(img)
        
        # Reshape data to: 53 (temporal), 52 (axial), 63 (medial), 53 (lateral)
        img = img.transpose((3,2,1,0))
        
        # Normalise values between -1 and 1
        if self.norm:
            img -= np.mean(0.051363078512534084) # Pre-calculated mean of whole dataset
            img /= 38.625 # Pre-calculated absolute maximum value of whole dataset

        return torch.from_numpy(img), torch.tensor([lbl])
    
    def __len__(self):
        return len(self.samples_index)