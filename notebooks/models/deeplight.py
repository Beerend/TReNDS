import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'original', 'resnet10'
]

def conv3x3(in_channels, out_channels, stride):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size = 3,
        dilation    = 1,
        stride      = stride,
        padding     = 1,
        bias        = True)

class DeepLight(nn.Module):
    
    def __init__(self, cnn_out_dim=512, num_slices=52, temp_size=53, temp_frame=None):
        super(DeepLight, self).__init__()
        self.cnn_out_dim = cnn_out_dim
        self.num_slices  = num_slices
        self.temp_frame  = temp_frame
        
        self.relu = nn.ReLU()
        
        # Convolutional layers
        self.conv1 = conv3x3(1,  16, 2)
        self.conv2 = conv3x3(16, 16, 1)
        self.conv3 = conv3x3(16, 16, 2)
        self.conv4 = conv3x3(16, 16, 1)
        self.conv5 = conv3x3(16, 32, 2)
        self.conv6 = conv3x3(32, 32, 1)
        self.conv7 = conv3x3(32, 32, 2)
        self.conv8 = conv3x3(32, 32, 1)
        
        # LSTM
        self.lstm = nn.LSTM(cnn_out_dim, 40, bidirectional=True)
        
        # Output
        self.fc = nn.Linear(2*num_slices*40, 1, bias=True)
        
        # Initiate all weights with normal distribution (Glorot and Bengio, 2010)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight = nn.init.xavier_normal_(m.weight)
            
    def forward(self, batch_data):
        B,T,A,M,L = batch_data.shape
        assert A==self.num_slices
        
        if self.temp_frame:
            T = 1
            batch_data = torch.unsqueeze(batch_data[:,self.temp_frame,:,:,:], 1) #B,1,A,M,L
            
        out = torch.zeros(B, T, device=batch_data.device)
        for t in range(T):
            embs = torch.zeros(B, A, self.cnn_out_dim, device=batch_data.device)
            for s in range(A):
                x = torch.unsqueeze(batch_data[:,t,s,:,:], 1) #B,1,M,L
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = self.relu(self.conv4(x))
                x = self.relu(self.conv5(x))
                x = self.relu(self.conv6(x))
                x = self.relu(self.conv7(x))
                x = self.relu(self.conv8(x))
                x = torch.flatten(x, start_dim=1) #B,512
                embs[:,s,:] = torch.unsqueeze(x, 1) #B,1,512
                
            x, _ = self.lstm(embs)
            x = torch.flatten(x, start_dim=1) #B,A*40
            x = self.fc(x) #B,1
            out[:,t] = x
                
        if not self.temp_frame:
            out = torch.mean(out, dim=1, keepdim=True) #B,1
        
        return out

def original(**kwargs):
    """ Constructs the original DeepLight model, as in Thomas et
        al. (2019) Analyzing Neuroimaging Data Through Recurrent
        Deep Learning Models, in Frontiers in Neuroscience. """
    
    return DeepLight(**kwargs)

def resnet10(**kwargs):
    """ Extended the original DeepLight model with ResNet-10 as
        feature extractor backbone. """
    
    return DeepLight(**kwargs)

"""
The Human Connectome Project preprocessing pipeline for fMRI data:
1. gradient unwarping
2. motion correction
3. fieldmap-based EPI distortion correction
4. brain-boundary based registration of EPI to structural T1-weighted scan
5. non-linear registration into MNI152 space
6. grand-mean intensity normalization

Additional preprocessing:
7. volume-based smoothing of the fMRI sequences with a 3 mm Gaussian kernel
8. linear detrending and standardization of the signle voxel signal time-series (resulting in a zero-centered voxel time-series with unit variance)
9. temporal filtering of the single voxel time-series with a butterworth highpass filter and a cutoff of 128s
10. applied outer brain mask to each fMRI volume
"""

"""
DeepLight Framework (original model):
- 60 epochs (bs=32)
- ADAM (lr=0.0001)
- global gradient norm clipping (threshold=5)
- early stopping
- normal-distributed random initalisation scheme

1. divide volumetric image in 52 axial slices

2. feature extractor (8 conv. layers)
    - conv (k=3x3, c=16, stride=2, padding=zero), ReLU, dropout=.3
    - conv (k=3x3, c=16, stride=1, padding=zero), ReLU, dropout=.3
    - conv (k=3x3, c=16, stride=2, padding=zero), ReLU, dropout=.4
    - conv (k=3x3, c=16, stride=1, padding=zero), ReLU, dropout=.4
    - conv (k=3x3, c=32, stride=2, padding=zero), ReLU, dropout=.5
    - conv (k=3x3, c=32, stride=1, padding=zero), ReLU, dropout=.5
    - conv (k=3x3, c=32, stride=2, padding=zero), ReLU, dropout=.5
    - conv (k=3x3, c=32, stride=1, padding=zero), ReLU, dropout=.5
    - output dim: 1x960
    
3. Bi-directional LSTM
    - lstm (in=52x960, out=1x40), tanh

4. Ouput:
    - fc (in=40, out=4), softmax
"""