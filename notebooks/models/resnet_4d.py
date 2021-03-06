import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import Tuple
from math import floor
from functools import partial

__all__ = ['4d_cnn_resnet10']

class Conv4d(nn.Module):
    """ Code taken and adapted from Timothy Gebhard:
    https://github.com/timothygebhard/pytorch-conv4d/blob/master/conv4d.py """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 stride=1, padding=0, bias=True):
        self.C_i = in_channels
        self.C_o = out_channels
        self.K   = kernel_size
        self.D   = dilation
        self.S   = stride
        self.P   = padding
        self.B   = bias
        super(Conv4d, self).__init__()
        
        # Initialise a Conv3d layer and store in list
        self.conv3d_layers = nn.ModuleList()
        for i in range(self.K):
            conv3d_layer = nn.Conv3d(self.C_i, self.C_o, kernel_size=self.K,
                                     dilation=self.D, stride=self.S, padding=self.P,
                                     bias=self.B)
            self.conv3d_layers.append(conv3d_layer)
            
    def forward(self, x):
        T = x.shape[2]
        
        # Size of output tensor (O) and output tensor without padding and stride (O_s)
        O = floor((T + 2*self.P - self.K) / self.S + 1)
#         O_s = T - self.K + 1
        
        # Output tensors for each Conv3d result
        frame_results = O * [None]
        
        for i in range(self.K):
            for j in range(T):
                # Calculate the output temporal frame without stride
                out_frame = j - i + self.P
                
                # Check if result is retained after stride
                if out_frame%self.S!=0:
                    continue
                
                # Calcualte the output temporal frame after stride
                out_frame = floor(out_frame/self.S)
                    
                # Check if output frame is without output boundaries
                if out_frame<0 or out_frame>=O:
                    continue
                
                # Convolute temporal frame with 3D convolutional layer
                frame_conv3d = self.conv3d_layers[i](x[:,:,j,:,:,:])
                
                # Check if first result for temporal frame, store result
                if frame_results[out_frame] is None:
                    frame_results[out_frame] = frame_conv3d
                else:
                    frame_results[out_frame] += frame_conv3d
                    
        return torch.stack(frame_results, dim=2)
    
#                 out_frame = j - (i - self.K // 2) - (T - O_s) // 2 - (1 - self.K % 2)
#                 k_center  = out_frame % self.S
#                 out_frame = floor(out_frame / self.S)
#                 if k_center != 0 or out_frame <0 or out_frame >= O:
#                     continue
    
class MaxPool4d(nn.Module):
    
    def __init__(self, kernel_size:int, stride:int=1, padding:int=0):
        self.K = kernel_size
        self.S = stride
        self.P = padding
        super(MaxPool4d, self).__init__()
        
        self.pool3d = nn.MaxPool3d(kernel_size=self.K, stride=self.S, padding=self.P)
        self.pool1d = nn.MaxPool1d(kernel_size=self.K, stride=self.S, padding=self.P)
        
    def forward(self, x):
        B,C,T  = x.shape[:3]
        X_size = x.shape[2:]
        O_size = [floor((X_size[i] + 2*self.P - self.K) / self.S + 1) for
                 i in range(4)]
        
        frame_results = T * [None]
        
        for i in range(T): 
            frame_results[i] = self.pool3d(x[:,:,i,:,:,:])
        
        x = torch.stack(frame_results, dim=2)
        x = x.permute(0,1,3,4,5,2)
        x = x.contiguous()
        x = x.view((B,C*O_size[1]*O_size[2]*O_size[3],T))
        x = self.pool1d(x)
        x = x.view((B,C,O_size[1],O_size[2],O_size[3],O_size[0]))
        x = x.permute(0,1,5,2,3,4)
        
        return x
    
class AvgPool4d(nn.Module):
    
    def __init__(self, kernel_size:Tuple, stride:int=1, padding:int=0):
        self.K = kernel_size
        self.S = stride
        self.P = padding
        super(AvgPool4d, self).__init__()
        
        self.pool3d = nn.AvgPool3d(kernel_size=self.K[1:], stride=self.S, padding=self.P)
        self.pool1d = nn.AvgPool1d(kernel_size=self.K[0], stride=self.S, padding=self.P)
        
    def forward(self, x):
        B,C,T  = x.shape[:3]
        X_size = x.shape[2:]
        O_size = [floor((X_size[i] + 2*self.P - self.K[i]) / self.S + 1) for
                 i in range(4)]
        
        frame_results = T * [None]
        
        for i in range(T): 
            frame_results[i] = self.pool3d(x[:,:,i,:,:,:])
        
        x = torch.stack(frame_results, dim=2)
        x = x.permute(0,1,3,4,5,2)
        x = x.contiguous()
        x = x.view((B,C*O_size[1]*O_size[2]*O_size[3],T))
        x = self.pool1d(x)
        x = x.view((B,C,O_size[1],O_size[2],O_size[3],O_size[0]))
        x = x.permute(0,1,5,2,3,4)
        
        return x
    
class BatchNorm4d(nn.Module):
    """ Not implemented yet """
    
    def __init__(self, num_features):
        super(BatchNorm4d, self).__init__()

    def forward(self, x):
        
        return x

class BasicBlock(nn.Module):
    
    def __init__(self, channels_in, channels_out, stride=1, dilation=1, downsample=None):
        self.S = stride
        self.D = dilation
        super(BasicBlock, self).__init__()
        
        self.conv1 = Conv4d(channels_in, channels_out, kernel_size=3, stride=stride,
                            dilation=dilation, padding=dilation)
        # self.bn1   = BatchNorm4d(channels_out)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = Conv4d(channels_out, channels_out, kernel_size=3, stride=1, 
                            dilation=dilation, padding=dilation)
        # self.bn2   = BatchNorm4d(channels_out)
        self.downs = downsample
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        
        if self.downs:
            res = self.downs(res)
                
        x += res
        x = self.relu(x)
        return x

class ResNet4D(nn.Module):
    
    def __init__(self, layers, num_class=1, no_cuda=False):
        self.no_cuda  = no_cuda
        self.channels = 64
        super(ResNet4D, self).__init__()
        
        self.conv1   = Conv4d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1     = BatchNorm4d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = MaxPool4d(kernel_size=3, stride=2, padding=1)
        
        self.layer1  = self._make_layer(64, layers[0])
        self.layer2  = self._make_layer(128, layers[1], stride=2)
        self.layer3  = self._make_layer(256, layers[2], stride=2)
        self.layer4  = self._make_layer(512, layers[3], stride=2)
        
        self.avgpool = AvgPool4d(kernel_size=(2,2,2,2))
        self.fc      = nn.Sequential(nn.Linear(512, num_class, bias=True))
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.SyncBatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def _make_layer(self, channels_out, blocks, stride=1, dilation=1):
        downsample = None
        if stride!=1 or channels_out!=self.channels:
            downsample = nn.Sequential(Conv4d(self.channels, channels_out,
                                       kernel_size=1, stride=stride, bias=False)) #BatchNorm4d(channels_out)
        
        layers = []
        layers.append(BasicBlock(self.channels, channels_out, stride=stride,
                      dilation=dilation, downsample=downsample))
        self.channels = channels_out
        
        for i in range(1, blocks):
            layers.append(BasicBlock(self.channels, channels_out, dilation=dilation))
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1) #B,C,T,A,M,L
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)        
        x = self.avgpool(x)
        x = x.view((-1, 512))
        x = self.fc(x)
        
        return x
    
def resnet10_4d(**kwargs):
    """ Constructs a ResNet-10 model with 4-dim convolutional kernels """
    model = ResNet4D([1,1,1,1], **kwargs)
    return model