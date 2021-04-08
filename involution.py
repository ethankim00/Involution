import torch
import torch.nn as nn


import matplotlib.pyplot as plt
import numpy as np


from typing import Union, Tuple

class Involution2d(nn.Module):
    """Involution layer for Neural Networks
    
    Implementation of the involution layer described by Li et al. 2021 https://arxiv.org/pdf/2103.06255v1.pdf
    
    """
    def __init__(self, \
                in_dim : int, \
                out_dim : int, \
                kernel_dim : Union[int, Tuple[int, int]], \
                stride : Union[int, Tuple[int, int]], \
                groups : int = 1, \
                reduction : int = 1, \
                dilation : Union[int, Tuple[int, int]] = 1, \
                padding : Union[int, Tuple[int, int]] = 3, \
                activation : nn.Module = None) -> None: 
        """Initialize Involution Layer.
        
        Unpack arguments and initialize needed transformations.
        
        Args:
            in_dim : number of input channels
            out_dim : number of output channels
            kernel_dim : Kernel size, can be specified by int or tuple
            stride : stride
            groups : Number of kernel generators to use
            reduction : bottleneck factor 
            dilation : 
            pading : 
            activation : Nonlinearity function to apply 
        """
        super(Involution2d, self).__init__()
        # Unpack Params
        self.in_dim = in_dim 
        self.out_dim = out_dim
        self.kernel_dim = kernel_dim if isinstance(kernel_dim, tuple) else tuple((kernel_dim, kernel_dim))
        self.stride = stride 
        self.groups = groups 
        self.reduction = reduction  
        self.dilation = dilation 
        self.padding = padding 
        # Initialize Transformations
        self.initial_mapping = nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),bias=False) if self.in_dim != self.out_dim else nn.Identity() # Initial linear upsampling/downsampling
        self.unfold = nn.Unfold(kernel_size=self.kernel_dim, dilation=self.dilation, padding=self.padding, stride=self.stride)
        self.o = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride) if stride !=  1 else nn.Identity()
        self.reduce = nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim // self.reduction, kernel_size = (1, 1)) 
        self.span = nn.Conv2d(in_channels= self.out_dim // self.reduction, out_channels = self.kernel_dim[0] * self.kernel_dim[1] * self.groups, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.sigma = activation if activation is not None else nn.Sequential(nn.BatchNorm2d(num_features=self.out_dim // self.reduction), nn.ReLU())
        
    def forward(self, X : torch.tensor) -> torch.tensor:
        """Forward Pass of Involution Layer. 

        Args:
            X : input of dimension, batch x channels x height x width

        Returns: 
            out : output of dimension batch x outchannels x height x width 
        """
        B, C, H, W = X.shape 
        assert(C == self.in_dim)
        unfolded = self.unfold(self.initial_mapping(X))
        unfolded = unfolded.view(B, self.groups, self.out_dim//self.groups, self.kernel_dim[0] * self.kernel_dim[1], H, W)
        pool = self.o(X)
        reduced = self.reduce(pool) # Apply bottleneck dimension reduction
        kernel = self.span(self.sigma(reduced)) # Convert each channel to a kernel C -> KxK pixelwise 
        kernel = kernel.view(B, self.groups, self.kernel_dim[0]*self.kernel_dim[1], H, W).unsqueeze(dim = 2)
        out = (kernel * unfolded).sum(dim =3)
        # Apply the kernels
        out = out.view(B, -1, H, W)
    
        return out 