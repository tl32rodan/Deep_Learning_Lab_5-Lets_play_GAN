import torchvision.utils
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Generator Code
class Generator(nn.Module):
    def __init__(self, ngpu, nz=100, nc = 3, ngf = 64, num_conds=24):
        r'''
            nc : Number of channels in the training images. For color images this is 3
            nz : Size of z latent vector (i.e. size of generator input)
            ngf: Size of feature maps in generator
        '''
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.num_conds = num_conds
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz+self.num_conds, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        

    def forward(self, input, cond):
        cond = cond.view(-1, self.num_conds,1,1)
        #int('input.shape = ',input.shape)
        #print('cond.shape = ',cond.shape)
        x = torch.cat((input, cond),1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ngpu, nz=100, nc = 3, ndf = 64, num_conds=24):
        r'''
            nc : Number of channels in the training images. For color images this is 3
            nz : Size of z latent vector (i.e. size of generator input)
            ndf: Size of feature maps in discriminator
        '''
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ndf = ndf
        self.nc = nc
        self.num_conds = num_conds
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc+1, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.decision_layer = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        # Reshape condition to a channel of image
        self.extend_cond = nn.Sequential(
            nn.Linear(self.num_conds, 64*64, bias=False),
            nn.ReLU()
        )

    def forward(self, input, cond, get_feature = False):
        cond = self.extend_cond(cond).view(-1,1,64,64)
        #print('input.shape = ',input.shape)
        #print('cond.shape = ',cond.shape)
        x = torch.cat((input,cond),1)
        x = self.main(x)
        if get_feature:
            last_f = x
            x = self.decision_layer(x)
            return x, last_f
        else:
            return self.decision_layer(x)
