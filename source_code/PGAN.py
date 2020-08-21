import torch
import torch.nn as nn
import torch.nn.functional as F
from progressive_conv_nets import GNet, DNet


class ProgressiveGAN(nn.Module):
    def __init__(self,
                 dimLatentVector=512,
                 depthScale0=512,
                 initBiasToZero=True,
                 leakyness=0.2,
                 perChannelNormalization=True,
                 miniBatchStdDev=False,
                 equalizedlR=True,
                 **kwargs):
        r"""
        Args:
        Specific Arguments:
            - depthScale0 (int)
            - initBiasToZero (bool): should layer's bias be initialized to
                                     zero ?
            - leakyness (float): negative slope of the leakyRelU activation
                                 function
            - perChannelNormalization (bool): do we normalize the output of
                                              each convolutional layer ?
            - miniBatchStdDev (bool): mini batch regularization for the
                                      discriminator
            - equalizedlR (bool): if True, forces the optimizer to see weights
                                  in range (-1, 1)
        """
        
        self.depthScale0 = depthScale0
        self.initBiasToZero = initBiasToZero
        self.leakyReluLeak = leakyness
        self.depthOtherScales = []
        self.perChannelNormalization = perChannelNormalization
        self.alpha = 0
        self.miniBatchStdDev = miniBatchStdDev
        self.equalizedlR = equalizedlR
        
        self.gnet = GNet(self.latentVectorDim,
                    self.depthScale0,
                    initBiasToZero=self.initBiasToZero,
                    leakyReluLeak=self.leakyReluLeak,
                    normalization=self.perChannelNormalization,
                    generationActivation=None,
                    dimOutput=self.dimOutput,
                    equalizedlR=self.equalizedlR)
        
        self.dnet = DNet(self.depthScale0,
                    initBiasToZero=self.initBiasToZero,
                    leakyReluLeak=self.leakyReluLeak,
                    sizeDecisionLayer=1,
                    miniBatchNormalization=self.miniBatchStdDev,
                    dimInput=self.dimOutput,
                    equalizedlR=self.equalizedlR)
