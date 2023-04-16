# -*- encoding:utf-8 -*-
# @Author: jhan
# @FileName: fegan.py
# @Date: 2022/6/17 16:21

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet
import torch.nn.functional as F

lrelu_value = 0.1
act = nn.LeakyReLU(lrelu_value)

def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t

class RRRB(nn.Module):
    """ Residual in residual re-parameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.

    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|

    Args:
        in_places/out_places (int): The number of feature maps.
    """

    def __init__(self, in_places, out_places, kernel_size, stride):
        super(RRRB, self).__init__()
        self.expand_conv = nn.Conv2d(in_places, out_places, 1, 1, 0)
        self.fea_conv = nn.Conv2d(out_places, out_places, kernel_size, stride, 0)
        self.reduce_conv = nn.Conv2d(out_places, in_places, 1, 1, 0)

    def forward(self, x):
        out = self.expand_conv(x)
        out_identity = out      # 256*256

        # explicitly padding with bias for re-parameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identity
        out = self.reduce_conv(out)
        out += x

        return out


class BasicBlock(nn.Module):
    """
        Diagram:
            --conv3x3--BN--ReLU--conv3x3--BN--ReLU--
            |------------conv3x3--BN---------------|

        Args:
            in_places/out_places (int): The number of feature maps.
        """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None

        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    """
        Diagram:
            --BasicBlock--BasicBlock--
        """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class ERB(nn.Module):
    """
    Diagram:
        --RRRB--LeakyReLU--RRRB--

    Args:
        in_places/out_places (int): The number of feature maps.
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(ERB, self).__init__()
        self.conv1 = RRRB(in_planes, out_planes, kernel_size, stride)
        self.conv2 = RRRB(in_planes, out_planes, kernel_size, stride)

    def forward(self, x):
        out = self.conv1(x)
        out = act(out)
        out = self.conv2(out)

        return out

class Encoder1(nn.Module):
    """
    Diagram:
        --ERB--ERB--

    Args:
    in_places/out_places (int): The number of feature maps.
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        super(Encoder1, self).__init__()
        self.block1 = ERB(in_planes, out_planes, kernel_size, stride)
        self.block2 = ERB(out_planes, out_planes, kernel_size, stride)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class Decoder(nn.Module):
    """
    Diagram:
        --conv1x1--BN--ReLU--conv3x3--BN--ReLU--conv1x1--BN--ReLU--+--
        |----------------------encoder output----------------------|

    Args:
        in_places/out_places (int): The number of feature maps.
    """

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class Generator(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)                    # 64

        self.encoder1 = Encoder1(64, 64, 3, 1)                  # 64
        self.encoder2 = Encoder(64, 128, 3, 2, 1)               # 32
        self.encoder3 = Encoder(128, 256, 3, 2, 1)              # 16
        self.encoder4 = Encoder(256, 512, 3, 2, 1)              # 8

        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)           # 32
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)           # 64
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)            # 128
        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)             # 256

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, 32, 4, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(32, 3, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder blocks
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)

        y = self.final(d1)

        return y


class Discriminator(nn.Module):
    """
    A 4-layer Markovian discriminator
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            # Returns downsampling layers of each discriminator block
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if bn: layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)