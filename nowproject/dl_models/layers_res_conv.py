import torch
from torch import nn
from torch.nn import ConvTranspose3d
import torch.nn.functional as F


class ConvTranspose3DPadding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, output_padding=0, 
                 bias=True, dilation=1, padding_mode='zeros'):
        super().__init__()
        self.conv_transpose = ConvTranspose3d(in_channels,  out_channels, kernel_size=kernel_size, 
                                              stride=stride, padding=padding, output_padding=output_padding,
                                              bias=bias, dilation=dilation, padding_mode=padding_mode)

    def forward(self, x1, x2):
        x1 = self.conv_transpose(x1)
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        return x1

def downsampling(in_channels, out_channels, typ='all'):
    k=(3,4,4)
    s=(1,2,2)
        
    return nn.Sequential(
        nn.Conv3d(in_channels,  out_channels, kernel_size=k, stride=s, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.BatchNorm3d(out_channels, affine=True)
    ) 


class UpsamplingResConv(nn.Module):
    def __init__(self, in_channels, out_channels, last=False, first_layer_kernel=(3,4,4), 
                 first_layer_stride=(1,2,2)):
        super().__init__()
        if isinstance(first_layer_kernel, list):
            first_layer_kernel = tuple(first_layer_kernel)
        
        if isinstance(first_layer_stride, list):
            first_layer_stride = tuple(first_layer_stride)

        if len(first_layer_kernel) == 1:
            first_layer_kernel = first_layer_kernel[0]
        if len(first_layer_stride) == 1:
            first_layer_stride = first_layer_stride[0]
        
        conv_module = ConvTranspose3DPadding if not last else nn.ConvTranspose3d
        self.conv_transpose = conv_module(in_channels,  out_channels, kernel_size=first_layer_kernel, 
                                          stride=first_layer_stride, padding=1)
        upsample_layers = [
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.02, inplace=True),
            nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        ]
        if not last:
            upsample_layers += [
                nn.LeakyReLU(negative_slope=0.02, inplace=True),
                nn.BatchNorm3d(out_channels, affine=True)
            ]
        
        self.upsample = nn.Sequential(*upsample_layers) 

    def forward(self, x1, x2=None):
        if x2 is not None:
            x = self.conv_transpose(x1, x2)
        else:
            x = self.conv_transpose(x1)
        x = self.upsample(x)
        return x

def upsampling(in_channels, out_channels, typ='all'):
    k=(3,4,4)
    s=(1,1,1)

    return nn.Sequential(
        nn.ConvTranspose3d(in_channels,  out_channels, kernel_size=k, stride=s, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.BatchNorm3d(out_channels, affine=True)
    ) 

def upsamplingLast(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels,  out_channels, kernel_size=4, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.02, inplace=True),
        nn.ConvTranspose3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    ) 
