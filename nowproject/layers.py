import torch
from torch import nn
from torch.nn import (
    Linear,
    Identity,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Upsample,
    ConvTranspose3d
)
import torch.nn.functional as F

#Complex multiplication
class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

##----------------------------------------------------------------------------.
class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv0 = SpectralConv2d_fast(self.in_channels, self.out_channels, modes1, modes2)
        self.w0 = Conv2d(self.in_channels, self.out_channels, 1)

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2

        return x

##----------------------------------------------------------------------------.
class ConvBlock(torch.nn.Module):
    """Convolution block.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input graph.
    out_channels : int
        Number of channels in the output graph.
    kernel_size : int
        Chebychev polynomial degree
    bias : bool, optional
        Whether to add bias parameters. The default is True.
        If batch_norm = True, bias is set to False.
    batch_norm : bool, optional
        Wheter to use batch normalization. The default is False.
    batch_norm_before_activation : bool, optional
        Whether to apply the batch norm before or after the activation function.
        The default is False.
    activation : bool, optional
        Whether to apply an activation function. The default is True.
    activation_fun : str, optional
        Name of an activation function implemented in torch.nn.functional
        The default is 'relu'.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        bias=True,
        batch_norm=False,
        batch_norm_before_activation=False,
        activation=True,
        activation_fun="relu",
        padding="valid",
        padding_mode="replicate", 
        dilation=1,
        conv_type="regular",
        modes1=None,
        modes2=None
    ):

        super().__init__()
        # If batch norm is used, set conv bias = False
        if batch_norm:
            bias = False
        # Define convolution
        # TODO: (add config)
        # - ‘valid’, ‘same’ or 0,1, ... 
        # - 'padding_mode': 'zeros', 'reflect', 'replicate' or 'circular'  (GG: maybe replicate/reflect is better)
        # torch.nn.Conv2d(in_chan stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        if conv_type == "spectral":
            self.conv = SpectralConv2d_fast(
                in_channels,
                out_channels,
                modes1,
                modes2
            )
        elif conv_type == "fno":
            self.conv = FNO2d(
                in_channels,
                out_channels,
                modes1=modes1,
                modes2=modes2
            )
        else:
            self.conv = Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                padding=padding,
                padding_mode=padding_mode,
                dilation=dilation,
            )
        if batch_norm:
            self.bn = BatchNorm2d(out_channels)
        self.bn_before_act = batch_norm_before_activation
        self.norm = batch_norm
        self.act = activation
        self.act_fun = getattr(F, activation_fun)

    def forward(self, x):
        """Define forward pass of a ConvBlock.

        It expects a tensor with shape: (sample, time, y, x).
        """
        x = self.conv(x)
        if self.norm and self.bn_before_act:
            # [batch, y, x, time-feature]
            x = self.bn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.act:
            x = self.act_fun(x)
        if self.norm and not self.bn_before_act:
            # [batch, y, x, time-feature]
            x = self.bn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class ResBlock(torch.nn.Module):
    """
    General definition of a Residual Block.

    Parameters
    ----------
    in_channels : int
        Input dimension of the tensor.
    out_channels :(int, tuple)
        Output dimension of each ConvBlock within a ResBlock.
        The length of the tuple determine the number of ConvBlock layers
        within the ResBlock.
    convblock_kwargs : TYPE
        Arguments for the ConvBlock layer.
    rezero: bool, optional
        If True, it apply the ReZero trick to potentially improve optimization.
        The default is True.
    act_on_last_conv : bool, optional
        If false, do not add the activation function after the last conv layer.
        The reason is that if the act_fun is Relu, the ResBlock can only output
        positive increments, which might be detrimental to optimization
        The default is False
    bn_on_last_conv : bool, optional
        Specify if applying a batch normalization layer after the last conv layer.
        It is used only if convblock_args['batch_norm'] = True
        The default is False
    bn_zero_init : bool , optional
        If there is a batch normalization layer after the last conv layer (bn_on_last_conv=True),
        if bn_zero_init = True, it initializes the bn layer to zero so that the Resblock
        behavae like an identity (if in_channels == out_channels).
        In practice this seems not necessary if rezero=True.
        The default is False.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        convblock_kwargs,
        rezero=True,
        act_on_last_conv=False,
        bn_on_last_conv=False,
        bn_zero_init=False,
        **kwargs
    ):
        super().__init__()
        ##--------------------------------------------------------
        # ResBlock options
        self.rezero = rezero
        ##--------------------------------------------------------
        # Check out_channels type (if an int --> a single conv)
        if not isinstance(out_channels, (int, tuple, list)):
            raise TypeError("'output_channels' must be int or list/tuple of int.")
        if not isinstance(out_channels, (tuple, list)):
            out_channels = [out_channels]
        out_channels = list(out_channels)
        ##--------------------------------------------------------
        ### Define list of convolutional layers
        # - Initialize list of convolutional layers within the ResBlock
        conv_names_list = []
        tmp_in = in_channels
        n_layers = len(out_channels)
        for i, tmp_out in enumerate(out_channels):
            tmp_convblock_kwargs = convblock_kwargs.copy()
            # --------------------------------------------------------.
            # - If last conv layer, do not add the activation function
            # --> If act_fun like Relu, the ResBlock can only output positive increments
            if not act_on_last_conv and (i == (n_layers - 1)):
                tmp_convblock_kwargs["activation"] = False
            # --------------------------------------------------------.
            # - If last conv layer, do not add the batch_norm (or only add on the last?)
            if not bn_on_last_conv and (i == (n_layers - 1)):
                tmp_convblock_kwargs["batch_norm"] = False
            # --------------------------------------------------------.
            # - Define conv layer name
            tmp_conv_name = "convblock" + str(i + 1)
            # --------------------------------------------------------.
            # - Create the conv layer
            tmp_conv = ConvBlock(
                tmp_in, tmp_out, **tmp_convblock_kwargs
            )
            setattr(self, tmp_conv_name, tmp_conv)
            conv_names_list.append(tmp_conv_name)
            tmp_in = tmp_out

        self.conv_names_list = conv_names_list
        ##--------------------------------------------------------
        ### Define the residual connection
        if in_channels == out_channels[-1]:
            self.res_connection = Identity()
        else:
            self.res_connection = Linear(in_channels, out_channels[-1])

        ##--------------------------------------------------------
        ### Define multiplier initialized at 0 for ReZero trick
        if self.rezero:
            self.rezero_weight = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

        ##--------------------------------------------------------
        # Zero-initialize the last BN in each residual branch
        # --> Each ResBlock behave like identity if input_channels == output_channels
        if bn_on_last_conv and bn_zero_init and convblock_kwargs["batch_norm"]:
            last_convblock = getattr(self, conv_names_list[-1])
            torch.nn.init.constant_(last_convblock.bn.weight, 0)
            torch.nn.init.constant_(last_convblock.bn.bias, 0)

        ##--------------------------------------------------------

    def forward(self, x):
        """Define forward pass of a ResBlock."""
        # Perform convolutions
        x_out = x
        for conv_name in self.conv_names_list:
            x_out = getattr(self, conv_name)(x_out)
        # Rezero trick
        if self.rezero:
            x_out *= self.rezero_weight
        # Add residual connection
        x_out += torch.permute(
                        self.res_connection(torch.permute(x, (0, 2, 3, 1))), 
                        (0, 3, 1, 2)
                    )
        return x_out


class Upsampling(torch.nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 mid_channels: int, 
                 out_channels: int, 
                 convblock_kwargs: dict,
                 kernel_size_pooling: int):
        super().__init__()
        self.uconv21 = ConvBlock(
            in_channels, mid_channels, **convblock_kwargs
        )
        self.uconv22 = ConvBlock(
            mid_channels, out_channels, **convblock_kwargs
        )
        self.up = Upsample(scale_factor=kernel_size_pooling, mode='bilinear', align_corners=True)
        # self.up = ConvTranspose2d(in_channels, in_channels // 2, kernel_size=kernel_size_pooling, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x_cat = torch.cat((x1, x2), dim=1)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)

        return x


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
    def __init__(self, in_channels, out_channels, last=False):
        super().__init__()
        k = (3,4,4) if not last else 4
        s = (1,2,2) if not last else 2
        
        conv_module = ConvTranspose3DPadding if not last else nn.ConvTranspose3d
        self.conv_transpose = conv_module(in_channels,  out_channels, kernel_size=k, stride=s, padding=1)
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
