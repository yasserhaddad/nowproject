from functools import partial
from typing import List, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

# https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/buildingblocks.py

def create_conv_3d(in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]], 
                   order: Union[List[str], Tuple[str, ...]], num_groups: int, padding: int) -> List[nn.Module]:
    """Creates a convolutional block composed of the specified modules. The available options are : Conv3D,
    GroupNorm, BatchNorm and activation functions (ReLU, LeakyReLU and ELU).

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : Union[int, Tuple[int, ...]]
        Kernel size of the convolution
    order : Union[List[str], Tuple[str, ...]]
        Order of the layers in the convolutional block
    num_groups : int
        Number of groups to divide the channels into for
        the GroupNorm
    padding : int
        Padding of the convolution

    Returns
    -------
    List[nn.Module]
        List of the different modules composing the convolutional block

    Raises
    ------
    TypeError
        Order must be a list or a tuple
    ValueError
        Order must contain "conv"
    ValueError
        Non-linearity cannot be the first operation in the layer
    ValueError
        Number of channels shuld be divisible by num_groups
    """
    if type(order) not in [list, tuple]:
        raise TypeError("'order' variable must be a list") 
    
    if type(order) == tuple:
        order = list(order)
    order = [element.lower() for element in order]   
    
    if 'conv' not in order:
        raise ValueError("Conv layer MUST be present")
    if order[0] in ['relu', 'leakyrelu', 'elu']:
        raise ValueError( 'Non-linearity cannot be the first operation in the layer')

    modules = []

    for i, module in enumerate(order):
        if module == 'relu':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif module == 'leakyrelu':
            modules.append(('LeakyReLU', nn.LeakyReLU(inplace=True)))
        elif module == 'elu':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif module == 'conv':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not 'batchnorm' in order
            modules.append(('conv', nn.Conv3d(in_channels, out_channels, kernel_size, 
                                                bias=bias, padding=padding)))
        elif module == 'groupnorm':
            is_before_conv = i < order.index('conv')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif module == 'batchnorm':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{module}'. MUST be one of ['batchnorm', 'groupnorm', 'relu', 'leakyrelu', 'elu', 'conv']")

    return modules


class SingleConv(nn.Sequential):
    """Creates a single convolutional block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]] = 3, 
                 order: Union[List[str], Tuple[str, ...]] = ("conv", "relu"), num_groups: int = 8, 
                 padding: int = 1):
        """Initializes the Single Conv.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : Union[int, Tuple[int, ...]], optional
            Kernel size of the convolution, by default 3
        order : Union[List[str], Tuple[str, ...]], optional
            Order of the layers in the convolutional block, by default ("conv", "relu")
        num_groups : int, optional
            Number of groups to divide the channels into for
            the GroupNorm, by default 8
        padding : int, optional
            Padding of the convolution, by default 1
        """
        super().__init__()

        for name, module in create_conv_3d(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """Creates a series of two convolutional blocks, either for encoding or decoding."""
    def __init__(self, in_channels: int, out_channels: int, encoder: bool, kernel_size: Union[int, Tuple[int, ...]] = 3, 
                 order: Union[List[str], Tuple[str, ...]] = ("conv", "relu"), num_groups: int = 8, 
                 padding: int = 1):
        """Initiliazes the DoubleConv. This series of two convolutional blocks
        can be used either in the encoding path, in which case the number of channels
        increase, or in the decoding path, in which case the number of channels decrease.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        encoder : bool
            Whether the series of convolutional blocks is
            in the encoding path or not
        kernel_size : Union[int, Tuple[int, ...]], optional
            Kernel size of the convolution, by default 3
        order : Union[List[str], Tuple[str, ...]], optional
            Order of the layers in the convolutional block, by default ("conv", "relu")
        num_groups : int, optional
            Number of groups to divide the channels into for
            the GroupNorm, by default 8
        padding : int, optional
            Padding of the convolution, by default 1
        """
        super().__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups,
                                   padding=padding))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups,
                                   padding=padding))



class ResNetBlock(nn.Module):
    """
    Basic residual UNet block with no activation function after adding the increment and after the last layer.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]] = 3, 
                 order: Union[List[str], Tuple[str, ...]] = ("conv", "relu"), num_groups: int = 8, 
                 rezero: bool = True, **kwargs):
        """Initializes the ResNetBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : Union[int, Tuple[int, ...]], optional
            Kernel size of the convolution, by default 3
        order : Union[List[str], Tuple[str, ...]], optional
            Order of the layers in the convolutional block, by default ("conv", "relu")
        num_groups : int, optional
            Number of groups to divide the channels into for
            the GroupNorm, by default 8
        rezero : bool, optional
            Whether to apply the ReZero trick, by default True
        """
        super().__init__()
        if type(order) != list:
            order = list(order)
        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in ["relu", "elu", "leakyrelu"]:
            if c in n_order:
                n_order.remove(c)
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order, num_groups=num_groups)

        if in_channels == out_channels:
            self.res_connection = nn.Identity()
        else:
            self.res_connection = nn.Linear(in_channels, out_channels)

        ### Define multiplier initialized at 0 for ReZero trick
        self.rezero = rezero
        if self.rezero:
            self.rezero_weight = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        if self.rezero:
            out *= self.rezero_weight
        # Add residual connection
        out += torch.permute(
                    self.res_connection(torch.permute(x, (0, 2, 3, 4, 1))), 
                    (0, 4, 1, 2, 3)
                )

        return out

class ExtResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.
    The SingleConv takes care of increasing/decreasing the number of channels and also ensures that the number
    of output channels is compatible with the residual block that follows.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, ...]] = 3, 
                 order: Union[List[str], Tuple[str, ...]] = ("conv", "relu"), num_groups: int = 8, **kwargs):
        """Initializes the ExResNetBlock.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : Union[int, Tuple[int, ...]], optional
            Kernel size of the convolution, by default 3
        order : Union[List[str], Tuple[str, ...]], optional
            Order of the layers in the convolutional block, by default ("conv", "relu")
        num_groups : int, optional
            Number of groups to divide the channels into for
            the GroupNorm, by default 8
        """
        super().__init__()
        if type(order) != list:
            order = list(order)
        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        n_order = order
        for c in ["relu", "elu", "leakyrelu"]:
            if c in n_order:
                n_order.remove(c)
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order, num_groups=num_groups)

        # create non-linearity separately
        if 'leakyrelu' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'elu' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out

        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer followed by a DoubleConv module.
    """
    def __init__(self, in_channels: int, out_channels: int, conv_kernel_size: Union[int, Tuple[int, ...]] = 3, 
                 apply_pooling: bool = True, pool_kernel_size: Union[int, Tuple[int, ...]] = 2, 
                 pool_type: str = 'max', basic_module: nn.Module = DoubleConv, 
                 conv_layer_order: Union[List[str], Tuple[str, ...]] = ("conv", "relu"), 
                 num_groups: int = 8, padding: int = 1):
        """Initialize the Encoder.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        conv_kernel_size : Union[int, Tuple[int, ...]], optional
            Kernel size of the convolution, by default 3
        apply_pooling : bool, optional
            Whether to apply pooling before convolution, by default True
        pool_kernel_size : Union[int, Tuple[int, ...]], optional
            Kernel size of the pooling, by default 2
        pool_type : str, optional
            Type of pooling layer to use, by default 'max'
        basic_module : nn.Module, optional
            Basic model for the encoder/decoder, by default DoubleConv
        conv_layer_order : Union[List[str], Tuple[str, ...]], optional
            Order of the layers in the basic convolutional block, by default ("conv", "relu")
        num_groups : int, optional
            Number of groups to divide the channels into for the GroupNorm, by default 8
        padding : int, optional
            Padding of the convolution, by default 1
        """
        super().__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer (either learned 
    ConvTranspose3d or nearest neighbor interpolation) followed by a basic module.
    """
    def __init__(self, in_channels: int, out_channels: int, conv_kernel_size: Union[int, Tuple[int, ...]] = 3, 
                 scale_factor: Union[int, Tuple[int, ...]] = (2, 2, 2), basic_module: nn.Module = DoubleConv, 
                 conv_layer_order: Union[List[str], Tuple[str, ...]] = ("conv", "relu"), num_groups: int = 8, 
                 mode: str = 'nearest', padding: int = 1, upsample: bool = True):
        """Initialize the Decoder.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        conv_kernel_size : Union[int, Tuple[int, ...]], optional
            Kernel size of the convolution, by default 3
        scale_factor : Union[int, Tuple[int, ...]], optional
            Used as the multiplier for the image H/W/D, must reverse the pooling from
            the corresponding encoder, by default (2, 2, 2)
        basic_module : nn.Module, optional
            Basic module for the decoder, by default DoubleConv
        conv_layer_order : Union[List[str], Tuple[str, ...]], optional
            Order of the layers in the basic convolutional block, by default ("conv", "relu")
        num_groups : int, optional
            Number of groups to divide the channels into for the GroupNorm, by default 8
        mode : str, optional
            Interpolation upsampling mode, by default 'nearest'
        padding : int, optional
            Padding of the convolution, by default 1
        upsample : bool, optional
            Whether the input should be upsampled, by default True
        """
        super().__init__()

        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=mode)
                # concat joining
                self.joining = partial(self._joining, concat=True)
            else:
                # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor)
                # sum joining
                self.joining = partial(self._joining, concat=False)
                # adapt the number of in_channels for the ExtResNetBlock
                in_channels = out_channels
        else:
            # no upsampling
            self.upsampling = NoUpsampling()
            # concat joining
            self.joining = partial(self._joining, concat=True)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.joining(encoder_features, x)
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


def create_encoders(in_channels: int, f_maps: List[int], basic_module: nn.Module, 
                    conv_kernel_size: Union[int, Tuple[int, ...]], conv_padding: int, 
                    layer_order: Union[List[str], Tuple[str, ...]], num_groups: int,
                    pool_kernel_size: Union[int, Tuple[int, ...]], 
                    pool_type: str = "max") -> nn.ModuleList:
    """Creates a list of encoders of depth len(f_maps).

    Parameters
    ----------
    in_channels : int
        Number of input channels
    f_maps : List[int]
        Number of feature maps
    basic_module : nn.Module
        Basic module for the encoder
    conv_kernel_size : Union[int, Tuple[int, ...]]
        Kernel size of the convolution
    conv_padding : int
        Padding of the convolution
    layer_order : Union[List[str], Tuple[str, ...]]
        Order of the layers in the basic convolutional block
    num_groups : int
        Number of groups to divide the channels into for the GroupNorm
    pool_kernel_size : Union[int, Tuple[int, ...]]
        Kernel size of the pooling
    pool_type : str, optional
        Type of pooling layer to use, by default "max"

    Returns
    -------
    nn.ModuleList
        List of encoders
    """
    encoders = []
    for i, out_feature_num in enumerate(f_maps):
        if i == 0:
            encoder = Encoder(in_channels, out_feature_num,
                              apply_pooling=False,  # skip pooling in the first encoder
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              padding=conv_padding)
        else:
            encoder = Encoder(f_maps[i - 1], out_feature_num,
                              basic_module=basic_module,
                              conv_layer_order=layer_order,
                              conv_kernel_size=conv_kernel_size,
                              num_groups=num_groups,
                              pool_kernel_size=pool_kernel_size,
                              padding=conv_padding,
                              pool_type=pool_type)

        encoders.append(encoder)

    return nn.ModuleList(encoders)


def create_decoders(f_maps: List[int], basic_module: nn.Module, conv_kernel_size: Union[int, Tuple[int, ...]], 
                    conv_padding: int, layer_order: Union[List[str], Tuple[str, ...]], num_groups: int, 
                    upsample: bool, scale_factor: Union[int, Tuple[int, ...]]) ->  nn.ModuleList:
    """Creates a list of decoders of length len(f_maps) - 1.

    Parameters
    ----------
    f_maps : List[int]
        Number of feature maps
    basic_module : nn.Module
        Basic module for the decoder
    conv_kernel_size : Union[int, Tuple[int, ...]]
        Kernel size of the convolution
    conv_padding : int
        Padding of the convolution
    layer_order : Union[List[str], Tuple[str, ...]]
        Order of the layers in the basic convolutional block
    num_groups : int
        Number of groups to divide the channels into for the GroupNorm
    upsample : bool
        Whether the input should be upsampled
    scale_factor : Union[int, Tuple[int, ...]]
        Used as the multiplier for the image H/W/D, must reverse the pooling from
        the corresponding encoder

    Returns
    -------
    nn.ModuleList
        List of decoders
    """
    # create decoder path consisting of the Decoder modules. The length of the decoder list is equal to `len(f_maps) - 1`
    decoders = []
    reversed_f_maps = list(reversed(f_maps))
    for i in range(len(reversed_f_maps) - 1):
        if basic_module == DoubleConv:
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
        else:
            in_feature_num = reversed_f_maps[i]

        out_feature_num = reversed_f_maps[i + 1]

        # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
        # currently strides with a constant stride: (2, 2, 2)

        _upsample = True
        if i == 0:
            # upsampling can be skipped only for the 1st decoder, afterwards it should always be present
            _upsample = upsample

        decoder = Decoder(in_feature_num, out_feature_num,
                          basic_module=basic_module,
                          conv_layer_order=layer_order,
                          conv_kernel_size=conv_kernel_size,
                          num_groups=num_groups,
                          padding=conv_padding,
                          upsample=_upsample,
                          scale_factor=scale_factor)
        decoders.append(decoder)
    return nn.ModuleList(decoders)


class DecoderMultiScale(nn.Module):
    """
    A single module for decoder path for MultiScaleResConv consisting of the upsampling layer (either learned 
    ConvTranspose3d or nearest neighbor interpolation) followed by a basic module.
    """
    def __init__(self, in_channels: int, out_channels: int, conv_kernel_size: Union[int, Tuple[int, ...]] = 3, 
                 scale_factor: Union[int, Tuple[int, ...]] = (2, 2, 2), basic_module: nn.Module = DoubleConv,
                 conv_layer_order: Union[List[str], Tuple[str, ...]] = ("conv", "relu"), num_groups: int = 8, 
                 mode: str = 'nearest', padding: int = 1, upsample: bool = True):
        """Initialize the DecoderMultiScale.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        conv_kernel_size : Union[int, Tuple[int, ...]], optional
            Kernel size of the convolution, by default 3
        scale_factor : Union[int, Tuple[int, ...]], optional
            Used as the multiplier for the image H/W/D, must reverse the pooling from
            the corresponding encoder, by default (2, 2, 2)
        basic_module : nn.Module, optional
            Basic module for the decoder, by default DoubleConv
        conv_layer_order : Union[List[str], Tuple[str, ...]], optional
            Order of the layers in the basic convolutional block, by default ("conv", "relu")
        num_groups : int, optional
            Number of groups to divide the channels into for the GroupNorm, by default 8
        mode : str, optional
            Interpolation upsampling mode, by default 'nearest'
        padding : int, optional
            Padding of the convolution, by default 1
        upsample : bool, optional
            Whether the input should be upsampled, by default True
        """
        super().__init__()

        if upsample:
            if basic_module == DoubleConv:
                # if DoubleConv is the basic_module use interpolation for upsampling and concatenation joining
                self.upsampling = InterpolateUpsampling(mode=mode)
            else:
                # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
                self.upsampling = TransposeConvUpsampling(in_channels=in_channels, out_channels=out_channels,
                                                          kernel_size=conv_kernel_size, scale_factor=scale_factor)
                # adapt the number of in_channels for the ExtResNetBlock
                in_channels = out_channels
            
        else:
            # no upsampling
            self.upsampling = NoUpsampling()

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         padding=padding)

    def forward(self, encoder_features, x):
        x = self.upsampling(encoder_features=encoder_features, x=x)
        x = self.basic_module(x)
        return x


def create_encoders_multiscale(in_channels: int, f_maps: List[int], basic_module: nn.Module, 
                               conv_kernel_size: Union[int, Tuple[int, ...]], 
                               conv_padding: int, layer_order: Union[List[str], Tuple[str]], 
                               num_groups: int, upsample_last: bool = False) -> nn.ModuleList:
    """Creates a list of encoders for MutliScaleResConv. It starts with a list of length len(f_maps) 
    encoders that increases the number of channels and then it is followed by a list of length
    len(f_maps) - 2 or len(f_maps) - 1 that decreases the number of channels. 

    Parameters
    ----------
    in_channels : int
        Number of input channels
    f_maps : List[int]
        Number of feature maps
    basic_module : nn.Module
        Basic module for the encoder
    conv_kernel_size : Union[int, Tuple[int, ...]]
        Kernel size of the convolution
    conv_padding : int
        Padding of the convolution
    layer_order : Union[List[str], Tuple[str]]
        Order of the layers in the basic convolutional block
    num_groups : int
        Number of groups to divide the channels into for the GroupNorm
    upsample_last : bool, optional
        Whether the output of the last encoder will be upsampled after 
        encoding. If it is not, the list of encoders that decrease the 
        number of channels will have a length of len(f_maps) - 1,
        otherwise len(f_maps) - 2, by default False

    Returns
    -------
    nn.ModuleList
        List of encoders
    """
    encoders_in = []
    encoders_out = []
    for i, out_feature_num in enumerate(f_maps):
        input_channels = in_channels if i == 0 else f_maps[i - 1]
        encoder = Encoder(input_channels, out_feature_num,
                            apply_pooling=False, 
                            basic_module=basic_module,
                            conv_layer_order=layer_order,
                            conv_kernel_size=conv_kernel_size,
                            num_groups=num_groups,
                            padding=conv_padding)

        encoders_in.append(encoder)

        if i > 0: 
            if not upsample_last or i != 1:
                encoder = Encoder(out_feature_num, input_channels,
                                    apply_pooling=False, 
                                    basic_module=basic_module,
                                    conv_layer_order=layer_order,
                                    conv_kernel_size=conv_kernel_size,
                                    num_groups=num_groups,
                                    padding=conv_padding)

                encoders_out.append(encoder)

    return nn.ModuleList(encoders_in + encoders_out[::-1])



class AbstractUpsampling(nn.Module):
    """
    Abstract class for upsampling. A given implementation should upsample a 
    given 5D input tensor using either interpolation or learned 
    transposed convolution.
    """

    def __init__(self, upsample):
        super().__init__()
        self.upsample = upsample

    def forward(self, encoder_features, x):
        # get the spatial dimensions of the output given the encoder_features
        # output_size = encoder_features.size()[2:]
        x = self.upsample(x)

        diffY = encoder_features.size()[-2] - x.size()[-2]
        diffX = encoder_features.size()[-1] - x.size()[-1]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        return x


class InterpolateUpsampling(AbstractUpsampling):
    """
    Upsamples input using interpolation.
    """
    def __init__(self, mode='nearest'):
        """Initialize InteropolateSampling.

        Parameters
        ----------
        mode : str, optional
            Mode of the interpolation. Options are 'nearest' | 'linear' | 
            'bilinear' | 'trilinear' | 'area', by default 'nearest'
        """
        upsample = partial(self._interpolate, mode=mode)
        super().__init__(upsample)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)

class TransposeConvUpsampling(AbstractUpsampling):
    """
    Args:
        in_channels (int): number of input channels for transposed conv
            used only if transposed_conv is True
        out_channels (int): number of output channels for transpose conv
            used only if transposed_conv is True
        kernel_size (int or tuple): size of the convolving kernel
            used only if transposed_conv is True
        scale_factor (int or tuple): stride of the convolution
            used only if transposed_conv is True
    """

    def __init__(self, in_channels: int = None, out_channels: int = None, 
                 kernel_size: Union[int, Tuple[int, ...]] = 3, 
                 scale_factor: Union[int, Tuple[int, ...]] = (2, 2, 2)):
        """Initialize the TransposeConvUpsampling.

        Parameters
        ----------
        in_channels : int, optional
            Number of input channels, by default None
        out_channels : int, optional
            Number of output channels, by default None
        kernel_size : Union[int, Tuple[int, ...]], optional
            Kernel size of the convolution, by default 3
        scale_factor : Union[int, Tuple[int, ...]], optional
            Stride of the convolution, by default (2, 2, 2)
        """
        upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, 
                                      stride=scale_factor, padding=1)

        super().__init__(upsample)


class NoUpsampling(AbstractUpsampling):
    def __init__(self):
        super().__init__(self._no_upsampling)

    @staticmethod
    def _no_upsampling(x, size):
        return x


class NoDownsampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x