import torch
import numpy as np
from typing import List, Union, Dict
from torch.nn import (
    Linear,
    Identity,
    BatchNorm1d,
    BatchNorm2d,
    F, 
    Conv2d
)
from nowproject.models import UNetModel, ConvNetModel
from nowproject.utils_models import check_pool_method
from nowproject.utils_models import check_skip_connection

##----------------------------------------------------------------------------.
class ConvBlock(torch.nn.Module):
    """Spherical graph convolution block.

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
        padding=0, 
        dilation=True,
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
                        
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            padding=padding,
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

        It expects a tensor with shape: (sample, x, y, time-feature).
        """
        # TODO adapt !!!! 
        x = self.conv(x)
        if self.norm and self.bn_before_act:
            # [batch, node, time-feature]
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.act:
            x = self.act_fun(x)
        if self.norm and not self.bn_before_act:
            # [batch, node, time-feature]
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
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
        ### Define list of convolutional layer
        # - Initialize list of convolutional layer within the ResBlock
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
        x_out += self.res_connection(x)
        return x_out


##----------------------------------------------------------------------------.
class UNet(UNetModel, torch.nn.Module):
    """UNet with residual layers.

    Parameters
    ----------
    tensor_info: dict
        Dictionary with all relevant shape, dimension and feature order informations
        regarding input and output tensors.
    kernel_size_conv : int
        Size ("width") of the square convolutional kernel.
        The number of pixels of the kernel is kernel_size_conv**2
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
    pool_method : str, optional
        Pooling method. Either 'max' or 'avg'
    kernel_size_pooling : int, optional
        The size of the window to max/avg over.
        kernel_size_pooling = 4 means halving the resolution when pooling.
        The default is 4.
    skip_connection : str, optional
        Possibilities: 'none','stack','sum','avg'
        The default is 'stack.
    increment_learning: bool, optional
        If increment_learning = True, the network is forced internally to learn the increment
        from the previous timestep.
        If increment_learning = False, the network learn the full state.
        The default is False.
    periodic_padding : bool, optional
        Matters only if sampling='equiangular' and conv_type='image'.
        whether to use periodic padding along the longitude dimension. The default is True.
    """

    def __init__(
        self,
        tensor_info: Dict,
        # Convolutions options
        kernel_size_conv: int = 3,  
        padding: str = "valid",
        # ConvBlock Options
        bias: bool = True,
        batch_norm: bool = True,
        batch_norm_before_activation: bool = True,
        activation: bool = True,
        activation_fun: str = "relu",
        # Pooling options
        pool_method: str = "max",
        kernel_size_pooling: int = 4,
        # Architecture options
        skip_connection: str = "stack",
        increment_learning: bool = False,
    ):
        ##--------------------------------------------------------------------.
        super().__init__()
        ##--------------------------------------------------------------------.
        # Retrieve tensor informations
        self.dim_names = tensor_info["dim_order"]["dynamic"]
        self.input_feature_dim = tensor_info["input_n_feature"]
        self.input_time_dim = tensor_info["input_n_time"]
        self.input_node_dim = tensor_info["input_shape_info"]["dynamic"]["node"]

        self.output_time_dim = tensor_info["output_n_time"]
        self.output_feature_dim = tensor_info["output_n_feature"]
        self.output_node_dim = tensor_info["output_shape_info"]["dynamic"]["node"]

        ##--------------------------------------------------------------------.
        # Define size of last dimension for ConvChen conv (merging time-feature dimension)
        # TODO: to redefine based if applying Conv2D or Conv3D 
        self.input_channels = self.input_feature_dim * self.input_time_dim
        self.output_channels = self.output_feature_dim * self.output_time_dim

        ##--------------------------------------------------------------------.
        # Decide whether to predict the difference from the previous timestep
        #  instead of the full state
        self.increment_learning = increment_learning

        ##--------------------------------------------------------------------.
        ### Check arguments
        pool_method = check_pool_method(pool_method)
        skip_connection = check_skip_connection(skip_connection)

        ##--------------------------------------------------------------------.
        ### Define ConvBlock options
        convblock_kwargs = {
            "kernel_size": kernel_size_conv,
            "bias": bias,
            "batch_norm": batch_norm,
            "batch_norm_before_activation": batch_norm_before_activation,
            "activation": activation,
            "activation_fun": activation_fun,
            "padding": "valid",  # TODO GENERALIZE
        }
        ##--------------------------------------------------------------------.            
        # - Define UNet levels
        UNet_depth = 3
       
        ##--------------------------------------------------------------------.
        ### Define Pooling - Unpooling layers
        # --> https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        # pool_method, kernel_size_pooling 
        # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1)
        # torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, dilation=1)
        
        ##--------------------------------------------------------------------.
        ### Define Encoding blocks
        # Encoding block 1
        self.conv1 = ResBlock(
            self.input_channels,
            (32 * 2, 64 * 2),
            convblock_kwargs=convblock_kwargs,
        )

        # Encoding block 2
        self.conv2 = ResBlock(
            64 * 2,
            (96 * 2, 128 * 2),
            convblock_kwargs=convblock_kwargs,
        )

        # Encoding block 3
        self.conv3 = ResBlock(
            128 * 2,
            (256 * 2, 128 * 2),
            convblock_kwargs=convblock_kwargs,
        )

        ##--------------------------------------------------------------------.
        ### Decoding blocks
        # Decoding block 2
        self.uconv21 = ConvBlock(
            256 * 2, 128 * 2, **convblock_kwargs
        )
        self.uconv22 = ConvBlock(
            128 * 2, 64 * 2, **convblock_kwargs
        )

        # Decoding block 1
        self.uconv11 = ConvBlock(
            128 * 2, 64 * 2, **convblock_kwargs
        )
        self.uconv12 = ConvBlock(
            64 * 2, 32 * 2,  **convblock_kwargs
        )
        
        # This is important for regression tasks 
        special_kwargs = convblock_kwargs.copy()
        special_kwargs["batch_norm"] = False
        special_kwargs["activation"] = False
        self.uconv13 = ConvBlock(
            32 * 2, self.output_channels, **special_kwargs
        )

    ##------------------------------------------------------------------------.
    def encode(self, x):
        """Define UNet encoder."""
        # TODO: Adapt to 2D spatial inputs 
        # --- TODO: Maybe create Conv2DBlock and Conv3DBlock
        
        # Current input shape: ['sample', 'time', 'node', 'feature']
        # Desired shape: ['sample', 'node', 'time-feature']

        ##--------------------------------------------------------------------.
        batch_size = x.shape[0]

        ##--------------------------------------------------------------------.
        # Reorder and reshape data
        x = (
            x.rename(*self.dim_names)
            .align_to("sample", "node", "time", "feature")
            .rename(None)
        )  # x.permute(0, 2, 1, 3)
        x = x.reshape(
            batch_size, self.input_node_dim, self.input_channels
        )  # reshape to ['sample', 'node', 'time-feature']
        ##--------------------------------------------------------------------.
        # Block 1
        x_enc1 = self.conv1(x)

        # x_enc11 = self.conv11(x)
        # x_enc1 = self.conv13(x_enc11)
        # x_enc1 += self.conv1_res(x)

        # Level 2
        x_enc2_ini, idx1 = self.pool1(x_enc1)
        x_enc2 = self.conv2(x_enc2_ini)

        # Level 3
        x_enc3_ini, idx2 = self.pool2(x_enc2)
        x_enc3 = self.conv3(x_enc3_ini)

        return x_enc3, x_enc2, x_enc1, idx2, idx1  #  x_enc11

    ##------------------------------------------------------------------------.
    def decode(self, x_enc3, x_enc2, x_enc1, idx2, idx1):  # x_enc11):
        """Define UNet decoder."""
        # Block 2
        x = self.unpool2(x_enc3, idx2)
        x_cat = torch.cat((x, x_enc2), dim=2)
        x = self.uconv21(x_cat)
        x = self.uconv22(x)

        # Block 1
        x = self.unpool1(x, idx1)
        x_cat = torch.cat((x, x_enc1), dim=2)
        x = self.uconv11(x_cat)
        x = self.uconv12(x)

        # Apply conv without batch norm and act fun
        x = self.uconv13(x)

        # x_cat = torch.cat((x, x_enc11), dim=2)
        # Apply conv without batch norm and act fun
        # x = self.uconv13(x_cat)

        ##--------------------------------------------------------------------.
        # Reshape data to ['sample', 'time', 'node', 'feature']
        batch_size = x.shape[0]  # ['sample', 'node', 'time-feature']
        x = x.reshape(
            batch_size,
            self.output_node_dim,
            self.output_time_dim,
            self.output_feature_dim,
        )  # ==> ['sample', 'node', 'time', 'feature']
        x = (
            x.rename(*["sample", "node", "time", "feature"])
            .align_to(*self.dim_names)
            .rename(None)
        )  # x.permute(0, 2, 1, 3)
        return x

####--------------------------------------------------------------------------.
# HERE AFTER TO ADAPT ONCE PREVIOUS WORKS 

class ResNet(ConvNetModel, torch.nn.Module):
    """ResNet.

    Parameters
    ----------
    tensor_info: dict
        Dictionary with all relevant shape, dimension and feature order informations
        regarding input and output tensors.
    kernel_size_conv : int
        Size ("width") of the square convolutional kernel.
        - The number of pixels of the kernel is kernel_size_conv**2
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
    pool_method : str, optional
        Not used
    kernel_size_pooling : int, optional
        Not used
    skip_connection : str, optional
        Possibilities: 'none','stack','sum','avg'
        The default is 'stack.
    increment_learning: bool, optional
        If increment_learning = True, the network is forced internally to learn the increment
        from the previous timestep.
        If increment_learning = False, the network learn the full state.
        The default is False.
    periodic_padding : bool, optional
        Matters only if sampling='equiangular' and conv_type='image'.
        whether to use periodic padding along the longitude dimension. The default is True.
    """

    def __init__(
        self,
        tensor_info: Dict,
        sampling: str,
        sampling_kwargs: Dict,
        # Convolutions options
        kernel_size_conv: int = 3,
        conv_type: str = "graph",
        graph_type: str = "knn",
        knn: int = 20,
        # Options for classical image convolution on equiangular sampling
        periodic_padding: bool = True,
        # ConvBlock Options
        bias: bool = True,
        batch_norm: bool = True,
        batch_norm_before_activation: bool = True,
        activation: bool = True,
        activation_fun: str = "relu",
        # Pooling options
        pool_method: str = "max",
        kernel_size_pooling: int = 4,
        # Architecture options
        skip_connection: str = "stack",
        increment_learning: bool = False,
    ):
        ##--------------------------------------------------------------------.
        super().__init__()
        ##--------------------------------------------------------------------.
        # Retrieve tensor informations
        self.dim_names = tensor_info["dim_order"]["dynamic"]
        self.input_feature_dim = tensor_info["input_n_feature"]
        self.input_time_dim = tensor_info["input_n_time"]
        self.input_node_dim = tensor_info["input_shape_info"]["dynamic"]["node"]

        self.output_time_dim = tensor_info["output_n_time"]
        self.output_feature_dim = tensor_info["output_n_feature"]
        self.output_node_dim = tensor_info["output_shape_info"]["dynamic"]["node"]

        ##--------------------------------------------------------------------.
        # Define size of last dimension for ConvChen conv (merging time-feature dimension)
        self.input_channels = self.input_feature_dim * self.input_time_dim
        self.output_channels = self.output_feature_dim * self.output_time_dim

        ##--------------------------------------------------------------------.
        # Decide whether to predict the difference from the previous timestep
        #  instead of the full state
        self.increment_learning = increment_learning

        ##--------------------------------------------------------------------.
        ### Check arguments
        sampling = check_sampling(sampling)
        conv_type = check_conv_type(conv_type, sampling)
        skip_connection = check_skip_connection(skip_connection)

        ##--------------------------------------------------------------------.
        # Derive lonlat ratio from sampling_kwargs if equiangular
        if sampling == "equiangular":
            lonlat_ratio = sampling_kwargs["nlon"] / sampling_kwargs["nlat"]
        else:
            lonlat_ratio = None
        ##--------------------------------------------------------------------.
        ### Define ConvBlock options
        convblock_kwargs = {
            "kernel_size": kernel_size_conv,
            "conv_type": conv_type,
            "bias": bias,
            "batch_norm": batch_norm,
            "batch_norm_before_activation": batch_norm_before_activation,
            "activation": activation,
            "activation_fun": activation_fun,
            # Options for conv_type = "image", sampling='equiangular'
            "periodic_padding": periodic_padding,
            "lonlat_ratio": lonlat_ratio,
        }
        ##--------------------------------------------------------------------.
        ### Define graph and laplacian
        # - Update knn based on model settings
        if sampling != "equiangular":
            # (pygsp.graphs.SphereEquiangular do not accept k)
            sampling_kwargs["k"] = knn

        sampling_list = [sampling]
        sampling_kwargs_list = [sampling_kwargs]

        ##--------------------------------------------------------------------.
        ### Initialize graphs and laplacians
        # - If conv_type == 'image', self.laplacians = [None] * UNet_depth
        # - self.init_graph_and_laplacians() defines self.graphs and self.laplacians
        self.init_graph_and_laplacians(
            sampling_list=sampling_list,
            sampling_kwargs_list=sampling_kwargs_list,
            graph_type=graph_type,
            conv_type=conv_type,
        )

        ##--------------------------------------------------------------------.
        ### Define ResNet Convolutional Layers
        n_conv_layers = 4
        n_conv_features = 128  # self.input_channels*16
        resblock_shapes = [n_conv_features for i in range(n_conv_layers)] + [
            self.input_channels
        ]
        # Define ResBlocks
        self.resblock1 = ResBlock(
            self.input_channels,
            resblock_shapes,
            laplacian=self.laplacians[0],
            convblock_kwargs=convblock_kwargs,
        )
        self.resblock2 = ResBlock(
            self.input_channels,
            resblock_shapes,
            laplacian=self.laplacians[0],
            convblock_kwargs=convblock_kwargs,
        )

        self.resblock3 = ResBlock(
            self.input_channels,
            resblock_shapes,
            laplacian=self.laplacians[0],
            convblock_kwargs=convblock_kwargs,
        )
        self.resblock4 = ResBlock(
            self.input_channels,
            resblock_shapes,
            laplacian=self.laplacians[0],
            convblock_kwargs=convblock_kwargs,
        )
        # Define Intermediate ConvBlocks
        self.conv1 = ConvBlock(
            self.input_channels,
            n_conv_features,
            laplacian=self.laplacians[0],
            **convblock_kwargs
        )
        self.conv2 = ConvBlock(
            n_conv_features,
            n_conv_features,
            laplacian=self.laplacians[0],
            **convblock_kwargs
        )
        self.conv3 = ConvBlock(
            n_conv_features,
            n_conv_features,
            laplacian=self.laplacians[0],
            **convblock_kwargs
        )
        self.conv4 = ConvBlock(
            n_conv_features,
            n_conv_features,
            laplacian=self.laplacians[0],
            **convblock_kwargs
        )

        # Define Final ConvBlock
        special_kwargs = convblock_kwargs.copy()
        special_kwargs["batch_norm"] = False
        special_kwargs["activation"] = False
        self.conv_final = ConvBlock(
            n_conv_features,
            self.output_channels,
            laplacian=self.laplacians[0],
            **special_kwargs
        )
        ##--------------------------------------------------------------------.

    ##------------------------------------------------------------------------.
    def forward(self, x):
        """Define ResNet forward pass."""
        # Current input shape: ['sample', 'time', 'node', 'feature']
        # Desired shape: ['sample', 'node', 'time-feature']
        ##--------------------------------------------------------------------.
        batch_size = x.shape[0]

        ##--------------------------------------------------------------------.
        # Reorder and reshape data
        x = (
            x.rename(*self.dim_names)
            .align_to("sample", "node", "time", "feature")
            .rename(None)
        )  # x.permute(0, 2, 1, 3)
        x = x.reshape(
            batch_size, self.input_node_dim, self.input_channels
        )  # reshape to ['sample', 'node', 'time-feature']
        ##--------------------------------------------------------------------.
        # Define forward pass
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv_final(x)

        ##--------------------------------------------------------------------.
        # Reshape data to ['sample', 'time', 'node', 'feature']
        batch_size = x.shape[0]  # ['sample', 'node', 'time-feature']
        x = x.reshape(
            batch_size,
            self.output_node_dim,
            self.output_time_dim,
            self.output_feature_dim,
        )  # ==> ['sample', 'node', 'time', 'feature']
        x = (
            x.rename(*["sample", "node", "time", "feature"])
            .align_to(*self.dim_names)
            .rename(None)
        )  # x.permute(0, 2, 1, 3)
        return x


class EPDNet(ConvNetModel, torch.nn.Module):
    """Encoder-Process-Decoder Net.

    This architecture is inspired from 'Kochov et al., 2021. ML accelerated computational fluid dynamics.'

    Parameters
    ----------
    tensor_info: dict
        Dictionary with all relevant shape, dimension and feature order informations
        regarding input and output tensors.
    sampling : str
        Name of the spherical sampling.
    sampling_kwargs : int
        Arguments to define the spherical pygsp graph.
    conv_type : str, optional
        Convolution type. Either 'graph' or 'image'.
        The default is 'graph'.
        conv_type='image' can be used only when sampling='equiangular'.
    knn : int
        DESCRIPTION
    graph_type : str , optional
        DESCRIPTION
        'voronoi' or 'knn'.
        'knn' build a knn graph.
        'voronoi' build a voronoi mesh graph and require the igl package
        The default is 'knn'.
    kernel_size_conv : int
        Size ("width") of the convolutional kernel.
        If conv_type='graph':
        - A kernel_size of 1 won't take the neighborhood into account.
        - A kernel_size of 2 will look up to the 1-neighborhood (1 hop away).
        - A kernel_size of 3 will look up to the 2-neighborhood (2 hops away).
        --> The order of the Chebyshev polynomials is kernel_size_conv - 1.
        If conv_type='image':
        - Width of the square convolutional kernel.
        - The number of pixels of the kernel is kernel_size_conv**2
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
    pool_method : str, optional
        Not used
    kernel_size_pooling : int, optional
        Not used
    skip_connection : str, optional
        Possibilities: 'none','stack','sum','avg'
        The default is 'stack.
    increment_learning: bool, optional
        If increment_learning = True, the network is forced internally to learn the increment
        from the previous timestep.
        If increment_learning = False, the network learn the full state.
        The default is False.
    periodic_padding : bool, optional
        Matters only if sampling='equiangular' and conv_type='image'.
        whether to use periodic padding along the longitude dimension. The default is True.
    """

    def __init__(
        self,
        tensor_info: Dict,
        sampling: str,
        sampling_kwargs: Dict,
        # Convolutions options
        kernel_size_conv: int = 3,
        conv_type: str = "graph",
        graph_type: str = "knn",
        knn: int = 20,
        # Options for classical image convolution on equiangular sampling
        periodic_padding: bool = True,
        # ConvBlock Options
        bias: bool = True,
        batch_norm: bool = True,
        batch_norm_before_activation: bool = True,
        activation: bool = True,
        activation_fun: str = "relu",
        # Pooling options
        pool_method: str = "max",
        kernel_size_pooling: int = 4,
        # Architecture options
        skip_connection: str = "stack",
        increment_learning: bool = False,
    ):
        ##--------------------------------------------------------------------.
        super().__init__()
        ##--------------------------------------------------------------------.
        # Retrieve tensor informations
        self.dim_names = tensor_info["dim_order"]["dynamic"]
        self.input_feature_dim = tensor_info["input_n_feature"]
        self.input_time_dim = tensor_info["input_n_time"]
        self.input_node_dim = tensor_info["input_shape_info"]["dynamic"]["node"]

        self.output_time_dim = tensor_info["output_n_time"]
        self.output_feature_dim = tensor_info["output_n_feature"]
        self.output_node_dim = tensor_info["output_shape_info"]["dynamic"]["node"]

        ##--------------------------------------------------------------------.
        # Define size of last dimension for ConvChen conv (merging time-feature dimension)
        self.input_channels = self.input_feature_dim * self.input_time_dim
        self.output_channels = self.output_feature_dim * self.output_time_dim

        ##--------------------------------------------------------------------.
        # Decide whether to predict the difference from the previous timestep
        #  instead of the full state
        self.increment_learning = increment_learning

        ##--------------------------------------------------------------------.
        ### Check arguments
        sampling = check_sampling(sampling)
        conv_type = check_conv_type(conv_type, sampling)
        skip_connection = check_skip_connection(skip_connection)

        ##--------------------------------------------------------------------.
        # Derive lonlat ratio from sampling_kwargs if equiangular
        if sampling == "equiangular":
            lonlat_ratio = sampling_kwargs["nlon"] / sampling_kwargs["nlat"]
        else:
            lonlat_ratio = None
        ##--------------------------------------------------------------------.
        ### Define ConvBlock options
        convblock_kwargs = {
            "kernel_size": kernel_size_conv,
            "conv_type": conv_type,
            "bias": bias,
            "batch_norm": batch_norm,
            "batch_norm_before_activation": batch_norm_before_activation,
            "activation": activation,
            "activation_fun": activation_fun,
            # Options for conv_type = "image", sampling='equiangular'
            "periodic_padding": periodic_padding,
            "lonlat_ratio": lonlat_ratio,
        }
        ##--------------------------------------------------------------------.
        ### Define graph and laplacian
        # - Update knn based on model settings
        if sampling != "equiangular":
            # (pygsp.graphs.SphereEquiangular do not accept k)
            sampling_kwargs["k"] = knn

        sampling_list = [sampling]
        sampling_kwargs_list = [sampling_kwargs]

        ##--------------------------------------------------------------------.
        ### Initialize graphs and laplacians
        # - If conv_type == 'image', self.laplacians = [None] * UNet_depth
        # - self.init_graph_and_laplacians() defines self.graphs and self.laplacians
        self.init_graph_and_laplacians(
            sampling_list=sampling_list,
            sampling_kwargs_list=sampling_kwargs_list,
            graph_type=graph_type,
            conv_type=conv_type,
        )

        ##--------------------------------------------------------------------.
        ### Define EPDnet Convolutional Layers
        n_conv_layers = 3
        n_conv_features = 128  # self.input_channels*16
        resblock_shapes = [n_conv_features for i in range(n_conv_layers)]
        # Define Encoder ConvBlocks
        self.enc_conv1 = ConvBlock(
            self.input_channels,
            n_conv_features,
            laplacian=self.laplacians[0],
            **convblock_kwargs
        )
        self.enc_conv2 = ConvBlock(
            n_conv_features,
            n_conv_features,
            laplacian=self.laplacians[0],
            **convblock_kwargs
        )
        # Define "Process" ResBlocks
        self.resblock1 = ResBlock(
            n_conv_features,
            resblock_shapes,
            laplacian=self.laplacians[0],
            convblock_kwargs=convblock_kwargs,
        )
        self.resblock2 = ResBlock(
            n_conv_features,
            resblock_shapes,
            laplacian=self.laplacians[0],
            convblock_kwargs=convblock_kwargs,
        )

        self.resblock3 = ResBlock(
            n_conv_features,
            resblock_shapes,
            laplacian=self.laplacians[0],
            convblock_kwargs=convblock_kwargs,
        )
        self.resblock4 = ResBlock(
            n_conv_features,
            resblock_shapes,
            laplacian=self.laplacians[0],
            convblock_kwargs=convblock_kwargs,
        )

        # Define Decoder ConvBlocks
        self.dec_conv1 = ConvBlock(
            n_conv_features,
            n_conv_features,
            laplacian=self.laplacians[0],
            **convblock_kwargs
        )
        # self.dec_conv2 = ConvBlock(n_conv_features, n_conv_features,
        #                            laplacian = self.laplacians[0],
        #                            **convblock_kwargs)

        # Define Final ConvBlock
        special_kwargs = convblock_kwargs.copy()
        special_kwargs["batch_norm"] = False
        special_kwargs["activation"] = False
        self.conv_final = ConvBlock(
            n_conv_features,
            self.output_channels,
            laplacian=self.laplacians[0],
            **special_kwargs
        )
        ##--------------------------------------------------------------------.

    ##------------------------------------------------------------------------.
    def forward(self, x):
        """Define EPDNet forward pass."""
        # Current input shape: ['sample', 'time', 'node', 'feature']
        # Desired shape: ['sample', 'node', 'time-feature']
        ##--------------------------------------------------------------------.
        batch_size = x.shape[0]

        ##--------------------------------------------------------------------.
        # Reorder and reshape data
        x = (
            x.rename(*self.dim_names)
            .align_to("sample", "node", "time", "feature")
            .rename(None)
        )  # x.permute(0, 2, 1, 3)
        x = x.reshape(
            batch_size, self.input_node_dim, self.input_channels
        )  # reshape to ['sample', 'node', 'time-feature']
        ##--------------------------------------------------------------------.
        # Define forward pass
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.dec_conv1(x)
        # x = self.dec_conv2(x)
        x = self.conv_final(x)

        ##--------------------------------------------------------------------.
        # Reshape data to ['sample', 'time', 'node', 'feature']
        batch_size = x.shape[0]  # ['sample', 'node', 'time-feature']
        x = x.reshape(
            batch_size,
            self.output_node_dim,
            self.output_time_dim,
            self.output_feature_dim,
        )  # ==> ['sample', 'node', 'time', 'feature']
        x = (
            x.rename(*["sample", "node", "time", "feature"])
            .align_to(*self.dim_names)
            .rename(None)
        )  # x.permute(0, 2, 1, 3)
        return x


class ConvNet(ConvNetModel, torch.nn.Module):
    """Convolutional Net.

    Parameters
    ----------
    tensor_info: dict
        Dictionary with all relevant shape, dimension and feature order informations
        regarding input and output tensors.
    sampling : str
        Name of the spherical sampling.
    sampling_kwargs : int
        Arguments to define the spherical pygsp graph.
    conv_type : str, optional
        Convolution type. Either 'graph' or 'image'.
        The default is 'graph'.
        conv_type='image' can be used only when sampling='equiangular'.
    knn : int
        DESCRIPTION
    graph_type : str , optional
        DESCRIPTION
        'voronoi' or 'knn'.
        'knn' build a knn graph.
        'voronoi' build a voronoi mesh graph and require the igl package
        The default is 'knn'.
    kernel_size_conv : int
        Size ("width") of the convolutional kernel.
        If conv_type='graph':
        - A kernel_size of 1 won't take the neighborhood into account.
        - A kernel_size of 2 will look up to the 1-neighborhood (1 hop away).
        - A kernel_size of 3 will look up to the 2-neighborhood (2 hops away).
        --> The order of the Chebyshev polynomials is kernel_size_conv - 1.
        If conv_type='image':
        - Width of the square convolutional kernel.
        - The number of pixels of the kernel is kernel_size_conv**2
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
    pool_method : str, optional
        Not used
    kernel_size_pooling : int, optional
        Not used
    skip_connection : str, optional
        Possibilities: 'none','stack','sum','avg'
        The default is 'stack.
    increment_learning: bool, optional
        If increment_learning = True, the network is forced internally to learn the increment
        from the previous timestep.
        If increment_learning = False, the network learn the full state.
        The default is False.
    periodic_padding : bool, optional
        Matters only if sampling='equiangular' and conv_type='image'.
        whether to use periodic padding along the longitude dimension. The default is True.
    """

    def __init__(
        self,
        tensor_info: Dict,
        sampling: str,
        sampling_kwargs: Dict,
        # Convolutions options
        kernel_size_conv: int = 3,
        conv_type: str = "graph",
        graph_type: str = "knn",
        knn: int = 20,
        # Options for classical image convolution on equiangular sampling
        periodic_padding: bool = True,
        # ConvBlock Options
        bias: bool = True,
        batch_norm: bool = True,
        batch_norm_before_activation: bool = True,
        activation: bool = True,
        activation_fun: str = "relu",
        # Pooling options
        pool_method: str = "max",
        kernel_size_pooling: int = 4,
        # Architecture options
        skip_connection: str = "stack",
        increment_learning: bool = False,
    ):
        ##--------------------------------------------------------------------.
        super().__init__()
        ##--------------------------------------------------------------------.
        # Retrieve tensor informations
        self.dim_names = tensor_info["dim_order"]["dynamic"]
        self.input_feature_dim = tensor_info["input_n_feature"]
        self.input_time_dim = tensor_info["input_n_time"]
        self.input_node_dim = tensor_info["input_shape_info"]["dynamic"]["node"]

        self.output_time_dim = tensor_info["output_n_time"]
        self.output_feature_dim = tensor_info["output_n_feature"]
        self.output_node_dim = tensor_info["output_shape_info"]["dynamic"]["node"]

        ##--------------------------------------------------------------------.
        # Define size of last dimension for ConvChen conv (merging time-feature dimension)
        self.input_channels = self.input_feature_dim * self.input_time_dim
        self.output_channels = self.output_feature_dim * self.output_time_dim

        ##--------------------------------------------------------------------.
        # Decide whether to predict the difference from the previous timestep
        #  instead of the full state
        self.increment_learning = increment_learning

        ##--------------------------------------------------------------------.
        ### Check arguments
        sampling = check_sampling(sampling)
        conv_type = check_conv_type(conv_type, sampling)
        skip_connection = check_skip_connection(skip_connection)

        ##--------------------------------------------------------------------.
        # Derive lonlat ratio from sampling_kwargs if equiangular
        if sampling == "equiangular":
            lonlat_ratio = sampling_kwargs["nlon"] / sampling_kwargs["nlat"]
        else:
            lonlat_ratio = None
        ##--------------------------------------------------------------------.
        ### Define ConvBlock options
        convblock_kwargs = {
            "kernel_size": kernel_size_conv,
            "conv_type": conv_type,
            "bias": bias,
            "batch_norm": batch_norm,
            "batch_norm_before_activation": batch_norm_before_activation,
            "activation": activation,
            "activation_fun": activation_fun,
            # Options for conv_type = "image", sampling='equiangular'
            "periodic_padding": periodic_padding,
            "lonlat_ratio": lonlat_ratio,
        }
        ##--------------------------------------------------------------------.
        ### Define graph and laplacian
        # - Update knn based on model settings
        if sampling != "equiangular":
            # (pygsp.graphs.SphereEquiangular do not accept k)
            sampling_kwargs["k"] = knn

        sampling_list = [sampling]
        sampling_kwargs_list = [sampling_kwargs]

        ##--------------------------------------------------------------------.
        ### Initialize graphs and laplacians
        # - If conv_type == 'image', self.laplacians = [None] * UNet_depth
        # - self.init_graph_and_laplacians() defines self.graphs and self.laplacians
        self.init_graph_and_laplacians(
            sampling_list=sampling_list,
            sampling_kwargs_list=sampling_kwargs_list,
            graph_type=graph_type,
            conv_type=conv_type,
        )

        ##--------------------------------------------------------------------.
        ### Define Convolutional Layers
        n_conv_layers = 6
        n_conv_features = 128  # self.input_channels*16
        # Initialize
        conv_names_list = []
        tmp_in = self.input_channels
        tmp_out = n_conv_features
        for i in range(1, n_conv_layers + 1):
            # --------------------------------------------------------.
            # - Create a copy of convblock args
            tmp_convblock_kwargs = convblock_kwargs.copy()
            # --------------------------------------------------------.
            # - Define conv layer name
            tmp_conv_name = "conv" + str(i + 1)
            # --------------------------------------------------------.
            # - Create the conv layer
            tmp_conv = ConvBlock(
                tmp_in, tmp_out, laplacian=self.laplacians[0], **tmp_convblock_kwargs
            )
            setattr(self, tmp_conv_name, tmp_conv)
            conv_names_list.append(tmp_conv_name)
            tmp_in = tmp_out

        # Attach a list with the name of all ConvBlocks
        self.conv_names_list = conv_names_list

        ##--------------------------------------------------------
        # Define Final ConvBlock
        special_kwargs = convblock_kwargs.copy()
        special_kwargs["batch_norm"] = False
        special_kwargs["activation"] = False
        self.conv_final = ConvBlock(
            n_conv_features,
            self.output_channels,
            laplacian=self.laplacians[0],
            **special_kwargs
        )
        ##--------------------------------------------------------

    ##------------------------------------------------------------------------.
    def forward(self, x):
        """Define ConvNet forward pass."""
        # Current input shape: ['sample', 'time', 'node', 'feature']
        # Desired shape: ['sample', 'node', 'time-feature']
        ##--------------------------------------------------------------------.
        batch_size = x.shape[0]

        ##--------------------------------------------------------------------.
        # Reorder and reshape data
        x = (
            x.rename(*self.dim_names)
            .align_to("sample", "node", "time", "feature")
            .rename(None)
        )  # x.permute(0, 2, 1, 3)
        x = x.reshape(
            batch_size, self.input_node_dim, self.input_channels
        )  # reshape to ['sample', 'node', 'time-feature']
        ##--------------------------------------------------------------------.
        # Define forward pass
        for conv_name in self.conv_names_list:
            x = getattr(self, conv_name)(x)
        x = self.conv_final(x)

        ##--------------------------------------------------------------------.
        # Reshape data to ['sample', 'time', 'node', 'feature']
        batch_size = x.shape[0]  # ['sample', 'node', 'time-feature']
        x = x.reshape(
            batch_size,
            self.output_node_dim,
            self.output_time_dim,
            self.output_feature_dim,
        )  # ==> ['sample', 'node', 'time', 'feature']
        x = (
            x.rename(*["sample", "node", "time", "feature"])
            .align_to(*self.dim_names)
            .rename(None)
        )  # x.permute(0, 2, 1, 3)
        return x
