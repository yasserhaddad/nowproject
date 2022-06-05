import torch
import numpy as np
from typing import List, Union, Dict
from torch.nn import (
    Conv2d,
    ReLU,
    MaxPool2d,
    AvgPool2d,
    Upsample,
    Sequential
)
import torch.nn.functional as F

from nowproject.layers import ConvBlock, ResBlock, Upsampling
from nowproject.models import UNetModel, ConvNetModel
from nowproject.utils.utils_models import check_skip_connection


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
        last_layer_activation: bool = False,
        conv_type: str = "regular",
        # Pooling options
        pool_method: str = "max",
        kernel_size_pooling: int = 2,
        # Architecture options
        skip_connection: str = "stack",
        increment_learning: bool = False,
        # Output options
        # categorical: bool = False
    ):
        ##--------------------------------------------------------------------.
        super().__init__()
        # self.categorical = categorical
        ##--------------------------------------------------------------------.
        # Retrieve tensor informations
        self.dim_names = tensor_info["dim_order"]["dynamic"]
        self.input_feature_dim = tensor_info["input_n_feature"]
        self.input_time_dim = tensor_info["input_n_time"]
        self.input_train_y_dim = tensor_info["input_shape_info"]["train"]["dynamic"]["y"]
        self.input_train_x_dim = tensor_info["input_shape_info"]["train"]["dynamic"]["x"]
        self.input_test_y_dim = tensor_info["input_shape_info"]["test"]["dynamic"]["y"]
        self.input_test_x_dim = tensor_info["input_shape_info"]["test"]["dynamic"]["x"]

        self.output_time_dim = tensor_info["output_n_time"]
        self.output_feature_dim = tensor_info["output_n_feature"]
        self.output_train_y_dim = tensor_info["output_shape_info"]["train"]["dynamic"]["y"]
        self.output_train_x_dim = tensor_info["output_shape_info"]["train"]["dynamic"]["x"]
        self.output_test_y_dim = tensor_info["output_shape_info"]["test"]["dynamic"]["y"]
        self.output_test_x_dim = tensor_info["output_shape_info"]["test"]["dynamic"]["x"]

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
        pool_method = pool_method.lower()
        skip_connection = check_skip_connection(skip_connection)

        ##--------------------------------------------------------------------.
        ### Define ConvBlock options
        convblock_kwargs = {
            "conv_type": conv_type,
            "kernel_size": kernel_size_conv,
            "bias": bias,
            "batch_norm": batch_norm,
            "batch_norm_before_activation": batch_norm_before_activation,
            "activation": activation,
            "activation_fun": activation_fun,
            "padding": 1,
            "padding_mode": "reflect",
            "modes1": self.input_train_y_dim//2 + 1,
            "modes2": self.input_train_x_dim //2 + 1
        }
       
        ##--------------------------------------------------------------------.
        ### Define Pooling - Unpooling layers
        if pool_method == "avg":
            self.pool1 = AvgPool2d(kernel_size_pooling)
            self.pool2 = AvgPool2d(kernel_size_pooling)
        elif pool_method == "max":
            self.pool1 = MaxPool2d(kernel_size_pooling)
            self.pool2 = MaxPool2d(kernel_size_pooling)
        
        self.unpool1 = Upsample(scale_factor=kernel_size_pooling, mode='bilinear', align_corners=True)
        self.unpool2 = Upsample(scale_factor=kernel_size_pooling, mode='bilinear', align_corners=True)
        
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
        self.up1 = Upsampling(256 * 2, 128 * 2, 64 * 2, convblock_kwargs, kernel_size_pooling)
        # Decoding block 1
        self.up2 = Upsampling(128 * 2, 64 * 2, 32 * 2, convblock_kwargs, kernel_size_pooling)
        
        # This is important for regression tasks 
        special_kwargs = convblock_kwargs.copy()
        special_kwargs["conv_type"] = "regular"
        special_kwargs["batch_norm"] = False
        special_kwargs["activation"] = last_layer_activation
        self.uconv13 = ConvBlock(
            32 * 2, self.output_channels, **special_kwargs
        )

    ##------------------------------------------------------------------------.
    def encode(self, x, training=True):
        """Define UNet encoder."""
        # TODO: Adapt to 2D spatial inputs 
        # --- TODO: Maybe create Conv2DBlock and Conv3DBlock
        
        # Current input shape: ['sample', 'time', 'node', 'feature']
        # Desired shape: ['sample', 'node', 'time-feature']

        ##--------------------------------------------------------------------.
        batch_size = x.shape[0]
        input_y_dim = self.input_train_y_dim if training else self.input_test_y_dim
        input_x_dim = self.input_train_x_dim if training else self.input_test_x_dim
        ##--------------------------------------------------------------------.
        # Reorder and reshape data
        x = (
            x.rename(*self.dim_names)
            .align_to("sample", "time", "y", "x", "feature")
            .rename(None)
        )  # x.permute(0, 2, 1, 3)
        x = x.reshape(
            batch_size, self.input_channels, input_y_dim, input_x_dim
        )  # reshape to ['sample', 'channels', 'y', 'x']
        ##--------------------------------------------------------------------.
        # Level 1
        x_enc1 = self.conv1(x)

        # Level 2
        x_enc2_ini = self.pool1(x_enc1)
        x_enc2 = self.conv2(x_enc2_ini)

        # Level 3
        x_enc3_ini = self.pool2(x_enc2)
        x_enc3 = self.conv3(x_enc3_ini)

        return x_enc3, x_enc2, x_enc1

    ##------------------------------------------------------------------------.
    def decode(self, x_enc3, x_enc2, x_enc1, training=True):  # x_enc11):
        """Define UNet decoder."""
        # Block 2
        x = self.up1(x_enc3, x_enc2)

        # Block 1
        x = self.up2(x, x_enc1)

        # Apply conv without batch norm and act fun
        x = self.uconv13(x)

        ##--------------------------------------------------------------------.
        # Reshape data to ['sample', 'time', 'node', 'feature']
        batch_size = x.shape[0]  # ['sample', 'node', 'time-feature']
        output_y_dim = self.output_train_y_dim if training else self.output_test_y_dim
        output_x_dim = self.output_train_x_dim if training else self.output_test_x_dim
        
        x = x.reshape(
            batch_size,
            self.output_time_dim,
            output_y_dim,
            output_x_dim,
            self.output_feature_dim,
        )  # ==> ['sample', 'node', 'time', 'feature']
        x = (
            x.rename(*["sample", "time", "y", "x", "feature"])
            .align_to(*self.dim_names)
            .rename(None)
        )  # x.permute(0, 2, 1, 3)
        # if self.categorical:
        #     x = torch.round(x)
        return x


class EPDNet(ConvNetModel, torch.nn.Module):
    """Encoder-Process-Decoder Net.

    This architecture is inspired from 'Kochov et al., 2021. ML accelerated computational fluid dynamics.'

    Parameters
    ----------
    tensor_info: dict
        Dictionary with all relevant shape, dimension and feature order informations
        regarding input and output tensors.
    conv_type : str, optional
        Convolution type. Either 'graph' or 'image'.
        The default is 'graph'.
        conv_type='image' can be used only when sampling='equiangular'.
    kernel_size_conv : int
        Size ("width") of the convolutional kernel.
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
    """

    def __init__(
        self,
        tensor_info: Dict,
        # Convolutions options
        kernel_size_conv: int = 3,
        # ConvBlock Options
        bias: bool = True,
        batch_norm: bool = True,
        batch_norm_before_activation: bool = True,
        activation: bool = True,
        activation_fun: str = "relu",
        last_layer_activation: bool = True,
        conv_type: str = "regular",
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
        self.input_train_y_dim = tensor_info["input_shape_info"]["train"]["dynamic"]["y"]
        self.input_train_x_dim = tensor_info["input_shape_info"]["train"]["dynamic"]["x"]
        self.input_test_y_dim = tensor_info["input_shape_info"]["test"]["dynamic"]["y"]
        self.input_test_x_dim = tensor_info["input_shape_info"]["test"]["dynamic"]["x"]

        self.output_time_dim = tensor_info["output_n_time"]
        self.output_feature_dim = tensor_info["output_n_feature"]
        self.output_train_y_dim = tensor_info["output_shape_info"]["train"]["dynamic"]["y"]
        self.output_train_x_dim = tensor_info["output_shape_info"]["train"]["dynamic"]["x"]
        self.output_test_y_dim = tensor_info["output_shape_info"]["test"]["dynamic"]["y"]
        self.output_test_x_dim = tensor_info["output_shape_info"]["test"]["dynamic"]["x"]

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
        skip_connection = check_skip_connection(skip_connection)
        ##--------------------------------------------------------------------.
        ### Define ConvBlock options
        convblock_kwargs = {
            "conv_type": conv_type,
            "kernel_size": kernel_size_conv,
            "bias": bias,
            "batch_norm": batch_norm,
            "batch_norm_before_activation": batch_norm_before_activation,
            "activation": activation,
            "activation_fun": activation_fun,
            "padding": 1,  # TODO GENERALIZE
            "padding_mode": "replicate",
            "modes1": self.input_train_y_dim//2 + 1,
            "modes2": self.input_train_x_dim //2 + 1
        }

        ##--------------------------------------------------------------------.
        ### Define EPDnet Convolutional Layers
        n_conv_layers = 3
        n_conv_features = 128  # self.input_channels*16
        resblock_shapes = [n_conv_features for i in range(n_conv_layers)]
        # Define Encoder ConvBlocks
        self.enc_conv1 = ConvBlock(
            self.input_channels,
            n_conv_features,
            **convblock_kwargs
        )
        self.enc_conv2 = ConvBlock(
            n_conv_features,
            n_conv_features,
            **convblock_kwargs
        )
        # Define "Process" ResBlocks
        convblock_kwargs = {
            "conv_type": "regular",
            "kernel_size": kernel_size_conv,
            "bias": bias,
            "batch_norm": batch_norm,
            "batch_norm_before_activation": batch_norm_before_activation,
            "activation": activation,
            "activation_fun": activation_fun,
            "padding": 1,  # TODO GENERALIZE
            "padding_mode": "replicate",
            "modes1": self.input_train_y_dim//2 + 1,
            "modes2": self.input_train_x_dim //2 + 1
        }

        self.resblock1 = ResBlock(
            n_conv_features,
            resblock_shapes,
            convblock_kwargs=convblock_kwargs,
        )
        self.resblock2 = ResBlock(
            n_conv_features,
            resblock_shapes,
            convblock_kwargs=convblock_kwargs,
        )

        self.resblock3 = ResBlock(
            n_conv_features,
            resblock_shapes,
            convblock_kwargs=convblock_kwargs,
        )
        self.resblock4 = ResBlock(
            n_conv_features,
            resblock_shapes,
            convblock_kwargs=convblock_kwargs,
        )

        # Define Decoder ConvBlocks
        self.dec_conv1 = ConvBlock(
            n_conv_features,
            n_conv_features,
            **convblock_kwargs
        )
        # self.dec_conv2 = ConvBlock(n_conv_features, n_conv_features,
        #                            laplacian = self.laplacians[0],
        #                            **convblock_kwargs)

        # Define Final ConvBlock
        special_kwargs = convblock_kwargs.copy()
        special_kwargs["conv_type"] = "regular"
        special_kwargs["batch_norm"] = False
        special_kwargs["activation"] = last_layer_activation
        self.conv_final = ConvBlock(
            n_conv_features,
            self.output_channels,
            **special_kwargs
        )
        ##--------------------------------------------------------------------.

    ##------------------------------------------------------------------------.
    def forward(self, x, training=True):
        """Define EPDNet forward pass."""
        # Current input shape: ['sample', 'time', 'y', 'x', 'feature']
        # Desired shape: ['sample', 'node', 'time-feature']
        ##--------------------------------------------------------------------.
        batch_size = x.shape[0]
        input_y_dim = self.input_train_y_dim if training else self.input_test_y_dim
        input_x_dim = self.input_train_x_dim if training else self.input_test_x_dim
        ##--------------------------------------------------------------------.
        # Reorder and reshape data
        x = (
            x.rename(*self.dim_names)
            .align_to("sample", "time", "y", "x", "feature")
            .rename(None)
        )  # x.permute(0, 2, 1, 3)
        x = x.reshape(
            batch_size, self.input_channels, input_y_dim, input_x_dim
        )  # reshape to ['sample', 'channels', 'y', 'x']
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
        # Reshape data to ['sample', 'time', 'y', 'x', 'feature']
        batch_size = x.shape[0] 
        output_y_dim = self.output_train_y_dim if training else self.output_test_y_dim
        output_x_dim = self.output_train_x_dim if training else self.output_test_x_dim

        x = x.reshape(
            batch_size,
            self.output_time_dim,
            output_y_dim,
            output_x_dim,
            self.output_feature_dim,
        )  # ==> ['sample', 'node', 'time', 'feature']
        x = (
            x.rename(*["sample", "time", "y", "x", "feature"])
            .align_to(*self.dim_names)
            .rename(None)
        )

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
        # Convolutions options
        kernel_size_conv: int = 3,
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
            "periodic_padding": periodic_padding,
        }

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
        # Convolutions options
        kernel_size_conv: int = 3,
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
        self.input_train_y_dim = tensor_info["input_shape_info"]["train"]["dynamic"]["y"]
        self.input_train_x_dim = tensor_info["input_shape_info"]["train"]["dynamic"]["x"]
        self.input_test_y_dim = tensor_info["input_shape_info"]["test"]["dynamic"]["y"]
        self.input_test_x_dim = tensor_info["input_shape_info"]["test"]["dynamic"]["x"]

        self.output_time_dim = tensor_info["output_n_time"]
        self.output_feature_dim = tensor_info["output_n_feature"]
        self.output_train_y_dim = tensor_info["output_shape_info"]["train"]["dynamic"]["y"]
        self.output_train_x_dim = tensor_info["output_shape_info"]["train"]["dynamic"]["x"]
        self.output_test_y_dim = tensor_info["output_shape_info"]["test"]["dynamic"]["y"]
        self.output_test_x_dim = tensor_info["output_shape_info"]["test"]["dynamic"]["x"]

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
            # Options for conv_type = "image", sampling='equiangular'
        }

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
