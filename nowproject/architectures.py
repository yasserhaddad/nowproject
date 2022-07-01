import torch
import numpy as np
from typing import List, Tuple, Union, Dict
from torch import nn
import torch.nn.functional as F

from nowproject.models.layers_res_conv import (
    downsampling,
    UpsamplingResConv
)

from nowproject.models.layers_optical_flow import (
    RAFTOpticalFlow
)

from nowproject.models.models_utils import (
    reshape_input_for_decoding,
    reshape_input_for_encoding
)

from nowproject.models.layers_3d import (
    DecoderMultiScale, 
    DoubleConv, 
    ExtResNetBlock, 
    NoDownsampling, 
    ResNetBlock, 
    create_encoders_multiscale
)
from nowproject.models.unet3d import Base3DUNet, number_of_features_per_level


class ResidualUNet3D(Base3DUNet):
    """
    3DUNet model with ResBlocks as base module and transposed convolutions for upsampling.
    """

    def __init__(self, 
                 tensor_info: dict, 
                 f_maps: Union[int, List[int]] = 64, 
                 layer_order: Union[List[str], Tuple[str, ...]] = ("conv", "relu"),
                 num_groups: int = 8, 
                 num_levels: int = 5, 
                 conv_padding: int = 1, 
                 pool_kernel_size: Union[int, Tuple[int, ...]] = (1, 2, 2), 
                 conv_kernel_size: Union[int, Tuple[int, ...]] = 3, 
                 upsample_scale_factor: Union[int, Tuple[int, ...]] = (1, 2, 2),
                 increment_learning: bool = True, 
                 pool_type: str = "max", 
                 **kwargs):
        """
        Initialize the ResidualUNet3D.

        Parameters
        ----------
        tensor_info : dict
            Dictionary containing all the input and output tensor information
        f_maps : Union[int, List[int]], optional
            Number of feature maps, by default 64
        layer_order : Union[List[str], Tuple[str]], optional
            The order of the layers in the basic convolutional block, by default ("conv", "relu")
        num_groups : int, optional
            Number of groups to divide the channels into for the GroupNorm, by default 8
        num_levels : int, optional
            Depth of the network, by default 5
        conv_padding : int, optional
            Padding of the convolution, by default 1
        pool_kernel_size : Union[int, Tuple[int]], optional
            Kernel size of the pooling, by default (1, 2, 2)
        conv_kernel_size : Union[int, Tuple[int]], optional
            Kernel size of the convolution, by default 3
        upsample_scale_factor : Union[int, Tuple[int]], optional
            Kernel size of the transposed convolution, by default (1, 2, 2)
        increment_learning : bool, optional
            Whether to perform increment learning or not, by default True
        pool_type : str, optional
            Type of pooling layer to use, by default "max"
        """
        super().__init__(tensor_info=tensor_info,
                        final_sigmoid=False,
                        basic_module=ResNetBlock,
                        f_maps=f_maps,
                        layer_order=layer_order,
                        num_groups=num_groups,
                        num_levels=num_levels,
                        is_segmentation=False,
                        conv_padding=conv_padding,
                        pool_kernel_size=pool_kernel_size,
                        conv_kernel_size=conv_kernel_size,
                        upsample_scale_factor=upsample_scale_factor,
                        increment_learning=increment_learning,
                        pool_type=pool_type,
                        **kwargs)


class MultiScaleResidualConv(nn.Module):
    """
    Model that treats the input signals at multiple scales in parallel before joining them at the end
    of the encoding. Upsamples the downscaled encodings with transposed convolutions.
    """
    def __init__(self, 
                 tensor_info: dict, 
                 f_maps: int = 64, 
                 layer_order: Union[List[str], Tuple[str, ...]] = ("conv", "relu"), 
                 basic_module: nn.Module = ResNetBlock,
                 num_groups: int = 8, 
                 num_levels: int = 3, 
                 conv_padding: int = 1, 
                 pooling_depth: int = 2, 
                 pool_kernel_size: Union[int, Tuple[int, ...]] = (1, 2, 2), 
                 conv_kernel_size: Union[int, Tuple[int, ...]] = 3, 
                 upsample_scale_factor: Union[int, Tuple[int, ...]] = (1, 2, 2), 
                 final_conv_kernel: Union[int, Tuple[int, ...]] = (4, 1, 1), 
                 increment_learning: bool = True, 
                 pool_type: str = "max"):
        """Initialize the MultiScaleResidualConv.

        Parameters
        ----------
        tensor_info : dict
            Dictionary containing all the input and output information.
        f_maps : int, optional
            Number of feature maps, by default 64
        layer_order : Union[List[str], Tuple[str]], optional
            Order of the layers in the basic convolutional block, by default ("conv", "relu")
        basic_module : nn.Module, optional
            Basic model for the encoder/decoder, by default ExtResNetBlock
        num_groups : int, optional
            Number of groups to divide the channels into for the GroupNorm, by default 8
        num_levels : int, optional
            Number of levels in the encoder/decoder path, by default 3
        conv_padding : int, optional
            Padding of the convolution, by default 1
        pooling_depth : int, optional
            Number of pooling levels, by default 2
        pool_kernel_size : Union[int, Tuple[int]], optional
            Kernel size of the pooling, by default (1, 2, 2)
        conv_kernel_size : Union[int, Tuple[int, ...]], optional
            Kernel size of the convolution, by default 3
        upsample_scale_factor : Union[int, Tuple[int]], optional
            Kernel size of the transposed convolution, by default (1, 2, 2)
        final_conv_kernel : Union[int, Tuple[int]], optional
            Kernel size of the final convolution, by default (4, 1, 1)
        increment_learning : bool, optional
            Whether to perform increment learning or not, by default True
        pool_type : str, optional
            Type of pooling layer to use, by default "max"
        """
        super().__init__()

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

        self.input_channels = self.input_feature_dim 
        self.output_channels = self.output_feature_dim

        self.pooling = nn.ModuleList()
        for i in range(pooling_depth):
            if i == 0: self.pooling.append(nn.ModuleList([NoDownsampling()]))
            else:
                if pool_type == 'max':
                    self.pooling.append(nn.ModuleList([
                                    nn.MaxPool3d(kernel_size=pool_kernel_size) for i in range(1, i+1)
                                    ]))
                else:
                    self.pooling.append(nn.ModuleList([
                                    nn.AvgPool3d(kernel_size=pool_kernel_size) for i in range(1, i+1)
                                    ]))


        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)
            
        # create multi-scale encoders-decoders
        self.encoders = nn.ModuleList()
        for i in range(pooling_depth):
            self.encoders.append(create_encoders_multiscale(self.input_channels, f_maps, basic_module, conv_kernel_size, conv_padding, 
                                                            layer_order, num_groups, upsample_last=(i!=0)))
        
        self.decoders = nn.ModuleList()
        for i in range(1, pooling_depth):
            decoder_list = nn.ModuleList()
            for j in range(i):
                input_channels = f_maps[1] if j == 0 else f_maps[0]
                decoder_list.append(DecoderMultiScale(input_channels, f_maps[0],
                                                    basic_module=basic_module,
                                                    conv_layer_order=layer_order,
                                                    conv_kernel_size=conv_kernel_size,
                                                    num_groups=num_groups,
                                                    padding=conv_padding,
                                                    upsample=True,
                                                    scale_factor=upsample_scale_factor))
            self.decoders.append(decoder_list)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], self.output_channels, final_conv_kernel)
        # self.final_conv = nn.Conv1d(f_maps[0], self.output_channels, final_conv_kernel)
        
        ##--------------------------------------------------------------------.
        # Decide whether to predict the difference from the previous timestep
        #  instead of the full state
        self.increment_learning = increment_learning
        if self.increment_learning:
            self.res_increment = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, training=True):
        self.batch_size = x.shape[0]
        input_y_dim = self.input_train_y_dim if training else self.input_test_y_dim
        input_x_dim = self.input_train_x_dim if training else self.input_test_x_dim

        ##--------------------------------------------------------------------.
        # Reorder and reshape data
        x = reshape_input_for_encoding(x, self.dim_names, 
                                    [self.batch_size, self.input_feature_dim, self.input_time_dim, 
                                    input_y_dim, input_x_dim])

        ## Extract last timestep (to add after decoding) to make the network learn the increment
        self.x_last_timestep = x[:, -1:, -1, :, :].reshape(self.batch_size, input_y_dim, input_x_dim, 1)\
                                                    .unsqueeze(dim=1)
        
        encoder_features = []
        for i in range(len(self.pooling)):
            x_f = x
            for pool_layer in self.pooling[i]:
                x_f = pool_layer(x_f)
            for encoding_layer in self.encoders[i]:
                x_f = encoding_layer(x_f)
            encoder_features.append(x_f)


        for i, decoder in enumerate(self.decoders):
            for j in range(len(decoder)):
                encoder_features[i+1] = decoder[j](encoder_features[i-j], encoder_features[i+1])
        
        x = torch.stack(encoder_features).sum(dim=0)
        x = self.final_conv(x)

        output_y_dim = self.output_train_y_dim if training else self.output_test_y_dim
        output_x_dim = self.output_train_x_dim if training else self.output_test_x_dim
        x = reshape_input_for_decoding(x, self.dim_names, 
                                       [self.batch_size, self.output_time_dim, output_y_dim, 
                                       output_x_dim, self.output_feature_dim])

        if self.increment_learning:
            x *= self.res_increment 
            x += self.x_last_timestep

        return x


class resConv(nn.Module):
    """
    Model inspired from Cuomo et al. 2021, "Use of Deep Learning for Weather Radar Nowcasting".
    Similar to UNet3D but downscales with 3D convolutions instead of pooling layers.
    """
    def __init__(self, tensor_info: dict, 
                # ConvBlock Options
                n_filter: int, 
                first_layer_upsampling_kernels: List[List[int]],
                first_layer_upsampling_stride: List[List[int]],
                last_convblock_kernel: List[int] = [5, 4],
                last_layer_activation: bool = False,
                optical_flow: bool = False,
                # Architecture options
                increment_learning: bool = True):
        """Initialize the resConv model.

        Parameters
        ----------
        tensor_info : dict
            Dictionary containing all the input and output tensor information
        n_filter : int
            Number of feature maps
        first_layer_upsampling_kernels : List[List[int]]
            Kernel size of the first upsampling layer
        first_layer_upsampling_stride : List[List[int]]
            Stride of the first upsampling layer
        last_convblock_kernel : List[int], optional
            Kernel size of the last convolutional block, by default [5, 4]
        last_layer_activation : bool, optional
            Whether to add an activation after the last convolutional block, by default False
        optical_flow : bool, optional
            Whether to estimate the optical flow and advect the field prior
            to the convolutions, by default False
        increment_learning : bool, optional
            Whether to perform increment learning or not, by default True
        """
        super().__init__()
        
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
        self.input_channels = self.input_feature_dim 
        self.output_channels = self.output_feature_dim

        ##--------------------------------------------------------------------.
        # Decide whether to predict the difference from the previous timestep
        #  instead of the full state
        self.increment_learning = increment_learning
        self.optical_flow = optical_flow
        ##--------------------------------------------------------------------.
        # Layers
        if self.optical_flow:
            self.raft = RAFTOpticalFlow(self.input_time_dim, small_model=False)

        self.e_conv0 = downsampling(in_channels=self.input_channels, out_channels=n_filter)
        self.e_conv1 = downsampling(in_channels=n_filter,   out_channels=n_filter*2)
        self.e_conv2 = downsampling(in_channels=n_filter*2, out_channels=n_filter*3)
        self.e_conv3 = downsampling(in_channels=n_filter*3, out_channels=n_filter*4)
        
        self.d_conv3 = UpsamplingResConv(in_channels=n_filter*4,   out_channels=n_filter*3,
                                         first_layer_kernel=first_layer_upsampling_kernels[0],
                                         first_layer_stride=first_layer_upsampling_stride[0])
        self.d_conv2 = UpsamplingResConv(in_channels=n_filter*3,   out_channels=n_filter*2,
                                         first_layer_kernel=first_layer_upsampling_kernels[1],
                                         first_layer_stride=first_layer_upsampling_stride[1])
        self.d_conv1 = UpsamplingResConv(in_channels=n_filter*2,   out_channels=n_filter,
                                         first_layer_kernel=first_layer_upsampling_kernels[2],
                                         first_layer_stride=first_layer_upsampling_stride[2])
        self.d_conv0 = UpsamplingResConv(in_channels=n_filter,     out_channels=16, last=True,
                                         first_layer_kernel=first_layer_upsampling_kernels[3],
                                         first_layer_stride=first_layer_upsampling_stride[3])

        conv_last_layers = [
            nn.Conv1d(16, 4, last_convblock_kernel[0]),
            nn.Conv1d(4, self.output_channels, last_convblock_kernel[1])
        ]

        if last_layer_activation:
            conv_last_layers = conv_last_layers + [nn.ReLU()]

        self.conv_last = nn.Sequential(*conv_last_layers)
        if self.increment_learning:
            self.res_increment = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        
    
    def encode(self, x, training=True):
        self.batch_size = x.shape[0]
        input_y_dim = self.input_train_y_dim if training else self.input_test_y_dim
        input_x_dim = self.input_train_x_dim if training else self.input_test_x_dim

        ##--------------------------------------------------------------------.
        # Reorder and reshape data
        x = reshape_input_for_encoding(x, self.dim_names, 
                                       [self.batch_size, self.input_feature_dim, self.input_time_dim, 
                                       input_y_dim, input_x_dim])
        if self.optical_flow:
            x = self.raft(x)
        
        ## Extract last timestep (to add after decoding) to make the network learn the increment
        self.x_last_timestep = x[:, -1:, -1, :, :].reshape(self.batch_size, input_y_dim, input_x_dim, 1)\
                                                  .unsqueeze(dim=1)

        e1 = self.e_conv0(x)
        e2 = self.e_conv1(e1)
        e3 = self.e_conv2(e2)
        e4 = self.e_conv3(e3)
        
        return e1, e2, e3, e4

    def decode(self, e1, e2, e3, e4, training=True):
        d3 = self.d_conv3(e4, x2=e3)
        d2 = self.d_conv2(d3+e3, x2=e2)
        d1 = self.d_conv1(d2+e2, x2=e1)
        y = self.d_conv0(d1+e1)

        y = y.rename("sample", "feature", "time", "y", "x")\
             .align_to("sample", "y", "x", "feature", "time")\
             .flatten(["sample", "y", "x"],  "pixel_batch")\
             .rename(None)

        y = self.conv_last(y)

        output_y_dim = self.output_train_y_dim if training else self.output_test_y_dim
        output_x_dim = self.output_train_x_dim if training else self.output_test_x_dim
        y = reshape_input_for_decoding(y, self.dim_names, 
                                       [self.batch_size, self.output_time_dim, output_y_dim, 
                                       output_x_dim, self.output_feature_dim])

        if self.increment_learning:
            y *= self.res_increment 
            y += self.x_last_timestep

        return y


    def forward(self, x, training=True):
        e1, e2, e3, e4 = self.encode(x, training=training)
        y = self.decode(e1, e2, e3, e4, training=training)
        return y
