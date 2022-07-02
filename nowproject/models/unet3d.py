
from typing import List, Tuple, Union

import torch
import torch.nn as nn

from nowproject.models.layers_3d import (
    DoubleConv, 
    ExtResNetBlock, 
    create_encoders,
    create_decoders
)
from nowproject.models.models_utils import reshape_input_for_decoding, reshape_input_for_encoding


def number_of_features_per_level(init_channel_number: int, num_levels: int) -> List[int]:
    """Generates the list of feature maps at each level of the encoder.

    Parameters
    ----------
    init_channel_number : int
        Initial number of feature maps
    num_levels : int
        Number of levels in the encoder path

    Returns
    -------
    List[int]
        The list of feature maps at each level of the encoder
    """
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class Base3DUNet(nn.Module):
    def __init__(self, 
                 tensor_info: dict, 
                 basic_module: nn.Module, 
                 f_maps: Union[int, List[int]] = 64,       
                 layer_order: Union[List[str], Tuple[str, ...]] = ("conv", "relu"),
                 num_groups: int = 8, 
                 num_levels: int = 4, 
                 conv_kernel_size: Union[int, Tuple[int, ...]] = 3, 
                 pool_kernel_size: Union[int, Tuple[int, ...]] = 2,
                 conv_padding: int = 1, 
                 upsample_scale_factor: Union[int, Tuple[int, ...]] = (2, 2, 2), 
                 final_conv_kernel: int = 4, 
                 increment_learning: bool = True, 
                 pool_type: str = "max", 
                 **kwargs):
        """Base class for standard and residual UNet.

        Parameters
        ----------
        tensor_info : dict
            Dictionary containing all the input and output tensor information
        basic_module : nn.Module
            Basic model for the encoder/decoder
        f_maps : Union[int, List[int]], optional
            Number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4, by default 64
        layer_order : Union[List[str], Tuple[str, ...]], optional
            Determines the order of layers, by default ("conv", "relu")
        num_groups : int, optional
            Number of groups to divide the channels into for the GroupNorm, by default 8
        num_levels : int, optional
            Number of levels in the encoder/decoder path, by default 4
        conv_kernel_size : Union[int, Tuple[int, ...]], optional
            Kernel size of the convolution, by default 3
        pool_kernel_size : Union[int, Tuple[int, ...]], optional
            Kernel size of the pooling, by default 2
        conv_padding : int, optional
            Padding of the convolution, by default 1
        upsample_scale_factor : Union[int, Tuple[int, ...]], optional
            Kernel size of the transposed convolution, by default (2, 2, 2)
        final_conv_kernel : int, optional
            Kernel size of the final convolution, by default 4
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

        ##--------------------------------------------------------------------.
        # Define size of last dimension for ConvChen conv (merging time-feature dimension)
        self.input_channels = self.input_feature_dim 
        self.output_channels = self.output_feature_dim

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.encoders = create_encoders(self.input_channels , f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,
                                        num_groups, pool_kernel_size, pool_type=pool_type)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,
                                        upsample=True, scale_factor=upsample_scale_factor)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        # self.final_conv = nn.Conv3d(f_maps[0], self.output_channels, final_conv_kernel)
        self.final_conv = nn.Conv1d(f_maps[0], self.output_channels, final_conv_kernel)
        
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

        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = x.rename("sample", "feature", "time", "y", "x")\
             .align_to("sample", "y", "x", "feature", "time")\
             .flatten(["sample", "y", "x"],  "pixel_batch")\
             .rename(None)

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