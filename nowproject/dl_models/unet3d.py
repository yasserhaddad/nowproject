import importlib

import torch
import torch.nn as nn

from nowproject.dl_models.layers_3d import (
    DoubleConv, 
    ExtResNetBlock, 
    create_encoders,
    create_decoders
)
from nowproject.utils.utils_models import reshape_input_for_decoding, reshape_input_for_encoding


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class Base3DUNet(nn.Module):
    """
    Base class for standard and residual UNet.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, tensor_info, final_sigmoid, basic_module, f_maps=64, layer_order=("conv", "relu"),
                 num_groups=8, num_levels=4, is_segmentation=True, conv_kernel_size=3, pool_kernel_size=2,
                 conv_padding=1, upsample_scale_factor=(2, 2, 2), final_conv_kernel=4, increment_learning=True, 
                 pool_type="max", **kwargs):
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

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None
        
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
        # apply final_activation (i.e. Sigmoid or Softmax) only during prediction. During training the network outputs logits
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)

        output_y_dim = self.output_train_y_dim if training else self.output_test_y_dim
        output_x_dim = self.output_train_x_dim if training else self.output_test_x_dim
        x = reshape_input_for_decoding(x, self.dim_names, 
                                       [self.batch_size, self.output_time_dim, output_y_dim, 
                                       output_x_dim, self.output_feature_dim])

        if self.increment_learning:
            x *= self.res_increment 
            x += self.x_last_timestep

        return x