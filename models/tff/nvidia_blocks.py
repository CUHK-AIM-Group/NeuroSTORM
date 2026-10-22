import torch
import torch.nn as nn
from collections import OrderedDict

# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def tuple_prod(x):
    prod = 1
    for xx in x:
        prod *= xx
    return prod

class GreenBlock(nn.Module):
    def __init__(self, in_channels, out_channels ,drop_rate=0.4):
        """
        green_block(inp, filters, name=None)
        ------------------------------------
        Implementation of the special residual block used in the paper. The block
        consists of two (GroupNorm --> ReLu --> 3x3x3 non-strided Convolution)
        units, with a residual connection from the input `inp` to the output. Used
        internally in the model. Can be used independently as well.
        Note that images must come with dimensions "c, H, W, D"
        Parameters
        ----------
        `inp`: An keras.layers.layer instance, required
            The keras layer just preceding the green block.
        `out_channels`: integer, required
            No. of filters to use in the 3D convolutional block. The output
            layer of this green block will have this many no. of channels.
        Returns
        -------
        `out`: A keras.layers.Layer instance
            The output of the green block. Has no. of channels equal to `filters`.
            The size of the rest of the dimensions remains same as in `inp`.
        """
        super(GreenBlock, self).__init__()
        self.Drop_Rate = drop_rate
        # Define block
        self.block = nn.Sequential(OrderedDict([
            ('group_norm0', nn.GroupNorm(num_channels=in_channels, num_groups=in_channels // 4)),
            #('norm0', nn.BatchNorm3d(num_features=in_channels)),
            ('relu0', nn.LeakyReLU(inplace=True)),
            ('conv0', nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            ('group_norm1', nn.GroupNorm(num_channels=out_channels, num_groups=in_channels // 4)),
            #('norm1', nn.BatchNorm3d(num_features=out_channels)),
            ('relu1', nn.LeakyReLU(inplace=True)),
            ('conv2', nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
        ]))

    def forward(self, inputs):
        #x_res = self.res(inputs)
        x_res = inputs
        x = torch.nn.functional.dropout(self.block(inputs), p=self.Drop_Rate, training=self.training)
        #return torch.cat([x, x_res], dim=1)
        return x + x_res



class UpGreenBlock(nn.Sequential):
    def __init__(self, in_features, out_features, shape, Drop_Rate):
        super(UpGreenBlock, self).__init__()

        self.add_module('conv', nn.Conv3d(in_features, out_features, kernel_size=1, stride=1))
        self.add_module('up', nn.Upsample(size=shape))
        self.add_module('green', GreenBlock(out_features, out_features, Drop_Rate))
