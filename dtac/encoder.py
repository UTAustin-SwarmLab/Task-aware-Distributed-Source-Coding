"""Residual Block Encoder"""
import torch
import numpy as np
import os
from torch import nn
from dtac.resnet import LNBlock
from collections import OrderedDict


# class ResEncoder(nn.Module):
#     def __init__(self, input_shape, feature_dim, n_downsamples=4, n_res_blocks=3,
#                 #  num_filters=(32, 64), n_hidden_layers=2, hidden_size=128): # , 128, 256
#                  num_filters=(8, 16, 32, 64), n_hidden_layers=2, hidden_size=128):
#         super().__init__()

#         assert len(input_shape) == 3
#         assert n_downsamples == len(num_filters)
#         self.input_shape = input_shape
#         self.feature_dim = feature_dim
#         self.n_downsamples = n_downsamples
#         self.n_res_blocks = n_res_blocks

#         self.conv_layers = nn.ModuleList(
#             [nn.Conv2d(input_shape[0], num_filters[0], 3, stride=2, padding=1)]
#         )
#         for i in range(self.n_downsamples - 1):
#             self.conv_layers.append(nn.Conv2d(num_filters[i], num_filters[i + 1], 3, stride=2, padding=1))

#         conv_shapes = self.compute_conv_shapes()

#         self.res_blocks = nn.ModuleList()
#         for i in range(self.n_downsamples):
#             self.res_blocks.append(nn.ModuleList())
#             for j in range(self.n_res_blocks):
#                 self.res_blocks[i].append(LNBlock(conv_shapes[i]))
        
#         self.ln_layers = nn.ModuleList()
#         for i in range(self.n_downsamples):
#             self.ln_layers.append(nn.LayerNorm(conv_shapes[i]))

#         x = torch.rand([1] + list(input_shape))
#         conv_flattened_size = np.prod(self.forward_conv(x).shape[-3:])
#         ff_layers = OrderedDict()
#         previous_feature_size = conv_flattened_size
#         for i in range(n_hidden_layers):
#             ff_layers[f'linear_{i + 1}'] = nn.Linear(in_features=previous_feature_size,
#                                                      out_features=hidden_size)
#             ff_layers[f'relu_{i + 1}'] = nn.ReLU()
#             previous_feature_size = hidden_size

#         ff_layers[f'linear_{n_hidden_layers + 1}'] = nn.Linear(in_features=previous_feature_size,
#                                                                out_features=2 * feature_dim)
#         self.ff_layers = nn.Sequential(ff_layers)

#     def compute_conv_shapes(self):
#         shapes = []
#         y = torch.rand([1] + list(self.input_shape))
#         for i in range(self.n_downsamples):
#             y = self.conv_layers[i](y)
#             shapes.append(y.shape[1:])
#         return shapes

#     def forward_conv(self, obs):
#         # assert obs.max() <= 1 and 0 <= obs.min(), f'Make sure images are between 0 and 1. Get [{obs.min()}, {obs.max()}]'
#         conv = obs
#         for i in range(self.n_downsamples):
#             conv = self.conv_layers[i](conv)
#             conv = self.ln_layers[i](conv)
#             conv = torch.relu(conv)
#             for j in range(self.n_res_blocks):
#                 conv = self.res_blocks[i][j](conv)
#         return conv

#     def forward(self, obs):
#         h = self.forward_conv(obs)
#         h = h.flatten(start_dim=1)
#         out = self.ff_layers(h)
#         mean, log_std = out.split([self.feature_dim, self.feature_dim], dim=1)
#         return mean, log_std


### Sravan's code
class ResEncoder(nn.Module):
    def __init__(self, input_shape, feature_dim, n_downsamples=4, n_res_blocks=3,
                n_hidden_layers=2, hidden_size=256):
        super().__init__()
        if 'airbus' in os.getcwd():
            num_filters=(16, 32, 64, 128)
        elif 'PnP' in os.getcwd():
            num_filters=(8, 16, 32, 64) # seed 0
            # num_filters=(16, 32, 64, 128) # seed 10
        else:
            raise ValueError('Unknown dataset')
        assert len(input_shape) == 3
        assert n_downsamples == len(num_filters)
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.n_downsamples = n_downsamples
        self.n_res_blocks = n_res_blocks
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(input_shape[0], num_filters[0], 3, stride=2, padding=1)]
        )
        for i in range(self.n_downsamples - 1):
            self.conv_layers.append(nn.Conv2d(num_filters[i], num_filters[i + 1], 3, stride=2, padding=1))
        conv_shapes = self.compute_conv_shapes()
        self.res_blocks = nn.ModuleList()
        for i in range(self.n_downsamples):
            self.res_blocks.append(nn.ModuleList())
            for j in range(self.n_res_blocks):
                self.res_blocks[i].append(LNBlock(conv_shapes[i]))
        self.ln_layers = nn.ModuleList()
        for i in range(self.n_downsamples):
            self.ln_layers.append(nn.LayerNorm(conv_shapes[i]))
        x = torch.rand([1] + list(input_shape))
        conv_flattened_size = np.prod(self.forward_conv(x).shape[-3:])
        ff_layers = OrderedDict()
        previous_feature_size = conv_flattened_size
        for i in range(n_hidden_layers):
            ff_layers[f'linear_{i + 1}'] = nn.Linear(in_features=previous_feature_size,
                                                     out_features=hidden_size)
            ff_layers[f'relu_{i + 1}'] = nn.ReLU()
            previous_feature_size = hidden_size
        ff_layers[f'linear_{n_hidden_layers + 1}'] = nn.Linear(in_features=previous_feature_size,
                                                               out_features=feature_dim)
        self.ff_layers = nn.Sequential(ff_layers)
    def compute_conv_shapes(self):
        shapes = []
        y = torch.rand([1] + list(self.input_shape))
        for i in range(self.n_downsamples):
            y = self.conv_layers[i](y)
            shapes.append(y.shape[1:])
        return shapes
    def forward_conv(self, obs):
        # assert obs.max() <= 1 and 0 <= obs.min(), f’Make sure images are between 0 and 1. Get [{obs.min()}, {obs.max()}]’
        conv = obs
        for i in range(self.n_downsamples):
            conv = self.conv_layers[i](conv)
            conv = self.ln_layers[i](conv)
            conv = torch.relu(conv)
            for j in range(self.n_res_blocks):
                conv = self.res_blocks[i][j](conv)
        return conv
    def forward(self, obs):
        h = self.forward_conv(obs)
        h = h.flatten(start_dim=1)
        out = self.ff_layers(h)
        return out, None

if __name__ == '__main__':
    encoder = ResEncoder(input_shape=(3, 512, 512), feature_dim=256)
    x = torch.rand([1, 3, 512, 512])
    mean, log_std = encoder(x)
    print(mean.shape, log_std.shape)
