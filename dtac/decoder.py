"""Residual Block Decoder"""
import torch
import numpy as np
from torch import nn
from dtac.resnet import LNBlock
from collections import OrderedDict


class ResDecoder(nn.Module):
    def __init__(self, output_shape, feature_dim, n_upsamples=4, n_res_blocks=3, final_upsample_filters=16,
                 num_filters=(32, 64, 128, 256), n_hidden_layers=2, hidden_size=128):
        super().__init__()

        assert len(output_shape) == 3
        assert n_upsamples == len(num_filters)
        assert output_shape[1] % 2 ** n_upsamples == 0
        self.output_shape = output_shape
        self.feature_dim = feature_dim
        self.n_upsamples = n_upsamples
        self.n_res_blocks = n_res_blocks
        self.smallest_conv_shape = (num_filters[n_upsamples-1], output_shape[1] // 2 ** n_upsamples,
                                 output_shape[2] // 2 ** n_upsamples)

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(num_filters[0], final_upsample_filters, 3, stride=1, padding=1)]
        )
        for i in range(self.n_upsamples - 1):
            self.conv_layers.append(nn.Conv2d(num_filters[i + 1], num_filters[i], 3, stride=1, padding=1))
        
        conv_shapes = self.compute_conv_shapes()

        self.final_conv = nn.Conv2d(final_upsample_filters, self.output_shape[0], 3, stride=1, padding=1)

        self.res_blocks = nn.ModuleList()
        for i in range(self.n_upsamples):
            self.res_blocks.append(nn.ModuleList())
            for j in range(self.n_res_blocks):
                self.res_blocks[i].append(LNBlock(conv_shapes[i + 1]))

        self.ln_layers = nn.ModuleList()
        for i in range(self.n_upsamples + 1):
            self.ln_layers.append(nn.LayerNorm(conv_shapes[i]))

        ff_layers = OrderedDict()
        last_hidden_dim = np.prod(self.smallest_conv_shape)
        previous_feature_size = feature_dim
        for i in range(n_hidden_layers):
            ff_layers[f'linear_{i + 1}'] = nn.Linear(in_features=previous_feature_size,
                                                     out_features=hidden_size)
            ff_layers[f'relu_{i + 1}'] = nn.ReLU()
            previous_feature_size = hidden_size

        ff_layers[f'linear_{n_hidden_layers + 1}'] = nn.Linear(in_features=previous_feature_size,
                                                               out_features=last_hidden_dim)
        self.ff_layers = nn.Sequential(ff_layers)

    def compute_conv_shapes(self):
        shapes = [self.smallest_conv_shape]
        y = torch.rand([1] + list(self.smallest_conv_shape))
        for i in range(self.n_upsamples -1, -1, -1):
            y = nn.functional.interpolate(y, scale_factor=2)
            y = self.conv_layers[i](y)
            shapes.insert(0, y.shape[1:])
        return shapes

    def forward_conv(self, h):
        conv = h
        for i in range(self.n_upsamples - 1, -1, -1):
            for j in range(self.n_res_blocks):
                conv = self.res_blocks[i][j](conv)
            conv = nn.functional.interpolate(conv, scale_factor=2)
            conv = self.conv_layers[i](conv)
            conv = self.ln_layers[i](conv)
            conv = torch.relu(conv)
        conv = self.final_conv(conv)
        return conv

    def forward(self, feature):
        h = self.ff_layers(feature)
        h = h.view(-1, *self.smallest_conv_shape)
        out = self.forward_conv(h)
        return out


if __name__ == '__main__':
    decoder = ResDecoder(output_shape=(3, 896, 512), feature_dim=256)
    x = torch.rand([1, 256])
    image = decoder(x)
    print(image.shape)
