import torch
import torch.linalg
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from dtac.encoder import ResEncoder
from dtac.decoder import ResDecoder
import dtac.ResNetEnc as ResNetenc
import dtac.ResNetDec as ResNetdec

def PSNR(img1, img2, PIXEL_MAX = 255.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        print("You are comparing two same images")
        return 100
    else:
        return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class CNNEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=3, num_filters=64, n_hidden_layers=2, hidden_size=128,
                 min_log_std=-10, max_log_std=2):
        super().__init__()

        # assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.conv_layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        x = torch.rand([1] + list(obs_shape))
        conv_flattened_size = np.prod(self.forward_conv(x).shape[-3:])
        ff_layers = OrderedDict()
        previous_feature_size = conv_flattened_size
        for i in range(n_hidden_layers):
            ff_layers[f'linear_{i + 1}'] = nn.Linear(in_features=previous_feature_size,
                                                     out_features=hidden_size)
            ff_layers[f'relu_{i + 1}'] = nn.ReLU()
            previous_feature_size = hidden_size

        ff_layers[f'linear_{n_hidden_layers + 1}'] = nn.Linear(in_features=previous_feature_size,
                                                               out_features=2 * feature_dim)
        self.ff_layers = nn.Sequential(ff_layers)

    def forward_conv(self, obs):
        # assert obs.max() <= 1 and 0 <= obs.min(), f'Make sure images are in [0, 1]. Get [{obs.min()}, {obs.max()}]'
        conv = torch.relu(self.conv_layers[0](obs))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.conv_layers[i](conv))
        conv = conv.reshape(conv.size(0), -1)
        return conv

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        out = self.ff_layers(h)
        mean, log_std = out.split([self.feature_dim, self.feature_dim], dim=1)
        log_std = log_std.clip(self.min_log_std, self.max_log_std)
        return mean, log_std


class CNNDecoder(nn.Module):
    def __init__(self, input_dim, out_shape, num_layers=3, num_filters=64, n_hidden_layers=2, hidden_size=128):
        super().__init__()

        # assert len(out_shape) == 3, "Please specify output image size, channel first."
        # assert out_shape[1] % (2 ** num_layers) == 0, "Only supports 2x up-scales."
        self.out_shape = out_shape
        self.num_layers = num_layers

        ff_layers = OrderedDict()
        previous_feature_size = input_dim
        for i in range(n_hidden_layers):
            ff_layers[f'linear_{i + 1}'] = nn.Linear(in_features=previous_feature_size,
                                                     out_features=hidden_size)
            ff_layers[f'relu_{i + 1}'] = nn.ReLU()
            previous_feature_size = hidden_size

        side_length = self.out_shape[1] // (2 ** self.num_layers)
        self.smallest_image_size = (num_filters, side_length, side_length)
        flattened_size = int(np.prod(self.smallest_image_size))
        ff_layers[f'linear_{n_hidden_layers + 1}'] = nn.Linear(in_features=previous_feature_size,
                                                               out_features=flattened_size)
        ff_layers[f'relu_{n_hidden_layers + 1}'] = nn.ReLU()
        self.ff_layers = nn.Sequential(ff_layers)

        self.conv_layers = nn.ModuleList()
        for i in range(num_layers - 1):
            if i == num_layers - 2 and out_shape[-1] == 100:
                self.conv_layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=2))
            elif i == num_layers - 2 and out_shape[-1] == 84:
                self.conv_layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=2))
            else:
                self.conv_layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1))
        self.conv_layers.append(nn.Conv2d(num_filters, out_shape[0], 3, stride=1, padding=1))

    def forward(self, z):
        h = self.ff_layers(z)
        h = h.reshape((h.shape[0], *self.smallest_image_size))

        for i in range(self.num_layers - 1):
            h = nn.functional.interpolate(h, scale_factor=2)
            h = self.conv_layers[i](h)
            h = nn.functional.relu(h)
        output = nn.functional.interpolate(h, scale_factor=2)
        output = self.conv_layers[-1](output)

        return output


class E2D1(nn.Module):
    def __init__(self, obs_shape1: tuple, obs_shape2: tuple, z_dim1: int, z_dim2: int, norm_sample: bool=True, num_layers=3, num_filters=64, n_hidden_layers=2, hidden_size=128):
        super().__init__()
        self.enc1 = CNNEncoder(obs_shape1, z_dim1, num_layers, num_filters, n_hidden_layers, hidden_size)
        self.enc2 = CNNEncoder(obs_shape2, z_dim2, num_layers, num_filters, n_hidden_layers, hidden_size)
        self.dec = CNNDecoder( int((z_dim1 + z_dim2)* 0.75 ), (obs_shape1[0] + obs_shape2[0], obs_shape1[1], obs_shape1[2])) ### gym
        self.norm_sample = norm_sample

    def forward(self, obs1, obs2):
        z1_mean, z1_log_std = self.enc1(obs1)
        z2_mean, z2_log_std = self.enc2(obs2)

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1_mean.shape[1] // 2 # 16
            batch_size = z1_mean.shape[0]
            z1_private = z1_mean[:, :num_features]
            z2_private = z2_mean[:, :num_features]
            z1_share = z1_mean[:, num_features:]
            z2_share = z2_mean[:, num_features:]

            ### similarity (invariance) loss of shared representations
            invar_loss =  F.mse_loss(z1_share, z2_share)

            ### decode 
            z_sample = torch.cat((z1_private, z1_share, z2_private), dim=1)
            obs_dec = self.dec(z_sample)
            obs = torch.cat((obs1, obs2), dim=1)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### variance loss
            z1_private_norm = z1_private - z1_private.mean(dim=0)
            z1_share_norm = z1_share - z1_share.mean(dim=0)
            z2_private_norm = z2_private - z2_private.mean(dim=0)
            z2_share_norm = z2_share - z2_share.mean(dim=0)

            std_z1_private = torch.sqrt(z1_private_norm.var(dim=0) + 0.0001)
            std_z1_share = torch.sqrt(z1_share_norm.var(dim=0) + 0.0001)
            std_z2_private = torch.sqrt(z2_private_norm.var(dim=0) + 0.0001)
            std_z2_share = torch.sqrt(z2_share_norm.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_z1_private)) / 4 + torch.mean(F.relu(1 - std_z1_share)) / 4 + torch.mean(F.relu(1 - std_z2_private)) / 4 + torch.mean(F.relu(1 - std_z2_share)) / 4

            ### covariance loss 
            z1_private_share = torch.cat((z1_private_norm, z1_share_norm), dim=1)
            z2_private_share = torch.cat((z2_private_norm, z2_share_norm), dim=1)
            z12_private = torch.cat((z1_private_norm, z2_private_norm), dim=1)
            covz1 = (z1_private_share.T @ z1_private_share) / (batch_size - 1)
            covz2 = (z2_private_share.T @ z2_private_share) / (batch_size - 1)
            covz12 = (z12_private.T @ z12_private) / (batch_size - 1)
            cov_loss = off_diagonal(covz1).pow_(2).sum().div(num_features) / 3 + off_diagonal(covz2).pow_(2).sum().div(num_features) / 3 + off_diagonal(covz12).pow_(2).sum().div(num_features) / 3

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), std_loss, invar_loss, cov_loss, psnr


class E2D1NonSym(nn.Module):
    def __init__(self, obs_shape1: tuple, obs_shape2: tuple, z_dim1: int, z_dim2: int, norm_sample: bool=True, num_layers=3, num_filters=64, n_hidden_layers=2, hidden_size=128):
        super().__init__()
        self.enc1 = CNNEncoder(obs_shape1, z_dim1, num_layers, num_filters, n_hidden_layers, hidden_size)
        self.enc2 = CNNEncoder(obs_shape2, z_dim2, num_layers, num_filters, n_hidden_layers, hidden_size)
        self.dec = CNNDecoder( int((z_dim1 + z_dim2)* 0.75 ), (obs_shape1[0], obs_shape1[2], obs_shape1[2])) ### airbus
        self.norm_sample = norm_sample

    def forward(self, obs1, obs2):
        z1_mean, z1_log_std = self.enc1(obs1)
        z2_mean, z2_log_std = self.enc2(obs2)

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1_mean.shape[1] // 2 # 16
            batch_size = z1_mean.shape[0]
            z1_private = z1_mean[:, :num_features]
            z2_private = z2_mean[:, :num_features]
            z1_share = z1_mean[:, num_features:]
            z2_share = z2_mean[:, num_features:]

            ### similarity (invariance) loss of shared representations
            invar_loss =  F.mse_loss(z1_share, z2_share)

            ### decode 
            z_sample = torch.cat((z1_private, z1_share, z2_private), dim=1)
            obs_dec = self.dec(z_sample)
            obs = torch.cat((obs1, obs2), dim=1)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### variance loss
            z1_private_norm = z1_private - z1_private.mean(dim=0)
            z1_share_norm = z1_share - z1_share.mean(dim=0)
            z2_private_norm = z2_private - z2_private.mean(dim=0)
            z2_share_norm = z2_share - z2_share.mean(dim=0)

            std_z1_private = torch.sqrt(z1_private_norm.var(dim=0) + 0.0001)
            std_z1_share = torch.sqrt(z1_share_norm.var(dim=0) + 0.0001)
            std_z2_private = torch.sqrt(z2_private_norm.var(dim=0) + 0.0001)
            std_z2_share = torch.sqrt(z2_share_norm.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_z1_private)) / 4 + torch.mean(F.relu(1 - std_z1_share)) / 4 + torch.mean(F.relu(1 - std_z2_private)) / 4 + torch.mean(F.relu(1 - std_z2_share)) / 4

            ### covariance loss 
            z1_private_share = torch.cat((z1_private_norm, z1_share_norm), dim=1)
            z2_private_share = torch.cat((z2_private_norm, z2_share_norm), dim=1)
            z12_private = torch.cat((z1_private_norm, z2_private_norm), dim=1)
            covz1 = (z1_private_share.T @ z1_private_share) / (batch_size - 1)
            covz2 = (z2_private_share.T @ z2_private_share) / (batch_size - 1)
            covz12 = (z12_private.T @ z12_private) / (batch_size - 1)
            cov_loss = off_diagonal(covz1).pow_(2).sum().div(num_features) / 3 + off_diagonal(covz2).pow_(2).sum().div(num_features) / 3 + off_diagonal(covz12).pow_(2).sum().div(num_features) / 3

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), std_loss, invar_loss, cov_loss, psnr


class E1D1(nn.Module):
    def __init__(self, obs_shape: tuple, z_dim: int, norm_sample: bool=True, num_layers=3, num_filters=64, n_hidden_layers=2, hidden_size=128): # noise=0.01):
        super().__init__()
        self.enc = CNNEncoder(obs_shape, z_dim, num_layers, num_filters, n_hidden_layers, hidden_size)
        self.dec = CNNDecoder(z_dim, (obs_shape[0], obs_shape[1], obs_shape[2]), num_layers, num_filters, n_hidden_layers, hidden_size)
        self.norm_sample = norm_sample

    def forward(self, obs):
        z1_mean, z1_log_std = self.enc(obs)

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1_mean.shape[1] // 2
            batch_size = z1_mean.shape[0]
            z1_private = z1_mean[:, :num_features]
            z1_share = z1_mean[:, num_features:]

            ### similarity (invariance) loss of shared representations
            invar_loss =  F.mse_loss(z1_share, z1_share) # this is always 0

            ### decode 
            z_sample = torch.cat((z1_private, z1_share), dim=1)
            obs_dec = self.dec(z_sample)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### variance loss
            z1_private_norm = z1_private - z1_private.mean(dim=0)
            z1_share_norm = z1_share - z1_share.mean(dim=0)

            std_z1_private = torch.sqrt(z1_private_norm.var(dim=0) + 0.0001)
            std_z1_share = torch.sqrt(z1_share_norm.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_z1_private)) / 2 + torch.mean(F.relu(1 - std_z1_share)) / 2

            ### covariance loss 
            z1_private_share = torch.cat((z1_private_norm, z1_share_norm), dim=1)
            covz1 = (z1_private_share.T @ z1_private_share) / (batch_size - 1)
            cov_loss = off_diagonal(covz1).pow_(2).sum().div(num_features) / 3

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), std_loss, invar_loss, cov_loss, psnr


class ResE2D1NonSym(nn.Module):
    def __init__(self, size1: tuple, size2: tuple, z_dim1: int, z_dim2: int, norm_sample:bool=True, n_samples: int=4, n_res_blocks: int=3):
        super().__init__()
        self.enc1 = ResEncoder(size1, z_dim1, n_downsamples=n_samples, n_res_blocks=n_res_blocks)
        self.enc2 = ResEncoder(size2, z_dim2, n_downsamples=n_samples, n_res_blocks=n_res_blocks)
        self.dec = ResDecoder((size2[0], size2[-1], size2[-1]), int((z_dim1 + z_dim2)* 0.75), n_upsamples=n_samples, n_res_blocks=n_res_blocks)
        self.norm_sample = norm_sample

    def forward(self, obs1, obs2, obs):
        z1_mean, z1_log_std = self.enc1(obs1)
        z2_mean, z2_log_std = self.enc2(obs2)

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1_mean.shape[1] // 2 # 16
            batch_size = z1_mean.shape[0]
            z1_private = z1_mean[:, :num_features]
            z2_private = z2_mean[:, :num_features]
            z1_share = z1_mean[:, num_features:]
            z2_share = z2_mean[:, num_features:]

            ### similarity (invariance) loss of shared representations
            invar_loss =  F.mse_loss(z1_share, z2_share)

            ### decode 
            z_sample = torch.cat((z1_private, z1_share, z2_private), dim=1)
            obs_dec = self.dec(z_sample)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### variance loss
            z1_private_norm = z1_private - z1_private.mean(dim=0)
            z1_share_norm = z1_share - z1_share.mean(dim=0)
            z2_private_norm = z2_private - z2_private.mean(dim=0)
            z2_share_norm = z2_share - z2_share.mean(dim=0)

            std_z1_private = torch.sqrt(z1_private_norm.var(dim=0) + 0.0001)
            std_z1_share = torch.sqrt(z1_share_norm.var(dim=0) + 0.0001)
            std_z2_private = torch.sqrt(z2_private_norm.var(dim=0) + 0.0001)
            std_z2_share = torch.sqrt(z2_share_norm.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_z1_private)) / 4 + torch.mean(F.relu(1 - std_z1_share)) / 4 + torch.mean(F.relu(1 - std_z2_private)) / 4 + torch.mean(F.relu(1 - std_z2_share)) / 4

            ### covariance loss 
            z1_private_share = torch.cat((z1_private_norm, z1_share_norm), dim=1)
            z2_private_share = torch.cat((z2_private_norm, z2_share_norm), dim=1)
            z12_private = torch.cat((z1_private_norm, z2_private_norm), dim=1)
            covz1 = (z1_private_share.T @ z1_private_share) / (batch_size - 1)
            covz2 = (z2_private_share.T @ z2_private_share) / (batch_size - 1)
            covz12 = (z12_private.T @ z12_private) / (batch_size - 1)
            cov_loss = off_diagonal(covz1).pow_(2).sum().div(num_features) / 3 + off_diagonal(covz2).pow_(2).sum().div(num_features) / 3 + off_diagonal(covz12).pow_(2).sum().div(num_features) / 3

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), std_loss, invar_loss, cov_loss, psnr


class ResE2D1(nn.Module):
    def __init__(self, obs_shape1: tuple, obs_shape2: tuple, z_dim1: int, z_dim2: int, norm_sample:bool=True, n_samples: int=4, n_res_blocks: int=3):
        super().__init__()
        self.enc1 = ResEncoder(obs_shape1, z_dim1, n_downsamples=n_samples, n_res_blocks=n_res_blocks)
        self.enc2 = ResEncoder(obs_shape2, z_dim2, n_downsamples=n_samples, n_res_blocks=n_res_blocks)
        self.dec = ResDecoder((obs_shape1[0] + obs_shape2[0], obs_shape1[1], obs_shape1[2]), int((z_dim1 + z_dim2)* 0.75 ), \
                              n_upsamples=n_samples, n_res_blocks=n_res_blocks)
        self.norm_sample = norm_sample

    def forward(self, obs1, obs2):
        z1_mean, z1_log_std = self.enc1(obs1)
        z2_mean, z2_log_std = self.enc2(obs2)

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1_mean.shape[1] // 2 # 16
            batch_size = z1_mean.shape[0]
            z1_private = z1_mean[:, :num_features]
            z2_private = z2_mean[:, :num_features]
            z1_share = z1_mean[:, num_features:]
            z2_share = z2_mean[:, num_features:]

            ### similarity (invariance) loss of shared representations
            invar_loss =  F.mse_loss(z1_share, z2_share)

            ### decode 
            z_sample = torch.cat((z1_private, z1_share, z2_private), dim=1)
            obs_dec = self.dec(z_sample)
            obs = torch.cat((obs1, obs2), dim=1)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### variance loss
            z1_private_norm = z1_private - z1_private.mean(dim=0)
            z1_share_norm = z1_share - z1_share.mean(dim=0)
            z2_private_norm = z2_private - z2_private.mean(dim=0)
            z2_share_norm = z2_share - z2_share.mean(dim=0)

            std_z1_private = torch.sqrt(z1_private_norm.var(dim=0) + 0.0001)
            std_z1_share = torch.sqrt(z1_share_norm.var(dim=0) + 0.0001)
            std_z2_private = torch.sqrt(z2_private_norm.var(dim=0) + 0.0001)
            std_z2_share = torch.sqrt(z2_share_norm.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_z1_private)) / 4 + torch.mean(F.relu(1 - std_z1_share)) / 4 + torch.mean(F.relu(1 - std_z2_private)) / 4 + torch.mean(F.relu(1 - std_z2_share)) / 4

            ### covariance loss 
            z1_private_share = torch.cat((z1_private_norm, z1_share_norm), dim=1)
            z2_private_share = torch.cat((z2_private_norm, z2_share_norm), dim=1)
            z12_private = torch.cat((z1_private_norm, z2_private_norm), dim=1)
            covz1 = (z1_private_share.T @ z1_private_share) / (batch_size - 1)
            covz2 = (z2_private_share.T @ z2_private_share) / (batch_size - 1)
            covz12 = (z12_private.T @ z12_private) / (batch_size - 1)
            cov_loss = off_diagonal(covz1).pow_(2).sum().div(num_features) / 3 + off_diagonal(covz2).pow_(2).sum().div(num_features) / 3 + off_diagonal(covz12).pow_(2).sum().div(num_features) / 3

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), std_loss, invar_loss, cov_loss, psnr


class ResE1D1(nn.Module):
    def __init__(self, obs_shape: tuple, z_dim: int, norm_sample: bool=True, n_samples: int=4, n_res_blocks: int=3): # noise=0.01):
        super().__init__()
        self.enc = ResEncoder(obs_shape, z_dim, n_downsamples=n_samples, n_res_blocks=n_res_blocks)
        self.dec = ResDecoder(obs_shape, z_dim, n_upsamples=n_samples, n_res_blocks=n_res_blocks)
        self.norm_sample = norm_sample

    def forward(self, obs):
        z1_mean, z1_log_std = self.enc(obs)

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1_mean.shape[1] // 2
            batch_size = z1_mean.shape[0]
            z1_private = z1_mean[:, :num_features]
            z1_share = z1_mean[:, num_features:]

            ### similarity (invariance) loss of shared representations
            invar_loss =  F.mse_loss(z1_share, z1_share) # this is always 0

            ### decode 
            z_sample = torch.cat((z1_private, z1_share), dim=1)
            obs_dec = self.dec(z_sample)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### variance loss
            z1_private_norm = z1_private - z1_private.mean(dim=0)
            z1_share_norm = z1_share - z1_share.mean(dim=0)

            std_z1_private = torch.sqrt(z1_private_norm.var(dim=0) + 0.0001)
            std_z1_share = torch.sqrt(z1_share_norm.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_z1_private)) / 2 + torch.mean(F.relu(1 - std_z1_share)) / 2

            ### covariance loss 
            z1_private_share = torch.cat((z1_private_norm, z1_share_norm), dim=1)
            covz1 = (z1_private_share.T @ z1_private_share) / (batch_size - 1)
            cov_loss = off_diagonal(covz1).pow_(2).sum().div(num_features) / 3

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), std_loss, invar_loss, cov_loss, psnr


class ResNetE1D1(nn.Module):
    def __init__(self, norm_sample: bool=False): # noise=0.01):
        super().__init__()
        self.enc = ResNetenc.ResNet(ResNetenc.Bottleneck, [3, 4, 6, 3], return_indices=True)
        self.dec = ResNetdec.ResNet(ResNetdec.Bottleneck, [3, 4, 6, 3])
        self.norm_sample = norm_sample

    def forward(self, obs):
        z1_mean, indices = self.enc(obs) ### 2048

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1_mean.shape[1] // 2
            batch_size = z1_mean.shape[0]
            z1_private = z1_mean[:, :num_features]
            z1_share = z1_mean[:, num_features:]

            ### similarity (invariance) loss of shared representations
            invar_loss =  F.mse_loss(z1_share, z1_share) # this is always 0

            ### decode 
            z_sample = torch.cat((z1_private, z1_share), dim=1)
            obs_dec = self.dec(z_sample, indices)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### variance loss
            z1_private_norm = (z1_private - z1_private.mean(dim=0)).reshape(batch_size, num_features)
            z1_share_norm = (z1_share - z1_share.mean(dim=0)).reshape(batch_size, num_features)

            std_z1_private = torch.sqrt(z1_private_norm.var(dim=0) + 0.0001)
            std_z1_share = torch.sqrt(z1_share_norm.var(dim=0) + 0.0001)
            std_loss = torch.mean(F.relu(1 - std_z1_private)) / 2 + torch.mean(F.relu(1 - std_z1_share)) / 2

            ### covariance loss 
            z1_private_share = torch.cat((z1_private_norm, z1_share_norm), dim=1)
            covz1 = (z1_private_share.T @ z1_private_share) / (batch_size - 1)
            cov_loss = off_diagonal(covz1).pow_(2).sum().div(num_features) / 3

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), std_loss, invar_loss, cov_loss, psnr


if __name__ == '__main__':
    # e2d1 = E2D1((3, 128, 128), (3, 128, 128), 32, 32).cuda()
    # rand_obs1 = torch.rand((16, 3, 128, 128)).cuda()
    # rand_obs2 = torch.rand((16, 3, 128, 128)).cuda()
    # obsdec, mseloss, kl1, kl2, crosscor = e2d1(rand_obs1, rand_obs2)
    # print(obsdec.shape, mseloss, kl1, kl2, crosscor)

    e1d1 = ResNetE1D1().cuda()
    rand_obs = torch.rand((16, 3, 221, 221)).cuda()
    obsdec, mseloss, std_loss, invar_loss, cov_loss, psnr = e1d1(rand_obs)
    print(obsdec.shape, mseloss, std_loss, invar_loss, cov_loss, psnr)


