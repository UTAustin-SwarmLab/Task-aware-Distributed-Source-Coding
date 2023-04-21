import torch
import torch.linalg
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from dtac.encoder import ResEncoder
from dtac.decoder import ResDecoder

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


def data_pca(z):
    #PCA using SVD
	data_mean = torch.mean(z, axis=0)
	z_norm = z - data_mean
	u, s, v = torch.svd(z_norm)
	return s, v, data_mean


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
            self.conv_layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=2)) # 1

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
        self.dec = CNNDecoder( (z_dim1 + z_dim2), (obs_shape1[0] + obs_shape2[0], obs_shape1[1], obs_shape1[2])) ### gym
        self.norm_sample = norm_sample

    def forward(self, obs1, obs2, random_bottle_neck=False):
        z1, _ = self.enc1(obs1)
        z2, _ = self.enc2(obs2)

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1.shape[1] + z2.shape[1]
            batch_size = z1.shape[0]
            obs = torch.cat((obs1, obs2), dim=1)

            ### decode 
            z_sample = torch.cat((z1, z2), dim=1)

            if random_bottle_neck:
                # reduce the dimensionality of the data using dpca
                # use PCA to reduce dimension
                dim_p = torch.randint(1, num_features, (1,)).item()

                s_1, v_1, mu_1 = data_pca(z1)
                s_2, v_2, mu_2 = data_pca(z2)

                # pick the indices of the top 1/2 singular values for s_1 and s_2 combined
                s_1_2 = torch.cat((s_1,s_2), 0)
                ind = torch.argsort(s_1_2,descending=True)
                ind = ind[:dim_p]
                ind_1 = ind[ind < s_1.shape[0]]
                ind_2 = ind[ind >= s_1.shape[0]] - s_1.shape[0]

                # project z1 and z2 into corresponding subspace
                z1_p = torch.matmul(z1 - mu_1, v_1[:,ind_1])
                z2_p = torch.matmul(z2 - mu_2, v_2[:,ind_2])

                # concatenate to form full z
                # z_o = torch.cat((z1_p,z2_p), 1)

                # project back the latent to full dim
                z1_b =  torch.matmul(z1_p, v_1[:,ind_1].T) + mu_1
                z2_b =  torch.matmul(z2_p, v_2[:,ind_2].T) + mu_2
                z_sample = torch.cat((z1_b,z2_b),1)

            obs_dec = self.dec(z_sample)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### Normalize
            z_sample = z_sample - z_sample.mean(dim=0)

            ### nuclear norm 
            z_sample = z_sample / torch.norm(z_sample, p=2)
            nuc_loss = torch.norm(z_sample, p='nuc', dim=(0, 1)) / batch_size

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), nuc_loss, 0, 0, psnr


class E2D1NonSym(nn.Module):
    def __init__(self, obs_shape1: tuple, obs_shape2: tuple, z_dim1: int, z_dim2: int, norm_sample: bool=True, num_layers=3, num_filters=64, n_hidden_layers=2, hidden_size=128):
        super().__init__()
        self.enc1 = CNNEncoder(obs_shape1, z_dim1, num_layers, num_filters, n_hidden_layers, hidden_size)
        self.enc2 = CNNEncoder(obs_shape2, z_dim2, num_layers, num_filters, n_hidden_layers, hidden_size)
        self.dec = CNNDecoder( (z_dim1 + z_dim2), (obs_shape1[0], obs_shape1[2], obs_shape1[2])) ### airbus
        self.norm_sample = norm_sample

    def forward(self, obs1, obs2, obs, random_bottle_neck=False):
        z1, _ = self.enc1(obs1)
        z2, _ = self.enc2(obs2)

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1.shape[1] + z2.shape[1]
            batch_size = z1.shape[0]

            ### decode 
            z_sample = torch.cat((z1, z2), dim=1)
            if random_bottle_neck:
                # reduce the dimensionality of the data using dpca
                # use PCA to reduce dimension
                dim_p = torch.randint(1, num_features, (1,)).item()

                s_1, v_1, mu_1 = data_pca(z1)
                s_2, v_2, mu_2 = data_pca(z2)

                # pick the indices of the top 1/2 singular values for s_1 and s_2 combined
                s_1_2 = torch.cat((s_1,s_2), 0)
                ind = torch.argsort(s_1_2,descending=True)
                ind = ind[:dim_p]
                ind_1 = ind[ind < s_1.shape[0]]
                ind_2 = ind[ind >= s_1.shape[0]] - s_1.shape[0]

                # project z1 and z2 into corresponding subspace
                z1_p = torch.matmul(z1 - mu_1, v_1[:,ind_1])
                z2_p = torch.matmul(z2 - mu_2, v_2[:,ind_2])

                # concatenate to form full z
                # z_o = torch.cat((z1_p,z2_p), 1)

                # project back the latent to full dim
                z1_b =  torch.matmul(z1_p, v_1[:,ind_1].T) + mu_1
                z2_b =  torch.matmul(z2_p, v_2[:,ind_2].T) + mu_2
                z_sample = torch.cat((z1_b,z2_b),1)

            obs_dec = self.dec(z_sample)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### Normalize
            z_sample = z_sample - z_sample.mean(dim=0)

            ### nuclear loss 
            z_sample = z_sample / torch.norm(z_sample, p=2)
            nuc_loss = torch.norm(z_sample, p='nuc', dim=(0, 1)) / batch_size

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), nuc_loss, 0, 0, psnr


class E1D1(nn.Module):
    def __init__(self, obs_shape: tuple, z_dim: int, norm_sample: bool=True, num_layers=3, num_filters=64, n_hidden_layers=2, hidden_size=128): # noise=0.01):
        super().__init__()
        self.enc = CNNEncoder(obs_shape, z_dim, num_layers, num_filters, n_hidden_layers, hidden_size)
        self.dec = CNNDecoder(z_dim, (obs_shape[0], obs_shape[1], obs_shape[2]), num_layers, num_filters, n_hidden_layers, hidden_size)
        self.norm_sample = norm_sample

    def forward(self, obs):
        z1, _ = self.enc(obs)

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1.shape[1] // 2
            batch_size = z1.shape[0]
            z1_private = z1[:, :num_features]
            z1_share = z1[:, num_features:]

            ### decode 
            z_sample = torch.cat((z1_private, z1_share), dim=1)
            obs_dec = self.dec(z_sample)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### Normalize
            z_sample = z_sample - z_sample.mean(dim=0)

            ### nuclear loss 
            z_sample = z_sample / torch.norm(z_sample, p=2)
            nuc_loss = torch.norm(z_sample, p='nuc', dim=(0, 1)) / batch_size

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), nuc_loss, 0, 0, psnr


class ResE2D1NonSym(nn.Module):
    def __init__(self, size1: tuple, size2: tuple, z_dim1: int, z_dim2: int, norm_sample:bool=True, n_samples: int=4, n_res_blocks: int=3):
        super().__init__()
        self.enc1 = ResEncoder(size1, z_dim1, n_downsamples=n_samples, n_res_blocks=n_res_blocks)
        self.enc2 = ResEncoder(size2, z_dim2, n_downsamples=n_samples, n_res_blocks=n_res_blocks)
        self.dec = ResDecoder((size2[0], size2[-1], size2[-1]), (z_dim1 + z_dim2), n_upsamples=n_samples, n_res_blocks=n_res_blocks)
        self.norm_sample = norm_sample

    def forward(self, obs1, obs2, obs, random_bottle_neck=False):
        z1, _ = self.enc1(obs1)
        z2, _ = self.enc2(obs2)

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1.shape[1] + z2.shape[1]
            batch_size = z1.shape[0]

            ### decode 
            z_sample = torch.cat((z1, z2), dim=1)
            if random_bottle_neck:
                # reduce the dimensionality of the data using dpca
                # use PCA to reduce dimension
                dim_p = torch.randint(1, num_features, (1,)).item()

                s_1, v_1, mu_1 = data_pca(z1)
                s_2, v_2, mu_2 = data_pca(z2)

                # pick the indices of the top 1/2 singular values for s_1 and s_2 combined
                s_1_2 = torch.cat((s_1,s_2), 0)
                ind = torch.argsort(s_1_2,descending=True)
                ind = ind[:dim_p]
                ind_1 = ind[ind < s_1.shape[0]]
                ind_2 = ind[ind >= s_1.shape[0]] - s_1.shape[0]

                # project z1 and z2 into corresponding subspace
                z1_p = torch.matmul(z1 - mu_1, v_1[:,ind_1])
                z2_p = torch.matmul(z2 - mu_2, v_2[:,ind_2])

                # concatenate to form full z
                # z_o = torch.cat((z1_p,z2_p), 1)

                # project back the latent to full dim
                z1_b =  torch.matmul(z1_p, v_1[:,ind_1].T) + mu_1
                z2_b =  torch.matmul(z2_p, v_2[:,ind_2].T) + mu_2
                z_sample = torch.cat((z1_b,z2_b),1)

            obs_dec = self.dec(z_sample)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### Normalize
            z_sample = z_sample - z_sample.mean(dim=0)

            ### nuclear loss 
            z_sample = z_sample / torch.norm(z_sample, p=2)
            nuc_loss = torch.norm(z_sample, p='nuc', dim=(0, 1)) / batch_size
            

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), nuc_loss, 0, 0, psnr


class ResE2D1(nn.Module):
    def __init__(self, obs_shape1: tuple, obs_shape2: tuple, z_dim1: int, z_dim2: int, norm_sample:bool=True, n_samples: int=4, n_res_blocks: int=3):
        super().__init__()
        self.enc1 = ResEncoder(obs_shape1, z_dim1, n_downsamples=n_samples, n_res_blocks=n_res_blocks)
        self.enc2 = ResEncoder(obs_shape2, z_dim2, n_downsamples=n_samples, n_res_blocks=n_res_blocks)
        self.dec = ResDecoder((obs_shape1[0] + obs_shape2[0], obs_shape1[1], obs_shape1[2]), (z_dim1 + z_dim2),
                                n_upsamples=n_samples, n_res_blocks=n_res_blocks)
        self.norm_sample = norm_sample

    def forward(self, obs1, obs2, random_bottle_neck=False):
        z1, _ = self.enc1(obs1)
        z2, _ = self.enc2(obs2)
        obs = torch.cat((obs1, obs2), dim=1)

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1.shape[1] + z2.shape[1]
            batch_size = z1.shape[0]

            ### decode 
            z_sample = torch.cat((z1, z2), dim=1)
            if random_bottle_neck:
                # reduce the dimensionality of the data using dpca
                # use PCA to reduce dimension
                dim_p = torch.randint(1, num_features, (1,)).item()

                s_1, v_1, mu_1 = data_pca(z1)
                s_2, v_2, mu_2 = data_pca(z2)

                # pick the indices of the top 1/2 singular values for s_1 and s_2 combined
                s_1_2 = torch.cat((s_1,s_2), 0)
                ind = torch.argsort(s_1_2,descending=True)
                ind = ind[:dim_p]
                ind_1 = ind[ind < s_1.shape[0]]
                ind_2 = ind[ind >= s_1.shape[0]] - s_1.shape[0]

                # project z1 and z2 into corresponding subspace
                z1_p = torch.matmul(z1 - mu_1, v_1[:,ind_1])
                z2_p = torch.matmul(z2 - mu_2, v_2[:,ind_2])

                # concatenate to form full z
                # z_o = torch.cat((z1_p,z2_p), 1)

                # project back the latent to full dim
                z1_b =  torch.matmul(z1_p, v_1[:,ind_1].T) + mu_1
                z2_b =  torch.matmul(z2_p, v_2[:,ind_2].T) + mu_2
                z_sample = torch.cat((z1_b,z2_b),1)

            obs_dec = self.dec(z_sample)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### Normalize
            z_sample = z_sample - z_sample.mean(dim=0)

            ### nuclear loss 
            z_sample = z_sample / torch.norm(z_sample, p=2)
            nuc_loss = torch.norm(z_sample, p='nuc', dim=(0, 1)) / batch_size

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), nuc_loss, 0, 0, psnr


class ResE1D1(nn.Module):
    def __init__(self, obs_shape: tuple, z_dim: int, norm_sample: bool=True, n_samples: int=4, n_res_blocks: int=3): # noise=0.01):
        super().__init__()
        self.enc = ResEncoder(obs_shape, z_dim, n_downsamples=n_samples, n_res_blocks=n_res_blocks)
        self.dec = ResDecoder(obs_shape, z_dim, n_upsamples=n_samples, n_res_blocks=n_res_blocks)
        self.norm_sample = norm_sample

    def forward(self, obs):
        z1, _ = self.enc(obs)

        if self.norm_sample:
            raise NotImplementedError
        else:
            ### Not using the normal distribution samples, instead using the variant, invariant, and covariant
            ### leave log_std unused. 
            num_features = z1.shape[1] // 2
            batch_size = z1.shape[0]
            z1_private = z1[:, :num_features]
            z1_share = z1[:, num_features:]

            ### decode 
            z_sample = torch.cat((z1_private, z1_share), dim=1)
            obs_dec = self.dec(z_sample)
            mse = 0.5 * torch.mean((obs - obs_dec) ** 2, dim=(1, 2, 3))
            psnr = PSNR(obs_dec, obs)

            ### Normalize
            z_sample = z_sample - z_sample.mean(dim=0)

            ### nuclear loss 
            z_sample = z_sample / torch.norm(z_sample, p=2)
            nuc_loss = torch.norm(z_sample, p='nuc', dim=(0, 1)) / batch_size

            ### weight parameters recommended by VIC paper: 25, 25, and 10
            return obs_dec, torch.mean(mse), nuc_loss, 0, 0, psnr


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


