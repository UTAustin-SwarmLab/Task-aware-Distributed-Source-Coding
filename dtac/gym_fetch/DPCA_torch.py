from pathlib import Path
import numpy as np
import torch
import random
import sys

from dtac.gym_fetch.curl_sac import Actor
from dtac.gym_fetch.utils import center_crop_image             
from dtac.gym_fetch.ClassAE import *

def PCA(input_data, device): 
    '''
    input_data: (n, d). n: number of data = batch size, d: dimension
    return singular_val_vec: (s, v). s: singular value, v: singular vector
    '''
    mean = input_data.mean(axis=0)
    assert mean.shape == (input_data.shape[1],)
    norm_data = input_data - mean
    u, s, vh = torch.linalg.svd(norm_data, full_matrices=True)
    u, s, vh = u.to(device), s.to(device), vh.to(device)
    singular_val_vec = []
    v = vh.T
    for i in range(len(s)):
        singular_val_vec.append( (s[i], v[:, i]) )

    return singular_val_vec,  mean

class DPCA_Process():
    def __init__(self, singular_val_vec, mean, num_features:int, device) -> None:
        '''
        singular_val_vec: (s, v, seg). s: singular value, v: singular vector, seg: segment number
        mean: (d,). mean of data.
        num features: number of features in each segment.
        device: cpu or gpu.
        '''
        self.singular_val_vec = singular_val_vec ### (singular value, singular vector, segment#)
        self.mean = mean
        self.num_features = num_features
        self.d = mean.shape[0]
        assert self.d == self.num_features * 3
        self.device = device

        return

    def LinearEncode(self, input_data, dpca_dim:int=0):                
        '''
        input: (n, d)
        self.projs: [(n, d1), (n, d2), (n, d3)]. d1+d2+d3 = dpca_dim
        dpca_dim: number of principal components
        mean: (d,)
        '''
        assert input_data.shape[1] == self.d
        n = input_data.shape[0]

        norm_input = input_data - self.mean
        self.projs = [[], [], []]
        self.proj_vec = [[], [], []]
        for i in range(dpca_dim):
            seg = self.singular_val_vec[i][2]
            start, end = seg * self.num_features, (seg+1) * self.num_features
            self.projs[seg].append(norm_input[:, start:end] @ self.singular_val_vec[i][1])
            self.proj_vec[seg].append(self.singular_val_vec[i][1])
        
        for i in range(len(self.projs)):
            ### if there is no projection vector in this segment
            if len(self.projs[i]) == 0:
                self.projs[i] = torch.zeros((n, self.num_features), device=self.device)
                self.proj_vec[i] = torch.zeros((self.num_features, self.num_features), device=self.device)
            else: 
                self.projs[i] = torch.stack(self.projs[i], dim=1)
                self.proj_vec[i] = torch.stack(self.proj_vec[i], dim=1)

        return

    def LinearDecode(self, dpca_dim:int):
        '''
        output: (n, d). The reconstructed input data
        '''
        assert dpca_dim > 0 and dpca_dim <= self.d

        n = self.projs[0].shape[0]
        output = torch.zeros((n, self.d), device=self.device)
        for i in range(len(self.projs)):
            output[:, i*self.num_features:(i+1)*self.num_features] = torch.matmul(self.projs[i], self.proj_vec[i].T)

        return output + self.mean

    def LinearEncDec(self, input, dpca_dim:int):
        '''
        First do linear encoding, then do linear decoding
        input: (n, d)
        dpca_dim: number of principal components
        output: (n, d). The reconstructed input data
        '''
        self.LinearEncode(input, dpca_dim)
        output = self.LinearDecode(dpca_dim)
        return output


class PCA_Process():
    def __init__(self, singular_val_vec, mean, num_features:int, device) -> None:
        '''
        singular_val_vec: (s, v, seg). s: singular value, v: singular vector, seg: segment number
        mean: (d,). mean of data.
        num features: number of features in each segment.
        device: cpu or gpu.
        '''
        self.singular_val_vec = singular_val_vec ### (singular value, singular vector, segment#)
        self.mean = mean
        self.num_features = num_features
        self.d = mean.shape[0]
        self.device = device

        return

    def LinearEncode(self, input_data, dpca_dim:int=0):                
        '''
        input: (n, d)
        self.projs: [(n, d1), (n, d2), (n, d3)]. d1+d2+d3 = dpca_dim
        dpca_dim: number of principal components
        mean: (d,)
        '''
        assert input_data.shape[1] == self.d
        n = input_data.shape[0]

        norm_input = input_data - self.mean
        self.projs = []
        self.proj_vec = []
        for i in range(dpca_dim):
            seg = self.singular_val_vec[i][2]
            self.projs.append(norm_input[:, :] @ self.singular_val_vec[i][1])
            self.proj_vec.append(self.singular_val_vec[i][1])
        
        self.projs = torch.stack(self.projs, dim=1)
        self.proj_vec = torch.stack(self.proj_vec, dim=1)

        return

    def LinearDecode(self, dpca_dim:int):
        '''
        output: (n, d). The reconstructed input data
        '''
        assert dpca_dim > 0 and dpca_dim <= self.d

        n = self.projs[0].shape[0]
        output = torch.zeros((n, self.d), device=self.device)
        output[:, :] = torch.matmul(self.projs, self.proj_vec.T)

        return output + self.mean

    def LinearEncDec(self, input, dpca_dim:int):
        '''
        First do linear encoding, then do linear decoding
        input: (n, d)
        dpca_dim: number of principal components
        output: (n, d). The reconstructed input data
        '''
        self.LinearEncode(input, dpca_dim)
        output = self.LinearDecode(dpca_dim)
        return output


def DistriburedPCA(dvae_model, rep_dim, device, env='gym_fetch'):
    ### Load dataset
    dataset_dir = '/store/datasets/gym_fetch/'
    if env == 'gym_fetch':
        reach = torch.load(dataset_dir + 'reach.pt')
        obs1 = reach[0][:, 0:3, :, :]
        obs2 = reach[0][:, 3:6, :, :]
    elif env == 'PickAndPlace':
        pick = torch.load(dataset_dir + 'pnp_128_20011.pt')
        ### center crop image
        cropped_image_size = 112
        pick[0] = center_crop_image(pick[0], cropped_image_size)
        obs1 = pick[0][:, 0:3, :, :]
        obs2 = pick[0][:, 3:6, :, :]
    else:
        loader = env

    if env == 'gym_fetch' or env == 'PickAndPlace':
        index = np.arange(len(obs1))
        batch = 100
        n_batches = len(obs1) // batch
        
        Z = torch.zeros((len(obs1), rep_dim), device=device)
        # invar = []

        for i in range(n_batches):
            b_idx = index[i * batch:(i + 1) * batch]
            o1_batch = torch.tensor(obs1[b_idx], device=device).float() / 255
            o2_batch = torch.tensor(obs2[b_idx], device=device).float() / 255

            ### get middle representations
            z1, _ = dvae_model.enc1(o1_batch)
            z2, _ = dvae_model.enc2(o2_batch)
            z1 = z1.detach()
            z2 = z2.detach()
            num_features = z1.shape[1] // 2
            batch = z1.shape[0]
            z1_private = z1[:, :num_features]
            z2_private = z2[:, :num_features]
            z1_share = z1[:, num_features:]
            z2_share = z2[:, num_features:]

            ### collect private and share representations
            ### concatenate representations
            z = torch.cat((z1_private, z1_share, z2_private), dim=1)
            Z[b_idx, :] = z

        mean = Z.mean(axis=0)
    else:
        data_point_num = loader.dataset.__len__()
        flag = 0
        Z = torch.zeros((data_point_num, rep_dim), device=device)

        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(device).type(torch.cuda.FloatTensor) / 255.0
            labels = labels.to(device)

            ### encode data
            with torch.no_grad():
                x1 = torch.zeros(x.shape[0], 3, 112, 112).to(device)
                x2 = torch.zeros(x.shape[0], 3, 112, 112).to(device)
                x1[:, :, :64, :112] = x[:, :, :64, :112]
                x2[:, :, 64:, :112] = x[:, :, 64:, :112]
                z1, _ = dvae_model.enc1(x1)
                z2, _ = dvae_model.enc2(x2)
                z1 = z1.detach()
                z2 = z2.detach()
                num_features = z1.shape[1] // 2
                batch = z1.shape[0]
                z1_private = z1[:, :num_features]
                z2_private = z2[:, :num_features]
                z1_share = z1[:, num_features:]
                z2_share = z2[:, num_features:]

                z = torch.cat((z1_private, z1_share, z2_private), dim=1)
                Z[flag:flag+batch, :] = z
            flag = flag + batch
        
        mean = Z.mean(axis=0)


    ### PCA for each segment
    singular_val_vec = []
    for seg in range(3): ### 3 segments: private 1, share, private 2
        start, end = seg * int(rep_dim/3), (seg+1) * int(rep_dim/3)
        seg_singular_val_vec, _ = PCA(Z[:, start:end], device)
        for s, v in seg_singular_val_vec:
            singular_val_vec.append( (s, v, seg) )
    singular_val_vec.sort(key=lambda x: x[0], reverse=True)

    dpca = DPCA_Process(singular_val_vec, mean, int(rep_dim/3), device)
    return dpca, singular_val_vec


def JointPCA(dvae_model, rep_dim, device, env='gym_fetch'):
    ### Load dataset
    dataset_dir = '/store/datasets/gym_fetch/'
    if env == 'gym_fetch':
        reach = torch.load(dataset_dir + 'reach.pt')
        obs1 = reach[0][:, 0:3, :, :]
        obs2 = reach[0][:, 3:6, :, :]
    elif env == 'PickAndPlace':
        pick = torch.load(dataset_dir + 'pnp_128_20011.pt')
        ### center crop image
        cropped_image_size = 112
        pick[0] = center_crop_image(pick[0], cropped_image_size)
        obs1 = pick[0][:, 0:3, :, :]
        obs2 = pick[0][:, 3:6, :, :]
    else:
        raise NotImplementedError

    index = np.arange(len(obs1))
    batch = 100
    n_batches = len(obs1) // batch
    
    Z = torch.zeros((len(obs1), rep_dim), device=device)
    # invar = []

    for i in range(n_batches):
        b_idx = index[i * batch:(i + 1) * batch]
        o1_batch = torch.tensor(obs1[b_idx], device=device).float() / 255
        o2_batch = torch.tensor(obs2[b_idx], device=device).float() / 255

        ### get middle representations
        obs = torch.cat((o1_batch, o2_batch), dim=1)
        z, _ = dvae_model.enc(obs)
        z = z.detach()
        num_features = z.shape[1]
        batch = z.shape[0]

        ### collect private and share representations
        ### concatenate representations
        Z[b_idx, :] = z

    mean = Z.mean(axis=0)

    ### PCA for each segment
    singular_val_vec = []
    seg_singular_val_vec, _ = PCA(Z, device)
    for s, v in seg_singular_val_vec:
        singular_val_vec.append( (s, v, 0) )
    singular_val_vec.sort(key=lambda x: x[0], reverse=True)

    dpca = PCA_Process(singular_val_vec, mean, rep_dim, device)
    return dpca, singular_val_vec


if __name__ == '__main__':
    ### Set the random seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print("random seed: ", seed)

    view_from = '2image' # '2image' or '_side' or '_arm'
    view, channel = 1, 3
    if view_from == '2image':
        view, channel = 2, 6

    device_num = 6
    cropTF = '_nocrop' # '_nocrop' or ''
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    model_path = './gym_fetch/models/'
    model_name = f'actor{cropTF}{view_from}/actor{view_from}-99.pth'

    image_cropped_size = 128 # 112
    image_orig_size = 128
    z_dim = 64
    vae_path = './gym_fetch/models/'

    beta_rec = 0.0 # 98304.0 10000.0
    batch_size = 128
    beta_kl = 25.0 # 1.0 25.0
    vae_model = "CNNBasedVAE" # "SVAE" or "CNNBasedVAE"
    weight_cross_penalty = 100.0
    task_weight = 100.0 # task aware
    VAEepoch = 99
    norm_sample = False # False True
    vae_name = f'gym_fetch_{z_dim}_aware{norm_sample}{vae_model}_{beta_kl}_{beta_rec}_{task_weight}_{batch_size}_{weight_cross_penalty}/DVAE_awa-{VAEepoch}.pth'

    ### Load policy network
    if vae_model == 'CNNBasedVAE':
        dvae_model = E2D1((3,128,128), (3,128,128), int(z_dim/2), int(z_dim/2), norm_sample=norm_sample).to(device)

    dvae_model.load_state_dict(torch.load(vae_path + vae_name))
    dvae_model.eval()
    act_model = Actor((channel, image_cropped_size, image_cropped_size), (4,), 1024, 'pixel', 50, -10, 2, 4, 32, None, False).to(device)
    act_model.load_state_dict(torch.load(model_path + model_name))
    act_model.eval()
    DistriburedPCA(dvae_model, rep_dim=int(z_dim/4*3), device=device)