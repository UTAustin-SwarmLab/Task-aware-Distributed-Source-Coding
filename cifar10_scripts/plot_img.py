import torch
import numpy as np
import os
import torchvision.utils as vutils
import random
import torch.optim as optim
from torchvision import datasets, transforms
### to start tensorboard:  tensorboard --logdir=./cifar10_scripts/summary --port=6006
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm

from dtac.ClassDAE import *


def plot(seed=20, dataset="cifar10", device=7, batch_size=50):
    ### Set the random seed
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print("random seed: ", seed)

    device = torch.device("cpu") if device <= -1 else torch.device("cuda:" + str(device))


    ### Load the dataset
    if dataset == "cifar10":
        image_size = 32
        p = 0.15 ### probability augmentation (combined with CNN-based VAE)
        transform_train=transforms.Compose([
            transforms.ToTensor(),
            ])
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            ])
        train_kwargs = {'batch_size': batch_size}
        test_kwargs = {'batch_size': batch_size}
        trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)
    else:
        raise NotImplementedError
    
    fig_dir = f'./figures/Plot_{dataset}_seed{seed}'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    cur_iter = 0
    for batch_idx, (obs, out) in enumerate(tqdm(train_loader)):
        obs, out = obs.to(device), out.to(device)
        assert obs.min() >= 0 and obs.max() <= 1
        
        ### add Gaussian noise to images
        obs1 = obs + torch.randn_like(obs) * 0.1
        obs2 = obs + torch.randn_like(obs) * 1

        for idx in range(batch_size):
            img = torch.cat((obs1[idx].unsqueeze(0), obs2[idx].unsqueeze(0)), dim=0)
            vutils.save_image(img, f'{fig_dir}/image_{batch_idx}_{idx}.jpg', nrow=2)

        return

if __name__ == "__main__":
    """        
    python plot_img.py
    """
    plot()