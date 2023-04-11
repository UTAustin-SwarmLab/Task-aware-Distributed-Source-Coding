#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import sys
from torch.utils.tensorboard import SummaryWriter

from dtac.gym_fetch.curl_sac import Actor
from dtac.gym_fetch.behavior_cloning_agent import ImageBasedRLAgent

def random_crop(imgs, out):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    
    return cropped


def train2image():
    dataset_dir = '/store/datasets/gym_fetch/'
    pick = torch.load(dataset_dir + 'pnp_128_20011.pt')

    # 10k examples of 'obs', 'next_obs', 'action', 'reward', 'done' 
    # 'obs', 'next_obs': type <class 'numpy.ndarray'> shape (10000, 6, 128, 128)
    # 'action' type <class 'numpy.ndarray'> shape (10000, 4)
    # 'reward' type <class 'numpy.ndarray'> shape (10000, 1)
    # 'done' type <class 'numpy.ndarray'> shape (10000, 1) float32 0.0 (done) or 1.0 (not done)
    obs = pick[0]
    next_obs = pick[1]
    action = pick[2]
    reward = pick[3]
    done = pick[4]

    lr = 1e-4
    batch = 128
    epoch = 1000
    device_num = 4
    image_cropped_size = 112

    ### Input: 2 images. Output: action 4 dim
    model_path = f'./models/pick_actor_nocrop2image_sac_lr{lr}/'
    LOG_DIR = f'./summary/pick_actor_nocrop2imag_sac_lr{lr}/'
    summary_writer = SummaryWriter(os.path.join(LOG_DIR, 'tb'))
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    actor2 = ImageBasedRLAgent(arch='joint', zdim=action.shape[-1], image_size=image_cropped_size, channels=(256, 128, 64, 32)).to(device)
    actor2.train()
    optimizer = optim.Adam(actor2.parameters(), lr=lr)

    ### load Soft Actor Critic model
    sac_path = "/store/datasets/gym_fetch/pnp_actor_300000.pt"
    sac = Actor((6, image_cropped_size, image_cropped_size), (4,), 1024, 'pixel', 50, -10, 2, 4, 32, None, False).to(device)
    sac.load_state_dict(torch.load(sac_path))
    sac.eval()

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for ep in range(epoch):
        index = np.arange(len(obs))
        np.random.shuffle(index)
        n_batches = len(obs) // batch
        
        for i in range(n_batches):
            b_idx = index[i * batch:(i + 1) * batch]
            o_batch = random_crop(obs[b_idx], image_cropped_size)
            o_batch = torch.tensor(o_batch, device=device).float() / 255
            a_batch = torch.tensor(action[b_idx], device=device)
            
            a_gt = sac(o_batch)[0]
            a_batch = a_gt.detach()
            
            output = actor2(o_batch)
            mu_pred = output[:, :4]
            a_pred = mu_pred # + eps * std_pred
            loss = torch.mean((a_pred - a_batch) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

        ### log tensorboard
        summary_writer.add_scalar('actor loss', loss.item(), ep)

        ### print loss
        print("Epoch: {}, Train Loss: {}".format(ep, loss.item()))

        ### save model
        if (ep + 1) % 50 == 0 or ep <= 10:
            torch.save(actor2.state_dict(), model_path + f'actor2image-{ep}.pth') 

    return

if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    train2image()
