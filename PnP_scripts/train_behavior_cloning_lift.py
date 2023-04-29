#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import random
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dtac.gym_fetch.behavior_cloning_agent import ImageBasedRLAgent
from dtac.gym_fetch.utils import center_crop_image
import gym

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
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    pick = torch.load('./lift_hardcode.pt')

    # 10k examples of 'obs', 'next_obs', 'action', 'reward', 'done' 
    # 'obs', 'next_obs': type <class 'numpy.ndarray'> shape (10000, 6, 128, 128)
    # 'action' type <class 'numpy.ndarray'> shape (10000, 4)
    # 'reward' type <class 'numpy.ndarray'> shape (10000, 1)
    # 'done' type <class 'numpy.ndarray'> shape (10000, 1) float32 0.0 (done) or 1.0 (not done)
    obs = pick[0]
    action = pick[2]

    lr = 1e-4
    batch = 128
    epoch = 1000
    device_num = 7
    image_cropped_size = 112
    num_episodes = 100

    ### Input: 2 images. Output: action 4 dim
    model_path = f'./models/lift_actor_nocrop2image_sac_lr{lr}/'
    LOG_DIR = f'./summary/lift_actor_nocrop2image_sac_lr{lr}/'
    summary_writer = SummaryWriter(os.path.join(LOG_DIR, 'tb'))
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    actor2 = ImageBasedRLAgent(arch='joint', zdim=action.shape[-1], image_size=image_cropped_size, channels=(256, 128, 64, 32)).to(device)
    actor2.train()
    optimizer = optim.Adam(actor2.parameters(), lr=lr)

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
        if (ep + 1) % 20 == 0 or ep <= 10:
            all_ep_rewards = []
            env = gym.make('Lift-both-v1')
            env.seed(seed)

            num_successes = 0
            for i in range(num_episodes):
                obs = env.reset()
                step = 0

                # save_imgs(obs, step, i)

                done = False
                episode_reward = 0
                episode_success = False
                j = 0
                while not done:
                    #### input 112x112 image
                    obs = center_crop_image(obs, image_cropped_size)
                    obs_rec = torch.tensor(obs).to(device).float().unsqueeze(0) / 255
                    output = actor2(obs_rec).detach().cpu().numpy()[0]
                    mu_pred = output[:4]
                    a_pred = mu_pred
                    obs, reward, done, info = env.step(a_pred)
                    j += 1

                    step += 1
                    # save_imgs(obs, step, i)

                    if info.get('is_success'):
                        episode_success = True
                    episode_reward += reward
                num_successes += episode_success
                all_ep_rewards.append(episode_reward)

            mean_ep_reward = np.mean(all_ep_rewards)
            best_ep_reward = np.max(all_ep_rewards)
            std_ep_reward = np.std(all_ep_rewards)
            success_rate = num_successes / num_episodes
            print(mean_ep_reward, best_ep_reward, std_ep_reward, success_rate)
            summary_writer.add_scalar('Success Rate', success_rate, ep)

    return

if __name__ == '__main__':
    train2image()
