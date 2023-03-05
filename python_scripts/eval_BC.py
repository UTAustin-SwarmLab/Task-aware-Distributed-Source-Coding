from pathlib import Path
import numpy as np
import torch
import random
import argparse
import csv

from dtac.gym_fetch.behavior_cloning_agent import ImageBasedRLAgent
from dtac.gym_fetch.utils import center_crop_image
from dtac.gym_fetch.curl_sac import Actor
from dtac.gym_fetch.DPCA_torch import *
# from dtac.gym_fetch.env_wrapper import env_wrapper
from dtac.gym_fetch.ClassAE import *

import dtac
import gym

env_name = 'FetchPickAndPlace' # FetchPickAndPlace FetchReach
if env_name == 'FetchPickAndPlace':
    change_model = True
else:
    change_model = False
pick = "" # ""
cameras = [8, 10]
cropped_image_size = 112 #84 128
original_image_size = 128 #100 128

def encode_and_decode(obs, VAE, dpca, dpca_dim:int=0):
    obs_tensor = torch.tensor(obs).to(device).float().unsqueeze(0) / 255
    if vae_model == "SVAE":
        if dpca is not None:
            raise NotImplementedError
        else:
            obs_rec = VAE(obs_tensor)[0][0, :, :, :].clip(0, 1)
    elif "Joint" not in vae_model and "BasedVAE" in vae_model:
        obs1 = obs_tensor[:, :3, :, :]
        obs2 = obs_tensor[:, 3:, :, :]
        if dpca is not None:
            z1, _ = VAE.enc1(obs1)
            z2, _ = VAE.enc2(obs2)
            z1 = z1.detach()
            z2 = z2.detach()
            num_features = z1.shape[1] // 2
            batch = z1.shape[0]
            z1_private = z1[:, :num_features]
            z2_private = z2[:, :num_features]
            z1_share = z1[:, num_features:]
            z2_share = z2[:, num_features:]
            z = torch.cat((z1_private, z1_share, z2_private), dim=1)
            
            recon_z = dpca.LinearEncDec(z, dpca_dim=dpca_dim)

            # z_sample = torch.cat((recon_z[:, :num_features], recon_z[:, num_features:2*num_features], recon_z[:, 2*num_features:], recon_z[:, num_features:2*num_features]), dim=1)
            z_sample = torch.cat((recon_z[:, :num_features], recon_z[:, num_features:2*num_features], recon_z[:, 2*num_features:]), dim=1)
            obs_rec = VAE.dec(z_sample).clip(0, 1)
        else:
            obs_rec = VAE(obs1, obs2)[0][:, :, :, :].clip(0, 1)
    elif "Joint" in vae_model:
        if dpca is not None:
            raise NotImplementedError
        else:
            obs_rec = VAE(obs_tensor)[0][:, :, :, :].clip(0, 1)
    return obs_rec

def evaluate(policy, device, dataset, DPCA_tf:bool=False, dpca_dim:int=0, num_episodes=100):
    all_ep_rewards = []

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = gym.make('PNP-both-v1')
    env.seed(seed)

    def run_eval_loop():
        num_successes = 0

        for i in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_success = False
            j = 0
            while not done:
                ### no VAE
                obs = center_crop_image(obs, cropped_image_size)
                obs_rec = torch.tensor(obs).to(device).float().unsqueeze(0) / 255
                output = policy(obs_rec).detach().cpu().numpy()[0]
                mu_pred = output[:4]
                a_pred = mu_pred
                obs, reward, done, info = env.step(a_pred)
                j += 1

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

        return mean_ep_reward, best_ep_reward, std_ep_reward, success_rate, rep_dims
    
    return run_eval_loop()


if __name__ == '__main__':
    """python eval_AE.py -z 64 -l 1e-3 -b 128 -r 10000 -k 25 -t 0 -corpen 10 -s 0 -vae CNNBasedVAE -vae_e 99 -ns False -crop True -dpca 0"""
    ### take the argument
    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument("-c", "--device", type=int, help="device: -1 for cpu, 0 and up for specific cuda device", default=7)
    parser.add_argument("-e", "--epoch", type=int, help="task model eposh", default=99)
    parser.add_argument("-lr", "--lr", type=float, help="learning rate", default=1e-2)
    args = parser.parse_args()

    device_num = args.device
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    model_path = ""
    model_name = "/store/datasets/gym_fetch/pnp_actor_300000.pt"

    cropped_image_size = 112 # 128 84
    image_orig_size = 128 # 100 128
    vae_path = './models/'
    dataset = "PickAndPlace" # gym_fetch PickAndPlace

    act_path = f"./models/pick_actor_nocrop2image_sac_lr{args.lr}/"
    act_name = f'actor2image-{args.epoch}.pth'
    act_model = ImageBasedRLAgent(arch='joint', zdim=4, image_size=cropped_image_size, channels=(256, 128, 64, 32)).to(device)
    act_model.load_state_dict(torch.load(act_path + act_name))
    act_model = act_model.eval()
    mean_ep_reward, best_ep_reward, std_ep_reward, success_rate, rep_dims = evaluate(act_model, device, dataset, DPCA_tf=None)
    