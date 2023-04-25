from pathlib import Path
import numpy as np
import torch
import random
import argparse
import csv

# from dtac.gym_fetch.behavior_cloning_agent import ImageBasedRLAgent
from dtac.gym_fetch.utils import center_crop_image
from dtac.gym_fetch.curl_sac import Actor
from dtac.DPCA_torch import *
# from dtac.ClassAE import *
from dtac.ClassDAE import *

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


def encode_and_decodeEQ(obs, VAE, dpca, device, vae_model, dpca_dim:int=0):
    obs_tensor = torch.tensor(obs).to(device).float().unsqueeze(0) / 255
    if "Joint" not in vae_model and "BasedVAE" in vae_model:
        obs1 = obs_tensor[:, :3, :, :]
        obs2 = obs_tensor[:, 3:, :, :]
        if dpca is not None:
            z1, _ = VAE.enc1(obs1)
            z2, _ = VAE.enc2(obs2)
            z1 = z1.detach()
            z2 = z2.detach()
            z = torch.cat((z1, z2), dim=1)
            recon_z = dpca.LinearEncDec(z, dpca_dim=dpca_dim)
            z_sample = recon_z
            obs_rec = VAE.dec(z_sample).clip(0, 1)
        else:
            obs_rec = VAE(obs1, obs2)[0][:, :, :, :].clip(0, 1)
    elif "Joint" in vae_model:
        if dpca is not None:
            z, _ = VAE.enc(obs_tensor)
            z = z.detach()
            recon_z = dpca.LinearEncDec(z, dpca_dim=dpca_dim)
            z_sample = recon_z
            obs_rec = VAE.dec(z_sample).clip(0, 1)
        else:
            obs_rec = VAE(obs_tensor)[0][:, :, :, :].clip(0, 1)
    return obs_rec

def evaluate(policy, VAE, device, dataset, vae_model, DPCA_tf:bool=False, dpca_dim:int=0, 
             num_episodes=100, view_from='2image', crop_first=True):
    all_ep_rewards = []

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = gym.make('PNP-both-v1')
    # e = gym.make('PNP-side-v1')
    # e = gym.make('PNP-hand-v1')
    env.seed(seed)

    def run_eval_loop():
        num_successes = 0
        rep_dims = [0, 0, 0]
        if DPCA_tf:
            if "Joint" not in vae_model:
                # dpca, singular_val_vec = DistriburedPCA(VAE, rep_dim=int(z_dim*3/4), device=device, env=dataset)
                dpca, singular_val_vec = DistriburedPCAEQ(VAE, rep_dim=z_dim, device=device, env=dataset)
            else:
                dpca, singular_val_vec = JointPCA(VAE, rep_dim=z_dim, device=device, env=dataset)
            ### count importance priority of dimensions
            print(dpca_dim)
            for i in range(dpca_dim):
                seg = singular_val_vec[i][2]
                rep_dims[seg] += 1
        else:
            dpca = None

        for i in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_success = False
            j = 0
            while not done:
                ### with VAE
                if not crop_first:
                    #### input 128x128 image
                    obs_rec = encode_and_decodeEQ(obs, VAE, dpca, dpca_dim)
                    obs_rec = center_crop_image(obs_rec, cropped_image_size)
                else:
                    #### input 112x112 image
                    obs = center_crop_image(obs, cropped_image_size)
                    obs_rec = encode_and_decodeEQ(obs, VAE, dpca, device, vae_model, dpca_dim)
                
                ### no VAE
                # obs = center_crop_image(obs, cropped_image_size)
                # obs_rec = torch.tensor(obs).to(device).float().unsqueeze(0) / 255

                ### Parse image to get the side view or both view
                if view_from != '2image':
                    if view_from == '_side':
                        obs_rec = obs_rec[:3, :, :]
                    elif view_from == '_arm':
                        obs_rec = obs_rec[3:, :, :]

                output = policy(obs_rec)[0].detach().cpu().numpy()[0]
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

    """
    python eval_DAE.py -z 64 -l 1e-3 -b 128 -r 10000 -k 25 -t 0 -corpen 10 -s 0 -vae CNNBasedVAE -vae_e 99 -ns False -crop True -dpca 0 -device 7
    """

    ### take the argument
    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument("-z", "--z_dim", type=int, help="latent dimensions", default=64)
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=1e-4)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=128)
    parser.add_argument("-r", "--beta_rec", type=float, help="beta coefficient for the reconstruction loss", default=0.0)
    parser.add_argument("-k", "--beta_kl", type=float, help="beta coefficient for the kl divergence", default=25)
    parser.add_argument("-t", "--beta_task", type=float, help="beta coefficient for the task loss", default=0)
    parser.add_argument("-corpen", "--cross_penalty", type=float, help="cross-correlation penalty", default=10)
    parser.add_argument("-s", "--seed", type=int, help="seed", default=0)
    parser.add_argument("-c", "--device", type=int, help="device: -1 for cpu, 0 and up for specific cuda device", default=7)
    parser.add_argument("-vae", "--vae_model", type=str, help="vae model: CNNBasedVAE or SVAE", default="CNNBasedVAE")
    parser.add_argument("-vae_e", "--vae_epoch", type=int, help="task model eposh", default=99)
    parser.add_argument("-ns", "--norm_sample", type=bool, help="Sample from Normal distribution (VAE) or not", default=False)
    parser.add_argument("-crop", "--rand_crop", type=bool, help="randomly crop images", default=True)
    parser.add_argument("-dpca", "--dpca", type=int, help="DPCA or not", default=False)
    args = parser.parse_args()

    view_from = '2image' # '2image' or '_side' or '_arm'
    view, channel = 1, 3
    if view_from == '2image':
        view, channel = 2, 6

    if args.norm_sample == 'True' or args.norm_sample == 'true':
        args.norm_sample = True
    else:
        args.norm_sample = False

    device_num = args.device
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    model_path = ""
    # model_name = "./gym_fetch/PickAndPlaceActor/actor_254000.pt"
    model_name = "/store/datasets/gym_fetch/pnp_actor_300000.pt"

    DPCA_tf = args.dpca # True False
    min_dpca_dim = 4
    max_dpca_dim = 48
    step_dpca_dim = 4
    if DPCA_tf:
        print("Running DPCA.")
    else:
        print("Not running DPCA.")

    z_dim = args.z_dim
    beta_rec = args.beta_rec # 98304.0 10000.0
    beta_kl = args.beta_kl # 1.0 25.0
    vae_model = args.vae_model # "CNNBasedVAE" or "ResBasedVAE" or "JointResBasedVAE" or "JointCNNBasedVAE"
    weight_cross_penalty = args.cross_penalty
    beta_task = args.beta_task # task aware
    VAEepoch = args.vae_epoch
    lr = args.lr
    VAE_seed = args.seed
    
    cropped_image_size = 112 # 128 84
    image_orig_size = 128 # 100 128
    vae_path = './models/'
    dataset = "PickAndPlace" # gym_fetch PickAndPlace
    batch_size = args.batch_size # 128
    norm_sample = bool(args.norm_sample) # False True
    VAEcrop = '_True' # '_True' or '' or '_False'
    crop_first = True # False True
    rand_crop = bool(args.rand_crop) # True False

    if norm_sample:
        model_type = "DVAE"
    else:
        model_type = "DAE"
    if rand_crop:
        rc = "randcrop"
    else:
        rc = "nocrop"
    # if "Joint" in vae_model:
    #     rc = "NoPCA_" + rc
    # vae_name = f'{dataset}_{z_dim}_taskaware_{model_type}_{rc}_{vae_model}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{VAE_seed}/DVAE_awa-{VAEepoch}.pth'
    vae_name = f'{dataset}_{z_dim}_randPCA_8_48_{model_type}_{rc}_{vae_model}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{VAE_seed}/DVAE_awa-{VAEepoch}.pth'
    print("VAE is", vae_name)

    ### Load policy network here
    if vae_model == 'CNNBasedVAE':
        nn_complexity = 0
        dvae_model = E2D1((3, cropped_image_size, cropped_image_size), (3, cropped_image_size, cropped_image_size), int(z_dim/2), int(z_dim/2), norm_sample, 4-nn_complexity, int(128/(nn_complexity+1)), 2, 128).to(device)
    elif vae_model == 'ResBasedVAE':
        nn_complexity = 2
        dvae_model = ResE2D1((3, cropped_image_size, cropped_image_size), (3, cropped_image_size, cropped_image_size), int(z_dim/2), int(z_dim/2), norm_sample, 4, 1).to(device)
    elif vae_model == 'JointCNNBasedVAE':
        nn_complexity = 0
        # dvae_model = E1D1((6, cropped_image_size, cropped_image_size), z_dim, norm_sample, 3, 64, 2, 128).to(device)
        dvae_model = E1D1((6, cropped_image_size, cropped_image_size), z_dim, norm_sample, 4-nn_complexity, int(128/(nn_complexity+1)), 2, 128).to(device)
    elif vae_model == 'JointResBasedVAE':
        nn_complexity = 2
        dvae_model = ResE1D1((6, cropped_image_size, cropped_image_size), z_dim, norm_sample, 4, 1).to(device)

    dvae_model.load_state_dict(torch.load(vae_path + vae_name))
    dvae_model.eval()
    act_model = Actor((channel, cropped_image_size, cropped_image_size), (4,), 1024, 'pixel', 50, -10, 2, 4, 32, None, False).to(device)
    act_model.load_state_dict(torch.load(model_path + model_name))
    act_model.eval()

    if DPCA_tf:
        eval_results = []
        for dpca_dim in range(max_dpca_dim, min_dpca_dim-1, -step_dpca_dim):
            mean_ep_reward, best_ep_reward, std_ep_reward, success_rate, rep_dims = evaluate(act_model, dvae_model, device, dataset, vae_model, DPCA_tf, dpca_dim)
            eval_results.append([dpca_dim, mean_ep_reward, best_ep_reward, std_ep_reward, success_rate, rep_dims[0], rep_dims[1], rep_dims[2]])

        header = ['dpca_dim', 'mean_ep_reward', 'best_ep_reward', 'std_ep_reward', 'success_rate', 'dim of z1 private', 'dim of z1 share', 'dim of z2 private']
        csv_name = vae_name.replace('.pth', '.csv').replace('/DVAE', '_DVAE')
        with open('../csv_data/' + csv_name, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(header)
            writer.writerows(eval_results)
    else:
        mean_ep_reward, best_ep_reward, std_ep_reward, success_rate, rep_dims = evaluate(act_model, dvae_model, device, dataset, DPCA_tf)
