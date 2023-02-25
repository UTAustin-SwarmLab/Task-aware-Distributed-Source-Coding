import env_wrapper
from pathlib import Path
import numpy as np
import torch
import random
import sys
import csv

from curl_sac import Actor
from DPCA_torch import *

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))              
from dtac.gym_fetch.ClassAE import *

env_name = 'FetchPickAndPlace' # FetchPickAndPlace FetchReach
if env_name == 'FetchPickAndPlace':
    change_model = True
else:
    change_model = False
pick = "" # ""
cameras = [8, 10]
image_cropped_size = 84 #84 128
original_image_size = 100 #100 128

eval_env = env_wrapper.make(
    domain_name=env_name + '-v1',
    task_name=None,
    seed=1,
    visualize_reward=False,
    from_pixels=True,
    cameras=cameras,
    height=original_image_size,
    width=original_image_size,
    change_model=change_model)
print("Env name is", env_name)

def center_crop_image(image, output_size=image_cropped_size):
    h, w = image.shape[-2:]
    if h > output_size: #center cropping
        new_h, new_w = output_size, output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        if len(image.shape) == 3:
            image = image[:, top:top + new_h, left:left + new_w]
        elif len(image.shape) == 4:
            image = image[:, :, top:top + new_h, left:left + new_w]
        else:
            raise ValueError("image should be 3 or 4 dimensional")
        return image
    else: #center translate
        shift = output_size - h
        shift = shift // 2
        if len(image.shape) == 3:
            new_image = np.zeros((image.shape[0], output_size, output_size))
            new_image[:, shift:shift + h, shift:shift+w] = image
        elif len(image.shape) == 4:
            new_image = np.zeros((image.shape[0], image.shape[1], output_size, output_size))
            new_image[:, :, shift:shift + h, shift:shift+w] = image
        return new_image


def encode_and_decode(obs, VAE, dpca, dpca_dim:int=0):
    obs_tensor = torch.tensor(obs).to(device).float().unsqueeze(0) / 255
    if vae_model == "SVAE":
        if dpca is not None:
            raise NotImplementedError
        else:
            obs_rec = VAE(obs_tensor)[0][0, :, :, :].clip(0, 1)
    elif vae_model == "CNNBasedVAE":
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
    return obs_rec

def evaluate(env, policy, VAE, device, dataset, DPCA_tf:bool=False, dpca_dim:int=0, num_episodes=100):
    all_ep_rewards = []

    def run_eval_loop():
        num_successes = 0
        if DPCA_tf:
            dpca, singular_val_vec = DistriburedPCA(VAE, rep_dim=int(z_dim*3/4), device=device, env=dataset)
            ### count importance priority of dimensions
            rep_dims = [0, 0, 0]
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
                    #### input 100x100 image
                    obs_rec = encode_and_decode(obs, VAE, dpca, dpca_dim)
                    obs_rec = center_crop_image(obs_rec)
                else:
                    #### input 84x84 image
                    obs = center_crop_image(obs)
                    obs_rec = encode_and_decode(obs, VAE, dpca, dpca_dim)
                
                ### no VAE
                # obs = center_crop_image(obs)
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

    """python ./fetch_sim/eval_fetch_VAE.py """

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

    DPCA_tf = False # True False
    min_dpca_dim = 2
    max_dpca_dim = 48
    step_dpca_dim = 2
    if DPCA_tf:
        print("Running DPCA.")
    else:
        print("Not running DPCA.")

    device_num = 2
    cropTF = '_nocrop' # '_nocrop' or ''
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    # model_path = './gym_fetch/models/'
    # model_name = f'{pick}actor{cropTF}{view_from}/actor{view_from}-99.pth'
    model_path = ""
    model_name = "/home/pl22767/DistributedTaskAwareCompression/fetch_sim/data/FetchPickAndPlace-v1/sparse-rad_sac-pixel-crop-01-26-FetchPickAndPlace-v1-im84-b128-nu1-change_model-s13618-id94666/model/actor_254000.pt"


    image_cropped_size = 84 # 128 84
    image_orig_size = 100 # 100 128
    z_dim = 64
    vae_path = './gym_fetch/models/'
    dataset = "PickAndPlace" # gym_fetch PickAndPlace

    beta_rec = 5000.0 # 98304.0 10000.0
    batch_size = 128
    beta_kl = 25.0 # 1.0 25.0
    vae_model = "CNNBasedVAE" # "SVAE" or "CNNBasedVAE"
    weight_cross_penalty = 10.0
    task_weight = 500.0 # task aware
    VAEepoch = 2999
    norm_sample = False # False True
    VAEcrop = '_True' # '_True' or '' or '_False'
    crop_first = True # False True
    vae_name = f'{dataset}_{z_dim}_aware{norm_sample}{VAEcrop}{vae_model}_{beta_kl}_{beta_rec}_{task_weight}_{batch_size}_{weight_cross_penalty}/DVAE_awa-{VAEepoch}.pth'
    print("VAE is", vae_name)

    ### Load policy network here
    if vae_model == 'SVAE':
        dvae_model = SoftIntroVAE(arch="dist", cdim=6, zdim=z_dim, image_size=image_orig_size, norm_sample=norm_sample).to(device)
    elif vae_model == 'CNNBasedVAE':
        if not crop_first:
            dvae_model = E2D1((3, image_orig_size, image_orig_size), (3, image_orig_size, image_orig_size), int(z_dim/2), int(z_dim/2), norm_sample=norm_sample).to(device)
        else:
            dvae_model = E2D1((3, image_cropped_size, image_cropped_size), (3, image_cropped_size, image_cropped_size), int(z_dim/2), int(z_dim/2), norm_sample=norm_sample).to(device)

    dvae_model.load_state_dict(torch.load(vae_path + vae_name))
    act_model = Actor((channel, image_cropped_size, image_cropped_size), (4,), 1024, 'pixel', 50, -10, 2, 4, 32, None, False).to(device)
    act_model.load_state_dict(torch.load(model_path + model_name))

    eval_results = []
    for dpca_dim in range(min_dpca_dim, max_dpca_dim+1, step_dpca_dim):
        mean_ep_reward, best_ep_reward, std_ep_reward, success_rate, rep_dims = evaluate(eval_env, act_model, dvae_model, device, dataset, DPCA_tf, dpca_dim)
        eval_results.append([dpca_dim, mean_ep_reward, best_ep_reward, std_ep_reward, success_rate, rep_dims[0], rep_dims[1], rep_dims[2]])

    header = ['dpca_dim', 'mean_ep_reward', 'best_ep_reward', 'std_ep_reward', 'success_rate', 'dim of z1 private', 'dim of z1 share', 'dim of z2 private']
    csv_name = vae_name.replace('.pth', '.csv').replace('/DVAE', '_DVAE')
    with open('/home/pl22767/DistributedTaskAwareCompression/fetch_sim/csv_data/' + csv_name, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(header)
        writer.writerows(eval_results)
    