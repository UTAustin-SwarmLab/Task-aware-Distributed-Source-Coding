import torch
import argparse
import random
import numpy as np
import os

from dtac.ClassDAE import *
from dtac.gym_fetch.curl_sac import Actor
from dtac.gym_fetch.utils import center_crop_image, random_crop_image
from dtac.covar import cov_func


def train_awa_vae(dataset="gym_fetch", z_dim=64, batch_size=32, num_epochs=250, beta_kl=1.0, beta_rec=0.0, beta_task=1.0, weight_cross_penalty=0.1, device=0, save_interval=5,
                  lr=2e-4, seed=0, model_path=None, dataset_dir=None, vae_model="CNNBasedVAE", norm_sample=True, rand_crop=False, randpca=False, data_seed=20):
    ### set paths
    if norm_sample:
        model_type = "DVAE"
    else:
        model_type = "DAE"
    if rand_crop:
        rc = "randcrop"
    else:
        rc = "nocrop"

    fig_dir = f'./figures/Plot_{dataset}_{z_dim}_randPCA_8_48_{model_type}_{rc}_{vae_model}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    model_path += f'{dataset}_{z_dim}_randPCA_8_48_{model_type}_{rc}_{vae_model}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    if not randpca:
        fig_dir = fig_dir.replace("randPCA", "NoPCA")
        model_path = model_path.replace("randPCA", "NoPCA")

    ### Set the random seed
    if data_seed != -1:
        random.seed(data_seed)
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        torch.cuda.manual_seed(data_seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", data_seed)
    device = torch.device("cpu") if args.device <= -1 else torch.device("cuda:" + str(args.device))

    ### Load the dataset
    if dataset == "gym_fetch":
        # 10k examples of 'obs', 'next_obs', 'action', 'reward', 'done' 
        # 'obs', 'next_obs': type <class 'numpy.ndarray'> shape (10000, 6, 128, 128)
        # 'action' type <class 'numpy.ndarray'> shape (10000, 4)
        # 'reward' type <class 'numpy.ndarray'> shape (10000, 1)
        # 'done' type <class 'numpy.ndarray'> shape (10000, 1) float32 0.0 (done) or 1.0 (not done)
        reach = torch.load(dataset_dir + 'reach.pt')
        obs1 = reach[0][:, 0:3, :, :]
        obs2 = reach[0][:, 3:6, :, :]
        action = reach[2]
        cropped_image_size = 128
    elif dataset == "PickAndPlace":
        pick = torch.load(dataset_dir + 'pnp_128_20011.pt')
        
        pick[2] = torch.tensor(pick[2], dtype=torch.float32)
        unique, idx, counts = torch.unique(pick[2], dim=0, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]

        obs1 = pick[0][first_indicies, 0:3, :, :]
        obs2 = pick[0][first_indicies, 3:6, :, :]
        a_gt = pick[2][first_indicies, :]
        cropped_image_size = 112
        task_model_path = "/store/datasets/gym_fetch/pnp_actor_300000.pt"
        ### load task model
        task_model = Actor((6, cropped_image_size, cropped_image_size), (4,), 1024, 'pixel', 50, -10, 2, 4, 32, None, False).to(device)
        task_model.load_state_dict(torch.load(task_model_path))
        task_model.eval()
    elif dataset == "Lift":
        pick = torch.load('./lift_hardcode.pt')
        cropped_image_size = 112
        pick[0] = center_crop_image(pick[0], cropped_image_size)
        obs1 = pick[0][:, 0:3, :, :]
        obs2 = pick[0][:, 3:6, :, :]
    else:
        raise NotImplementedError

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    if vae_model == "CNNBasedVAE":
        nn_complexity = 0
        DVAE_awa = E2D1((3, cropped_image_size, cropped_image_size), (3, cropped_image_size, cropped_image_size), int(z_dim/2), int(z_dim/2), norm_sample, 4-nn_complexity, int(128/(nn_complexity+1)), 2, 128).to(device)
        print("CNNBasedVAE Input shape", (3, cropped_image_size, cropped_image_size))
    elif vae_model == "ResBasedVAE":
        DVAE_awa = ResE2D1((3, cropped_image_size, cropped_image_size), (3, cropped_image_size, cropped_image_size), int(z_dim/2), int(z_dim/2), norm_sample, 4, 1).to(device)
        print("ResBasedVAE Input shape", (3, cropped_image_size, cropped_image_size))
    elif vae_model == "JointCNNBasedVAE":
        nn_complexity = 0
        DVAE_awa = E1D1((6, cropped_image_size, cropped_image_size), z_dim, norm_sample, 4-nn_complexity, int(128/(nn_complexity+1)), 2, 128).to(device)
        print("JointCNNBasedVAE Input shape", (6, cropped_image_size, cropped_image_size))
    elif vae_model == "JointResBasedVAE":
        DVAE_awa = ResE1D1((6, cropped_image_size, cropped_image_size), z_dim, norm_sample, 4, 1).to(device)
        print("JointResBasedVAE Input shape", (6, cropped_image_size, cropped_image_size))
    elif vae_model == "SepResBasedVAE":
        DVAE_awa = ResE2D2((3, cropped_image_size, cropped_image_size), (3, cropped_image_size, cropped_image_size), int(z_dim/2), int(z_dim/2), norm_sample, 4, 1).to(device)
        print("SepResBasedVAE Input shape", (3, cropped_image_size, cropped_image_size), (3, cropped_image_size, cropped_image_size))
    else:
        raise NotImplementedError

    DVAE_awa = DVAE_awa.eval()
    DVAE_awa.load_state_dict(torch.load(model_path + f'/DVAE_awa-{num_epochs}.pth'))

    index = np.arange(len(obs1))
    n_batches = len(obs1) // batch_size
    Z = torch.zeros((len(obs1), z_dim))
    for i in range(n_batches):
        b_idx = index[i * batch_size:(i + 1) * batch_size]
        o1_batch = torch.tensor(obs1[b_idx], device=device).float() / 255
        o2_batch = torch.tensor(obs2[b_idx], device=device).float() / 255

        if "Joint" not in vae_model:
            z1, _ = DVAE_awa.enc1(o1_batch)
            z2, _ = DVAE_awa.enc2(o2_batch)
            z = torch.cat((z1, z2), dim=1)
            b_idx = index[i * batch_size:(i + 1) * batch_size]
            # print("b_idx: ", b_idx)
            # print("z shape: ", z.shape)
            # print("Z shape: ", Z[b_idx, :].shape)
            # input()
            Z[b_idx, :] = z.cpu().detach()
        else:
            raise NotImplementedError
    cov = cov_func(Z[:, :z_dim//2], Z[:, z_dim//2:])
    print("Z12: ", cov)
    auto_cov = cov_func(Z[:, z_dim//2:], Z[:, z_dim//2:])
    auto_cov2 = cov_func(Z[:, :z_dim//2], Z[:, :z_dim//2])
    print("Z11: ", auto_cov)
    print("Z22: ", auto_cov2)
    ### see if any column is all zeros
    Z = Z.cpu().detach().numpy()
    _ = Z[:, ~np.all(Z == 0, axis=0)]
    print("_ shape: ", _.shape)
    return

if __name__ == "__main__":
    """        
    python calculate_cov.py --dataset Lift --device 0 -l 1e-4 -n 2999 -r 0 -k 0.0 -t 500.0 -z 48 -b 512 --seed 0 -corpen 0.0 -vae ResBasedVAE -ns False -p True
    """

    model_path = './models/'
    dataset_dir = '/store/datasets/gym_fetch/'

    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument("-d", "--dataset", type=str,
                        help="dataset to train on: ['cifar10', 'mnist', 'fmnist', 'svhn', 'monsters128', 'celeb128', "
                             "'celeb256', 'celeb1024']")
    parser.add_argument("-n", "--num_epochs", type=int, help="total number of epochs to run", default=250)
    parser.add_argument("-z", "--z_dim", type=int, help="latent dimensions", default=64)
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=2e-4)
    parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=128)
    parser.add_argument("-r", "--beta_rec", type=float, help="beta coefficient for the reconstruction loss", default=0.0)
    parser.add_argument("-k", "--beta_kl", type=float, help="beta coefficient for the kl divergence", default=1.0)
    parser.add_argument("-t", "--beta_task", type=float, help="beta coefficient for the task loss", default=1.0)
    parser.add_argument("-corpen", "--cross_penalty", type=float, help="cross-correlation penalty", default=0.1)
    parser.add_argument("-s", "--seed", type=int, help="seed", default=100)
    parser.add_argument("-c", "--device", type=int, help="device: -1 for cpu, 0 and up for specific cuda device", default=-1)
    parser.add_argument("-e", "--epoch", type=int, help="epoch: total epoch of training", default=1000)
    parser.add_argument("-vae", "--vae_model", type=str, help="vae model: CNNBasedVAE or SVAE", default="CNNBasedVAE")
    parser.add_argument("-ns", "--norm_sample", type=str, help="Sample from Normal distribution (VAE) or not", default="True")
    parser.add_argument("-p", "--randpca", type=bool, help="perform random pca when training", default=False)
    
    parser.add_argument("-ds", "--data_seed", type=int, help="seed of dataset", default=20)
    parser.add_argument("-crop", "--rand_crop", type=bool, help="randomly crop images", default=True)
    args = parser.parse_args()

    if args.norm_sample == 'True' or args.norm_sample == 'true':
        args.norm_sample = True
    else:
        args.norm_sample = False

    train_awa_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_epochs=args.num_epochs, weight_cross_penalty=args.cross_penalty, 
                  beta_kl=args.beta_kl, beta_rec=args.beta_rec, beta_task=args.beta_task, device=args.device, save_interval=50, lr=args.lr, seed=args.seed,
                  model_path=model_path, dataset_dir=dataset_dir, vae_model=args.vae_model, norm_sample=args.norm_sample,rand_crop=args.rand_crop, 
                  randpca=args.randpca, data_seed=args.data_seed)
