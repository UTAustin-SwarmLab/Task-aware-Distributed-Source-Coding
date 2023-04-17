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


def train_awa_vae(dataset="gym_fetch", z_dim=64, batch_size=32, num_epochs=250, beta_kl=1.0, beta_rec=0.0, beta_task=1.0, weight_cross_penalty=0.1, 
                 device=0, save_interval=30, lr=2e-4, seed=0, vae_model="CNNBasedVAE", norm_sample=True):
    ### set paths
    if norm_sample:
        model_type = "VAE"
    else:
        model_type = "DAE"

    LOG_DIR = f'./summary/{dataset}_{z_dim}_taskaware_{model_type}_{vae_model}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    fig_dir = f'./figures/{dataset}_{z_dim}_taskaware_{model_type}_{vae_model}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'

    model_path = f'./models/{dataset}_{z_dim}_taskaware_{model_type}_{vae_model}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    summary_writer = SummaryWriter(os.path.join(LOG_DIR, 'tb'))

    ### Set the random seed
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print("random seed: ", seed)

    device = torch.device("cpu") if args.device <= -1 else torch.device("cuda:" + str(args.device))


    ### Load the dataset
    if dataset == "cifar10":
        transform_train=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        transform_test=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        train_kwargs = {'batch_size': batch_size}
        test_kwargs = {'batch_size': batch_size}
        trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(testset, **test_kwargs)
    else:
        raise NotImplementedError

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    ### distributed models
    if vae_model == "CNNBasedVAE":
        # DVAE_awa = E2D1(obs1.shape[1:], obs2.shape[1:], int(z_dim/2), int(z_dim/2), norm_sample=norm_sample).to(device)
        DVAE_awa = E2D1NonSym((3, 32, 32), (3, 32, 32), int(z_dim/2), int(z_dim/2), norm_sample, 4-seed, int(128/(seed+1)), 2, 128).to(device)
        print("CNNBasedVAE Input shape", (3, 32, 32))
    elif vae_model == "ResBasedVAE":
        DVAE_awa = ResE2D1NonSym((3, 32, 32), (3, 32, 32), int(z_dim/2), int(z_dim/2), norm_sample, 4-seed, 3-seed).to(device)
        print("ResBasedVAE Input shape", (3, 32, 32), (3, 32, 32))
    ### Joint models
    elif vae_model == "JointCNNBasedVAE":
        DVAE_awa = E1D1((3, 32, 32), z_dim, norm_sample, 4-seed, int(128/(seed+1)), 2, 128).to(device)
        print("JointCNNBasedVAE Input shape", (3, 32, 32))
    elif vae_model == "JointResBasedVAE":
        DVAE_awa = ResE1D1((3, 32, 32), z_dim, norm_sample, 4-seed, 3-seed).to(device)
        print("JointResBasedVAE Input shape", (3, 32, 32))
    else:
        DVAE_awa = ResNetE1D1().to(device)

    optimizer = optim.Adam(DVAE_awa.parameters(), lr=lr)

    cur_iter = 0
    for ep in range(num_epochs):
        ep_loss = []
        DVAE_awa.train()
        
        for batch_idx, (obs, out) in enumerate(tqdm(train_loader)):
            obs, out = obs.to(device), out.to(device)
            assert obs.min() >= 0 and obs.max() <= 1
            
            if "Joint" not in vae_model:
                o1_batch = torch.zeros(obs.shape[0], obs.shape[1], 32, 32).to(device)
                o2_batch = torch.zeros(obs.shape[0], obs.shape[1], 32, 32).to(device)
                o1_batch[:, :, 8:, 8:] = obs[:, :, 8:, 8:]
                o1_batch += torch.randn(o1_batch.shape).to(device) * 0.1
                o2_batch[:, :, :20, :20] = obs[:, :, :20, :20] 
                o2_batch += torch.randn(o2_batch.shape).to(device) * 0.2

                obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(o1_batch, o2_batch, obs)
            else:
                obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(obs)

            psnr -= 20 * np.log10(255.0)
            task_loss = 0
            # obs_pred = obs_pred.clip(0, 1)
            loss = beta_task * task_loss + beta_rec * loss_rec + beta_kl * (kl1 + kl2) + weight_cross_penalty * loss_cor

            ### check models' train/eval modes
            if not DVAE_awa.training:
                print(DVAE_awa.training)
                raise KeyError("Models' train/eval modes are not correct")
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### log tensorboard
            summary_writer.add_scalar('loss', loss, cur_iter)
            summary_writer.add_scalar('task', task_loss, cur_iter)
            summary_writer.add_scalar('Rec', loss_rec, cur_iter)
            summary_writer.add_scalar('kl1/var', kl1, cur_iter)
            summary_writer.add_scalar('kl2/invar', kl2, cur_iter)
            summary_writer.add_scalar('cov', loss_cor, cur_iter)
            summary_writer.add_scalar('PSNR', psnr, cur_iter)

            ep_loss.append(loss.item())
            cur_iter += 1

        ### print loss
        print("Epoch: {}, Loss: {}".format(ep, np.mean(ep_loss)))

        ### save model
        if (ep + 1) % save_interval == 0 or ep == 0:
            ### save model
            torch.save(DVAE_awa.state_dict(), model_path + f'/DVAE_awa-{ep}.pth')

            ### eval model
            DVAE_awa.eval()
            test_psnr = []
            with torch.no_grad():
                for batch_idx, (obs, out) in enumerate(tqdm(test_loader)):
                    obs, out = obs.to(device), out.to(device)
                    
                    if "Joint" not in vae_model:
                        o1_batch = torch.zeros(obs.shape[0], obs.shape[1], 32, 32).to(device)
                        o2_batch = torch.zeros(obs.shape[0], obs.shape[1], 32, 32).to(device)
                        o1_batch[:, :, 8:, 8:] = obs[:, :, 8:, 8:]
                        o1_batch += torch.randn(o1_batch.shape).to(device) * 0.1
                        o2_batch[:, :, :20, :20] = obs[:, :, :20, :20] 
                        o2_batch += torch.randn(o1_batch.shape).to(device) * 0.2

                        obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(o1_batch, o2_batch, obs)
                    else:
                        obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(obs)
                    
                    psnr -= 20 * np.log10(255.0)
                    test_psnr.append(psnr.item())
                    obs_pred = obs_pred.clip(0, 1)
                
                summary_writer.add_scalar('test PSNR', np.mean(test_psnr), ep)
                print(f"====================================\nTest PSNR: {np.mean(test_psnr)}\n====================================")
                    
        ### export figure
        if (ep + 1) % save_interval == 0 or ep == num_epochs - 1 or ep == 0:
            max_imgs = min(batch_size, 8)
            vutils.save_image(torch.cat([obs[:max_imgs], obs_pred[:max_imgs]], dim=0).data.cpu(),
                '{}/image_{}.jpg'.format(fig_dir, ep), nrow=8)

    return

if __name__ == "__main__":
    """        
    python train_DAE.py --dataset cifar10 --device 6 -l 1e-4 -n 100 -r 1000.0 -k 0.0 -t 0.0 -z 64 -bs 1024 --seed 2 -corpen 0.0 -vae ResBasedVAE -ns False
    """

    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument("-d", "--dataset", type=str, help="dataset to train on: ['cifar10', 'airbus', 'PickAndPlace', 'gym_fetch']", default="")
    parser.add_argument("-n", "--num_epochs", type=int, help="total number of epochs to run", default=250)
    parser.add_argument("-z", "--z_dim", type=int, help="latent dimensions", default=256)
    parser.add_argument("-l", "--lr", type=float, help="learning rate", default=2e-4)
    parser.add_argument("-bs", "--batch_size", type=int, help="batch size", default=64)
    parser.add_argument("-r", "--beta_rec", type=float, help="beta coefficient for the reconstruction loss", default=0.0)
    parser.add_argument("-k", "--beta_kl", type=float, help="beta coefficient for the kl divergence", default=1.0)
    parser.add_argument("-t", "--beta_task", type=float, help="beta coefficient for the task loss", default=1.0)
    parser.add_argument("-corpen", "--cross_penalty", type=float, help="cross-correlation penalty", default=0.1)
    parser.add_argument("-s", "--seed", type=int, help="seed", default=100)
    parser.add_argument("-c", "--device", type=int, help="device: -1 for cpu, 0 and up for specific cuda device", default=-1)
    parser.add_argument("-vae", "--vae_model", type=str, help="vae model: CNNBasedVAE or SVAE", default="CNNBasedVAE")
    parser.add_argument("-ns", "--norm_sample", type=str, help="Sample from Normal distribution (VAE) or not", default="True")
    args = parser.parse_args()

    if args.norm_sample == 'True' or args.norm_sample == 'true':
        args.norm_sample = True
    else:
        args.norm_sample = False

    train_awa_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_epochs=args.num_epochs, 
                  weight_cross_penalty=args.cross_penalty, beta_kl=args.beta_kl, beta_rec=args.beta_rec, beta_task=args.beta_task, 
                  device=args.device, save_interval=5, lr=args.lr, seed=args.seed, vae_model=args.vae_model, norm_sample=args.norm_sample)
