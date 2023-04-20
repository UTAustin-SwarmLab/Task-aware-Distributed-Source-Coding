import torch
import numpy as np
import csv
import random
from torchvision import datasets, transforms
import argparse
from tqdm import tqdm

from dtac.DPCA_torch import *
from dtac.ClassDAE import *


def dpca_od_vae(dataset="gym_fetch", z_dim=64, batch_size=1024, num_epochs=250, beta_kl=1.0, beta_rec=0.0, beta_task=1.0, weight_cross_penalty=0.1, 
                 device=0, save_interval=30, lr=2e-4, seed=0, vae_model="CNNBasedVAE", norm_sample=True, start=0, end=97):
    ### set paths
    if norm_sample:
        model_type = "VAE"
    else:
        model_type = "DAE"

    model_path = f'./models/{dataset}_{z_dim}_taskaware_{model_type}_{vae_model}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'

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
        image_size = 32
    else:
        raise NotImplementedError

    ### distributed models
    if vae_model == "CNNBasedVAE":
        # DVAE_awa = E2D1(obs1.shape[1:], obs2.shape[1:], int(z_dim/2), int(z_dim/2), norm_sample=norm_sample).to(device)
        DVAE_awa = E2D1NonSym((3, image_size, image_size), (3, image_size, image_size), int(z_dim/2), int(z_dim/2), norm_sample, 4-seed, int(128/(seed+1)), 2, 128).to(device)
        print("CNNBasedVAE Input shape", (3, image_size, image_size))
    elif vae_model == "ResBasedVAE":
        DVAE_awa = ResE2D1NonSym((3, image_size, image_size), (3, image_size, image_size), int(z_dim/2), int(z_dim/2), norm_sample, 4-seed, 3-seed).to(device)
        print("ResBasedVAE Input shape", (3, image_size, image_size), (3, image_size, image_size))
    ### Joint models
    elif vae_model == "JointCNNBasedVAE":
        DVAE_awa = E1D1((3, image_size, image_size), z_dim, norm_sample, 4-seed, int(128/(seed+1)), 2, 128).to(device)
        print("JointCNNBasedVAE Input shape", (3, image_size, image_size))
    elif vae_model == "JointResBasedVAE":
        DVAE_awa = ResE1D1((3, image_size, image_size), z_dim, norm_sample, 4-seed, 3-seed).to(device)
        print("JointResBasedVAE Input shape", (3, image_size, image_size))
    else:
        DVAE_awa = ResNetE1D1().to(device)

    ### load vae model
    DVAE_awa.load_state_dict(torch.load(model_path + f'/DVAE_awa-{num_epochs}.pth'))
    DVAE_awa.eval()
    results = []

    if "Joint" not in vae_model:
        dpca, singular_val_vec = DistriburedPCAEQ(DVAE_awa, rep_dim=z_dim, device=device, env=dataset)

    else:
        dpca, singular_val_vec = JointPCA(DVAE_awa, rep_dim=z_dim, device=device, env=dataset)

    for dpca_dim in range(start, end+1, 4):
        test_psnr = []
        ### count importance priority of dimensions
        rep_dims = [0, 0, 0]
        print(dpca_dim)
        for i in range(dpca_dim):
            seg = singular_val_vec[i][2]
            rep_dims[seg] += 1
        
        with torch.no_grad():
            for batch_idx, (obs, out) in enumerate(tqdm(test_loader)):
                obs, out = obs.to(device), out.to(device)
               
                if "Joint" not in vae_model and "BasedVAE" in vae_model:
                    o1_batch = torch.zeros(obs.shape[0], obs.shape[1], image_size, image_size).to(device)
                    o2_batch = torch.zeros(obs.shape[0], obs.shape[1], image_size, image_size).to(device)
                    o1_batch[:, :, 8:, 8:] = obs[:, :, 8:, 8:]
                    o1_batch += torch.randn(o1_batch.shape).to(device) * 0.1
                    o2_batch[:, :, :20, :20] = obs[:, :, :20, :20] 
                    o2_batch += torch.randn(o2_batch.shape).to(device) * 0.2
                    if dpca is not None:
                        z1, _ = DVAE_awa.enc1(o1_batch)
                        z2, _ = DVAE_awa.enc2(o2_batch)
                        z1 = z1.detach()
                        z2 = z2.detach()
                        z = torch.cat((z1, z2), dim=1)
                        recon_z = dpca.LinearEncDec(z, dpca_dim=dpca_dim)
                        z_sample = recon_z
                        obs_rec = DVAE_awa.dec(z_sample)
                    else:
                        obs_rec = DVAE_awa(o1_batch, o2_batch)[0][:, :, :, :]
                elif "Joint" in vae_model:
                    if dpca is not None:
                        z, _ = DVAE_awa.enc(obs) ########## need to be modified
                        z = z.detach()
                        recon_z = dpca.LinearEncDec(z, dpca_dim=dpca_dim)
                        z_sample = recon_z
                        obs_rec = DVAE_awa.dec(z_sample).clip(0, 1)
                    else:
                        obs_rec = DVAE_awa(obs)[0][:, :, :, :].clip(0, 1)
                
                psnr = PSNR(obs_rec, obs)
                psnr -= 20 * np.log10(255.0)

                test_psnr.append(psnr.item())
                
        print("dpca_dim: ", dpca_dim, "rep_dims: ", rep_dims, "test_psnr: ", np.mean(test_psnr))
        results.append([dpca_dim, rep_dims[0], rep_dims[1], rep_dims[2], np.mean(test_psnr)])

    header = ['dpca_dim', 'dim of z1 private', 'dim of z1 share', 'dim of z2 private', 'testmAP']
    csv_name = f"{dataset}_{z_dim}_taskaware_{model_type}_{vae_model}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}-ep{num_epochs}"+ '.csv'
    with open('../csv_data/' + csv_name, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(header)
        writer.writerows(results)

    return

if __name__ == "__main__":
    """        
    python dpca_DAE.py --dataset cifar10 --device 4 -n 89 -l 1e-3 -r 1000.0 -k 0.0 -t 0.0 -z 64 -bs 1024 --seed 2 -corpen 0.0 -vae ResBasedVAE -ns False -st 4 -end 64
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
    parser.add_argument("-st", "--start", type=int, help="start epoch", default=4)
    parser.add_argument("-end", "--end", type=int, help="end epoch", default=96)
    args = parser.parse_args()

    if args.norm_sample == 'True' or args.norm_sample == 'true':
        args.norm_sample = True
    else:
        args.norm_sample = False

    dpca_od_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_epochs=args.num_epochs, 
                  weight_cross_penalty=args.cross_penalty, beta_kl=args.beta_kl, beta_rec=args.beta_rec, beta_task=args.beta_task, 
                  device=args.device, save_interval=50, lr=args.lr, seed=args.seed, vae_model=args.vae_model, norm_sample=args.norm_sample,
                  start=args.start, end=args.end)
