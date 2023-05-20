import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
import torchvision.utils as vutils
import random
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

from dtac.ClassDAE import *
from dtac.covar import cov_func
from dtac.object_detection.od_utils import *


def train_awa_vae(dataset="gym_fetch", z_dim=64, batch_size=32, num_epochs=250, beta_kl=1.0, beta_rec=0.0, beta_task=1.0, weight_cross_penalty=0.1, 
                 device=0, save_interval=30, lr=2e-4, seed=0, vae_model="CNNBasedVAE", norm_sample=True, width=448, height=448, data_seed=-1):
    ### set paths
    if norm_sample:
        model_type = "VAE"
    else:
        model_type = "AE"

    fig_dir = f'./figures/Plot_{dataset}_{z_dim}_randPCA_{model_type}_{vae_model}{width}x{height}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    task_model_path = "/home/pl22767/project/dtac-dev/airbus_scripts/models/YoloV1_224x224/yolov1_aug_0.05_0.05_resize448_224x224_ep60_map0.98_0.83.pth"
    model_path = f'./models/{dataset}_{z_dim}_randPCA_{model_type}_{vae_model}{width}x{height}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'

    ### Set the random seed
    if data_seed != -1:
        random.seed(data_seed)
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(data_seed)
        print("random seed: ", data_seed)

    device = torch.device("cpu") if args.device <= -1 else torch.device("cuda:" + str(args.device))

    ### Load the dataset
    if dataset == "airbus":
        file_parent_dir = f'../airbus_dataset/224x224_overlap28_percent0.3_/'
        files_dir = file_parent_dir + 'val/'
        images = [image for image in sorted(os.listdir(files_dir))
                        if image[-4:]=='.jpg']
        annots = []
        for image in images:
            annot = image[:-4] + '.txt'
            annots.append(annot)
            
        images = pd.Series(images, name='images')
        annots = pd.Series(annots, name='annots')
        df = pd.concat([images, annots], axis=1)
        df = pd.DataFrame(df)

        print("val set: ", file_parent_dir.split('/')[-2])
        transform_img = A.Compose(transforms=[
            A.Resize(width=height, height=height),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=500, min_visibility=0.3))

        test_dataset = ImagesDataset(
            files_dir=files_dir,
            df=df,
            transform=transform_img
        )
        g = torch.Generator()
        g.manual_seed(0)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g
        )
        cropped_image_size_w = width
        cropped_image_size_h = height
        cropped_image_size = height
    else:
        raise NotImplementedError
    
    iou, conf = 0.5, 0.4
    print("iou: ", iou, "conf: ", conf)

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    ### distributed models
    if vae_model == "CNNBasedVAE":
        DVAE_awa = E2D1((3, cropped_image_size_w, cropped_image_size_h), (3, cropped_image_size_w, cropped_image_size_h), int(z_dim/2), int(z_dim/2), norm_sample, 4-seed, int(128/(seed+1)), 2, 128).to(device)
        print("CNNBasedVAE Input shape", (3, cropped_image_size_w, cropped_image_size_h))
    elif vae_model == "ResBasedVAE":
        DVAE_awa = ResE2D1((3, cropped_image_size_h, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h), int(z_dim/2), int(z_dim/2), norm_sample, 4, 1).to(device)
        print("ResBasedVAE Input shape", (3, cropped_image_size_w, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h))
    ### Joint models
    elif vae_model == "JointCNNBasedVAE":
        DVAE_awa = E1D1((6, cropped_image_size, cropped_image_size), z_dim, norm_sample, 4, int(128/(seed+1)), 2, 128).to(device)
        print("JointCNNBasedVAE Input shape", (6, cropped_image_size, cropped_image_size))
    elif vae_model == "JointResBasedVAE":
        DVAE_awa = ResE1D1((6, cropped_image_size, cropped_image_size), z_dim, norm_sample, 4, 1).to(device)
        print("JointResBasedVAE Input shape", (6, cropped_image_size, cropped_image_size))
    elif vae_model == "SepResBasedVAE":
        DVAE_awa = ResE2D2((3, cropped_image_size_h, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h), int(z_dim/2), int(z_dim/2), norm_sample, 4, 1).to(device)
        print("SepResBasedVAE Input shape", (3, cropped_image_size_h, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h))
    else:
        raise NotImplementedError

    DVAE_awa.eval()
    DVAE_awa.load_state_dict(torch.load(model_path + f'/DVAE_awa-{num_epochs}.pth'))
    print("model loaded from: ", model_path + f'/DVAE_awa-{num_epochs}.pth')

    Z = torch.zeros((test_loader.dataset.__len__(), z_dim), device=device)
    index = np.arange(test_loader.dataset.__len__())
    for batch_idx, (obs, out) in enumerate(tqdm(test_loader)):
        obs_orig_112_0_255, out = obs.to(device), out.to(device)
        obs = obs_orig_112_0_255 / 255.0
        
        o1_batch = torch.zeros(obs.shape[0], obs.shape[1], cropped_image_size_h, cropped_image_size_h).to(device)
        o2_batch = torch.zeros(obs.shape[0], obs.shape[1], cropped_image_size_h, cropped_image_size_h).to(device)
        o1_batch[:, :, :cropped_image_size_w, :cropped_image_size_h] = obs[:, :, :cropped_image_size_w, :cropped_image_size_h]
        o2_batch[:, :, cropped_image_size_w-20:, :cropped_image_size_h] = obs[:, :, cropped_image_size_w-20:, :cropped_image_size_h]
        
        if "Joint" not in vae_model:
            z1, _ = DVAE_awa.enc1(o1_batch)
            z2, _ = DVAE_awa.enc2(o2_batch)
            z = torch.cat((z1, z2), dim=1)
            b_idx = index[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            # print("b_idx: ", b_idx)
            # print("z shape: ", z.shape)
            # print("Z shape: ", Z[b_idx, :].shape)
            # input()
            Z[b_idx, :] = z
        else:
            raise NotImplementedError
    cov = cov_func(Z[:, :z_dim//2], Z[:, z_dim//2:])
    if beta_rec == 0:
        print("Task-aware")
    else:
        print("Task-agnostic")
    print("Z12: ", cov)
    auto_cov = cov_func(Z[:, z_dim//2:], Z[:, z_dim//2:])
    auto_cov2 = cov_func(Z[:, :z_dim//2], Z[:, :z_dim//2])
    print("Z11: ", auto_cov)
    print("Z22: ", auto_cov2)
    ### see if any column is all zeros
    # Z = Z.cpu().detach().numpy()
    # _ = Z[:, ~np.all(Z == 0, axis=0)]
    # print("_ shape: ", _.shape)

    return

if __name__ == "__main__":
    """        
    python calculate_cov.py --dataset airbus --device 0 -l 1e-4 -n 299 -r 0.5 -k 0.0 -t 0.0 -z 80 -bs 64 --seed 1 -corpen 0.0 -vae ResBasedVAE -wt 80 -ht 112
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
    parser.add_argument("-ds", "--data_seed", type=int, help="seed of dataset", default=20)
    parser.add_argument("-s", "--seed", type=int, help="seed", default=100)
    parser.add_argument("-c", "--device", type=int, help="device: -1 for cpu, 0 and up for specific cuda device", default=-1)
    parser.add_argument("-vae", "--vae_model", type=str, help="vae model: CNNBasedVAE or SVAE", default="CNNBasedVAE")
    parser.add_argument("-ns", "--norm_sample", type=str, help="Sample from Normal distribution (VAE) or not", default="False")
    parser.add_argument("-wt", "--width", type=int, help="image width", default=256)
    parser.add_argument("-ht", "--height", type=int, help="image height", default=448)
    # parser.add_argument("-p", "--randpca", type=bool, help="image height", default=448)

    args = parser.parse_args()

    if args.norm_sample == 'True' or args.norm_sample == 'true':
        args.norm_sample = True
    else:
        args.norm_sample = False

    train_awa_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_epochs=args.num_epochs, 
                  weight_cross_penalty=args.cross_penalty, beta_kl=args.beta_kl, beta_rec=args.beta_rec, beta_task=args.beta_task, 
                  device=args.device, save_interval=50, lr=args.lr, seed=args.seed, vae_model=args.vae_model, norm_sample=args.norm_sample,
                  width=args.width, height=args.height, data_seed=args.data_seed)
