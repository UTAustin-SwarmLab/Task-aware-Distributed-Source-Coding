import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
import torchvision.utils as vutils
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from dtac.object_detection.yolov8_loss import Loss
### to start tensorboard:  tensorboard --logdir=./airbus_detection/summary --port=6006
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO
import argparse

from dtac.ClassAE import *
from dtac.object_detection.od_utils import *

def train_awa_vae(dataset="gym_fetch", z_dim=64, batch_size=32, num_epochs=250, beta_kl=1.0, beta_rec=0.0, beta_task=1.0, weight_cross_penalty=0.1, 
                  device=0, save_interval=50, lr=2e-4, seed=0, vae_model="CNNBasedVAE", norm_sample=True, width=448, height=448):
    ### set paths
    if norm_sample:
        model_type = "VAE"
    else:
        model_type = "AE"

    LOG_DIR = f'./summary/{dataset}_{z_dim}_v8_{model_type}_{vae_model}{width}x{height}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    fig_dir = f'./figures/{dataset}_{z_dim}_v8_{model_type}_{vae_model}{width}x{height}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    task_model_path = "./runs/detect/train/weights/best.pt"

    model_path = f'./models/{dataset}_{z_dim}_v8_{model_type}_{vae_model}{width}x{height}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
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
    if dataset == "airbus":
        file_parent_dir = f'../airbus_dataset/512x512_overlap64_percent0.3_/'
        files_dir = file_parent_dir + 'train/'
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

        print(f"resize to {height}x{height} then 448x448")
        print("testing set: ", files_dir.split('/')[-2])
        p = 0.01
        print("p: ", p)
        transform_img = A.Compose([
            A.Resize(width=height, height=height),
            A.Blur(p=p, blur_limit=(3, 7)), 
            A.MedianBlur(p=p, blur_limit=(3, 7)), A.ToGray(p=p), 
            A.CLAHE(p=p, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
            ToTensorV2(p=1.0)
        ])
        train_dataset = ImagesDataset(
            files_dir=files_dir,
            df=df,
            transform=transform_img
        )
        g = torch.Generator()
        g.manual_seed(0)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g
        )
        cropped_image_size_w = width
        cropped_image_size_h = height
        cropped_image_size = height

        ### laod test dataset
        test_dir = file_parent_dir + 'val/'
        test_images = [image for image in sorted(os.listdir(test_dir))
                                if image[-4:]=='.jpg']
        test_annots = []
        for image in test_images:
            annot = image[:-4] + '.txt'
            test_annots.append(annot)
        test_images = pd.Series(test_images, name='test_images')
        test_annots = pd.Series(test_annots, name='test_annots')
        test_df = pd.concat([test_images, test_annots], axis=1)
        test_df = pd.DataFrame(test_df)
        test_transform_img = A.Compose([
            A.Resize(width=height, height=height),
            ToTensorV2(p=1.0)
        ])
        test_dataset = ImagesDataset(
            transform=test_transform_img,
            df=test_df,
            files_dir=test_dir
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            worker_init_fn=seed_worker,
            generator=g
        )
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
        DVAE_awa = E2D1NonSym((3, cropped_image_size_w, cropped_image_size_h), (3, cropped_image_size_w, cropped_image_size_h), int(z_dim/2), int(z_dim/2), norm_sample, 4-seed, int(128/(seed+1)), 2, 128).to(device)
        print("CNNBasedVAE Input shape", (3, cropped_image_size_w, cropped_image_size_h))
    elif vae_model == "ResBasedVAE":
        DVAE_awa = ResE2D1NonSym((3, cropped_image_size_w, cropped_image_size_h), (3, cropped_image_size_w, cropped_image_size_h), int(z_dim/2), int(z_dim/2), norm_sample, 4-seed, 3-seed).to(device)
        print("ResBasedVAE Input shape", (3, cropped_image_size_w, cropped_image_size_h))
    ### Joint models
    elif vae_model == "JointCNNBasedVAE":
        DVAE_awa = E1D1((3, cropped_image_size, cropped_image_size), z_dim, norm_sample, 4-seed, int(128/(seed+1)), 2, 128).to(device)
        print("JointCNNBasedVAE Input shape", (3, cropped_image_size, cropped_image_size))
    elif vae_model == "JointResBasedVAE":
        DVAE_awa = ResE1D1((3, cropped_image_size, cropped_image_size), z_dim, norm_sample, 4-seed, 3-seed).to(device)
        print("JointResBasedVAE Input shape", (3, cropped_image_size, cropped_image_size))
    else:
        DVAE_awa = ResNetE1D1().to(device)

    # DVAE_awa = nn.Identity().to(device)
    optimizer = optim.Adam(DVAE_awa.parameters(), lr=lr)

    cur_iter = 0
    iou = 0.5
    conf = 0.4
    print(f"iou: {iou}, thres: {conf}")

    for ep in range(num_epochs):
        DVAE_awa.train()

        ### load task model
        task_model_v8 = YOLO(task_model_path)
        task_model_v8.to(device)
        task_model = task_model_v8.model.to(device)
        # print("task model: ", task_model_path)
        # task_model.eval()
        loss_fn = Loss(task_model)
        # for param in task_model.parameters():
        #     param.requires_grad = False

        ep_loss = []
        
        for batch_idx, (obs, labels) in enumerate(train_loader):
            obs_112_0_255, labels = obs.to(device), labels.to(device)
            obs = obs_112_0_255 / 255.0
            bs = int(obs.shape[0])
        
            batch = {"batch_idx": [], "cls": [], "bboxes": []}
            true_bboxes = cellboxes_to_boxes(labels)
            for idx in range(bs):
                for box in true_bboxes[idx]:
                    if box[1] > conf:
                        batch["batch_idx"].append(idx)
                        batch["cls"].append(0)
                        batch["bboxes"].append(box[2:]) 

            batch["batch_idx"] = torch.tensor(batch["batch_idx"])
            batch["cls"] = torch.tensor(batch["cls"])
            batch["bboxes"] = torch.tensor(batch["bboxes"]).view(-1, 4) ### xywhn
            
            if "Joint" not in vae_model:
                o1_batch = obs[:, :, :cropped_image_size_w, :cropped_image_size_h]
                o2_batch = obs[:, :, cropped_image_size_w:, :cropped_image_size_h]
                obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(o1_batch, o2_batch, obs)
            else:
                obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(obs)

            obs_pred_112_0_255 = obs_pred.clip(0, 1) * 255.0 ##################### important: clip the value to 0-255
            obs_pred_512_0_255 = F.interpolate(obs_pred_112_0_255, size=(512, 512)) ### resize to 448x448
            out_pred = task_model(obs_pred_512_0_255)

            task_loss, _ = loss_fn(out_pred, batch)
            loss = beta_task * task_loss + beta_rec * loss_rec + beta_kl * (kl1 + kl2) + weight_cross_penalty * loss_cor

            ## check models' train/eval modes
            if (not DVAE_awa.training) or task_model.training:
                print(DVAE_awa.training, task_model.training)
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
        if (ep + 1) % save_interval == 0 or (ep + 1) == 10 or ep == 0:
            print("Evaluation on train set")
            ### test on train set
            if "Joint" in vae_model:
                pred_boxes, target_boxes = get_bboxes_v8(train_loader, DVAE_awa, task_model_v8, iou, conf, True, device=device, 
                                        cropped_image_size_w = cropped_image_size_h, cropped_image_size_h = cropped_image_size_h)
            else:
                pred_boxes, target_boxes = get_bboxes_v8(train_loader, DVAE_awa, task_model_v8, iou, conf, False, device=device,
                                        cropped_image_size_w = cropped_image_size_w, cropped_image_size_h = cropped_image_size_h)

            train_mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            summary_writer.add_scalar(f'train_mean_avg_prec_{iou}_{conf}', train_mean_avg_prec, ep)  
            print(train_mean_avg_prec, ep)

            ### test on test set
            print("Evaluation on test set")
            if "Joint" in vae_model:
                pred_boxes, target_boxes = get_bboxes_v8(test_loader, DVAE_awa, task_model_v8, iou, conf, True, device=device, 
                                        cropped_image_size_w = cropped_image_size_h, cropped_image_size_h = cropped_image_size_h)
            else:
                pred_boxes, target_boxes = get_bboxes_v8(test_loader, DVAE_awa, task_model_v8, iou, conf, False, device=device,
                                        cropped_image_size_w = cropped_image_size_w, cropped_image_size_h = cropped_image_size_h)

            test_mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            summary_writer.add_scalar(f'test_mean_avg_prec_{iou}_{conf}', test_mean_avg_prec, ep)
            print(test_mean_avg_prec, ep)
        
            torch.save(DVAE_awa.state_dict(), model_path + f'/DVAE_awa-{ep}.pth')  

        ### export figure
        if (ep + 1) % save_interval == 0 or ep == num_epochs - 1 or ep == 0:
            max_imgs = min(batch_size, 8)
            vutils.save_image(torch.cat([obs[:max_imgs], obs_pred[:max_imgs]], dim=0).data.cpu(),
                '{}/image_{}.jpg'.format(fig_dir, ep), nrow=8)

    return

if __name__ == "__main__":
    """        
    python train_od_awaAE_v8.py --dataset airbus --device 7 -l 1e-4 -n 3000 -r 1000.0 -k 0.0 -t 0.1 -z 96 -bs 64 --seed 2 -corpen 0.0 -vae JointCNNBasedVAE -ns False -wt 64 -ht 112
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
    parser.add_argument("-wt", "--width", type=int, help="image width", default=256)
    parser.add_argument("-ht", "--height", type=int, help="image height", default=448)
    args = parser.parse_args()

    if args.norm_sample == 'True' or args.norm_sample == 'true':
        args.norm_sample = True
    else:
        args.norm_sample = False

    train_awa_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_epochs=args.num_epochs, 
                  weight_cross_penalty=args.cross_penalty, beta_kl=args.beta_kl, beta_rec=args.beta_rec, beta_task=args.beta_task, 
                  device=args.device, save_interval=50, lr=args.lr, seed=args.seed, vae_model=args.vae_model, norm_sample=args.norm_sample,
                  width=args.width, height=args.height)
