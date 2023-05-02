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
### to start tensorboard:  tensorboard --logdir=./airbus_scripts/summary --port=6006
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm

from dtac.ClassDAE import *
from dtac.object_detection.yolo_model import YoloV1, YoloLoss
from dtac.object_detection.od_utils import *


def train_awa_vae(dataset="gym_fetch", z_dim=64, batch_size=32, num_epochs=250, beta_kl=1.0, beta_rec=0.0, beta_task=1.0, weight_cross_penalty=0.1, 
                 device=0, save_interval=30, lr=2e-4, seed=0, vae_model="CNNBasedVAE", norm_sample=True, width=448, height=448, orig_z=80, orig_epoch=4000):
    ### set paths
    if norm_sample:
        model_type = "VAE"
    else:
        model_type = "AE"

    LOG_DIR = f'./summary/{dataset}_{orig_z}_{z_dim}_cancat_{model_type}_{vae_model}{width}x{height}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    fig_dir = f'./figures/{dataset}_{orig_z}_{z_dim}_cancat_{model_type}_{vae_model}{width}x{height}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    # task_model_path = "/home/pl22767/project/dtac-dev/airbus_scripts/models/YoloV1_896x512/yolov1_512x896_ep240_map0.97_0.99.pth"
    task_model_path = "/home/pl22767/project/dtac-dev/airbus_scripts/models/YoloV1_224x224/yolov1_aug_0.05_0.05_resize448_224x224_ep60_map0.98_0.83.pth"
    
    orig_model_path = f'./models/{dataset}_{orig_z}_randPCA_{model_type}_{vae_model}{width}x{height}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    summary_writer = SummaryWriter(os.path.join(LOG_DIR, 'tb'))
    new_model_path = f'./models/{dataset}_{orig_z}_{z_dim}_cancat_{model_type}_{vae_model}{width}x{height}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'

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
        file_parent_dir = f'../airbus_dataset/224x224_overlap28_percent0.3_/'
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
        print("training set: ", file_parent_dir.split('/')[-2])
        p = 0.05
        print("p: ", p)
        transform_img = A.Compose(transforms=[
            # A.Resize(width=height, height=height),
            A.RandomResizedCrop(width=height, height=height),
            A.Blur(p=p, blur_limit=(3, 7)), 
            A.MedianBlur(p=p, blur_limit=(3, 7)), A.ToGray(p=p), 
            A.CLAHE(p=p, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
            ToTensorV2(p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=500, min_visibility=0.3))

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
        test_images = [image for image in sorted(os.listdir(test_dir)) if image[-4:]=='.jpg']
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

    ### load task model
    task_model = YoloV1(split_size=7, num_boxes=2, num_classes=3).to(device)
    checkpoint = torch.load(task_model_path)
    task_model.load_state_dict(checkpoint["state_dict"])
    print("=> Loading checkpoint\n", "Train mAP:", checkpoint['Train mAP'], "\tTest mAP:", checkpoint['Test mAP'])
    task_model.eval()
    for param in task_model.parameters():
        param.requires_grad = False
    
    iou, conf = 0.5, 0.4
    print("iou: ", iou, "conf: ", conf)

    if not os.path.exists(new_model_path):
        os.makedirs(new_model_path)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    ### distributed models
    if vae_model == "CNNBasedVAE":
        raise NotImplementedError
        DVAE_awa = E2D1((3, cropped_image_size_w, cropped_image_size_h), (3, cropped_image_size_w, cropped_image_size_h), int(z_dim/2), int(z_dim/2), norm_sample, 4-seed, int(128/(seed+1)), 2, 128).to(device)
        print("CNNBasedVAE Input shape", (3, cropped_image_size_w, cropped_image_size_h))
    elif vae_model == "ResBasedVAE":
        orig_model = ResE2D1((3, cropped_image_size_h, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h), int(orig_z/2), int(orig_z/2), norm_sample, 4, 1).to(device)
        orig_model.load_state_dict(torch.load(orig_model_path+f'/DVAE_awa-{orig_epoch}.pth'))
        DVAE_awa = ConcatenateDAE(DAE=orig_model, z_dim=int(z_dim/2), orig_dim=int(orig_z/2)).to(device)
        print("ResBasedVAE Input shape", (3, cropped_image_size_w, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h))
    ### Joint models
    elif vae_model == "JointCNNBasedVAE":
        raise NotImplementedError
        DVAE_awa = E1D1((6, cropped_image_size, cropped_image_size), z_dim, norm_sample, 4, int(128/(seed+1)), 2, 128).to(device)
        print("JointCNNBasedVAE Input shape", (6, cropped_image_size, cropped_image_size))
    elif vae_model == "JointResBasedVAE":
        orig_model = ResE1D1((6, cropped_image_size, cropped_image_size), orig_z, norm_sample, 4, 1).to(device)
        orig_model.load_state_dict(torch.load(orig_model_path+f'/DVAE_awa-{orig_epoch}.pth'))
        DVAE_awa = ConcatenateJAE(JAE=orig_model, z_dim=z_dim, orig_dim=orig_z).to(device)
        print("JointResBasedVAE Input shape", (6, cropped_image_size, cropped_image_size))
    elif vae_model == "SepResBasedVAE":
        orig_model = ResE2D2((3, cropped_image_size_h, cropped_image_size_h), (3, cropped_image_size_h, cropped_image_size_h), int(z_dim/2), int(z_dim/2), norm_sample, 4, 1).to(device) ### 4, 1 to balance # of paras
        orig_model.load_state_dict(torch.load(orig_model_path+f'/DVAE_awa-{orig_epoch}.pth'))
        DVAE_awa = ConcatenateSepAE(SepAE=orig_model, z_dim=int(z_dim/2), orig_dim=int(orig_z/2)).to(device)
    else:
        raise NotImplementedError

    DVAE_awa.train()
    optimizer = optim.Adam(DVAE_awa.parameters(), lr=lr)

    cur_iter = 0
    loss_fn = YoloLoss()
    for ep in range(num_epochs):
        ep_loss = []
        
        for batch_idx, (obs, out) in enumerate(tqdm(train_loader)):
            obs_112_0_255, out = obs.to(device), out.to(device)
            obs = obs_112_0_255 / 255.0
            
            o1_batch = torch.zeros(obs.shape[0], obs.shape[1], cropped_image_size_h, cropped_image_size_h).to(device)
            o2_batch = torch.zeros(obs.shape[0], obs.shape[1], cropped_image_size_h, cropped_image_size_h).to(device)
            o1_batch[:, :, :cropped_image_size_w, :cropped_image_size_h] = obs[:, :, :cropped_image_size_w, :cropped_image_size_h]
            o2_batch[:, :, cropped_image_size_w-20:, :cropped_image_size_h] = obs[:, :, cropped_image_size_w-20:, :cropped_image_size_h]

            if "Joint" not in vae_model:
                obs_, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(o1_batch, o2_batch)

                obs_pred = torch.zeros_like(obs).to(device) # 3x112x112
                obs_pred[:, :, :cropped_image_size_w-20, :cropped_image_size_h] = obs_[:, :3, :cropped_image_size_w-20, :cropped_image_size_h]
                obs_pred[:, :, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h] = 0.5 * (obs_[:, :3, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h] 
                                                                                                        + obs_[:, 3:, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h])
                obs_pred[:, :, cropped_image_size_w:, :cropped_image_size_h] = obs_[:, 3:, cropped_image_size_w:, :cropped_image_size_h]
            else:
                obs6chan = torch.cat((o1_batch, o2_batch), dim=1)
                obs6chan_, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(obs6chan)
                obs_pred = torch.zeros_like(obs).to(device) # 3x112x112
                obs_pred[:, :, :cropped_image_size_w-20, :cropped_image_size_h] = obs6chan_[:, :3, :cropped_image_size_w-20, :cropped_image_size_h]
                obs_pred[:, :, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h] = 0.5 * (obs6chan_[:, :3, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h] 
                                                                                                        + obs6chan_[:, 3:, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h])
                obs_pred[:, :, cropped_image_size_w:, :cropped_image_size_h] = obs6chan_[:, 3:, cropped_image_size_w:, :cropped_image_size_h]

            obs_112_0_255 = obs_pred.clip(0, 1) * 255.0 ##################### important: clip the value to 0-255
            obs_pred_448_0_255 = F.interpolate(obs_112_0_255, size=(448, 448)) ### resize to 448x448
            out_pred = task_model(obs_pred_448_0_255)
            task_loss = loss_fn(out_pred, out)
            loss = beta_task * task_loss + beta_rec * loss_rec + beta_kl * (kl1 + kl2) + weight_cross_penalty * loss_cor

            ### check models' train/eval modes
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
        if (ep + 1) % save_interval == 0 or (ep + 1) == 20 or ep == 0:
            ### test on train set
            # if "Joint" in vae_model:
            #     pred_boxes, target_boxes = get_bboxes_AE(
            #         train_loader, task_model, DVAE_awa, True, iou_threshold=iou, threshold=conf, device=device,
            #         cropped_image_size_w=cropped_image_size, cropped_image_size_h=cropped_image_size
            #     )
            # else:
            #     pred_boxes, target_boxes = get_bboxes_AE(
            #         train_loader, task_model, DVAE_awa, False, iou_threshold=iou, threshold=conf, device=device,
            #         cropped_image_size_w = cropped_image_size_w, cropped_image_size_h = cropped_image_size_h
            #     )
            # train_mean_avg_prec = mean_average_precision(
            #     pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            # )
            # summary_writer.add_scalar(f'train_mean_avg_prec_{iou}_{conf}', train_mean_avg_prec, ep)    
            # print(train_mean_avg_prec, ep)

            ### test on test set
            if "Joint" in vae_model:
                pred_boxes, target_boxes = get_bboxes_AE(
                    test_loader, task_model, DVAE_awa, True, iou_threshold=iou, threshold=conf, device=device,
                    cropped_image_size_w=cropped_image_size, cropped_image_size_h=cropped_image_size
                )
            else:
                pred_boxes, target_boxes = get_bboxes_AE(
                    test_loader, task_model, DVAE_awa, False, iou_threshold=iou, threshold=conf, device=device,
                    cropped_image_size_w = cropped_image_size_w, cropped_image_size_h = cropped_image_size_h
                )
            test_mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            summary_writer.add_scalar(f'test_mean_avg_prec_{iou}_{conf}', test_mean_avg_prec, ep)
            print(test_mean_avg_prec, ep)

            torch.save(DVAE_awa.state_dict(), new_model_path + f'/DVAE_awa-{ep}.pth')  

        ### export figure
        if (ep + 1) % save_interval == 0 or ep == num_epochs - 1 or ep == 0:
            max_imgs = min(batch_size, 8)
            vutils.save_image(torch.cat([obs[:max_imgs], obs_pred[:max_imgs]], dim=0).data.cpu(),
                '{}/image_{}.jpg'.format(fig_dir, ep), nrow=8)

    return

if __name__ == "__main__":
    """
    python train_concatDAE.py --dataset airbus --device 6 --lr 1e-4 --num_epochs 1500 --beta_rec 0.0 --beta_kl 0.0 --beta_task 0.1 --z_dim 12 -origz 80 --batch_size 64 --seed 40 --cross_penalty 0.0 --vae_model ResBasedVAE --norm_sample False -orig_e 299
    """

    model_path = './models/'
    dataset_dir = '/store/datasets/gym_fetch/'

    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument("-d", "--dataset", type=str, help="dataset to train on", default="airbus")
    parser.add_argument("-n", "--num_epochs", type=int, help="total number of epochs to run", default=250)
    parser.add_argument("-z", "--z_dim", type=int, help="latent dimensions", default=64)
    parser.add_argument("-origz", "--orig_z", type=int, help="zdim of original model", default=80)
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
    parser.add_argument("-wt", "--width", type=int, help="width of observation", default=80)
    parser.add_argument("-ht", "--height", type=int, help="width of observation", default=112)
    parser.add_argument("-orig_e", "--orig_epoch", type=int, help="epoch of original model", default=112)
    args = parser.parse_args()

    if args.norm_sample == 'True' or args.norm_sample == 'true':
        args.norm_sample = True
    else:
        args.norm_sample = False

    train_awa_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_epochs=args.num_epochs, weight_cross_penalty=args.cross_penalty, 
                  beta_kl=args.beta_kl, beta_rec=args.beta_rec, beta_task=args.beta_task, device=args.device, save_interval=50, lr=args.lr, seed=args.seed,
                  vae_model=args.vae_model, norm_sample=args.norm_sample, width=args.width, height=args.height, orig_z=args.orig_z, orig_epoch=args.orig_epoch)
    