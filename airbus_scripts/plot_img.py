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
from dtac.object_detection.yolo_model import YoloV1
from dtac.object_detection.od_utils import *


def train_awa_vae(dataset="gym_fetch", z_dim=64, batch_size=32, num_epochs=250, beta_kl=1.0, beta_rec=0.0, beta_task=1.0, weight_cross_penalty=0.1, 
                 device=0, save_interval=30, lr=2e-4, seed=0, vae_model="CNNBasedVAE", norm_sample=True, width=448, height=448, data_seed=-1, randpca=False):
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
    
    if not randpca:
        fig_dir = fig_dir.replace("randPCA", "NoPCA")
        model_path = model_path.replace("randPCA", "NoPCA")

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

    cur_iter = 0
    for ep in range(1):
        
        for batch_idx, (obs, out) in enumerate(tqdm(test_loader)):
            if batch_idx > 10:
                break

            obs_orig_112_0_255, out = obs.to(device), out.to(device)
            obs = obs_orig_112_0_255 / 255.0
            
            o1_batch = torch.zeros(obs.shape[0], obs.shape[1], cropped_image_size_h, cropped_image_size_h).to(device)
            o2_batch = torch.zeros(obs.shape[0], obs.shape[1], cropped_image_size_h, cropped_image_size_h).to(device)
            o1_batch[:, :, :cropped_image_size_w, :cropped_image_size_h] = obs[:, :, :cropped_image_size_w, :cropped_image_size_h]
            o2_batch[:, :, cropped_image_size_w-20:, :cropped_image_size_h] = obs[:, :, cropped_image_size_w-20:, :cropped_image_size_h]
            
            if  "Joint" in vae_model:
                obs6chan = torch.cat((o1_batch, o2_batch), dim=1)
                obs6chan_, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(obs6chan)
                obs_pred = torch.zeros_like(obs).to(device) # 3x112x112
                obs_pred[:, :, :cropped_image_size_w-20, :cropped_image_size_h] = obs6chan_[:, :3, :cropped_image_size_w-20, :cropped_image_size_h]
                obs_pred[:, :, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h] = 0.5 * (obs6chan_[:, :3, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h] 
                                                                                                        + obs6chan_[:, 3:, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h])
                obs_pred[:, :, cropped_image_size_w:, :cropped_image_size_h] = obs6chan_[:, 3:, cropped_image_size_w:, :cropped_image_size_h] 
            elif "Sep" in vae_model:
                obs_, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(o1_batch, o2_batch)

                obs_pred = torch.zeros_like(obs).to(device) # 3x112x112
                obs_pred[:, :, :cropped_image_size_w-20, :cropped_image_size_h] = obs_[:, :3, :cropped_image_size_w-20, :cropped_image_size_h]
                obs_pred[:, :, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h] = 0.5 * (obs_[:, :3, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h] 
                                                                                                        + obs_[:, 3:, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h])
                obs_pred[:, :, cropped_image_size_w:, :cropped_image_size_h] = obs_[:, 3:, cropped_image_size_w:, :cropped_image_size_h]
            elif "Joint" not in vae_model:
                obs_, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(o1_batch, o2_batch)

                obs_pred = torch.zeros_like(obs).to(device) # 3x112x112
                obs_pred[:, :, :cropped_image_size_w-20, :cropped_image_size_h] = obs_[:, :3, :cropped_image_size_w-20, :cropped_image_size_h]
                obs_pred[:, :, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h] = 0.5 * (obs_[:, :3, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h] 
                                                                                                        + obs_[:, 3:, cropped_image_size_w-20:cropped_image_size_w, :cropped_image_size_h])
                obs_pred[:, :, cropped_image_size_w:, :cropped_image_size_h] = obs_[:, 3:, cropped_image_size_w:, :cropped_image_size_h]

            obs_112_0_255 = obs_pred.clip(0, 1) * 255.0 ##################### important: clip the value to 0-255
            obs_pred_448_0_255 = F.interpolate(obs_112_0_255, size=(448, 448)) ### resize to 448x448
            out_pred = task_model(obs_pred_448_0_255)

            for idx in range(batch_size): ### for each image
                img = torch.cat((o1_batch[idx].unsqueeze(0), o2_batch[idx].unsqueeze(0)), dim=0)
                vutils.save_image(img, f'{fig_dir}/image_2view_{batch_idx}_{idx}.jpg', nrow=2)

            batch_size = obs.shape[0]
            true_bboxes = cellboxes_to_boxes(out)
            bboxes = cellboxes_to_boxes(out_pred)
            all_pred_boxes = []
            all_true_boxes = []
            for idx in range(batch_size):
                nms_boxes = non_max_suppression(bboxes[idx], iou_threshold=iou, threshold=conf, box_format="midpoint")

                for nms_box in nms_boxes:
                    all_pred_boxes.append([idx] + nms_box)

                for box in true_bboxes[idx]:
                    # many will get converted to 0 pred
                    if box[1] > conf:
                        all_true_boxes.append([idx] + box)

            cur_iter += 1

            obs_orig_112_0_255 = obs_orig_112_0_255.data.cpu().type(torch.uint8)
            obs_112_0_255 = obs_112_0_255.data.cpu().type(torch.uint8)
            ### substract null image reconstruction
            # obs_112_0_255 = (obs_112_0_255 - obs_112_0_255[0]).clip(0, 255)
            for idx in range(batch_size): ### for each image
                bbox = []
                truebbox = []
                for box in all_pred_boxes:
                    i  = box[0]
                    if idx == i:
                        x, y, w, h = box[-4:]
                        if x<0 or y<0 or w<=0 or h<=0:
                            continue
                        else:
                            x1, x2 = x - w/2, x + w/2
                            y1, y2 = y - h/2 ,y + h/2
                            # print(f'x1={x1}, x2={x2}, y1={y1}, y2={y2}, x={x}, y={y}, w={w}, h={h}')
                            bbox.append((np.array( (x1, y1, x2, y2) ) * 112).astype(int))
                    elif idx < i:
                        break
                for box in all_true_boxes:
                    i  = box[0]
                    if idx == i:
                        x, y, w, h = box[-4:]
                        if x<0 or y<0 or w<=0 or h<=0:
                            continue
                        else:
                            x1, x2 = x - w/2, x + w/2
                            y1, y2 = y - h/2 ,y + h/2
                            # print(f'x1={x1}, x2={x2}, y1={y1}, y2={y2}, x={x}, y={y}, w={w}, h={h}')
                            truebbox.append((np.array( (x1, y1, x2, y2) ) * 112).astype(int))
                    elif idx < i:
                        break
                bbox = torch.tensor(bbox, dtype=torch.int)
                truebbox = torch.tensor(truebbox, dtype=torch.int)
                if bbox.shape[0] != 0 and truebbox.shape[0] != 0:
                    true = vutils.draw_bounding_boxes(obs_orig_112_0_255[idx], truebbox, width=2, colors='red') / 255.0
                    pred = vutils.draw_bounding_boxes(obs_112_0_255[idx], bbox, width=2, colors='yellow') / 255.0
                    img = torch.cat((true.unsqueeze(0), pred.unsqueeze(0)), dim=0)
                    vutils.save_image(img, f'{fig_dir}/image_{batch_idx}_{idx}_airbus.jpg', nrow=1)
                else:
                    img = torch.cat((obs_orig_112_0_255[idx].unsqueeze(0), obs_112_0_255[idx].unsqueeze(0)), dim=0) / 255.0
                    vutils.save_image(img, f'{fig_dir}/image_{batch_idx}_{idx}.jpg', nrow=1)



    return

if __name__ == "__main__":
    """        
    python plot_img.py --dataset airbus --device 0 -l 1e-4 -n 299 -r 0.5 -k 0.0 -t 0.0 -z 80 -bs 64 --seed 1 -corpen 0.0 -vae ResBasedVAE -ns False -wt 80 -ht 112
    python plot_img.py --dataset airbus --device 7 -l 1e-4 -n 299 -r 0.0 -k 0.0 -t 0.1 -z 40 -bs 64 --seed 1 -corpen 0.0 -vae JointResBasedVAE -ns False -wt 80 -ht 112 -p True
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
    parser.add_argument("-p", "--randpca", type=bool, help="perform random pca when training", default=False)
    args = parser.parse_args()

    if args.norm_sample == 'True' or args.norm_sample == 'true':
        args.norm_sample = True
    else:
        args.norm_sample = False

    train_awa_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_epochs=args.num_epochs, 
                  weight_cross_penalty=args.cross_penalty, beta_kl=args.beta_kl, beta_rec=args.beta_rec, beta_task=args.beta_task, 
                  device=args.device, save_interval=50, lr=args.lr, seed=args.seed, vae_model=args.vae_model, norm_sample=args.norm_sample,
                  width=args.width, height=args.height, data_seed=args.data_seed, randpca=args.randpca)
