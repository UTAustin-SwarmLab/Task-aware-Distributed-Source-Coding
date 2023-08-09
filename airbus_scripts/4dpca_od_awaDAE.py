import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import pandas as pd
import numpy as np
import os
import csv
import random
from torch.utils.data import DataLoader
### to start tensorboard:  tensorboard --logdir=./airbus_scripts/summary --port=6006
import argparse

from dtac.ClassDAE import *
from dtac.object_detection.yolo_model import YoloV1
from dtac.object_detection.od_utils import *


def dpca_od_vae(dataset="gym_fetch", z_dim=64, batch_size=32, num_epochs=250, beta_kl=1.0, beta_rec=0.0, beta_task=1.0, weight_cross_penalty=0.1, 
                 device=0, save_interval=30, lr=2e-4, seed=0, vae_model="CNNBasedVAE", width=448, height=448, start=0, end=97, randpca=False):
    ### set paths
    model_type = "AE"

    # task_model_path = "/home/pl22767/project/dtac-dev/airbus_scripts/models/YoloV1_896x512/yolov1_512x896_ep240_map0.97_0.99.pth"
    task_model_path = "/home/pl22767/project/dtac-dev/airbus_scripts/models/YoloV1_224x224/yolov1_aug_0.05_0.05_resize448_224x224_ep60_map0.98_0.83.pth"
    model_path = f'./models/{dataset}4_{z_dim}_randPCA_{model_type}_{vae_model}{width}x{height}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    if not randpca:
        model_path = model_path.replace("randPCA", "NoPCA")
    print("Random PCA: ", randpca)

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
        p = 0.0
        transform_img = A.Compose(transforms=[
            A.Resize(width=height, height=height),
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
        cropped_image_size_w2 = height - width # 70
        cropped_image_size_h = int(height/2)

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

    ### distributed models
    if vae_model == "ResBasedVAE":
        DVAE_awa = ResE4D1((3, cropped_image_size_w, cropped_image_size_h), (3, cropped_image_size_w, cropped_image_size_h), 
                           (3, cropped_image_size_w2, cropped_image_size_h), (3, cropped_image_size_w2, cropped_image_size_h), 
                           int(z_dim/4), int(z_dim/4), int(z_dim/4), int(z_dim/4), 4, 1).to(device)
        print("ResBasedVAE Input shape",(3, cropped_image_size_w, cropped_image_size_h), (3, cropped_image_size_w, cropped_image_size_h), 
                           (3, cropped_image_size_w2, cropped_image_size_h), (3, cropped_image_size_w2, cropped_image_size_h))
    ### load vae model
    DVAE_awa.load_state_dict(torch.load(model_path + f'/DVAE_awa-{num_epochs}.pth'))
    DVAE_awa.eval()

    ### test on test set
    results = AE_dpcaEQ4(test_loader, train_loader, task_model, DVAE_awa, z_dim, joint=False,
                        iou_threshold=iou, threshold=conf, device=device, start = start, end = end, 
                        cropped_image_size_w = cropped_image_size_w, cropped_image_size_h = cropped_image_size_h)

    header = ['dpca_dim', 'dim of z1 private', 'dim of z2 private', 'dim of z3 private', 'dim of z4 private', 'testmAP']
    csv_name = model_path.replace("./models/", "") + f'-ep{num_epochs}' + '.csv'
    with open('../csv_data/' + csv_name, 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(header)
        writer.writerows(results)

    return

if __name__ == "__main__":
    """        
    python 4dpca_od_awaDAE.py --dataset airbus --device 6 -n 0 -l 1e-4 -r 0.0 -k 0.0 -t 0.1 -z 80 -bs 64 --seed 0 -corpen 0.0 -vae ResBasedVAE -wt 56 -ht 112 -st 4 -end 40
    python 4dpca_od_awaDAE.py --dataset airbus --device 7 -n 299 -l 1e-4 -r 0.0 -k 0.0 -t 0.1 -z 40 -bs 64 --seed 0 -corpen 0.0 -vae JointResBasedVAE -wt 64 -ht 112 -st 4 -end 40
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
    parser.add_argument("-wt", "--width", type=int, help="image width", default=56)
    parser.add_argument("-ht", "--height", type=int, help="image height", default=112)
    parser.add_argument("-st", "--start", type=int, help="start epoch", default=4)
    parser.add_argument("-end", "--end", type=int, help="end epoch", default=96)
    parser.add_argument("-p", "--randpca", type=bool, help="perform random pca when training", default=True)
    args = parser.parse_args()

    dpca_od_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_epochs=args.num_epochs, 
                  weight_cross_penalty=args.cross_penalty, beta_kl=args.beta_kl, beta_rec=args.beta_rec, beta_task=args.beta_task, 
                  device=args.device, save_interval=50, lr=args.lr, seed=args.seed, vae_model=args.vae_model, 
                  width=args.width, height=args.height, start=args.start, end=args.end, randpca=args.randpca)
