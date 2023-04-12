### code modified from https://www.kaggle.com/code/vexxingbanana/yolov1-from-scratch-pytorch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import pandas as pd
import numpy as np
import os
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dtac.object_detection.yolo_model import YoloV1, YoloLoss
from dtac.object_detection.od_utils import *
from dtac.ClassAE import *

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

TILE_WIDTH = 224
TILE_HEIGHT = 224
TILE_OVERLAP = 28
TRUNCATED_PERCENT = 0.3
print(f"Tile size: {TILE_WIDTH}x{TILE_HEIGHT} with {TILE_OVERLAP} pix overlap and {TRUNCATED_PERCENT} truncated percent")

files_dir = f'../airbus_dataset/{TILE_WIDTH}x{TILE_HEIGHT}_overlap{TILE_OVERLAP}_percent{TRUNCATED_PERCENT}_/train/'
test_dir = f'../airbus_dataset/{TILE_WIDTH}x{TILE_HEIGHT}_overlap{TILE_OVERLAP}_percent{TRUNCATED_PERCENT}_/val/'

LEARNING_RATE = 1e-4 # 2e-5
device_num = 4
DEVICE = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 # 64 in original paper but resource exhausted error otherwise.
WEIGHT_DECAY = 0
EPOCHS = 250
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False 
resize_shape = 112
MODEL_PATH = f"./models/YoloV1_{TILE_WIDTH}x{TILE_HEIGHT}/"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

g = torch.Generator()
g.manual_seed(0)

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

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE).type(torch.cuda.FloatTensor), y.to(DEVICE).type(torch.cuda.FloatTensor)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss = loss.item())
        
    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")

p = 0.05
transform_img = A.Compose([
    A.RandomResizedCrop(width=resize_shape, height=resize_shape),
    # A.Resize(width=resize_shape, height=resize_shape),
    # A.augmentations.crops.transforms.Crop(x_min=0, y_min=0, x_max=64, y_max=112, p=1.0),
    A.Resize(width=448, height=448),
    A.Blur(p=p, blur_limit=(3, 7)), 
    A.MedianBlur(p=p, blur_limit=(3, 7)), A.ToGray(p=p), 
    A.CLAHE(p=p, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
    ToTensorV2(p=1.0)]
    , bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=500, min_visibility=0.3)
)

test_transform_img = A.Compose([
    # A.Resize(width=resize_shape, height=resize_shape),
    # A.augmentations.crops.transforms.Crop(x_min=0, y_min=0, x_max=64, y_max=112, p=1.0),
    A.Resize(width=448, height=448),
    A.Blur(p=p, blur_limit=(3, 7)), 
    A.MedianBlur(p=p, blur_limit=(3, 7)), A.ToGray(p=p), 
    A.CLAHE(p=p, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
    ToTensorV2(p=1.0)
])

print(f"resize to {resize_shape}x{resize_shape} and 448x448")

def main():
    model = YoloV1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    # model.load_state_dict(torch.load("/home/pl22767/project/dtac-dev/airbus_detection/models/YoloV1_512x512/yolov1_upsample224_512x512_ep149_map0.98_0.74.pth")["state_dict"])

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=1, mode='max', verbose=True) # patience=3
    loss_fn = YoloLoss()

    train_dataset = ImagesDataset(
        files_dir=files_dir,
        df=df,
        transform=transform_img
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    test_dataset = ImagesDataset(
        transform=test_transform_img,
        df=test_df,
        files_dir=test_dir
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    for epoch in range(EPOCHS):
        ### Train
        model.train()
        train_fn(train_loader, model, optimizer, loss_fn)
        
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            ### Test
            model.eval()
            ############# do not need to train in test ###############
            # train_fn(test_loader, model, optimizer, loss_fn)

            ### Calculate test mAP
            pred_boxes, target_boxes = get_bboxes(
                train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
            )

            mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Train mAP ({epoch}): {mean_avg_prec}")
            scheduler.step(mean_avg_prec)
            
            ### Calculate test mAP
            pred_boxes, target_boxes = get_bboxes(
                test_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
            )

            test_mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Test mAP: {test_mean_avg_prec}")

            if (mean_avg_prec >= 0.80 and test_mean_avg_prec >= 0.80) or epoch == EPOCHS - 1 or epoch % 50 == 0:
                checkpoint = {
                        "state_dict": model.state_dict(),
                        # "optimizer": optimizer.state_dict(),
                        "Train mAP": mean_avg_prec,
                        "Test mAP": test_mean_avg_prec,
                }
                save_checkpoint(checkpoint, filename=MODEL_PATH+f"yolov1_aug_{p}_{p}_rc_resize{resize_shape}_{TILE_WIDTH}x{TILE_HEIGHT}_ep{epoch}_map{mean_avg_prec:.2f}_{test_mean_avg_prec:.2f}.pth")

def predictions():
    LOAD_MODEL = True
    TASK_MODEL_FILE = MODEL_PATH + f"yolov1_aug_0.05_0.05_resize448_224x224_ep60_map0.98_0.83.pth"

    model = YoloV1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    if LOAD_MODEL:
        model, optimizer = load_checkpoint(torch.load(TASK_MODEL_FILE), model, optimizer)

    train_dataset = ImagesDataset(
        transform=transform_img,
        df=df,
        files_dir=files_dir
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    test_dataset = ImagesDataset(
        transform=test_transform_img,
        df=test_df,
        files_dir=test_dir
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    for epoch in range(1):
        model.eval()
        ### test on train set
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        ### test on test set
        pred_boxes, target_boxes = get_bboxes(
            test_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        )
        test_mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Test mAP: {test_mean_avg_prec}")

def predictionsV8():
    from ultralytics import YOLO

    LOAD_MODEL = True
    TASK_MODEL_FILE = "./runs/detect/train/weights/best.pt"

    task_model = YOLO(TASK_MODEL_FILE)
    task_model.to(DEVICE)
    model = task_model.model
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_dataset = ImagesDataset(
        transform=transform_img,
        df=df,
        files_dir=files_dir
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    test_dataset = ImagesDataset(
        transform=test_transform_img,
        df=test_df,
        files_dir=test_dir
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    for epoch in range(1):
        model.eval()
        iou = 0.5
        thres = 0.4
        print(f"iou: {iou}, thres: {thres}")
        ### test on train set
        loss, pred_boxes, target_boxes = cal_loss(
            train_loader, model, task_model, iou_threshold=iou, threshold=thres, device=DEVICE
        )
        # pred_boxes, target_boxes = get_bboxes(
        #     train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        # )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        ### test on test set
        loss, pred_boxes, target_boxes = cal_loss(
            test_loader, model, task_model, iou_threshold=iou, threshold=thres, device=DEVICE
        )
        # pred_boxes, target_boxes = get_bboxes(
        #     test_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        # )
        test_mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=iou, box_format="midpoint"
        )
        # print(pred_boxes[:5], target_boxes[:5]) [train_idx, class_idx, prob, x, y, w, h]
        print(f"Test mAP: {test_mean_avg_prec}")
        # print(f"Test loss: {loss}")

def predictions_AE():
    ### load task
    # TASK_MODEL_FILE = MODEL_PATH + f"yolov1_512x512_ep80_map0.98_0.99.pth"
    TASK_MODEL_FILE = "/home/pl22767/project/dtac-dev/airbus_detection/models/YoloV1_512x512/yolov1_upsample224_512x512_ep149_map0.98_0.74.pth"

    task_model = YoloV1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    optimizer = optim.Adam(
        task_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    task_model, optimizer = load_checkpoint(torch.load(TASK_MODEL_FILE), task_model, optimizer)

    ### load AE
    AE_MODEL_FILE = "./models/" + "airbus_96_taskaware_AE_JointResBasedVAE64x112_kl1.0_rec1000.0_task0.1_bs64_cov0.0_lr0.0001_seed0/DVAE_awa-1199.pth"
    vae_model = "JointResBasedVAE64x112"
    cropped_image_size = 112    
    cropped_image_size_w = 64
    cropped_image_size_h = 112
    seed = 0
    DVAE_awa = ResE1D1((3, cropped_image_size, cropped_image_size), 96, False, 4-seed, 3-seed).to(DEVICE)
    # DVAE_awa = E1D1((3, cropped_image_size, cropped_image_size), 96, False, 4-seed, int(128/(seed+1)), 2, 128).to(DEVICE)
    DVAE_awa.load_state_dict(torch.load(AE_MODEL_FILE))
    # DVAE_awa = nn.Identity().to(DEVICE)

    ### load datasets
    transform_img = A.Compose([
        A.Resize(width=cropped_image_size, height=cropped_image_size),
        # A.Blur(p=0.5, blur_limit=(3, 7)), 
        # A.MedianBlur(p=0.5, blur_limit=(3, 7)), A.ToGray(p=0.5), 
        # A.CLAHE(p=0.5, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
        ToTensorV2(p=1.0)
    ])

    test_transform_img = A.Compose([
        A.Resize(width=cropped_image_size, height=cropped_image_size),
        ToTensorV2(p=1.0)
    ])

    train_dataset = ImagesDataset(
        transform=transform_img,
        df=df,
        files_dir=files_dir
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    test_dataset = ImagesDataset(
        transform=test_transform_img,
        df=test_df,
        files_dir=test_dir
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    for epoch in range(1):
        DVAE_awa.eval()
        task_model.eval()

        ### test on train set
        if "Joint" in vae_model:
            pred_boxes, target_boxes = get_bboxes_AE(
                train_loader, task_model, DVAE_awa, True, iou_threshold=0.5, threshold=0.4, device=DEVICE,
                cropped_image_size_w=cropped_image_size, cropped_image_size_h=cropped_image_size
            )
        else:
            pred_boxes, target_boxes = get_bboxes_AE(
                train_loader, task_model, DVAE_awa, False, iou_threshold=0.5, threshold=0.4, device=DEVICE,
                cropped_image_size_w = cropped_image_size_w, cropped_image_size_h = cropped_image_size_h
            )
        train_mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {train_mean_avg_prec}")

        ### test on test set
        if "Joint" in vae_model:
            pred_boxes, target_boxes = get_bboxes_AE(
                test_loader, task_model, DVAE_awa, True, iou_threshold=0.5, threshold=0.4, device=DEVICE,
                cropped_image_size_w=cropped_image_size, cropped_image_size_h=cropped_image_size
            )
        else:
            pred_boxes, target_boxes = get_bboxes_AE(
                test_loader, task_model, DVAE_awa, False, iou_threshold=0.5, threshold=0.4, device=DEVICE,
                cropped_image_size_w = cropped_image_size_w, cropped_image_size_h = cropped_image_size_h
            )
        test_mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Test mAP: {test_mean_avg_prec}")

if __name__ == "__main__":
    main()
    # predictions()
    # predictions_AE()

    # predictionsV8()