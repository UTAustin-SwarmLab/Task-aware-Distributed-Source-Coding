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

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Create TILE_WIDTHxTILE_HEIGHT tiles with 64 pix overlap
TILE_WIDTH = 512
TILE_HEIGHT = 512
TILE_OVERLAP = 64
TRUNCATED_PERCENT = 0.3
print(f"Tile size: {TILE_WIDTH}x{TILE_HEIGHT} with {TILE_OVERLAP} pix overlap and {TRUNCATED_PERCENT} truncated percent")

files_dir = f'../airbus_dataset/{TILE_WIDTH}x{TILE_HEIGHT}_overlap{TILE_OVERLAP}_percent{TRUNCATED_PERCENT}_/train/'
test_dir = f'../airbus_dataset/{TILE_WIDTH}x{TILE_HEIGHT}_overlap{TILE_OVERLAP}_percent{TRUNCATED_PERCENT}_/val/'

LEARNING_RATE = 2e-5
device_num = 7
DEVICE = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 # 64 in original paper but resource exhausted error otherwise.
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
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
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss = loss.item())
        
    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")

transform_img = A.Compose([
    A.Resize(width=224, height=224),
    A.Resize(width=448, height=448),
    A.Blur(p=0.5, blur_limit=(3, 7)), 
    A.MedianBlur(p=0.5, blur_limit=(3, 7)), A.ToGray(p=0.5), 
    A.CLAHE(p=0.5, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
    ToTensorV2(p=1.0)
])

test_transform_img = A.Compose([
    A.Resize(width=224, height=224),
    A.Resize(width=448, height=448),
    ToTensorV2(p=1.0)
])

print("resize to 224x224 and 448x448")

def main():
    model = YoloV1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load("/home/pl22767/project/dtac-dev/airbus_detection/models/YoloV1_512x512/yolov1_upsample224_512x512_ep149_map0.98_0.74.pth")["state_dict"])
    model.float()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, mode='max', verbose=True)
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
        epoch += 150
        ### Train
        model.train()
        train_fn(train_loader, model, optimizer, loss_fn)
        
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP ({epoch}): {mean_avg_prec}")
        scheduler.step(mean_avg_prec)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            ### Test
            model.eval()
            train_fn(test_loader, model, optimizer, loss_fn)
            
            pred_boxes, target_boxes = get_bboxes(
                test_loader, model, iou_threshold=0.5, threshold=0.4, device=DEVICE
            )

            test_mean_avg_prec = mean_average_precision(
                pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
            )
            print(f"Test mAP: {test_mean_avg_prec}")

            if (mean_avg_prec >= 0.9 and test_mean_avg_prec >= 0.9) or epoch == EPOCHS - 1:
                checkpoint = {
                        "state_dict": model.state_dict(),
                        # "optimizer": optimizer.state_dict(),
                        "Train mAP": mean_avg_prec,
                        "Test mAP": test_mean_avg_prec,
                }
                save_checkpoint(checkpoint, filename=MODEL_PATH+f"yolov1_upsample224_{TILE_WIDTH}x{TILE_HEIGHT}_ep{epoch}_map{mean_avg_prec:.2f}_{test_mean_avg_prec:.2f}.pth")


def predictions():
    LOAD_MODEL = True
    LOAD_MODEL_FILE = MODEL_PATH + f"yolov1_512x512_ep80_map0.98_0.99.pth"

    model = YoloV1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    if LOAD_MODEL:
        model, optimizer = load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

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
        # print(pred_boxes[:5], target_boxes[:5]) [train_idx, class_idx, prob, x1, y1, x2, y2]
        test_mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Test mAP: {test_mean_avg_prec}")


if __name__ == "__main__":
    # main()
    predictions()

