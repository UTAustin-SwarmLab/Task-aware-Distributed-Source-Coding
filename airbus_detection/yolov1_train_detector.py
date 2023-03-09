### code modified from https://www.kaggle.com/code/vexxingbanana/yolov1-from-scratch-pytorch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
# import torchvision.transforms as transforms
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
TILE_WIDTH = 896
TILE_HEIGHT = 512
TILE_OVERLAP = 64
TRUNCATED_PERCENT = 0.3

files_dir = f'../airbus_dataset/{TILE_WIDTH}x{TILE_HEIGHT}_overlap{TILE_OVERLAP}_percent{TRUNCATED_PERCENT}_/train/'
test_dir = f'../airbus_dataset/{TILE_WIDTH}x{TILE_HEIGHT}_overlap{TILE_OVERLAP}_percent{TRUNCATED_PERCENT}_/val/'

LEARNING_RATE = 2e-5
device_num = 6
DEVICE = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64 # 64 in original paper but resource exhausted error otherwise.
WEIGHT_DECAY = 0
EPOCHS = 150
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
MODEL_PATH = f"./models/YoloV1_{TILE_WIDTH}x{TILE_HEIGHT}/"
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, df=df, files_dir=files_dir, S=7, B=2, C=3, transform=None):
        self.annotations = df
        self.files_dir = files_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.files_dir, self.annotations.iloc[index, 1])
        boxes = []
        class_dictionary = {'0':0}

        file1 = open(label_path, 'r')
        Lines = file1.readlines()
        for line in Lines:
            if line != '':
                if line == '\n':
                    raise ValueError('Empty line in label file')
                
                line = line.split()
                klass = class_dictionary[line[0]]
                centerx = float(line[1])
                centery = float(line[2])
                boxwidth = float(line[3])
                boxheight = float(line[4])
                boxes.append([klass, centerx, centery, boxwidth, boxheight])
                
        boxes = torch.tensor(boxes, dtype=torch.float)
        img_path = os.path.join(self.files_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        image = image.convert("RGB")

        if self.transform is not None:
            # image, boxes = self.transform(image, boxes)]
            
            image = np.array(image)
            image = self.transform(image=image)["image"]
            image = image.float()
            boxes = boxes


        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, self.C] == 0:
                # Set that there exists an object
                label_matrix[i, j, self.C] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )

                label_matrix[i, j, 4:8] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

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
    A.Resize(width=448, height=448),
    A.Blur(p=0.5, blur_limit=(3, 7)), 
    A.MedianBlur(p=0.5, blur_limit=(3, 7)), A.ToGray(p=0.5), 
    A.CLAHE(p=0.5, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
    ToTensorV2(p=1.0)
])

test_transform_img = A.Compose([
    A.Resize(width=448, height=448),
    ToTensorV2(p=1.0)
])

def main():
    model = YoloV1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    model.float()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.1, patience=3, mode='max', verbose=True)
    loss_fn = YoloLoss()

    train_dataset = ImagesDataset(
        transform=transform_img,
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

    for epoch in range(EPOCHS):
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

        if (mean_avg_prec >= 0.8 and test_mean_avg_prec >= 0.8 and epoch % 5 == 0) or epoch == EPOCHS - 1:
            checkpoint = {
                    "state_dict": model.state_dict(),
                    # "optimizer": optimizer.state_dict(),
                    "Train mAP": mean_avg_prec,
                    "Test mAP": test_mean_avg_prec,
            }
            save_checkpoint(checkpoint, filename=MODEL_PATH+f"yolov1_{TILE_WIDTH}x{TILE_HEIGHT}_ep{epoch}_map{mean_avg_prec:.2f}_{test_mean_avg_prec:.2f}.pth")


def predictions():
    LOAD_MODEL = True
    LOAD_MODEL_FILE = MODEL_PATH + f"yolov1_512x896_ep100_map0.97_0.99.pth"

    EPOCHS = 1
    model = YoloV1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        model, optimizer = load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
    # model.float()

    train_dataset = ImagesDataset(
        transform=transform_img,
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

    for epoch in range(EPOCHS):
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

