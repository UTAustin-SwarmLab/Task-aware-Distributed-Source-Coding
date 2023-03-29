### code modified from https://www.kaggle.com/code/vexxingbanana/yolov1-from-scratch-pytorch
import torch
import torch.nn.functional as F
from collections import Counter
from PIL import Image
import numpy as np
import random
import os

from dtac.object_detection.yolov8_loss import Loss

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class ImagesDataset(torch.utils.data.Dataset):
    def __init__(self, df, files_dir, S=7, B=2, C=3, transform=None):
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

def intersection_over_union(boxes_preds, boxes_labels, box_format='midpoint'):
    """
    Calculates intersection over union
    
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes are (x,y,w,h) or (x1,y1,x2,y2) respectively.
    
    Returns:
        tensor: Intersection over union for all examples
    """
    # boxes_preds shape is (N, 4) where N is the number of bboxes
    #boxes_labels shape is (n, 4)
    
    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
        
    if box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4] # Output tensor should be (N, 1). If we only use 3, we go to (N)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    #.clamp(0) is for the case when they don't intersect. Since when they don't intersect, one of these will be negative so that should become 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=1
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def cal_loss(
    loader,
    task_tr_model,
    Yolomodel,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_true_boxes = []
    all_pred_boxes = []
    loss_fn = Loss(task_tr_model)

    # make sure model is in eval before get bboxes
    task_tr_model.eval()
    train_idx = 0
    loss = []

    for batch_idx, (x, labels) in enumerate(loader):
        batch = {"batch_idx": [], "cls": [], "bboxes": []}

        x = x.to(device)
        x = F.interpolate(x, size=(512, 512))
        x_np = x.cpu().numpy().transpose(0, 2, 3, 1)
        assert max(x_np[0].flatten()) >= 1.0, "Image normalized"
        labels = labels.to(device)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        
        output = task_tr_model(x)
        for o in output[1]:
            print("o", o.shape)
        input("wait")

        for idx in range(batch_size):
            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)
                    # print("box", box) # [0.0, 1.0, 0.7441406846046448, 0.1533203125, 0.26171875, 0.2089843899011612]

                    batch["batch_idx"].append(idx)
                    batch["cls"].append(0)
                    batch["bboxes"].append(box[2:])

            if Yolomodel.predictor is None:
                results = Yolomodel(x_np[idx])
            Yolomodel.predictor.args.verbose = False
            Yolomodel.predictor.args.conf = threshold
            Yolomodel.predictor.args.iou = iou_threshold

            results = Yolomodel(x_np[idx])

            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                for ind, _ in enumerate(boxes.cls):
                    if boxes.conf[ind] > threshold:
                        pred_box = torch.cat((torch.ones(1, 1).to(device)*train_idx, boxes.cls[ind].view(-1, 1).to(device), 
                                              boxes.conf[ind].view(-1, 1).to(device), boxes.xywhn[ind].view(-1, 4).to(device)), dim=1).view(-1).tolist()
                        all_pred_boxes.append(pred_box)

            train_idx += 1

        batch["batch_idx"] = torch.tensor(batch["batch_idx"])
        batch["cls"] = torch.tensor(batch["cls"])
        batch["bboxes"] = torch.tensor(batch["bboxes"]).view(-1, 4) ### xywhn
        l, _ = loss_fn(output, batch)
        loss.append(l.item())

    loss = sum(loss) / len(loss)

    return loss, all_pred_boxes, all_true_boxes

def get_bboxes_v8(
    loader,
    AE,
    Yolomodel,
    iou_threshold,
    threshold,
    joint = True,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
    cropped_image_size_w = None, 
    cropped_image_size_h = None
):
    
    all_true_boxes = []
    all_pred_boxes = []

    # make sure model is in eval before get bboxes
    AE.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        x /= 255.0
        
        with torch.no_grad():
            if joint:
                x = AE(x)[0]
            else:
                x1 = x[:, :, :cropped_image_size_w, :cropped_image_size_h]
                x2 = x[:, :, cropped_image_size_w:, :cropped_image_size_h]
                x = AE(x1, x2, x)[0]

            x = x.clip(0, 1) * 255.0
            x = F.interpolate(x, size=(512, 512))
            x_np = x.cpu().numpy().transpose(0, 2, 3, 1)
            assert max(x_np[0].flatten()) >= 1.0, "Image normalized"
        
        labels = labels.to(device)
        true_bboxes = cellboxes_to_boxes(labels)
        batch_size = x.shape[0]

        for idx in range(batch_size):
            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            results = Yolomodel(x_np[idx])
            Yolomodel.predictor.args.verbose = False
            Yolomodel.predictor.args.conf = threshold
            Yolomodel.predictor.args.iou = iou_threshold

            # results = Yolomodel(x_np[idx])

            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                for ind, _ in enumerate(boxes.cls):
                    if boxes.conf[ind] > threshold:
                        pred_box = torch.cat((torch.ones(1, 1).to(device)*train_idx, boxes.cls[ind].view(-1, 1).to(device), \
                                              boxes.conf[ind].view(-1, 1).to(device), boxes.xywhn[ind].view(-1, 4).to(device)), dim=1).view(-1).tolist()
                        all_pred_boxes.append(pred_box)

            train_idx += 1

    # make sure model is back in train mode
    AE.train()

    return all_pred_boxes, all_true_boxes


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def get_bboxes_AE(
    loader,
    task_model,
    AE,
    joint:bool, 
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
    cropped_image_size_w = None, 
    cropped_image_size_h = None
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    task_model.eval()
    AE.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device) / 255.0
        
        labels = labels.to(device)

        ### encode and decode data
        with torch.no_grad():
            if joint: 
                x = AE(x)[0]
            else:
                x1 = x[:, :, :cropped_image_size_w, :cropped_image_size_h]
                x2 = x[:, :, cropped_image_size_w:, :cropped_image_size_h]
                x = AE(x1, x2, x)[0]

            if AE.training or task_model.training:
                print(AE.training, task_model.training)
                raise KeyError("VAE and task model should be in the training mode")
        
            x = x.clip(0, 1) * 255.0
            x = F.interpolate(x, size=(448, 448)) ### resize to 448x448
            predictions = task_model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    AE.train()
    return all_pred_boxes, all_true_boxes


def get_bboxes_localAE(
    loader,
    task_model,
    AE,
    joint:bool, 
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
    cropped_image_size_w = None, 
    cropped_image_size_h = None
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    task_model.eval()
    AE.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x, labels = x.to(device), labels.to(device)
        x = task_model.darknet(x)
        x = F.pad(x, (0, 1, 0, 1, 0, 0, 0, 0), "constant", 0)

        ### encode and decode data
        with torch.no_grad():
            if joint: 
                x = AE(x)[0]
            else:
                x1 = x[:, :, :cropped_image_size_w, :cropped_image_size_h]
                x2 = x[:, :, cropped_image_size_w:, :cropped_image_size_h]
                x = AE(x1, x2, x)[0]

            if AE.training or task_model.training:
                print(AE.training, task_model.training)
                raise KeyError("VAE and task model should be in the training mode")
        
            x = x[:, :, :-1, :-1]
            predictions = task_model.fcs(torch.flatten(x, start_dim=1))

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    AE.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7, C=3):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, C + 10)
    bboxes1 = predictions[..., C + 1:C + 5]
    bboxes2 = predictions[..., C + 6:C + 10]
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C + 5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C + 5]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds

def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint\n", "Train mAP:", checkpoint['Train mAP'], "\tTest mAP:", checkpoint['Test mAP'])
    model.load_state_dict(checkpoint["state_dict"])
    # optimizer.load_state_dict(checkpoint["optimizer"]
    return model, optimizer


### main function for debugging
if __name__ == '__main__':
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import pandas as pd
    from torch.utils.data import DataLoader
    from ultralytics import YOLO
    from dtac.object_detection.yolo_model import YoloV1
    from dtac.gym_fetch.ClassAE import *

    size = 112
    file_parent_dir = f'../../airbus_dataset/512x512_overlap64_percent0.3_/'
    files_dir = file_parent_dir + 'train/' # 'train/'
    images = [image for image in sorted(os.listdir(files_dir))
                if image[-4:]=='.jpg']
    annots = []
    for image in images:
        annot = image[:-4] + '.txt'
        annots.append(annot)
    
    device_num = 7
    device = torch.device("cpu") if device_num <= -1 else torch.device("cuda:" + str(device_num))
    images = pd.Series(images, name='images')
    annots = pd.Series(annots, name='annots')
    df = pd.concat([images, annots], axis=1)
    df = pd.DataFrame(df)

    p = 0.01
    transform_img = A.Compose([
        A.Resize(width=size, height=size),
        # A.Blur(p=p, blur_limit=(3, 7)), 
        # A.MedianBlur(p=p, blur_limit=(3, 7)), A.ToGray(p=p), 
        # A.CLAHE(p=p, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
        ToTensorV2(p=1.0)
    ])
    test_dataset = ImagesDataset(
        files_dir=files_dir,
        df=df,
        transform=transform_img
    )
    g = torch.Generator()
    g.manual_seed(0)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g
    )

    AE = E1D1((3, size, size), 96, False, 4-2, int(128/(2+1)), 2, 128).to(device)
    AE.load_state_dict(torch.load("../../airbus_detection/models/airbus_96_taskaware_AE_JointCNNBasedVAE64x112_kl0.0_rec1000.0_task0.1_bs64_cov0.0_lr0.0001_seed2/DVAE_awa-2399.pth"))
    
    # task_model_path = "./runs/train/weights/best.pt"
    task_model_path = "/home/pl22767/project/dtac-dev/airbus_detection/models/YoloV1_512x512/yolov1_512x512_ep80_map0.98_0.99.pth"
    task_model = YoloV1(split_size=7, num_boxes=2, num_classes=3).to(device)
    checkpoint = torch.load(task_model_path)
    task_model.load_state_dict(checkpoint["state_dict"])
    print("=> Loading checkpoint\n", "Train mAP:", checkpoint['Train mAP'], "\tTest mAP:", checkpoint['Test mAP'])
    task_model.eval()

    pred, true = get_bboxes_AE(test_loader, task_model, AE, joint=True, iou_threshold=0.5, threshold=0.4, device=device)
    import pdb; pdb.set_trace()
