import torch
import numpy as np
from torchsummary import summary
from ultralytics import YOLO
import cv2

device_num = 1
device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
dic = torch.load("../airbus_detection/runs/detect/train10/weights/best.pt")

# print(dic.keys(), dic["train_args"])
task_model = dic["model"].to(device)
# task_model.eval()
results = task_model(torch.rand(1, 3, 512, 512).half().to(device))
print(results[0].grad)

# random_input = np.random.rand(1, 3, 512, 512)
# results = task_model(random_input)
print(results[0].shape)
print(results[1][0].shape)
print(results[1][1].shape)
print(results[1][2].shape)

# for result in results:
#     # detection
#     result.boxes.xyxy   # box with xyxy format, (N, 4)
#     result.boxes.xywh   # box with xywh format, (N, 4)
#     result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
#     result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
#     result.boxes.conf   # confidence score, (N, 1)
#     result.boxes.cls    # cls, (N, 1)
