import clip
import torch
from PIL import Image
import time

device_num = 1
device = torch.device("cpu") if device_num <= -1 else torch.device("cuda:" + str(device_num))

t_s = time.time()
model, preprocess = clip.load("ViT-B/32", device=device)
t_e = time.time()
print("Time loading:", t_e - t_s)

image = preprocess(Image.open("bus.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a bus", "a car", "two men in front of a bus"]).to(device)

with torch.no_grad():
    t_s = time.time()
    image_features = model.encode_image(image)
    t_e = time.time()
    text_features = model.encode_text(text)
    t_e2 = time.time()
    print("Time img:", t_e - t_s)
    print("Time txt:", t_e2 - t_e)

    logits_per_image, logits_per_text = model(image, text)
    # print(logits_per_image.shape, logits_per_text.shape) # transposed
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print(image_features.shape, text_features.shape)
print("Label probs:", probs)  # prints: [[9.902e-01 9.857e-03 1.407e-04]]