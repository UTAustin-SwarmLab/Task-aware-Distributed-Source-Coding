import torch
import torch.nn.functional as F
import albumentations as A
from dtac.gym_fetch.ClassAE import *

### main ###
device_num = 4
cropped_image_size = 64
z_dim = 64
norm_sample = False
seed = 0
device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")

# model.eval()
# print(model.training)

# def test(model):
#     # model.train()
#     # print(model.training)
#     model.eval()
#     print(model.training)
#     for para in model.parameters():
#         print(para.requires_grad)
#         # if para.requires_grad:
#         #     print(para.requires_grad)
#     # print(model.training)

# optimizer = torch.optim.Adam(model.dec.parameters(), lr=1e-3)
# L = torch.nn.MSELoss()

# x = torch.rand(1, 3, cropped_image_size, cropped_image_size).to(device)
# z = model(x)[0]

# loss = L(z, x)

# for para in model.parameters():
#     PARA = para.clone().detach()
#     print(para.requires_grad)
#     break

# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# for para in model.parameters():
#     delta = PARA - para
#     print(delta.max(), delta.min())
#     break



# model = E1D1((3, cropped_image_size, cropped_image_size), z_dim, norm_sample, 4-seed, int(128/(seed+1)), 2, 128).to(device)
# model2 = E1D1((3, 2*cropped_image_size, 2*cropped_image_size), z_dim, norm_sample, 4-seed, int(128/(seed+1)), 2, 128).to(device)
# x = torch.rand(1, 3, cropped_image_size, cropped_image_size, requires_grad=True).to(device)
# output = model(x)[0].clip(0, 1)
# L_output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
# # L_output = A.Resize(width=2*cropped_image_size, height=2*cropped_image_size)(image=output)['image'].to(device)
# output2 = model2(L_output)[0]
# loss = output2.mean()
# loss.backward()
# print(output2.retain_grad())
# print(output.retain_grad())
# print(x.retain_grad())


x = torch.randn(20, 3, 24, 24, requires_grad=True)
x_res = F.interpolate(x, size=(30, 30))
print(x.shape, x_res.shape)
loss = x_res.mean()
loss.backward()
# print(x.grad)
# print(loss.grad)
