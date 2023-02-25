import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import argparse
import random
import numpy as np
import os
import sys

from dtac.gym_fetch.ClassAE import *
from curl_sac import Actor

def random_crop(imgs, out=84, w1=None, h1=None):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    if w1 is None and h1 is None:
        w1 = np.random.randint(0, crop_max, n)
        h1 = np.random.randint(0, crop_max, n)
    cropped = torch.empty((n, c, out, out), dtype=imgs.dtype, device=imgs.device)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped, w1, h1

def center_crop_image(image, output_size):
    h, w = image.shape[-2:]
    if h > output_size: #center cropping
        new_h, new_w = output_size, output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        if len(image.shape) == 3:
            image = image[:, top:top + new_h, left:left + new_w]
        elif len(image.shape) == 4:
            image = image[:, :, top:top + new_h, left:left + new_w]
        else:
            raise ValueError("image should be 3 or 4 dimensional")
        return image
    else: #center translate
        shift = output_size - h
        shift = shift // 2
        if len(image.shape) == 3:
            new_image = np.zeros((image.shape[0], output_size, output_size))
            new_image[:, shift:shift + h, shift:shift+w] = image
        elif len(image.shape) == 4:
            new_image = np.zeros((image.shape[0], image.shape[1], output_size, output_size))
            new_image[:, :, shift:shift + h, shift:shift+w] = image
        return new_image

def train_awa_vae(dataset="gym_fetch", z_dim=64, batch_size=32, num_epochs=250, beta_kl=1.0, beta_rec=0.0, beta_task=1.0, weight_cross_penalty=0.1, 
                  device=0, save_interval=5, lr=2e-4, seed=0, model_path=None, dataset_dir=None, vae_model="CNNBasedVAE", task_model_epoch=99, norm_sample=True, rand_crop=False):

    # if norm_sample == False:

    ### set paths
    LOG_DIR = f'./summary/{dataset}_{z_dim}_aware{norm_sample}_{rand_crop}{vae_model}_{beta_kl}_{beta_rec}_{beta_task}_{batch_size}_{weight_cross_penalty}'
    fig_dir = f'./figures/{dataset}_{z_dim}_aware{norm_sample}_{rand_crop}{vae_model}_{beta_kl}_{beta_rec}_{beta_task}_{batch_size}_{weight_cross_penalty}'
    # task_model_path = model_path + f'actor_nocrop2image/actor2image-{task_model_epoch}.pth'
    task_model_path = "/home/pl22767/DistributedTaskAwareCompression/fetch_sim/data/FetchPickAndPlace-v1/sparse-rad_sac-pixel-crop-01-26-FetchPickAndPlace-v1-im84-b128-nu1-change_model-s13618-id94666/model/actor_254000.pt"
    model_path += f'./{dataset}_{z_dim}_aware{norm_sample}_{rand_crop}{vae_model}_{beta_kl}_{beta_rec}_{beta_task}_{batch_size}_{weight_cross_penalty}'
    summary_writer = SummaryWriter(os.path.join(LOG_DIR, 'tb'))

    ### Set the random seed
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)
    device = torch.device("cpu") if args.device <= -1 else torch.device("cuda:" + str(args.device))

    ### Load the dataset
    if dataset == "gym_fetch":
        # 10k examples of 'obs', 'next_obs', 'action', 'reward', 'done' 
        # 'obs', 'next_obs': type <class 'numpy.ndarray'> shape (10000, 6, 128, 128)
        # 'action' type <class 'numpy.ndarray'> shape (10000, 4)
        # 'reward' type <class 'numpy.ndarray'> shape (10000, 1)
        # 'done' type <class 'numpy.ndarray'> shape (10000, 1) float32 0.0 (done) or 1.0 (not done)
        reach = torch.load(dataset_dir + 'reach.pt')
        obs1 = reach[0][:, 0:3, :, :]
        obs2 = reach[0][:, 3:6, :, :]
        action = reach[2]
        cropped_image_size = 128
    elif dataset == "PickAndPlace":
        pick = torch.load(dataset_dir + '0_20001.pt')
        obs1 = pick[0][:, 0:3, :, :]
        obs2 = pick[0][:, 3:6, :, :]
        cropped_image_size = 84
    else:
        raise NotImplementedError

    ### load task model
    task_model = Actor((6, cropped_image_size, cropped_image_size), (4,), 1024, 'pixel', 50, -10, 2, 4, 32, None, False).to(device)
    task_model.load_state_dict(torch.load(task_model_path))
    task_model.eval()

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if vae_model == "CNNBasedVAE":
        # DVAE_awa = E2D1(obs1.shape[1:], obs2.shape[1:], int(z_dim/2), int(z_dim/2), norm_sample=norm_sample).to(device)
        DVAE_awa = E2D1((3, 84, 84), (3, 84, 84), int(z_dim/2), int(z_dim/2), norm_sample=norm_sample).to(device)
        print("CNNBasedVAE Input shape", (3, 84, 84))
    elif vae_model == "SVAE":
        DVAE_awa = SoftIntroVAE(arch="dist", cdim=6, zdim=z_dim, image_size=128, norm_sample=norm_sample).to(device)
    else:
        raise NotImplementedError
    optimizer = optim.Adam(DVAE_awa.parameters(), lr=lr)

    index = np.arange(len(obs1))
    n_batches = len(obs1) // batch_size

    cur_iter = 0
    for ep in range(num_epochs):
        ep_loss = []
        np.random.shuffle(index)
        for i in range(n_batches):
            b_idx = index[i * batch_size:(i + 1) * batch_size]
            o1_batch = torch.tensor(obs1[b_idx], device=device).float() / 255
            o2_batch = torch.tensor(obs2[b_idx], device=device).float() / 255

            if dataset == "PickAndPlace":
                if rand_crop == True:
                    ### random crop
                    o1_batch, w, h = random_crop(o1_batch, 84)
                    o2_batch = random_crop(o2_batch, 84, w, h)[0]
                    # obs_pred = random_crop(obs_pred, 84, w, h)[0]
                else:
                    ### center crop
                    o1_batch = center_crop_image(o1_batch, 84)
                    o2_batch = center_crop_image(o2_batch, 84)
                    # obs_pred = center_crop_image(obs_pred, 84)

                if vae_model == "CNNBasedVAE":
                    obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(o1_batch, o2_batch)
                elif vae_model == "SVAE":
                    obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(torch.cat((o1_batch, o2_batch), dim=1))

                task_output = task_model(obs_pred.clip(0, 1))[0]
                obs = torch.cat((o1_batch, o2_batch), dim=1)
                action_gt = task_model(obs)[0]
                task_loss = torch.mean((action_gt - task_output) ** 2)
                loss = beta_task * task_loss + beta_rec * loss_rec + beta_kl * (kl1 + kl2) + weight_cross_penalty * loss_cor

            elif dataset == "gym_fetch": # gym_fetch
                if vae_model == "CNNBasedVAE":
                    obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(o1_batch, o2_batch)
                elif vae_model == "SVAE":
                    obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(torch.cat((o1_batch, o2_batch), dim=1))

                task_output = task_model(obs_pred.clip(0, 1))[0]
                action_gt = torch.tensor(action[b_idx], device=device).float()
                task_loss = torch.mean((action_gt - task_output) ** 2)
                loss = beta_task * task_loss + beta_rec * loss_rec + beta_kl * (kl1 + kl2) + weight_cross_penalty * loss_cor
            else:
                raise NotImplementedError

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
        if (ep + 1) % save_interval == 0 or (ep + 1) <= 10:
            torch.save(DVAE_awa.state_dict(), model_path + f'/DVAE_awa-{ep}.pth')  

        ### export figure
        if (ep + 1) % save_interval == 0 or ep == num_epochs - 1 or ep == 0:
            max_imgs = min(batch_size, 8)
            vutils.save_image( torch.cat([o1_batch[:max_imgs], o2_batch[:max_imgs], obs_pred[:max_imgs, 0:3], obs_pred[:max_imgs, 3:6]], dim=0).data.cpu(),
                '{}/image_{}.jpg'.format(fig_dir, ep), nrow=8)

    return

if __name__ == "__main__":
    """        
    python train_awaVAE.py --dataset gym_fetch --device 0 --lr 1e-4 --num_epochs 3000 --beta_rec 0.0 --beta_task 10 --z_dim 64 --batch_size 128 --seed 0 --cross_penalty 0.0 --vae_model SVAE --norm_sample False --rand_crop True
    python train_awaVAE.py --dataset PickAndPlace --device 0 --lr 1e-4 --num_epochs 3000 --beta_rec 10000.0 --beta_kl 25.0 --beta_task 100 --z_dim 64 --batch_size 128 --seed 0 --cross_penalty 10.0 --vae_model CNNBasedVAE --norm_sample False --rand_crop True
    """

    model_path = '/home/pl22767/DistributedTaskAwareCompression/gym_fetch/models/'
    dataset_dir = '/store/datasets/gym_fetch/'

    parser = argparse.ArgumentParser(description="train Soft-IntroVAE")
    parser.add_argument("-d", "--dataset", type=str,
                        help="dataset to train on: ['cifar10', 'mnist', 'fmnist', 'svhn', 'monsters128', 'celeb128', "
                             "'celeb256', 'celeb1024']")
    parser.add_argument("-n", "--num_epochs", type=int, help="total number of epochs to run", default=250)
    parser.add_argument("-z", "--z_dim", type=int, help="latent dimensions", default=64)
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
    parser.add_argument("-task_e", "--task_model_epoch", type=int, help="task model eposh", default=99)
    parser.add_argument("-ns", "--norm_sample", type=str, help="Sample from Normal distribution (VAE) or not", default="True")
    parser.add_argument("-crop", "--rand_crop", type=bool, help="randomly crop images", default=False)
    args = parser.parse_args()

    if args.norm_sample == 'True' or args.norm_sample == 'true':
        args.norm_sample = True
    else:
        args.norm_sample = False

    train_awa_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_epochs=args.num_epochs, weight_cross_penalty=args.cross_penalty, 
                  beta_kl=args.beta_kl, beta_rec=args.beta_rec, beta_task=args.beta_task, device=args.device, save_interval=50, lr=args.lr, seed=args.seed,
                  model_path=model_path, dataset_dir=dataset_dir, vae_model=args.vae_model, task_model_epoch=args.task_model_epoch, norm_sample=args.norm_sample,
                  rand_crop=args.rand_crop)
