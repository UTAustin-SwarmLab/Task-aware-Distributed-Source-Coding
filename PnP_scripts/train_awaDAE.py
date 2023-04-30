import torch
import torch.optim as optim
### to start tensorboard: tensorboard --logdir=./PnP_scripts/summary --port=6006
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import argparse
import random
import numpy as np
import os

from dtac.ClassDAE import *
from dtac.gym_fetch.curl_sac import Actor
from dtac.gym_fetch.utils import center_crop_image, random_crop_image

from eval_DAE import evaluate

def train_awa_vae(dataset="gym_fetch", z_dim=64, batch_size=32, num_epochs=250, beta_kl=1.0, beta_rec=0.0, beta_task=1.0, weight_cross_penalty=0.1, device=0, save_interval=5,
                  lr=2e-4, seed=0, model_path=None, dataset_dir=None, vae_model="CNNBasedVAE", norm_sample=True, rand_crop=False, randpca=False, histepoch=0):
    ### set paths
    if norm_sample:
        model_type = "DVAE"
    else:
        model_type = "DAE"
    if rand_crop:
        rc = "randcrop"
    else:
        rc = "nocrop"
    if not randpca:
        rc = "NoPCA_" + rc
    LOG_DIR = f'./summary/{dataset}_{z_dim}_randPCA_8_48_{model_type}_{rc}_{vae_model}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    fig_dir = f'./figures/{dataset}_{z_dim}_randPCA_8_48_{model_type}_{rc}_{vae_model}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
    model_path += f'{dataset}_{z_dim}_randPCA_8_48_{model_type}_{rc}_{vae_model}_kl{beta_kl}_rec{beta_rec}_task{beta_task}_bs{batch_size}_cov{weight_cross_penalty}_lr{lr}_seed{seed}'
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
        pick = torch.load(dataset_dir + 'pnp_128_20011.pt')
        
        pick[2] = torch.tensor(pick[2], dtype=torch.float32)
        unique, idx, counts = torch.unique(pick[2], dim=0, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]

        obs1 = pick[0][first_indicies, 0:3, :, :]
        obs2 = pick[0][first_indicies, 3:6, :, :]
        a_gt = pick[2][first_indicies, :]
        cropped_image_size = 112
        task_model_path = "/store/datasets/gym_fetch/pnp_actor_300000.pt"
        ### load task model
        task_model = Actor((6, cropped_image_size, cropped_image_size), (4,), 1024, 'pixel', 50, -10, 2, 4, 32, None, False).to(device)
        task_model.load_state_dict(torch.load(task_model_path))
        task_model.eval()
    elif dataset == "Lift":
        pick = torch.load('./lift_hardcode.pt')
        pick[2] = torch.tensor(pick[2], dtype=torch.float32)
        obs1 = pick[0][:, 0:3, :, :]
        obs2 = pick[0][:, 3:6, :, :]
        a_gt = pick[2][:, :]
        cropped_image_size = 112
        task_model_path = '/home/pl22767/project/dtac-dev/PnP_scripts/models/lift_actor_nocrop2image_sac_lr0.001_seed1/actor2image-849_0.82.pth'
        task_model = torch.load(task_model_path).to(device)
        task_model.eval()
    else:
        raise NotImplementedError


    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if vae_model == "CNNBasedVAE":
        nn_complexity = 0
        DVAE_awa = E2D1((3, cropped_image_size, cropped_image_size), (3, cropped_image_size, cropped_image_size), int(z_dim/2), int(z_dim/2), norm_sample, 4-nn_complexity, int(128/(nn_complexity+1)), 2, 128).to(device)
        print("CNNBasedVAE Input shape", (3, cropped_image_size, cropped_image_size))
    elif vae_model == "ResBasedVAE":
        DVAE_awa = ResE2D1((3, cropped_image_size, cropped_image_size), (3, cropped_image_size, cropped_image_size), int(z_dim/2), int(z_dim/2), norm_sample, 4, 1).to(device)
        print("ResBasedVAE Input shape", (3, cropped_image_size, cropped_image_size))
    elif vae_model == "JointCNNBasedVAE":
        nn_complexity = 0
        DVAE_awa = E1D1((6, cropped_image_size, cropped_image_size), z_dim, norm_sample, 4-nn_complexity, int(128/(nn_complexity+1)), 2, 128).to(device)
        print("JointCNNBasedVAE Input shape", (6, cropped_image_size, cropped_image_size))
    elif vae_model == "JointResBasedVAE":
        DVAE_awa = ResE1D1((6, cropped_image_size, cropped_image_size), z_dim, norm_sample, 4, 1).to(device)
        print("JointResBasedVAE Input shape", (6, cropped_image_size, cropped_image_size))
    elif vae_model == "SepResBasedVAE":
        DVAE_awa = ResE2D2((3, cropped_image_size, cropped_image_size), (3, cropped_image_size, cropped_image_size), int(z_dim/2), int(z_dim/2), norm_sample, 4, 1).to(device)
        print("SepResBasedVAE Input shape", (3, cropped_image_size, cropped_image_size), (3, cropped_image_size, cropped_image_size))
    else:
        raise NotImplementedError
    if histepoch > 0:
        DVAE_awa.load_state_dict(torch.load(model_path + f'/DVAE_awa-{histepoch}.pth'))

    # _ = ResE2D1((3, cropped_image_size, cropped_image_size), (3, cropped_image_size, cropped_image_size), z_dim, z_dim, norm_sample, 4, 1).to(device)
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("ResE1D1 trainable parameters: ", count_parameters(DVAE_awa))
    # print("ResE2D1 trainable parameters: ", count_parameters(_))
    # exit(0)
    DVAE_awa = DVAE_awa.train()
    optimizer = optim.Adam(DVAE_awa.parameters(), lr=lr)

    index = np.arange(len(obs1))
    n_batches = len(obs1) // batch_size
    print("Random PCA: ", randpca)

    cur_iter = 0
    # torch.autograd.set_detect_anomaly(True)
    for ep in range(histepoch, num_epochs+histepoch):
        ep_loss = []
        np.random.shuffle(index)
        for i in range(n_batches):
            b_idx = index[i * batch_size:(i + 1) * batch_size]
            o1_batch = torch.tensor(obs1[b_idx], device=device).float() / 255
            o2_batch = torch.tensor(obs2[b_idx], device=device).float() / 255
            # a_gt_batch = torch.tensor(a_gt[b_idx], device=device).float()
            a_gt_batch = a_gt[b_idx].clone().detach().to(device).float()

            if dataset == "PickAndPlace" or dataset == "Lift":
                if rand_crop == True:
                    ### random crop
                    o1_batch, w, h = random_crop_image(o1_batch, cropped_image_size)
                    o2_batch = random_crop_image(o2_batch, cropped_image_size, w, h)[0]
                else:
                    ### center crop
                    o1_batch = center_crop_image(o1_batch, cropped_image_size)
                    o2_batch = center_crop_image(o2_batch, cropped_image_size)
                    # obs_pred = center_crop_image(obs_pred, cropped_image_size)

                if "Joint" not in vae_model and "BasedVAE" in vae_model:
                    obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(o1_batch, o2_batch, random_bottle_neck=randpca)
                elif "Joint" in vae_model:
                    obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(torch.cat((o1_batch, o2_batch), dim=1))

                task_output = task_model(obs_pred.clip(0, 1))[0]
                obs = torch.cat((o1_batch, o2_batch), dim=1)

                ### learn from task model or dataset
                action_gt = a_gt_batch
                # action_gt = task_model(obs)[0]
                # action_gt[:, 3] = (100*action_gt[:, 3]).clip(-1, 1).detach()
                # action_gt = action_gt.detach()

                task_loss = torch.mean((action_gt - task_output) ** 2)
                loss = beta_task * task_loss + beta_rec * loss_rec + beta_kl * (kl1 + kl2) + weight_cross_penalty * loss_cor
                # assert not torch.isnan(loss).any(), "loss is nan"
            elif dataset == "gym_fetch": # gym_fetch
                raise NotImplementedError
                if "Joint" not in vae_model and "BasedVAE" in vae_model:
                    obs_pred, loss_rec, kl1, kl2, loss_cor, psnr = DVAE_awa(o1_batch, o2_batch, random_bottle_neck=randpca)
                elif "Joint" in vae_model:
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
        if (ep + 1) % save_interval == 0 or (ep + 1) == 10 or ep == 0:
            success_rate = evaluate(task_model, DVAE_awa, device, dataset, vae_model, DPCA_tf=False, dpca_dim=0, num_episodes=100)[3]
            summary_writer.add_scalar('Success Rate', success_rate, ep)
            torch.save(DVAE_awa.state_dict(), model_path + f'/DVAE_awa-{ep}.pth')

        ### export figure
        if (ep + 1) % save_interval == 0 or ep == num_epochs - 1 or ep == 0:
            max_imgs = min(batch_size, 8)
            vutils.save_image( torch.cat([o1_batch[:max_imgs], o2_batch[:max_imgs], obs_pred[:max_imgs, 0:3], obs_pred[:max_imgs, 3:6]], dim=0).data.cpu(),
                '{}/image_{}.jpg'.format(fig_dir, ep), nrow=8)

    return

if __name__ == "__main__":
    """        
    python train_awaDAE.py --dataset PickAndPlace --device 0 --lr 1e-4 --num_epochs 3000 --beta_rec 10000.0 --beta_kl 25.0 --beta_task 100 --z_dim 64 --batch_size 128 --seed 0 --cross_penalty 10.0 --vae_model CNNBasedVAE --norm_sample False --rand_crop True
    python train_awaDAE.py --dataset PickAndPlace --device 0 --lr 1e-4 --num_epochs 3000 --beta_rec 1000.0 --beta_kl 0.0 --beta_task 500 --z_dim 96 --batch_size 128 --seed 0 --cross_penalty 0.0 --vae_model ResBasedVAE --norm_sample False --rand_crop True -p True
    """

    model_path = './models/'
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
    parser.add_argument("-p", "--randpca", type=bool, help="perform random pca when training", default=False)
    parser.add_argument("-he", "--histepoch", type=int, help="load existing models", default=0)
    args = parser.parse_args()

    if args.norm_sample == 'True' or args.norm_sample == 'true':
        args.norm_sample = True
    else:
        args.norm_sample = False

    train_awa_vae(dataset=args.dataset, z_dim=args.z_dim, batch_size=args.batch_size, num_epochs=args.num_epochs, weight_cross_penalty=args.cross_penalty, 
                  beta_kl=args.beta_kl, beta_rec=args.beta_rec, beta_task=args.beta_task, device=args.device, save_interval=50, lr=args.lr, seed=args.seed,
                  model_path=model_path, dataset_dir=dataset_dir, vae_model=args.vae_model, norm_sample=args.norm_sample,rand_crop=args.rand_crop, 
                  randpca=args.randpca, histepoch=args.histepoch)
