import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, load_dir="None", image_size=84,
                 transform=None, hybrid_state_shape=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        if hybrid_state_shape is None:
            self.hybrid_states = None
            self.next_hybrid_states = None
        else:
            self.hybrid_states = np.empty((capacity, *hybrid_state_shape), dtype=np.float32)
            self.next_hybrid_states = np.empty((capacity, *hybrid_state_shape), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        if load_dir != "None" and load_dir != None:
            self.load(load_dir)

    def add(self, obs, action, reward, next_obs, done):
        if isinstance(obs, list):
            np.copyto(self.hybrid_states[self.idx], obs[1])
            obs = obs[0]
            np.copyto(self.next_hybrid_states[self.idx], next_obs[1])
            next_obs = next_obs[0]

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        if self.hybrid_states is not None:
            hybrid_obses = torch.as_tensor(self.hybrid_states[idxs], device=self.device).float()
            next_hybrid_obses = torch.as_tensor(self.next_hybrid_states[idxs], device=self.device).float()
            obses = [obses, hybrid_obses]
            next_obses = [next_obses, next_hybrid_obses]

        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self, translate=False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        if translate:
            from data_augs import random_translate
            obses = random_translate(obses)
            next_obses = random_translate(next_obses)
            pos = random_translate(pos)
        else:
            obses = random_crop(obses, self.image_size)
            next_obses = random_crop(next_obses, self.image_size)
            pos = random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        if self.hybrid_states is not None:
            hybrid_obses = torch.as_tensor(self.hybrid_states[idxs], device=self.device).float()
            next_hybrid_obses = torch.as_tensor(self.next_hybrid_states[idxs], device=self.device).float()
            obses = [obses, hybrid_obses]
            next_obses = [next_obses, next_hybrid_obses]

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def sample_rad(self, aug_funcs):
        # augs specified as flags
        # curl_sac organizes flags into aug funcs
        # passes aug funcs into sampler

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        if aug_funcs:
            for aug, func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug:
                    obses = func(obses)
                    next_obses = func(next_obses)

                if 'translate' in aug:
                    obses, tw, th = func(obses)
                    next_obses, _, _ = func(next_obses, tw, th)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        obses = obses / 255.
        next_obses = next_obses / 255.

        # augmentations go here
        if aug_funcs:
            for aug, func in aug_funcs.items():
                # skip crop and cutout augs
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug:
                    continue
                obses = func(obses)
                next_obses = func(next_obses)

        if self.hybrid_states is not None:
            hybrid_obses = torch.as_tensor(self.hybrid_states[idxs], device=self.device).float()
            next_hybrid_obses = torch.as_tensor(self.next_hybrid_states[idxs], device=self.device).float()
            obses = [obses, hybrid_obses]
            next_obses = [next_obses, next_hybrid_obses]

        return obses, actions, rewards, next_obses, not_dones

    def sample_bc(self):
        for idx in range(0, self.capacity if self.full else self.idx, self.batch_size):
            obses = torch.as_tensor(random_crop(self.obses[idx: idx + self.batch_size], self.image_size), device=self.device).float() / 255.
            next_obses = torch.as_tensor(random_crop(self.next_obses[idx: idx + self.batch_size], self.image_size), device=self.device).float() / 255.
            actions, rewards, not_dones = [torch.as_tensor(k[idx: idx + self.batch_size], device=self.device) for k in [self.actions, self.rewards, self.not_dones]]
            yield obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps
        self.hybrid_state = None
        self.special_reset_save = None

    def reset(self, save_special_steps=False):
        obs = self.env.reset(save_special_steps=save_special_steps)
        if save_special_steps:
            self.unpack_special_steps()
        if isinstance(obs, list):
            self.hybrid_state = obs[1]
            obs = obs[0]
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if isinstance(obs, list):
            self.hybrid_state = obs[1]
            obs = obs[0]
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def unpack_special_steps(self):
        special_steps_dict = self.env.special_reset_save
        obs_list = special_steps_dict['obs']
        stacked_obs = []
        for _ in range(self._k):
            self._frames.append(obs_list[0])
        for o in obs_list:
            self._frames.append(o)
            stacked_obs.append(self._get_obs())
        self.special_reset_save = {'obs': stacked_obs, 'act': special_steps_dict['act'],
                                   'reward': special_steps_dict['reward']}

    def _get_obs(self):
        assert len(self._frames) == self._k
        frames = np.concatenate(list(self._frames), axis=0)
        if self.hybrid_state is None:
            return frames
        else:
            return [frames, self.hybrid_state]

    def set_special_reset(self, mode):
        self.env.set_special_reset(mode)


# def random_crop(imgs, output_size):
#     """
#     Vectorized way to do random crop using sliding windows
#     and picking out random ones

#     args:
#         imgs, batch images with shape (B,C,H,W)
#     """
#     # batch size
#     n = imgs.shape[0]
#     img_size = imgs.shape[-1]
#     crop_max = img_size - output_size
#     imgs = np.transpose(imgs, (0, 2, 3, 1))
#     w1 = np.random.randint(0, crop_max, n)
#     h1 = np.random.randint(0, crop_max, n)
#     # creates all sliding windows combinations of size (output_size)
#     windows = view_as_windows(
#         imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
#     # selects a random window for each batch element
#     cropped_imgs = windows[np.arange(n), w1, h1]
#     return cropped_imgs


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
    

def random_crop_image(imgs, out=84, w1=None, h1=None):
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