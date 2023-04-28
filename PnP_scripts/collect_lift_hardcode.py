import gym
import dtac.custom_envs
import numpy as np
import torch

class HardcodePolicy:
    def __init__(self):
        self.stage = 0
        self.stage_count = 0
        self.obj_pos = None

    def get_action(self, gripper_pos=None):
        if self.stage == 0:  # Opening the gripper and raising the gripper
            if self.stage_count < 2:
                action = np.array([0, 0, 1, 1])
                self.stage_count += 1
            else:
                self.stage = 1
                self.stage_count = 0
                return self.get_action(gripper_pos)
        if self.stage == 1:  # Moving the gripper
            if np.linalg.norm(gripper_pos - (self.obj_pos + np.array([0, 0, 0.1]))) < 0.02:
                self.stage = 2
                return self.get_action(gripper_pos)
            else:
                action = np.zeros(4)
                action[:3] = (self.obj_pos + np.array([0, 0, 0.1]) - gripper_pos) * 10
        if self.stage == 2:  # Moving the gripper
            if np.linalg.norm(gripper_pos - self.obj_pos) < 0.02:
                self.stage = 3
                return self.get_action(gripper_pos)
            else:
                action = np.zeros(4)
                action[:3] = (self.obj_pos - gripper_pos) * 10
        if self.stage == 3:  # Closing the gripper
            if self.stage_count < 2:
                action = np.array([0, 0, 0, -1])
                self.stage_count += 1
            else:
                self.stage = 4
                self.stage_count = 0
                return self.get_action(gripper_pos)
        if self.stage == 4:  # Raising the gripper
            action = np.array([0, 0, 1, -1])

        return action
            
    def reset(self, obj_pos):
        self.stage = 0
        self.stage_count = 0
        self.obj_pos = obj_pos
        


e = gym.make('Lift-both-v1')

total_samples = 20000
observations, actions, next_observations, rewards, dones = [], [], [], [], []

policy = HardcodePolicy()

while True:
    obs = e.reset()
    done = False
    obj_pos = e.unwrapped._state_obs['observation'][3:6]
    policy.reset(obj_pos)

    while not done:
        action = policy.get_action(e.unwrapped._state_obs['observation'][:3])
        action = (action + np.random.normal(0, 0.1, 4)).clip(-1, 1)

        next_obs, r, done, _ = e.step(action)
        observations.append(obs)
        actions.append(action)
        next_observations.append(next_obs)
        rewards.append(r)
        dones.append(done)
        obs = next_obs

    if len(observations) > total_samples:
        break

observations = np.array(observations)
actions = np.array(actions)
next_observations = np.array(next_observations)
rewards = np.array(rewards)
dones = np.array(dones)

torch.save([observations, dones, actions], 'lift_hardcode.pt')
