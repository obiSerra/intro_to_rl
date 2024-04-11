import gym
import torch as th
import torch.nn.functional as F

# from stable_baselines.common.policies import CnnPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn as nn


if __name__ == "__main__":
    env = make_vec_env("CarRacing-v1", n_envs=4)
    model_name = "ppo_car_racing_base_v1"
    # normal reset, this changes the colour scheme by default
    obs = env.reset()

    # reset with colour scheme change

    class Net(BaseFeaturesExtractor):
        def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 160):
            super(Net, self).__init__(observation_space, features_dim)
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
            self.pol1 = nn.MaxPool2d(6, stride=2)
            self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
            self.pol2 = nn.MaxPool2d(12, stride=2)
            self.fc1 = nn.Linear(3072, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, features_dim)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.pol1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.pol2(x))
            x = nn.Flatten()(x)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            return x

    policy_kwargs = {
        "activation_fn": th.nn.ReLU,
        "net_arch": [dict(pi=[128, 128, 128], vf=[128, 128, 128])],
        "features_extractor_class": Net,
    }

    model_params = {
        # "batch_size": 60,
        # "n_steps": 1200,
        # "policy_kwargs": policy_kwargs
        # "clip_range": 0.3,
    }

    model = PPO("CnnPolicy", env, verbose=0, tensorboard_log="sb3_logs", **model_params)

    model.learn(total_timesteps=100_000, tb_log_name=model_name, reset_num_timesteps=False)

    model.save(model_name)
    print("training DONE!!!!!")
