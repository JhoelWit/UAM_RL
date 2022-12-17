# -*- coding: utf-8 -*-
from atc_environment import ATC_Environment
import random

import yaml
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed


with open('atc_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

env = ATC_Environment(config)
env._seed(seed=1, cuda=True)
env.reset()
# check_env(env) #used to prepare env for training



#Randomness test
while True:
    action = random.randint(0, 10)
    new_state, reward, done, info = env.step(action)
    if done:
        env.reset()