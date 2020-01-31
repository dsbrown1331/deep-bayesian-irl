import argparse
import torch.distributions as tdist
# coding: utf-8

# Take length 50 snippets and record the cumulative return for each one. Then determine ground truth labels based on this.

# In[1]:


import sys
import pickle
import gym
from gym import spaces
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.utils import save_image
from run_test import *
from baselines.common.trex_utils import preprocess
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--model_path', default='')
    parser.add_argument('--episode_count', default=100)
    parser.add_argument('--record_video', action='store_true')
    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()

    stochastic = True #it helps Atari policies to not get stuck if there is a little noise

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    #env = gym.make(args.env_id)

    #env id, env type, num envs, and seed
    env = make_vec_env(args.env_id, args.env_type, 1, 0,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })
    if args.record_video:
        env = VecVideoRecorder(env,'./videos/',lambda steps: True, 200000) # Always record every episode

    if args.env_type == 'atari':
        env = VecFrameStack(env, 4)
    elif args.env_type == 'mujoco':
        env = VecNormalize(env,ob=True,ret=False,eval=True)
    else:
        assert False, 'not supported env type'

    try:
        env.load(args.model_path) # Reload running mean & rewards if available
    except AttributeError:
        pass

    agent = PPO2Agent(env,args.env_type, stochastic)
    agent.load(args.model_path)
    #agent = RandomAgent(env.action_space)

    episode_count = args.episode_count
    reward = 0
    done = False

    env_test=gym.make(args.env_id)
    print(env_test.unwrapped.get_action_meanings())

    for i in range(int(episode_count)):
        ob = env.reset()
        steps = 0
        acc_reward = 0
        while True:

            action = agent.act(ob, reward, done)
            #action = env.action_space.sample()
            ob, reward, done, _ = env.step(action)
            if args.render:
                env.render()

            steps += 1
            acc_reward += reward
            if done:
                print(steps,acc_reward)
                break

    env.close()
    env.venv.close()
