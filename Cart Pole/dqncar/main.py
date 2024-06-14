import numpy as np
import random as rnd
import time 
import matplotlib.pyplot as plt
import math
import gym
import pygame

import torch
import tensorflow as tf

# all of the libraries above can be installed with pip
# ex: pip install numpy or pip install torch


from DQN import DQNAgent



# Hyperparams
input_dims = 4
output_dims = 2
# likely want to put in some other cool things here like batch size, learning rate, etc. 
episodes = 0
epsilon = 0.1
discount = 0.001 #learning rate

# Global Constants, change these
MAX_EPISODES = 100
BUFFER_BATCH_SIZE = 10000
BATCH_SIZE = 32



if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='human')
    state = env.reset()
    agent = DQNAgent(input_dims, output_dims, env, epsilon, discount) #added env to the DQNAgent class

    # Make the main game loop.  

    while episodes < MAX_EPISODES:
        time_step = 0
        rewards = []
        agent.replay_memory.erase_memory()
        observation, info = env.reset()
        state = observation
        time_step = 0
        done = False

        while not done:

            # Get action through agent
            
            #action = env.action_space.get_action(state)
            #action = env.action_space.sample()
            #print(action)
            #print(state)
            action = agent.get_action(state, env)
            
            # Take the action and observe the result
            observation, reward, terminated, trunicated, info = env.step(action)
            
            # Accumulate the reward
            rewards.append(reward)

            # Check if we lost
            if terminated or trunicated:
                done = True


            # Store our memory
            #agent.replay_memory.store_memory((state, action, reward, observation))
            #print(observation)
            state = observation

            # learn?
            #agent.learn()
            time_step += 1

            env.render()
        
    # TODO: Check if reward normalization makes sense!
    agent.save()
    env.close()
