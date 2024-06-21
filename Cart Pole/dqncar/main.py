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
MAX_EPISODES = 5000
BUFFER_BATCH_SIZE = 10000
BATCH_SIZE = 128



if __name__ == "__main__":
    # env = gym.make('CartPole-v1', render_mode='human')
    env = gym.make('CartPole-v1', render_mode='')
    state = env.reset()
    agent = DQNAgent(input_dims, output_dims, env, epsilon, discount) #added env to the DQNAgent class
    memory_count = 0
    agent.memory_buffer.erase_memory()
    # Make the main game loop.  
    total_rewards = []
    losses = []
    while episodes < MAX_EPISODES:
        time_step = 0
        rewards = []
        
        observation, info = env.reset()
        state = observation
        
        done = False

        total_reward = 0

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

            total_reward += reward
            #GOALS FOR NEXT TIME
            #store memory
            #graph results
            #episode x axis, sum of reward y axis
            #plt.plot(rewards, label='rewards',)

            # Store our memory
            #store as a touple

            agent.memory_buffer.store_memory((state, action, reward, observation))
            #print("In main: ",len(agent.memory_buffer))
            state = observation

            # graphs learining over time
            # learn?
            #agent.learn()
            #print ("memory count: ", memory_count)
            if memory_count == BATCH_SIZE:
                #print ("learning")
                loss = agent.learn()
                losses.append(loss)
                memory_count = 0
            else:
                memory_count += 1

            '''render environment'''
            #env.render()
        
        total_rewards.append(total_reward)
        episodes += 1
    plt.plot(total_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

    plt.plot(losses)
    plt.title('Loss over Time')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.show()

    #MORE GRAPHS
    #mean squared error 
    #total rewards
    #epsilon decay
    #run epoch at the end of the episode
    #potentially store loss in a list and graph it
        
    # TODO: Check if reward normalization makes sense!
    #agent.save()
    env.close()
