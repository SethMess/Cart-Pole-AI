# Misc imports
import numpy as np
import random as rnd
import gym

# These are your other files.
from buffer import ReplayBuffer
from model import Model

# Tensorflow if you're using tensorflow
import tensorflow as tf

# pytorch if you're using pytorch
import torch

class DQNAgent():
    def __init__(self, input_dims, output_dims, env, epsilon=0.1, learning_rate=0.001):
            self.output_dims = output_dims
            self.input_dims = input_dims
            self.observation_space = env.observation_space

            self.epsilon = epsilon
            self.learning_rate = learning_rate
            self.update_target_counter = 0

            self.model = Model(input_dims, output_dims)  # Create your model
            self.target_model = Model(input_dims, output_dims)  # Create your target model the one that gets updated

            self.memory_buffer = ReplayBuffer()  # Create your replay buffer

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)  # Create your optimizer

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set your device
            self.model.to(self.device)  # Move your model to the device
            self.target_model.to(self.device)  # Move your target model to the device

    # Method for predicting an action 
    def get_action(self, state, env, training=True) -> int:
        ''' 
        Get action function call.
        Ideally your state is processed by your target network. 

        Your state can be inputted into this function as an array/tuple, in which case
        needs to be turned into a tensor before being inputted into your network.

        or it can be inputted into this function as a tensor already. 
        mostly fashion. do what you please.
        '''
        
        # Train does not exit yet
        #self.model.train(training) 

        # Explore vs Expliot rule
        random = np.random.rand()    #assigns a random float in the range [0,1)
        
        if random < self.epsilon:
          action = env.action_space.sample()

        else:
            #turn state into pytorch tensor
            
            state = torch.from_numpy(state).float().to(self.device)
            #print(f"state before: {state}")
            #state = state.unsqueeze(0).to(self.device)
            #print(f"state after: {state}")

            QModel = self.model(state)
            action = torch.argmax(QModel).item()
        
        #print(f"action taken: {action}")
        return action
    
    def learn(self) -> float:
        ''' 
        This function will be the source of 90% of your problems at the
        start. this is where the magic happens. it's also where the tears happen.

        ask questions. please.

        I'll leave a lot more things up here to make it less painful.

        it returns a tuple in case you want to keep track of your losses (you do)
        '''
        loss = 0
        BUFFER_BATCH_SIZE = 1000
        BATCH_SIZE = 128
        TARGET_UPDATE = 3

        print("about to learn")
        # We just pass through the learn function if the batch size has not been reached. 
        if len(self.memory_buffer) < BATCH_SIZE:
           print(len(self.memory_buffer))
           return

        
        state = []
        action = []
        reward = []
        next_state = []

        memory = self.memory_buffer.collect_memory(BATCH_SIZE)
        state, action, reward, next_state = zip(*memory)        

        # Convert list of tensors to tensor.
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        
        #print(self.model(state_tensor).shape)
        #print("Model output shape:", self.model(state_tensor).shape)
        #print("Action tensor shape after unsqueeze:", action_tensor.long().unsqueeze(-1).shape)
        # print("state_tensor: ", state_tensor)
        # print("action_tensor: ", action_tensor)
        # print("reward_tensor: ", reward_tensor)
        # print("next_state_tensor: ", next_state_tensor)
        # One hot encoding our actions. (probably not important)

        # Find our predictions (r + max(q(s', a)) find next best values for tensors of new states
        #predictions = self.model(next_state_tensor)
        
        #I just commented out some code I wrote it might be right but it was not needed yet

        # get max value
        #max_q_values = torch.max(predictions, dim=1)[0]
        state_action_values = self.model(state_tensor).gather(dim=1, index=action_tensor.long().unsqueeze(-1))

        next_state_values = self.model(next_state_tensor).max(dim=1)[0].detach()
        # Calculate our target
        #target_values = reward_tensor + max_q_values
        expected_state_action_values = (next_state_values * (1-self.epsilon)) + reward_tensor

        # Calculate MSE Loss
        #loss = torch.nn.MSELoss()(predictions, target_values)
        loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(-1))

        # backward pass (back propogation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backward pass
        self.optimizer.step()  # Update weights

        # Update the target network
        self.update_target_counter += 1
        if self.update_target_counter % TARGET_UPDATE == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        return loss.item()
        

    def save(self, save_to_path: str) -> None:
        # if pytorch:
        torch.save(self.target_model.state_dict(), save_to_path)

    def load(self, load_path: str) -> None:

        # if tensorflow
        #loaded_target = tf.keras.models.load_model(load_path)
        #loaded_model = tf.keras.models.load_model(load_path)

        # if pytorch
        self.target_model.load_state_dict(torch.load(load_path))
        self.model.load_state_dict(torch.load(load_path))




if __name__ == "__main__":
    '''
    For those unfamiliar with this format, this is so that if you want to run this file
    instead of the main.py file to test this file specifically, everything in this block will be run.
    So, if you had a print statement outside of this block and called functions or classes,
    they will be ignored. 
    '''
    input_dims = 4
    output_dims = 2

    #These 2 steps are for testing
    env = gym.make('CartPole-v1', render_mode='human')
    state = env.reset()
    
    buffer = DQNAgent(input_dims, output_dims)
    print('dqn agent')

