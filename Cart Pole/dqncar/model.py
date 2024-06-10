import numpy as np

import torch
import tensorflow 


# pro tip: the target network and the network network need to have the 
# exact same architecture otherwise you cannot copy the weights between them.



def Model(input_dims, output_dims):
    '''This shouldn't need to be a class. This model should be very simple.
    Especially for cartpole, like, 2 dense, an output, and no convolutions 
    should be more than enough.'''

    inner_size = 64
    
    # Define the neural network architecture
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=input_dims, out_features=inner_size),   # Input layer
        torch.nn.ReLU(),                                                    # Activation function
        torch.nn.Linear(in_features=inner_size, out_features=inner_size),   # Hidden layer
        torch.nn.ReLU(),                                                    # Activation function
        torch.nn.Linear(in_features=inner_size, out_features=output_dims)   # Output layer
    )
    
    return model