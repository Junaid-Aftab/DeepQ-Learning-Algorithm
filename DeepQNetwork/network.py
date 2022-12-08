import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import clear_output
import numpy as np
import random
from matplotlib import pyplot as plt
import os, sys
import seaborn as sns
import import_ipynb
import Grid as GridWorld

class DeepQNetwork(nn.Module):
  
    def __init__(self, input_dims, num_actions, device):
        
        # We first define the input parameters
        super(DeepQNetwork, self).__init__()
        
        self.input_dims   = input_dims        #Dimensions of inputs, (N+2*agent_s_range)*(N+2*agent_s_range)
        self.num_actions  = num_actions       #Dimension of output layer, num_actions (= 4)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(64,128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1)        # not used right now
        self.conv4 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)        # not used right now
        self.fc5   = nn.Linear((self.input_dims)**2 * 128, self.num_actions)
        self.fc6   = nn.Linear(self.input_dims, self.num_actions)               # not used right now

        # Initialize parameters
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        nn.init.xavier_uniform_(self.fc6.weight)
        
        self.to(device)

    #We now implement the forward pass in the neural network. We apply non-linearities by applying the 
    #RELU function element-wise to the output of each linear network.
    
    def forward(self, state): 
        #mean = torch.mean(state)
        #std  = torch.std(state)
        #state = (state-mean)/std
        state = state/torch.max(state)
        #print(state.size())
        state = state.view(1,1,self.input_dims,self.input_dims)
        #print(state.size())
        state = F.relu(self.conv1(state.float()))
        #print(state.size())
        state = F.relu(self.conv2(state.float()))
        #print(state.size())
        #state = F.relu(self.conv3(state.float()))
        #state = F.relu(self.conv4(state.float()))
        #print(state.size())
        #state = F.relu(self.fc5(state.view(state.size(0), -1)))
        state = self.fc5(state.view(state.size(0), -1))
        #print(state.size())
        #return self.fc6(state)
        return state
