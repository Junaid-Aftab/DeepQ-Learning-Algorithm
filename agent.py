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
from Network import DeepQNetwork

class Agent():
    
    def __init__(self, gamma, epsilon, epsilon_end, epsilon_dec, learningrate, input_dims, 
                 minibatch_size, num_actions, capacity, episodes, 
                 C, worldsize, frac_fill, num_enemies, num_food, input_energy, food_energy, s_range, max_epoch,
                 agent_s_range,move_reward,food_reward,wall_reward,death_reward,survive_reward,enemy_reward,device,train):
        
        # We first define the input parameters to the agent's deep learning network
        
        self.learningrate = learningrate                      # Learning rate
        self.input_dims = input_dims                          # Dimensions of inputs, (worldsize+2)*(worldsize+2)
        self.num_actions = num_actions                        # Dimension of output layer, num_actions (=4)
        self.batch_size = minibatch_size                      # Minibatch size
        self.device     = device 

        # We define other parameters for the learning algorithm
        
        self.gamma = gamma                                   # Discount factor
        self.epsilon = epsilon                               # Epsilon
        self.epsilon_min = epsilon_end                       # Minimum epsilon
        self.epsilon_dec = epsilon_dec                       # Decremenet in epsilon
        self.episodes = episodes                             # Number of episodes
        self.capacity = capacity                             # Maximum number of previous experiences/iteration stored
        self.C = C                                           # Every C steps, update target network
        self.max_epoch = max_epoch                           # Max number of epochs
        self.train = train                                   # Training or testing?
         
        #We now define the input to the grid world the agent explores
       
      self.worldsize = worldsize                  #World size
        self.frac_fill = frac_fill                #Fraction Fill
        self.num_enemies = num_enemies            #Number of Enemies
        self.num_food = num_food                  #Number of food
        self.energy = input_energy                #Input energy
        self.food_energy = food_energy            #Food energy
        self.s = s_range                          #Enemy sight
        self.agent_s_range = agent_s_range        #Agent range
        self.move_reward   = move_reward          #Moving reward
        self.food_reward   = food_reward          #Food reward
        self.wall_reward   = wall_reward          #Wall reward
        self.death_reward  = death_reward         #Death reward
        self.survive_reward  = survive_reward     #Survive reward
        self.enemy_reward = enemy_reward          #Enemy reward

        #We now initialize the the agent's DeepQNetwork.
        
        self.DQN = DeepQNetwork(self.input_dims,self.num_actions, device)

        
    #We now implement the  DeepQ Learning algorithm.
    
    def DQN_learning(self):
        
        #We first specify the MSE loss function and the optimization algorithm to be used.
        #We currently use the ADAM solver
        
        self.loss = nn.MSELoss()
        optimizer = torch.optim.Adam(self.DQN.parameters(),lr=self.learningrate)
        s_time    = time.time()
        
        # We first initialize the memory list which holds the previous capacity number of experiences
       
        D = []
        
        # We copy the output for target action-value function
        
        Qhat = copy.deepcopy(self.DQN)
        
        count = 0
        
        loss_epochs = []
        wins        = []
        ep_survived = []
        rewards     = []
        
        # First fill up the replay experience buffer
        
        while len(D) < self.capacity:
            
            world = GridWorld.World(self.worldsize, self.frac_fill, self.num_enemies, self.num_food,
                              self.energy, self.food_energy, self.s,self.agent_s_range, self.move_reward,
                              self.food_reward, self.wall_reward, self.death_reward, self.survive_reward,
                              self.enemy_reward,self.train)
            
            old_input = torch.tensor(world.view)
            
            t = 0
            
            while not world.agent_dead() and t < self.max_epoch:
                
                action_index = np.random.randint(0,self.num_actions)     #The index of the random action picked.
                action = world.actions[action_index]                     #The executable action in the grid.
                
                reward = world.movement(world.agent_pos, 0, action)
                
                if t == self.max_epoch-1 and not world.agent_dead():
                    reward += self.survive_reward
                
                new_input = torch.tensor(world.view)
                
                D.append((old_input,action_index,reward,new_input, world.agent_dead() or t == self.max_epoch-1))
                
                if len(D) >= self.capacity:
                    break
                
                old_input = new_input
                t += 1     
        
        
        #We now train the DQN by simulating a fixed number of episodes.
        
        for i in range(self.episodes):
            
            # First grab the initial world for a new episode
            
            world = GridWorld.World(self.worldsize, self.frac_fill, self.num_enemies, self.num_food,
                              self.energy, self.food_energy, self.s,self.agent_s_range, self.move_reward,
                              self.food_reward, self.wall_reward, self.death_reward, self.survive_reward,
                              self.enemy_reward,self.train)
            
            old_input = torch.tensor(world.view)
            
            t = 0
            
            while not world.agent_dead() and t < self.max_epoch:
                
                #The policy used is the epsilon-greedy policy.
                
                if np.random.random() < self.epsilon:
                    action_index = np.random.randint(0,self.num_actions)     #The index of the random action picked.
                    action = world.actions[action_index]                     #The executable action in the grid.
                
                else:
                    action_index = torch.argmax(self.DQN(old_input.to(self.device))).item().    #The index of the optimal action picked.
                    #print(a.DQN(old_input.to(device)))
                    action = world.actions[action_index]                                        #The executable action in the grid.
                
                # Epsilon annealing
                if self.epsilon > self.epsilon_min:
                    self.epsilon -= self.epsilon_dec

                #We now execute the action for the agent. See the GridWorld document for more details on how the movement function is executed.
                
                reward = world.movement(world.agent_pos, 0, action)
                
                if t == self.max_epoch-1 and not world.agent_dead():
                    
                    reward += self.survive_reward
                    world.reward += self.survive_reward
                
                new_input = torch.tensor(world.view)
                
                ########################
                #if (i%1000 == 999):
                #clear_output(wait=True)
                #world.display_world(t)
                #    time.sleep(1)
                ########################
                
                #We now add the events to the replay memory. Since the replay memory is pre-filled,
                #the earliest memory is always deleted and replaced by the most recent experience
                
                D.pop(0)
                D.append((old_input,action_index,reward,new_input, world.agent_dead() or t == self.max_epoch-1))
                
                old_input = new_input
                
                # Sample a minibatch from D
                
                if len(D) > self.batch_size:
                    samples = random.sample(D,self.batch_size)
                
                else:
                    samples = random.sample(D,len(D))
        
                #We now update the target values
                
                y = torch.zeros(len(samples)).to(self.device)
                Q = torch.zeros(len(samples)).to(self.device)  # Q stores max Q value for each of the samples
                
                for s in range(len(samples)):
                    
                    Q[s] = self.DQN(samples[s][0].to(self.device))[0][samples[s][1]]
                    y[s] = torch.tensor(float(samples[s][2]), device=self.device)
                    
                    if not samples[s][4]:
                        y[s] += self.gamma*torch.max(Qhat(samples[s][3].to(self.device)))
                        #print(torch.max(Qhat(samples[s][3])))
                
                # We now perform gradient descent on the minibatch.
                loss = nn.MSELoss()(y,Q)
                
                #if i > 5:
                loss_epochs.append(loss.cpu().detach().numpy())
                
                optimizer.zero_grad()
                loss.backward()
                #for param in self.DQN.parameters(): # clamp gradient
                #  param.grad.data.clamp_(-1, 1)
                
                optimizer.step()
                if (count+1) % self.C == 0:
                    
                    # print('Updating.')
                    # Reset Qhat to Q (target action-value function = action-value function)
                    Qhat = copy.deepcopy(self.DQN)
                
                count += 1
                t += 1   
            
            #Log win record
            
            if i == 0:
                wins.append(1-int(world.agent_dead()))
            
            else:
                wins.append(wins[-1] + (1-int(world.agent_dead())))
            
            #world.reward += (self.death_reward if world.agent_dead() else self.survive_reward)
            #reward = world.reward # want this in the above loop
                
            ep_survived.append(t)
            rewards.append(world.reward)
            avg_ep = np.round(np.mean(ep_survived[max([i-100, 0]):i+1]),2)
            avg_re = np.round(np.mean(rewards[max([i-100, 0]):i+1]),2)
            
            #################################
            clear_output(wait=True)
            print("Training Episode: {}/{}. Wins: {}. Avg Epochs Survived: {}. Avg Reward: {}. Time Elapsed: {} s"
                  .format(i+1, self.episodes, int(wins[-1]), avg_ep, avg_re, int(time.time()-s_time)))
            # time.sleep(3)
            #################################
      
        return count, loss_epochs, wins, rewards, ep_survived
