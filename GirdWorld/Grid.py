import numpy as np
import random
import time
from IPython.display import clear_output
from matplotlib import pyplot as plt
import seaborn as sns

class World():
    
    def __init__(self, worldsize, frac_fill, num_enemies,
                 num_food, init_energy, food_energy, s_range,
                 agent_s_range, move_reward, food_reward, 
                 wall_reward, dead_reward, surv_reward,
                 enemy_reward,train):
        
        self.N             = worldsize
        self.fill          = frac_fill
        self.E             = num_enemies
        self.F             = num_food
        self.en_f          = food_energy
        self.energy        = init_energy
        self.s_range       = s_range
        self.agent_s_range = agent_s_range
        
        self.view = None
        
        self.reward = 0
        self.move_reward  = move_reward
        self.food_reward  = food_reward
        self.wall_reward  = wall_reward
        self.dead_reward  = dead_reward
        self.surv_reward  = surv_reward
        self.enemy_reward = enemy_reward
        
        self.train = train
        
        if self.train == True:
             
            self.agent  = 10
            self.enemy  = 20
            self.food   = 30
            self.empty  = 0
            self.wall   = 40
        
        else:
        
            self.agent  = 0
            self.enemy  = 1
            self.food   = 2
            self.empty  = 3
            self.wall   = 4
        
        self.agent_pos = 0
        self.enemy_pos = 1
        self.food_pos  = 2
        
        # Possible movement directions or "actions" (0-3)
        # Assume that staying in the same spot is not a possible action
        self.actions = [[-1,0],[0,1],[1,0],[0,-1]] # Possible reward for hiding: stay in place and be "invisible" to the enemy
                
        # Initialize positions of agent, enemies, food as a nested list for modular access later
        self.pos = [[],[],[]]
        
        # Generate the world grid structure
        self.gen()

    # Like random.randint but of a given output shape
    def randint(self, row, col, minimum, maximum):
        r = [[y*random.randint(minimum, maximum) for y in x] for x in row*[col*[1]]]
        return (r[0] if row == 1 else r)

    # Populate with random straight walls depending on fill parameter
    def gen_walls(self):
        r = self.agent_s_range
        while np.count_nonzero(self.world[r:self.N+r,r:self.N+r] == self.wall)/(self.N**2) < self.fill:
            i,j = self.randint(1,2,r,self.N+r-1)
            surround = np.ravel(self.world[i-1:i+2,j-1:j+2])
            if not self.wall in surround[[0,2,6,8]]:
                self.world[i,j] = self.wall

    # Place any amount of specified elements (food, agent, or enemy) with given placement rule(s)
    def place(self, kind, amount, rule):
        r = self.agent_s_range
        for a in range(amount):
            i,j = [0,0]
            while not (self.world[i,j] == self.empty and rule(i,j)):
                i,j = self.randint(1,2,r,self.N+r-1)
            if kind == self.agent_pos:
                self.world[i,j] = self.agent
            elif kind == self.enemy_pos:
                self.world[i,j] = self.enemy
            elif kind == self.food_pos:
                self.world[i,j] = self.food
            self.pos[kind].append([i,j])
    
    # Get agent's field of view based on current location and arbitrary sight range
    def agent_view(self, i, j, s_range): return self.world[i-s_range:i+s_range+1,j-s_range:j+s_range+1]
    
    # Initialize world and fill with walls, food, enemies, agent
    def gen(self):
        r = self.agent_s_range
        self.world = self.wall*np.ones((self.N+r*2,self.N+r*2))
        self.world[r:self.N+r, r:self.N+r] = self.empty
        self.gen_walls()
        self.place(self.food_pos,  self.F, lambda i,j: True)
        self.place(self.agent_pos, 1,      lambda i,j: True)

        # An enemy cannot spawn within the agent's field of view
        self.place(self.enemy_pos, self.E, lambda i,j: not self.agent in self.agent_view(i, j, r))
        
        self.view = self.agent_view(self.pos[self.agent_pos][0][0], self.pos[self.agent_pos][0][1], r)
        
    # Generic movement function for agent or enemy with food/energy updating included.
    # Reward updating for the agent is now included.
    # For the agent, can call this directly from the RNN/DNN.
    def movement(self, kind, idx, action):
        old = self.pos[kind][idx]
        new = tuple(np.add(old, action))
        old = tuple(old)
        
        reward = 0
        
        # If movement is possible in the given direction, check for food in that spot, then move.
        #If the agent is selected, additionally reward the agent
        if self.world[new] in [self.empty, self.food]:
            if self.world[new] == self.empty and kind == self.agent_pos:
                    reward += self.move_reward
            if self.world[new] == self.food:
                if kind == self.agent_pos:
                    self.energy += self.en_f
                    reward += self.food_reward
                
                # Remove the eaten food and place a new one somewhere else
                self.pos[self.food_pos].remove(list(new))
                self.place(self.food_pos, 1, lambda i,j: True)
            
            self.world[old] = self.empty
            if kind == self.agent_pos:
                self.world[new] = self.agent
            elif kind == self.enemy_pos:
                self.world[new] = self.enemy
            #self.world[new] = kind
            self.pos[kind][idx] = list(new)
            
        elif self.world[new] == self.wall and kind == self.agent_pos:
            reward += self.wall_reward
            
        elif self.world[new] == self.enemy and kind == self.agent_pos:
            reward += self.enemy_reward
        
        if kind == self.agent_pos:
            self.view = self.agent_view(self.pos[kind][0][0], self.pos[kind][0][1], self.agent_s_range)
            self.energy -= 1
            self.update_enemies()
            if self.agent_dead():
                reward = self.dead_reward
        
        self.reward += reward
        return reward
    
    # Update all enemy positions (includes naive hunting AI)
    def update_enemies(self):
        for i in range(len(self.pos[self.enemy_pos])):
            pos = self.pos[self.enemy_pos][i]
            
            # Define the distance vector "vec" between agent/enemy and calculate possible movement
            vec = np.subtract(self.pos[self.agent_pos][0], pos)
            acs = [[np.sign(vec[0]), 0], [0, np.sign(vec[1])]]
            
            # Check if the agent is within range
            if np.linalg.norm(vec) <= self.s_range:
                div = np.max(np.abs(vec))
                for j in range(1, div):
                    new = tuple(np.add(pos, np.array(np.round([vec[0]*j/div, vec[1]*j/div]), dtype='int')))
                    if self.world[new] == self.wall: acs = self.actions; break
                    # If there's a wall in the line of sight, move randomly
            
            # If the agent is not within range, move randomly
            else: acs = self.actions
            
            # Shuffle possible actions and look for open space to move
            np.random.shuffle(acs)
            for ac in acs:
                if self.world[tuple(np.add(pos, ac))] in [self.empty, self.food]:
                    self.movement(self.enemy_pos, i, ac)
                    break
            
    # Check if the agent is "dead" due to an enemy being within 1 block
    def agent_dead(self):
        i,j = self.pos[self.agent_pos][0]
        surround = np.ravel(self.world[i-1:i+2,j-1:j+2])[[1,3,5,7]]
        return self.enemy in surround or self.energy < 1

    # Display the world
    def display_world(self, epoch):
        plt.figure(figsize=(10,10))
        colors = ["white", "red", "lime", "black", "grey"]
        palette = sns.color_palette(colors)
        g = sns.heatmap(self.world, cbar=False, cmap=palette)
        g.set(yticks=[]); g.set(xticks=[])
        plt.title("GridWorld Survival Game, Epoch {} | Agent Energy: {} | Agent Reward: {}"
            .format(epoch+1, self.energy, self.reward))
        plt.show()






