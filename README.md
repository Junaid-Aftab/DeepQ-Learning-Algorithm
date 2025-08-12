# DeepQ-Learning-Algorithm

## Description

This repository contains code we implemented as part of the couse project for CMSC 727: Neural Modelling taught by James Reggia at University of Maryland, College Park. 

We implemented the deep Q-learning algorithm to examine the the discrepancy between recurrent and non-recurrent neural
networks for the purpose of Q-learning in a survival-focused GridWorld game. Our study compared the two architectures’ ability to guarantee a
high rate of “survival” (subject to our definition of this objective) in randomly generated grid environments that share basic inherent properties.

A description of the various files uploaded is as follows:

1. The file ```grid.py``` generates a two-dimensional grid wolrd populated with the virtual agent, enemies, food and obstacles/walls. 
   The `grid.py` file takes in the following parameters:
   
   - `worldsize:` Dimensions of the grid world
   - `frac_fill:` Fraction filled with walls
   - `num_enemies:` Number of enemies
   - `num_food:` Number of food
   - `input_energy:` Agent's input energy
   - `food_energy:` Energy per food
   - `s_range:` Enemy sight range when vision not blocked by wall
   - `agent_s_range:` Agent sight range

   - `move_reward:` Reward for moving
   - `food_reward:` Reward for acquiring food
   - `wall_reward:` Reward for hitting wall
   - `death_reward:` Reward for dying
   - `survive_reward:` Reward for surviving
   - `enemy_reward:` Reward for being eaten by enemy

   Here is a sample [image](images/grid.png) of what the `gird.py` file geneates.
   

2. The file `network.py`uses `PyTorch` to both the forward pass of a feedforward and convolutional neural network.

