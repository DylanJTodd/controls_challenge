#test: python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller simple
#batch Metrics python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller simple
#Generate comparison report python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller simple --baseline_controller open

#Car Velocity:                              v_ego
#Forward Acceleration:                      a_ego
#lateral acceleration due to road roll:     road_lataccel
#Current car lateral acceleration:          current_lataccel
#Steer input:                               steer_action

import numpy
import pandas
import torch
import os
import glob
from torch import nn
from torch.nn import functional
import sklearn.model_selection
from torch.utils.data import Dataset, DataLoader
import torch.optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX

#Actor Network
    # Determines the policy, maximizes expected future rewards
    # Inputs: Current state
    # Outputs: Probability distribution over action space 

#Critic Network
    #Multiple networks, each taking all inputs, however 8 differently trained networks for ensemble learning.
    #Evaluates actions taken by actor. Estimates the expected return (Q-value) of taking a specific action given a specific state.
    #Inputs: Current state, and current action
    #Outputs: Individual estimate of expected return (Q-value) associated with the state-action pair.

#Replay Buffer
    #Stores experiences during an agents interactions with environment. Breaks temporal correlations by shuffling and taking random buffer samples.
    #Inputs: Agents interactions with the environment

#SAC Agent Class
    #Encapsulates the entire SAC algorithm. Coordinates above actions/classes
    #Contains useful methods for selecting actions, updating networks, and storing/retreiving data

#Environment Class
    #Represents the environment in which the agent operates. It provides the agent with inputs/observations of the current state and rewards.

#Training Loop

class ActorNetwork(nn.Module):
    def __init__(self, max_actions=2):
        super(ActorNetwork, self).__init__()
        self.max_actions = max_actions
        self.layer1 = nn.Linear(4, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 64)
        self.layer4 = nn.Linear(64, 1)

    def forward(self, input_state):
        input_state = nn.functional.relu(self.layer1(input_state))
        input_state = nn.functional.relu(self.layer2(input_state))
        input_state = nn.functional.relu(self.layer3(input_state))
        output_action = self.max_actions * torch.tanh(self.layer4(input_state)) #[-2, 2]

class CriticNetwork(nn.Module):
    #Ensure to add:
        #Random weight initialization 
        #Maybe bootstrapping (different subsets of data), 
        #Hyperparameter variability
        #Boosting (THIS IS A MUST, SOUNDS SUPER COOL)

