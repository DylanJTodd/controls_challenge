#test: python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller simple
#batch Metrics python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller simple
#Generate comparison report python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller simple --baseline_controller open

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
class ActorNetwork(nn.Module):
    def __init__(self, max_steering=1.0, max_acceleration=1.0):
        super(ActorNetwork, self).__init__()
        self.max_steering = max_steering
        self.max_acceleration = max_acceleration
        self.layer1 = nn.Linear(4, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 64)
        self.layer4_steering = nn.Linear(64, 1)
        self.layer4_acceleration = nn.Linear(64, 1)

        # Parameters for Gaussian policy
        self.log_std = nn.Parameter(torch.zeros(1, 2))  # Log standard deviation

    def forward(self, input_state):
        x = nn.functional.relu(self.layer1(input_state))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        steering = self.max_steering * torch.tanh(self.layer4_steering(x))
        acceleration = self.max_acceleration * torch.tanh(self.layer4_acceleration(x))
        return steering, acceleration

    def log_prob(self, input_state, action):
        steering, acceleration = self.forward(input_state)
        mean = torch.cat((steering, acceleration), dim=-1)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        return log_prob



#Critic Network
    #Multiple networks, each taking all inputs, however 8 differently trained networks for ensemble learning.
    #Evaluates actions taken by actor. Estimates the expected return (Q-value) of taking a specific action given a specific state.
    #Inputs: Current state, and current action
    #Outputs: Individual estimate of expected return (Q-value) associated with the state-action pair.
        #Ensure to add:
        #Random variable weight initialization 
        #Maybe bootstrapping (different subsets of data), 
        #Hyperparameter variability
        #Ensemble learning
class CriticNetwork(nn.Module): 
    def __init__(self, num_networks=6):
        super(CriticNetwork, self).__init__()
        self.num_critics = num_networks
        self.critics = nn.ModuleList([self.create_critic() for _ in range(num_networks)])
    
    def create_critic(self):
            return nn.Sequential(
                nn.Linear(5, 512),  # 3 state + 2 action, excludes time
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

    def forward(self, state, action):
        q_values = [critic(torch.cat([state, action], 1)) for critic in self.critics]
        return torch.stack(q_values, dim=1)


#Replay Buffer
    #Stores experiences during an agents interactions with environment. Breaks temporal correlations by shuffling and taking random buffer samples.
    #Inputs: Agents interactions with the environment
class ReplayBuffer(object):
    def __init__(self, state_dimensions, action_dimensions, max_size=200000):
        self.max_size = max_size
        self.count = 0
        self.size = 0
        self.state = numpy.zeros((self.max_size, state_dimensions))
        self.action = numpy.zeros((self.max_size, action_dimensions))
        self.reward = numpy.zeros((self.max_size, 1))
        self.next_state = numpy.zeros((self.max_size, state_dimensions))
        self.terminal_state = numpy.zeros((self.max_size, 1))


    #Stores transition (current state, action taken, reward received, next observed state, terminal state indicator)
    #Count points to the next available slot in the buffer, with index wrap around max_size
    #Size reflects current number of transitions stored
    def store(self, state, action, reward, next_state, terminal_state):
        self.state[self.count] = state
        self.action[self.count] = action
        self.reward[self.count] = reward
        self.next_state[self.count] = next_state
        self.terminal_state[self.count] = terminal_state
        self.count = (self.count + 1) % self.max_size #When 'count' reaches max_size, it will be reset to 0
        self.size = min(self.size + 1, self.max_size) #Record the number of transitions


    #Randomly selects a batch of transitions
    #Each transition contains above mentioned features
    #transitions converted to pytorch tensors
    def sample(self, batch_size):
        index = numpy.random.choice(self.size, size=batch_size)
        batch_state = torch.tensor(self.state[index], dtype=torch.float32)
        batch_action = torch.tensor(self.action[index], dtype=torch.float32)
        batch_reward = torch.tensor(self.reward[index], dtype=torch.float32)
        batch_next_state = torch.tensor(self.next_state[index], dtype=torch.float32)
        batch_terminal_state = torch.tensor(self.terminal_state[index], dtype=torch.float32)

        return batch_state, batch_action, batch_reward, batch_next_state, batch_terminal_state
    


#SAC Agent Class
    #Encapsulates the entire SAC algorithm. Coordinates above actions/classes
    #Contains useful methods for selecting actions, updating networks, and storing/retreiving data

class SAC(object):
    def __init__(self, actor, critic, replay_buffer, discount_factor=0.99, soft_update=0.0005, temperature=0.2, actor_learning=0.001, critic_learning=0.001):
        self.actor = actor
        self.critic = critic
        self.target_critic = CriticNetwork()
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.replay_buffer = replay_buffer

        self.discount_factor = discount_factor  #Priority of long-term rewards
        self.soft_update = soft_update          #Gradually updates and prevents drastic changes in target values
        self.temperature = temperature          #Entropy regularization, encourages exploration, avoids local optima
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning)

        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=1000, gamma=0.9)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=1000, gamma=0.9)
    
    
    #Takes state and uses actor network to make a prediction
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) #Add batch dimension
        with torch.no_grad():
            steering, acceleration = self.actor(state)
        return torch.cat((steering, acceleration), dim=1).squeeze(0).numpy() #Remove batch dimension

    #Samples a batch of transitions from replay buffer
    #Updates critic
    def update_networks(self, batch_size):
        state, action, reward, next_state, terminal = self.replay_buffer.sample(batch_size) # gather a batch of transitions

        # Update critic
        # Computes next actions and log probabilities  using actor network
        # Calculates target Q value using reward, discount factors, and next Q values from the target critic, and entrop y term
        # Computes current Q value
        # Calculates critic loss as MSE between current and target Q values
        # Backprop critic loss and updates network params
        # Steps scheduler.
        with torch.no_grad():
            next_steering, next_acceleration = self.actor(next_state)
            next_action = torch.cat((next_steering, next_acceleration), dim=1)
            next_log_prob = self.actor.log_prob(next_state, next_action)
            target_Q = reward + self.discount_factor * (1 - terminal) * (
                torch.min(self.target_critic(next_state, next_action)) - self.temperature * next_log_prob
            )
        
        current_Q = self.critic(state, action)
        critic_loss = nn.functional.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        # Update actor
        # Computes current actions and log probs
        # Calculates actor loss using mean value of entropy-regularized Q-values
        # Backprops and updates network params
        # Steps scheduler
        current_action, current_log_prob = self.actor(state)
        actor_loss = (self.temperature * current_log_prob - torch.min(self.critic(state, current_action))).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        # Soft update of target critic
        # Gradually updates critic network parameters to slowly track critic network params, ensuring stable Q-values
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.soft_update * param.data + (1 - self.soft_update) * target_param.data)
    
    def train(self, num_episodes, max_steps_per_episode, batch_size, env):
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps_per_episode):
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.replay_buffer.store(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state

                if self.replay_buffer.size > batch_size:
                    self.update_networks(batch_size)

                if done:
                    break

            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")


    #Inputs:
        #Time:                                      t
        #Car Velocity:                              v_ego
        #Forward Acceleration:                      a_ego
        #lateral acceleration due to road roll:     road_lataccel

    #Outputs
        #Current car lateral acceleration:          current_lataccel
        #Steer input:                               steer_action
    

    





#Environment Class
    #Represents the environment in which the agent operates. It provides the agent with inputs/observations of the current state and rewards.

#Training Loop


