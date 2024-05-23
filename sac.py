# test: python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller simple
# batch Metrics python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller simple
# Generate comparison report python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller simple --baseline_controller open

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

# Inputs:
    # Time:                                      t
    # Car Velocity:                              v_ego
    # Forward Acceleration:                      a_ego
    # lateral acceleration due to road roll:     road_lataccel

# Outputs
    # Current car lateral acceleration:          current_lataccel
    # Steer input:                               steer_action
    

# Actor Network
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
        self.log_std = nn.Parameter(torch.zeros(1, 2))  # Log standard deviation ([1,2])

    def forward(self, input_state):
        x = nn.functional.relu(self.layer1(input_state))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        steering = self.max_steering * torch.tanh(self.layer4_steering(x)) #[1,1]
        acceleration = self.max_acceleration * torch.tanh(self.layer4_acceleration(x)) #[1,1]
        return steering, acceleration

    def log_prob(self, input_state, action):
        steering, acceleration = self.forward(input_state)
        mean = torch.cat((steering, acceleration), dim=1)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(mean).sum(dim=-1, keepdim=True)
        return log_prob

# Critic Network
    # Multiple networks, each taking all inputs, however 8 differently trained networks for ensemble learning.
    # Evaluates actions taken by actor. Estimates the expected return (Q-value) of taking a specific action given a specific state.
    # Inputs: Current state, and current action
    # Outputs: Individual estimate of expected return (Q-value) associated with the state-action pair.
        # Ensure to add:
        # Random variable weight initialization 
        # Maybe bootstrapping (different subsets of data), 
        # Hyperparameter variability
        # Ensemble learning
class CriticNetwork(nn.Module): 
    def __init__(self, num_networks=6):
        super(CriticNetwork, self).__init__()
        self.num_critics = num_networks
        self.critics = nn.ModuleList([self.create_critic() for _ in range(num_networks)])
    
    def create_critic(self):
            return nn.Sequential(
                nn.Linear(6, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

    def forward(self, state, action):
        steering, acceleration = action
        action = torch.cat((steering, acceleration), dim=1)
        q_values = [critic(torch.cat([state, action], 1)) for critic in self.critics]
        return torch.stack(q_values, dim=1)

# Replay Buffer
    # Stores experiences during an agents interactions with environment. Breaks temporal correlations by shuffling and taking random buffer samples.
    # Inputs: Agents interactions with the environment
class ReplayBuffer(object):
    def __init__(self, state_dimensions, max_size=200000):
        self.max_size = max_size
        self.count = 0
        self.size = 0
        self.state = numpy.zeros((self.max_size, state_dimensions))
        self.steering = numpy.zeros((self.max_size, 1))
        self.acceleration = numpy.zeros((self.max_size, 1))
        self.reward = numpy.zeros((self.max_size, 1))
        self.next_state = numpy.zeros((self.max_size, state_dimensions))
        self.terminal_state = numpy.zeros((self.max_size, 1))


    # Stores transition (current state, action taken, reward received, next observed state, terminal state indicator)
    # Count points to the next available slot in the buffer, with index wrap around max_size
    # Size reflects current number of transitions stored
    def store(self, state, steering, acceleration, reward, next_state, terminal_state):
        self.state[self.count] = state
        self.steering[self.count] = steering
        self.acceleration[self.count] = acceleration
        self.reward[self.count] = reward
        self.next_state[self.count] = next_state
        self.terminal_state[self.count] = terminal_state
        self.count = (self.count + 1) % self.max_size # When 'count' reaches max_size, it will be reset to 0
        self.size = min(self.size + 1, self.max_size) # Record the number of transitions


    # Randomly selects a batch of transitions
    # Each transition contains above mentioned features
    # transitions converted to pytorch tensors
    def sample(self, batch_size):
        index = numpy.random.randint(0, self.size, size=batch_size)
        batch_state = torch.tensor(self.state[index], dtype=torch.float32)
        batch_steering = torch.tensor(self.steering[index], dtype=torch.float32)
        batch_acceleration = torch.tensor(self.acceleration[index], dtype=torch.float32)
        batch_reward = torch.tensor(self.reward[index], dtype=torch.float32)
        batch_next_state = torch.tensor(self.next_state[index], dtype=torch.float32)
        batch_terminal_state = torch.tensor(self.terminal_state[index], dtype=torch.float32)

        return batch_state, batch_steering, batch_acceleration, batch_reward, batch_next_state, batch_terminal_state
        


# SAC Agent Class
    # Encapsulates the entire SAC algorithm. Coordinates above actions/classes
    # Contains useful methods for selecting actions, updating networks, and storing/retreiving data
class SAC(object):
    def __init__(self, actor, critic, replay_buffer, discount_factor=0.99, soft_update=0.0005, temperature=0.2, actor_learning=0.001, critic_learning=0.001):
        self.actor = actor
        self.critic = critic
        self.target_critic = CriticNetwork(num_networks=critic.num_critics)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.replay_buffer = replay_buffer

        self.discount_factor = discount_factor  # Priority of long-term rewards
        self.soft_update = soft_update          # Gradually updates and prevents drastic changes in target values
        self.temperature = temperature          # Entropy regularization, encourages exploration, avoids local optima
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning)

        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=1000, gamma=0.9)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=1000, gamma=0.9)
    
    
    # Takes state and uses actor network to make a prediction
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        with torch.no_grad():
            steering, acceleration = self.actor(state)
        return torch.cat((steering, acceleration), dim=1).squeeze(0).numpy() # Remove batch dimension

    # Samples a batch of transitions from replay buffer
    # Updates critic & actor networks
    def update_networks(self, batch_size):
        state, steering, acceleration, reward, next_state, terminal = self.replay_buffer.sample(batch_size)

        # Update critic
        with torch.no_grad():
            next_steering, next_acceleration = self.actor(next_state)
            next_action = tuple(next_steering, next_acceleration)
            next_log_prob = self.actor.log_prob(next_state, next_action)

            target_Q = reward + self.discount_factor * (1 - terminal) * (
                torch.min(self.target_critic(next_state, next_action), dim=1)[0].unsqueeze(-1) - self.temperature * next_log_prob
            )
        
        current_Q = self.critic(state, tuple(steering, acceleration))
        critic_loss = nn.functional.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        # Update actor
        current_steering, current_acceleration = self.actor(state)
        current_action = tuple(current_steering, current_acceleration)
        current_log_prob = self.actor.log_prob(state, current_action)

        actor_loss = (self.temperature * current_log_prob - torch.min(self.critic(state, current_action), dim=1)[0]).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        # Soft update of target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.soft_update * param.data + (1 - self.soft_update) * target_param.data)
    
    # Trains networks based on sampled experience through running multiple episodes and selecting multiple acitons. 
    def train(self, num_episodes, max_steps_per_episode, batch_size, environment):
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0

            for step in range(max_steps_per_episode):
                action = self.select_action(state)
                steering, acceleration = action
                next_state, reward, done, _ = environment.step(action)
                self.replay_buffer.store(state, steering, acceleration, reward, next_state, done)
                episode_reward += reward
                state = next_state

                if self.replay_buffer.size > batch_size:
                    self.update_networks(batch_size)

                if done:
                    break

            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")


# Environment Class
    # Represents the environment in which the agent operates. It provides the agent with inputs/observations of the current state and rewards.
class Environment:
    def __init__(self, model_path, data_path, controller, debug=False):
        self.model = TinyPhysicsModel(model_path, debug)
        self.simulator = TinyPhysicsSimulator(self.model, data_path, controller, debug)
        self.debug = debug


    # Reinitializes the simulator to starting state and returns initial state of environment
    def reset(self):
        self.simulator.reset()
        state = self.get_state()
        return state

    # Takes one step in the simulation based on the given action. Updates simulator, computes and returns next state and reward
    def step(self, action):
        steering_action, acceleration_action = action
        self.simulator.control_step(self.simulator.step_idx)
        self.simulator.sim_step(self.simulator.step_idx)
        self.simulator.step_idx += 1
        next_state = self.get_state()
        reward = self.compute_reward()
        done = self.simulator.step_idx >= len(self.simulator.data)
        return next_state, reward, done, {}

    
    # Retrieves the current state of simulator as a vector
    def get_state(self):
        state, _ = self.simulator.get_state_target(self.simulator.step_idx)
        state_vector = numpy.array([state.roll_lataccel, state.v_ego, state.a_ego, self.simulator.current_lataccel])
        return state_vector

    # Calculates reward based off the difference between target and current lateral acceleration
    def compute_reward(self):
        if self.simulator.step_idx <= 0 or self.simulator.step_idx > len(self.simulator.target_lataccel_history):
            return 0
        
        target_lataccel = self.simulator.target_lataccel_history[self.simulator.step_idx - 1]
        current_lataccel = self.simulator.current_lataccel
        reward = -numpy.abs(target_lataccel - current_lataccel).item()
        return reward


    

# Policy evaluation
# Runs episodes using learned policy and calculates evaluation metrics
class PolicyEvaluator:
    def __init__(self, environment, agent, num_episodes=10, max_steps_per_episode=1000):
        self.environment = environment
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.total_rewards = []

    def evaluate(self):
        self.total_rewards = []
        for episode in range(self.num_episodes):
            state = self.environment.reset()
            episode_reward = 0
            for step in range(self.max_steps_per_episode):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.environment.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    break
            self.total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
        
        avg_reward = numpy.mean(self.total_rewards)
        std_reward = numpy.std(self.total_rewards)
        print(f"Average Reward over {self.num_episodes} episodes: {avg_reward}")
        print(f"Standard Deviation of Reward: {std_reward}")
        
        self.plot_rewards()
        self.plot_reward_distribution()
        
        return avg_reward, std_reward

    def plot_rewards(self):
        matplotlib.pyplot.figure(figsize=(10, 6))
        matplotlib.pyplot.plot(range(1, self.num_episodes + 1), self.total_rewards, marker='o', linestyle='-')
        matplotlib.pyplot.title('Total Rewards per Episode')
        matplotlib.pyplot.xlabel('Episode')
        matplotlib.pyplot.ylabel('Total Reward')
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.show()

    def plot_reward_distribution(self):
        matplotlib.pyplot.figure(figsize=(10, 6))
        matplotlib.pyplot.hist(self.total_rewards, bins=20, density=True, alpha=0.6, color='g')
        matplotlib.pyplot.title('Distribution of Total Rewards')
        matplotlib.pyplot.xlabel('Total Reward')
        matplotlib.pyplot.ylabel('Frequency')
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.show()

