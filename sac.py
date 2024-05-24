import numpy
import pandas
import torch
import os
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader
import torch.optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot
matplotlib.use('Agg')
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX
import seaborn as sns
from controllers import BaseController

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(DEVICE))
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 64)
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) * self.max_action
        log_prob = normal.log_prob(z).sum(axis=-1) - torch.sum(2 * (numpy.log(2) - z - F.softplus(-2 * z)), dim=1)
        return action, log_prob, mean, log_std

    def get_log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = torch.atanh(action / self.max_action)
        log_prob = normal.log_prob(z).sum(axis=-1) - torch.sum(2 * (numpy.log(2) - z - F.softplus(-2 * z)), dim=1)
        return log_prob
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 64)
        self.q1 = nn.Linear(64, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 512)
        self.l5 = nn.Linear(512, 256)
        self.l6 = nn.Linear(256, 64)
        self.q2 = nn.Linear(64, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.q1(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = F.relu(self.l6(q2))
        q2 = self.q2(q2)

        return q1, q2

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(200000)
        self.count = 0
        self.size = 0
        self.s = numpy.zeros((self.max_size, state_dim))
        self.a = numpy.zeros((self.max_size, action_dim))
        self.r = numpy.zeros((self.max_size, 1))
        self.s_ = numpy.zeros((self.max_size, state_dim))
        self.dw = numpy.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = numpy.random.choice(self.size, size=batch_size)
        batch_s = torch.tensor(self.s[index], dtype=torch.float).to(DEVICE)
        batch_a = torch.tensor(self.a[index], dtype=torch.float).to(DEVICE)
        batch_r = torch.tensor(self.r[index], dtype=torch.float).to(DEVICE)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float).to(DEVICE)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float).to(DEVICE)
        return batch_s, batch_a, batch_r, batch_s_, batch_dw

class SAC(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.GAMMA = 0.99
        self.TAU = 0.005
        self.alpha = 0.2

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def choose_action(self, state):
        state = torch.unsqueeze(state.clone().detach().float(), 0).to(DEVICE)
        action, _, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = replay_buffer.sample(batch_size)
        with torch.no_grad():
            next_a, next_log_prob, _, _ = self.actor.sample(batch_s_)
            q1_target, q2_target = self.critic_target(batch_s_, next_a)
            min_q_target = torch.min(q1_target, q2_target)
            next_log_prob = next_log_prob.view(batch_size, 1)  # Reshape next_log_prob to [batch_size, 1]
            q_target = batch_r + self.GAMMA * (1 - batch_dw) * (min_q_target - self.alpha * next_log_prob)
            q_target = q_target.view(batch_size, 1)  # Ensure q_target has the shape [batch_size, 1]

        q1, q2 = self.critic(batch_s, batch_a)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        new_actions, log_pi, _, _ = self.actor.sample(batch_s)
        q1_new, q2_new = self.critic(batch_s, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_pi - min_q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()





def generate_filename(index):
    return f"./data/{index:05d}.csv"

def count_steps(filename):
    df = pandas.read_csv(filename)
    return len(df)

class Environment:
    def __init__(self, max_action, start_index):
        self.max_action = max_action
        self.last_actual = None
        self.num_steps = 0
        self.current_file_index = start_index
        self.max_file_index = 19999
        self.prepare()

    def prepare(self):
        file_name = generate_filename(self.current_file_index)
        self.num_steps = count_steps(file_name) - 1
        self.controller = BaseController()
        self.model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=False)
        self.sim = TinyPhysicsSimulator(self.model, file_name, controller=self.controller, debug=False)
        self.last_actual = None

    def next_episode(self):
        self.current_file_index = torch.randint(0, 20000, (1,)).item()
        self.prepare()

    def random_action(self):
        return -2.0 + 4.0 * torch.rand(1).item()

    def get_state(self):
        state_object, target = self.sim.get_state_target(self.sim.step_idx)
        roll_lataccel, v_ego, a_ego = state_object

        # Check the type of the last action and extract the float value appropriately
        last_action = self.sim.action_history[-1]
        if isinstance(last_action, (numpy.ndarray, torch.Tensor)):
            last_action = float(last_action.item())
        else:
            last_action = float(last_action)  # already a float

        # Convert other elements to float as needed
        actual = float(self.sim.current_lataccel_history[-1])
        target = float(target)
        v_ego = float(v_ego)
        a_ego = float(a_ego)
        roll_lataccel = float(roll_lataccel)

        state = numpy.array([v_ego, a_ego, roll_lataccel, target, actual, target - actual, last_action])
        return torch.tensor(state, dtype=torch.float32).to(DEVICE)




    def step(self, action):
        # Use torch.clamp instead of numpy.clip
        action = torch.clamp(action, -self.max_action, self.max_action)

        target = self.get_state()[3].item()  # Extract the single element from tensor
        self.sim.step(action.cpu().numpy())  # Ensure action is moved to CPU and converted to numpy
        actual = self.sim.current_lataccel_history[-1]
        last_actual = self.sim.current_lataccel_history[-2]
        
        lateral_cost = 100.0 * (actual - target) ** 2
        jerk_cost = 100.0 * (actual - last_actual) ** 2 / 0.1
        delta_action = self.sim.action_history[-1] - self.sim.action_history[-2]
        action_cost = delta_action ** 2
        
        cost = (5.0 * lateral_cost + 3.0 * jerk_cost) + 10 * action_cost
        reward = -1.0 * cost
        done = self.sim.step_idx >= len(self.sim.data)
        return actual, reward, done




class PolicyEvaluator:
    def __init__(self, agent, num_eval_runs=20, model_path="./models/tinyphysics.onnx"):
        self.agent = agent
        self.num_eval_runs = num_eval_runs
        self.model_path = model_path

    class MyController(BaseController):
        def __init__(self, agent):
            super().__init__()
            self.agent = agent

        def update(self, target_lataccel, current_lataccel, state, last_action):
            roll_lataccel, v_ego, a_ego = state
            state = torch.tensor([v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel, target_lataccel-current_lataccel, last_action]).to(DEVICE)
            action = self.agent.choose_action(state)[0]
            return action

    def plot_rollout(self, sim):
        fig, ax = matplotlib.subplots(figsize=(10, 5))
        ax.plot(sim.target_lataccel_history, label="Target Lateral Acceleration", alpha=0.5)
        ax.plot(sim.current_lataccel_history, label="Actual Lateral Acceleration", alpha=0.5)
        ax.legend()
        ax.set_xlabel("Step")
        ax.set_ylabel("Lateral Acceleration")
        ax.set_title("Rollout")
        #plt.show()

    def evaluate_policy(self):
        sns.set_theme()

        model = TinyPhysicsModel(self.model_path, debug=True)
        controller = self.MyController(self.agent)
        sim_files = ["./data/00000.csv", "./data/00002.csv", "./data/00010.csv"]

        for file in sim_files:
            sim = TinyPhysicsSimulator(model, file, controller=controller, debug=False)
            print(sim.rollout())
            self.plot_rollout(sim)

        times = 20  
        evaluate_reward = 0
        for _ in range(times):
            done = False
            random_integer = torch.randint(0, 19999, (1,)).item()
            env = Environment(2, random_integer)
            episode_reward = 0
            while not done:
                state = env.get_state()
                action = self.agent.choose_action(state)[0]
                actual, reward, done = env.step(action)
                episode_reward += reward
                
            evaluate_reward += env.sim.compute_cost()["total_cost"]

        return float(evaluate_reward / float(times))

state_dim = 7
action_dim = 1
max_action = 2
agent = SAC(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(state_dim, action_dim)

batch_size = 128
total_timesteps = 1000000
eval_freq = 100000
num_eval_runs = 20
max_episode_steps = 1000
data_folder = './data'

csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

def load_csv_data(file_path):
    data = pandas.read_csv(file_path)
    return data[['t', 'vEgo', 'aEgo', 'roll', 'targetLateralAcceleration', 'steerCommand']].values

timestep = 0
while timestep < total_timesteps:
    for csv_file in csv_files:
        data = load_csv_data(csv_file)
        env = Environment(max_action, start_index=numpy.random.randint(0, len(data)))

        state = env.get_state()
        episode_reward = 0
        for step in range(min(max_episode_steps, len(data) - 1)):
            action = agent.choose_action(state)
            action = torch.from_numpy(action).unsqueeze(1).to(DEVICE)

            next_actual, reward, done = env.step(action)
            if done:
                break

            next_state = env.get_state()
            done_bool = float(done) if step + 1 < max_episode_steps else 0.0

            replay_buffer.store(state.cpu().numpy(), action.cpu().numpy(), reward, next_state.cpu().numpy(), done_bool)

            state = next_state
            episode_reward += reward
            timestep += 1

            if replay_buffer.size > batch_size:
                agent.update(replay_buffer, batch_size)

            if timestep % eval_freq == 0:
                evaluator = PolicyEvaluator(agent, num_eval_runs=num_eval_runs)
                avg_reward = evaluator.evaluate_policy()
                print(f"Evaluation at timestep {timestep}: Average Reward = {avg_reward}")

            if timestep >= total_timesteps:
                break

        # Ensure reward is a scalar
        if isinstance(reward, numpy.ndarray):
            reward = reward.item()  # Extract the scalar value from the numpy array

        print(f"File {csv_file[7:12]}: Reward {reward:.2f}")

        if timestep >= total_timesteps:
            break

print("Training complete.")