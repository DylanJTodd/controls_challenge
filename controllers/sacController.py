import torch
import numpy

class sacController:
    def __init__(self, actor_network, critic_network, log_std):
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.log_std = log_std

    def update(self, target_lataccel, current_lataccel, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        steering, acceleration = self.actor_network(state_tensor)

        action = torch.cat((steering, acceleration), dim=1).squeeze(0).numpy()
        return action
