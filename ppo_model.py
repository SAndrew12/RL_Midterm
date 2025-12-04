# model.py
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Simple Actor-Critic network for continuous action spaces.
    - Actor: Gaussian policy (mean from network, log_std is learned parameter)
    - Critic: state-value function V(s)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(128, 128)):
        super().__init__()

        # Actor network: obs -> mean(action)
        actor_layers = []
        last_size = obs_dim
        for h in hidden_sizes:
            actor_layers.append(nn.Linear(last_size, h))
            actor_layers.append(nn.ReLU())
            last_size = h
        actor_layers.append(nn.Linear(last_size, act_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Log standard deviation for Gaussian policy
        # One parameter per action dimension
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic network: obs -> value
        critic_layers = []
        last_size = obs_dim
        for h in hidden_sizes:
            critic_layers.append(nn.Linear(last_size, h))
            critic_layers.append(nn.ReLU())
            last_size = h
        critic_layers.append(nn.Linear(last_size, 1))
        self.critic = nn.Sequential(*critic_layers)

    def _distribution(self, obs: torch.Tensor) -> Normal:
        """
        Create a Normal distribution for actions given observations.
        """
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Critic forward pass: returns V(s).
        """
        return self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor):
        """
        Sample an action, and return:
        - action (tensor)
        - log probability of action
        - value V(s)
        """
        dist = self._distribution(obs)
        action = dist.rsample()  # reparameterized sample
        log_prob = dist.log_prob(action).sum(axis=-1)
        value = self.value(obs)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Compute log_probs, entropy, and value for given (obs, actions).
        Used during PPO updates.
        """
        dist = self._distribution(obs)
        log_probs = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.value(obs)
        return log_probs, entropy, value
