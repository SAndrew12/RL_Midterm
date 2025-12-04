"""
Double DQN Model Implementation

This module contains:
1. QNetwork - Deep Q-Network architecture
2. ReplayBuffer - Experience replay mechanism
3. DoubleDQNAgent - Complete Double DQN agent with training logic

Double DQN prevents overestimation bias by:
- Using online network to SELECT the best action
- Using target network to EVALUATE that action's Q-value
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import Tuple, List

# Named tuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.

    Breaks temporal correlation between consecutive samples by randomly
    sampling from past experiences during training.
    """

    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """
        Sample a random batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        experiences = random.sample(self.buffer, batch_size)

        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor([e.done for e in experiences])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Deep Q-Network for approximating Q(s, a).

    Architecture:
        Input Layer -> Hidden Layers (ReLU) -> Output Layer

    Input: State observation vector (9D for single aircraft ATC)
    Output: Q-values for each action (9 actions for discrete ATC)
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: List[int] = [256, 256]
    ):
        """
        Initialize Q-Network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
        """
        super(QNetwork, self).__init__()

        # Build network architecture
        layers = []
        input_dim = state_dim

        # Hidden layers with ReLU activation
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))  # Layer normalization for stability
            input_dim = hidden_dim

        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            Q-values for each action [batch_size, action_dim]
        """
        return self.network(state)


class DoubleDQNAgent:
    """
    Double DQN Agent with Experience Replay.

    Key Components:
    1. Online Network: Used for selecting actions and computing Q(s,a) for training
    2. Target Network: Used for computing target Q-values (updated periodically)
    3. Experience Replay: Stores past experiences for training
    4. Epsilon-Greedy: Balances exploration vs exploitation

    Double DQN Algorithm:
    - Action selection: a* = argmax_a Q_online(s', a)
    - Action evaluation: Q_target(s', a*)
    - This decoupling reduces overestimation bias
    """

    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            learning_rate: float = 1e-4,
            gamma: float = 0.99,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01,
            epsilon_decay: float = 0.995,
            buffer_capacity: int = 100000,
            batch_size: int = 64,
            target_update_freq: int = 100,
            hidden_dims: List[int] = [256, 256],
            device: str = None
    ):
        """
        Initialize Double DQN Agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            learning_rate: Learning rate for Adam optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay factor for epsilon
            buffer_capacity: Maximum size of replay buffer
            batch_size: Number of samples per training batch
            target_update_freq: Frequency of target network updates (in training steps)
            hidden_dims: Hidden layer dimensions for Q-network
            device: Device to use ('cuda' or 'cpu')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize networks
        self.online_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)

        # Copy weights from online to target network
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode

        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Training counters
        self.train_step = 0
        self.episode_count = 0

        print(f"=" * 60)
        print(f"Double DQN Agent Initialized")
        print(f"=" * 60)
        print(f"Device: {self.device}")
        print(f"State Dimension: {state_dim}")
        print(f"Action Dimension: {action_dim}")
        print(f"Hidden Layers: {hidden_dims}")
        print(f"Learning Rate: {learning_rate}")
        print(f"Gamma: {gamma}")
        print(f"Buffer Capacity: {buffer_capacity}")
        print(f"Batch Size: {batch_size}")
        print(f"=" * 60)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state observation
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Selected action (integer)
        """
        # Exploration: random action
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        # Exploitation: greedy action based on Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_experience(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self) -> float:
        """
        Perform one training step using Double DQN algorithm.

        Double DQN Update Rule:
        1. Sample batch from replay buffer
        2. Compute current Q-values: Q_online(s, a)
        3. Select next actions using online network: a* = argmax_a Q_online(s', a)
        4. Evaluate actions using target network: Q_target(s', a*)
        5. Compute target: y = r + gamma * Q_target(s', a*) * (1 - done)
        6. Update online network: minimize (Q_online(s, a) - y)^2

        Returns:
            Training loss (TD error)
        """
        # Check if enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q-values from online network
        # Q(s, a) for the actions that were actually taken
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: decouple action selection from evaluation
        with torch.no_grad():
            # Use online network to SELECT best next actions
            next_actions = self.online_net(next_states).argmax(dim=1)

            # Use target network to EVALUATE those actions
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Compute target Q-values
            # y = r + gamma * Q_target(s', a*) if not done, else y = r
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss (Mean Squared Error between current Q and target Q)
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the online network
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Update target network periodically (hard update)
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """Hard update: copy weights from online network to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate exponentially."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'episode_count': self.episode_count,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }
        torch.save(checkpoint, filepath)
        print(f"âœ“ Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load agent checkpoint.

        Args:
            filepath: Path to load checkpoint from
        """
        # PyTorch 2.6+ requires weights_only=False for backward compatibility
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        self.episode_count = checkpoint.get('episode_count', 0)
        print(f"âœ“ Model loaded from {filepath}")
        print(f"  - Training step: {self.train_step}")
        print(f"  - Episode count: {self.episode_count}")
        print(f"  - Epsilon: {self.epsilon:.4f}")

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for all actions in a given state.

        Args:
            state: State observation

        Returns:
            Q-values for all actions
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_tensor)
            return q_values.cpu().numpy()[0]