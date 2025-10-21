import numpy as np
from collections import defaultdict


class MonteCarloAgent:
    """
    Monte Carlo RL Agent using First-Visit MC Control with Epsilon-Greedy policy
    """

    def __init__(self, n_states, n_actions, epsilon=0.1, gamma=0.99, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Monte Carlo Agent

        Args:
            n_states: Number of states in the environment
            n_actions: Number of actions available
            epsilon: Exploration rate for epsilon-greedy policy
            gamma: Discount factor for future rewards
            epsilon_decay: Decay rate for epsilon after each episode
            epsilon_min: Minimum epsilon value
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        # Q-table: state-action values
        self.q = np.zeros((n_states, n_actions))

        # Returns: for calculating average returns for each state-action pair
        self.returns = defaultdict(list)

        # Episode memory: stores (state, action, reward) tuples
        self.episode_memory = []

    def select_action(self, state):
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state

        Returns:
            action: Selected action
        """
        if np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, self.n_actions)
        else:
            # Exploitation: best action based on Q-values
            return np.argmax(self.q[state])

    def store_transition(self, state, action, reward):
        """
        Store a transition in episode memory

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
        """
        self.episode_memory.append((state, action, reward))

    def update_q_values(self):
        """
        Update Q-values using First-Visit Monte Carlo method after episode completion
        """
        # Calculate returns for each step in the episode
        G = 0  # Initialize return
        visited_state_action = set()

        # Iterate through episode in reverse to calculate returns
        for t in reversed(range(len(self.episode_memory))):
            state, action, reward = self.episode_memory[t]

            # Update return with discounted reward
            G = reward + self.gamma * G

            # First-visit MC: only update if this state-action pair hasn't been seen in this episode yet
            state_action = (state, action)
            if state_action not in visited_state_action:
                visited_state_action.add(state_action)

                # Store the return
                self.returns[state_action].append(G)

                # Update Q-value as average of all returns for this state-action pair
                self.q[state, action] = np.mean(self.returns[state_action])

        # Clear episode memory for next episode
        self.episode_memory = []

        # Decay epsilon for less exploration over time
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table(self):
        """
        Return the current Q-table

        Returns:
            q: Q-table numpy array
        """
        return self.q