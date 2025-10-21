import numpy as np
from collections import defaultdict


class MonteCarloAgent:

    def __init__(self, n_states, n_actions, epsilon=0.1, gamma=0.99, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.q = np.zeros((n_states, n_actions))
        self.returns = defaultdict(list)
        self.episode_memory = []

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.q[state])

    def store_transition(self, state, action, reward):
        self.episode_memory.append((state, action, reward))

    def update_q_values(self):
        G = 0
        visited = set()

        for t in reversed(range(len(self.episode_memory))):
            state, action, reward = self.episode_memory[t]
            G = reward + self.gamma * G

            sa = (state, action)
            if sa not in visited:
                visited.add(sa)
                self.returns[sa].append(G)
                self.q[state, action] = np.mean(self.returns[sa])

        self.episode_memory = []
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_table(self):
        return self.q