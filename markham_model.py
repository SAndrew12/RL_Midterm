import numpy as np
import random


class QLearningAgent:

    def __init__(
            self,
            n_states: int,
            n_actions: int,
            seed: int = 123,
            *,
            gamma: float = 0.99,
            alpha: float = 0.8,
            epsilon: float = 1.0,
            optimistic_init: float = 8.0,
            use_double_q: bool = True,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.use_double_q = bool(use_double_q)
        self._rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)


        if self.use_double_q:
            self.Q1 = np.full((self.n_states, self.n_actions),
                              float(optimistic_init), dtype=np.float32)
            self.Q2 = np.full((self.n_states, self.n_actions),
                              float(optimistic_init), dtype=np.float32)
        else:
            self.Q = np.full((self.n_states, self.n_actions),
                             float(optimistic_init), dtype=np.float32)

    # ---------- helpers ----------
    def _row(self, s: int) -> np.ndarray:
        """Returns the effective Q-value row for state s (Q1+Q2 for Double Q)."""
        if self.use_double_q:
            return self.Q1[s] + self.Q2[s]
        return self.Q[s]

    def _argmax_tiebreak(self, row: np.ndarray) -> int:
        """Picks the index of the maximum value, resolving ties randomly."""
        m = np.max(row)
        best = np.flatnonzero(row == m)
        return int(self._rng.choice(best))

    # ---------- policy ----------
    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if self._rng.random() < self.epsilon:
            # Explore: pick a random action
            return int(self._rng.integers(self.n_actions))
        # Exploit: pick the best action
        return self._argmax_tiebreak(self._row(state))

    # ---------- learning ----------
    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        """
        Performs one update step using Q-Learning or Double Q-Learning.
        The current learning rate (self.alpha) is used.
        """
        # The current learning rate is used uniformly
        alpha_eff = self.alpha

        if self.use_double_q:
            # Double Q-Learning: update one table using the other's max evaluation
            if self._rng.random() < 0.5:
                # Update Q1 using Q2's value estimate
                if done:
                    target = r
                else:
                    # Q1 selects the action, Q2 evaluates it
                    a_star = self._argmax_tiebreak(self.Q1[s_next])
                    target = r + self.gamma * float(self.Q2[s_next, a_star])

                # Update Q1
                self.Q1[s, a] += alpha_eff * (target - float(self.Q1[s, a]))
            else:
                # Update Q2 using Q1's value estimate
                if done:
                    target = r
                else:
                    # Q2 selects the action, Q1 evaluates it
                    a_star = self._argmax_tiebreak(self.Q2[s_next])
                    target = r + self.gamma * float(self.Q1[s_next, a_star])

                # Update Q2
                self.Q2[s, a] += alpha_eff * (target - float(self.Q2[s, a]))
        else:
            # Classic single-table Q-Learning
            if done:
                target = r
            else:
                target = r + self.gamma * float(np.max(self.Q[s_next]))

            # Update Q
            self.Q[s, a] += alpha_eff * (target - float(self.Q[s, a]))

    # ---------- utilities ----------
    def q_values(self) -> np.ndarray:
        """Returns the overall Q-table (sum for Double Q-Learning, or single Q)."""
        if self.use_double_q:
            return self.Q1 + self.Q2
        return self.Q

    def policy(self) -> np.ndarray:
        """Returns the best action for every state (the greedy policy)."""
        return np.argmax(self.q_values(), axis=1)
