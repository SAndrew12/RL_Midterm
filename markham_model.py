import numpy as np
import random

class QLearningAgent:
    """
    High-performance tabular Q-Learning for discrete GridWorld.

    Key optimizations:
      • Optional Double Q-Learning (reduces overestimation bias).
      • Epsilon-greedy with robust tie-breaking (no directional bias).
      • Step-wise epsilon decay with gentle floor.
      • Per-(s,a) adaptive learning rate with a safe floor.
      • Optimistic Q initialization to encourage efficient exploration.
      • Reproducible RNGs.

    Usage:
      agent = QLearningAgent(n_states, n_actions, seed=123, use_double_q=True)
      a = agent.select_action(s)
      agent.update(s, a, r, s_next, done)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        seed: int = 123,
        *,
        gamma: float = 0.99,
        # Learning rate schedule (per-visit decay with floor)
        alpha: float = 0.8,
        alpha_min: float = 0.05,
        alpha_decay: float = 0.998,
        # Exploration schedule (step-wise decay with floor)
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.999,
        # Optimistic initialization (encourages deep exploration early)
        optimistic_init: float = 5.0,
        # Double Q-Learning toggle
        use_double_q: bool = True,
    ):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)

        # LR schedule
        self.alpha = float(alpha)
        self.alpha_min = float(alpha_min)
        self.alpha_decay = float(alpha_decay)

        # Exploration schedule
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        # Reproducibility
        self._rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.use_double_q = bool(use_double_q)

        # Q tables
        if self.use_double_q:
            self.Q1 = np.full((self.n_states, self.n_actions),
                              float(optimistic_init), dtype=np.float32)
            self.Q2 = np.full((self.n_states, self.n_actions),
                              float(optimistic_init), dtype=np.float32)
        else:
            self.Q = np.full((self.n_states, self.n_actions),
                             float(optimistic_init), dtype=np.float32)

        # Visit counts for adaptive learning rates
        self.visits = np.zeros((self.n_states, self.n_actions), dtype=np.int32)

    # ---------- Internals ----------
    def _row(self, s: int) -> np.ndarray:
        if self.use_double_q:
            return self.Q1[s] + self.Q2[s]
        return self.Q[s]

    def _argmax_tiebreak(self, row: np.ndarray) -> int:
        m = np.max(row)
        best = np.flatnonzero(row == m)
        return int(self._rng.choice(best))

    # ---------- Policy ----------
    def select_action(self, state: int) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return self._argmax_tiebreak(self._row(state))

    # ---------- Learning ----------
    def _alpha_eff(self, s: int, a: int) -> float:
        # Gentle per-(s,a) decay with a minimum floor
        return max(self.alpha * (self.alpha_decay ** self.visits[s, a]), self.alpha_min)

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        self.visits[s, a] += 1
        alpha_eff = self._alpha_eff(s, a)

        if self.use_double_q:
            # Hasselt-style Double Q-Learning:
            # randomly pick which table to update; action selection from one,
            # evaluation from the other
            if self._rng.random() < 0.5:
                # Update Q1 using argmax from Q1 and value from Q2
                if done:
                    target = r
                else:
                    a_star = self._argmax_tiebreak(self.Q1[s_next])
                    target = r + self.gamma * float(self.Q2[s_next, a_star])
                td = target - float(self.Q1[s, a])
                self.Q1[s, a] += alpha_eff * td
            else:
                # Update Q2 using argmax from Q2 and value from Q1
                if done:
                    target = r
                else:
                    a_star = self._argmax_tiebreak(self.Q2[s_next])
                    target = r + self.gamma * float(self.Q1[s_next, a_star])
                td = target - float(self.Q2[s, a])
                self.Q2[s, a] += alpha_eff * td
        else:
            # Standard 1-table Q-Learning
            if done:
                target = r
            else:
                target = r + self.gamma * float(np.max(self.Q[s_next]))
            self.Q[s, a] += alpha_eff * (target - float(self.Q[s, a]))

        # Step-wise epsilon decay (within episode)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def q_values(self) -> np.ndarray:
        if self.use_double_q:
            return self.Q1 + self.Q2
        return self.Q

    def policy(self) -> np.ndarray:
        return np.argmax(self.q_values(), axis=1)
