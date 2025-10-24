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
    def _row(self, s: int) -> np.ndarray:

        if self.use_double_q:
            return self.Q1[s] + self.Q2[s]
        return self.Q[s]

    def _argmax_tiebreak(self, row: np.ndarray) -> int:

        m = np.max(row)
        best = np.flatnonzero(row == m)
        return int(self._rng.choice(best))
    def select_action(self, state: int) -> int:

        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))
        return self._argmax_tiebreak(self._row(state))
    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        alpha_eff = self.alpha

        if self.use_double_q:
            if self._rng.random() < 0.5:
                if done:
                    target = r
                else:
                    a_star = self._argmax_tiebreak(self.Q1[s_next])
                    target = r + self.gamma * float(self.Q2[s_next, a_star])

                self.Q1[s, a] += alpha_eff * (target - float(self.Q1[s, a]))
            else:
                if done:
                    target = r
                else:
                    a_star = self._argmax_tiebreak(self.Q2[s_next])
                    target = r + self.gamma * float(self.Q1[s_next, a_star])

                self.Q2[s, a] += alpha_eff * (target - float(self.Q2[s, a]))
        else:
            if done:
                target = r
            else:
                target = r + self.gamma * float(np.max(self.Q[s_next]))

            self.Q[s, a] += alpha_eff * (target - float(self.Q[s, a]))
    def q_values(self) -> np.ndarray:
        if self.use_double_q:
            return self.Q1 + self.Q2
        return self.Q

    def policy(self) -> np.ndarray:
        return np.argmax(self.q_values(), axis=1)
