import numpy as np
from dataclasses import dataclass


def _reset_obs(env, seed=None):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    if not isinstance(obs, (int, np.integer)):
        try: obs = int(obs)
        except Exception:
            raise RuntimeError(f"State not discrete/int-like: {type(obs)} -> {obs!r}")
    return int(obs)

def _step(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, _ = out
        done = bool(terminated or truncated)
    elif isinstance(out, tuple) and len(out) == 3:
        obs, reward, done = out
    else:
        raise RuntimeError(f"Unexpected step() return: {type(out)} len={len(out) if isinstance(out, tuple) else 'n/a'}")
    if not isinstance(obs, (int, np.integer)):
        try: obs = int(obs)
        except Exception:
            raise RuntimeError(f"State not discrete/int-like: {type(obs)} -> {obs!r}")
    return int(obs), float(reward), bool(done)

def _eps_greedy(q_row, n_actions, eps, rng=np.random):
    if rng.rand() < eps:
        return int(rng.randint(n_actions))

    m = q_row.max()
    ties = np.flatnonzero(q_row == m)
    return int(rng.choice(ties))


@dataclass
class NStepBootConfig:
    n: int = 10
    alpha_start: float = 0.20
    alpha_end: float = 0.05
    alpha_decay_episodes: int = 80
    gamma: float = 0.995
    epsilon_start: float = 0.80
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 60
    optimistic_init: float = 1.0

class NStepBootQLearningAgent:
    """
    n-step bootstrapped Q-learning:
      G = sum_{i=1..n} gamma^{i-1} r_{t+i} + gamma^n * max_a Q(s_{t+n}, a)   (if nonterminal before horizon)
      Q(s_t, a_t) <- Q + alpha * (G - Q)
    """
    def __init__(self, n_states:int, n_actions:int, config:NStepBootConfig = NStepBootConfig(), rng=np.random):
        self.n_states = n_states
        self.n_actions = n_actions
        self.cfg = config
        self.rng = rng
        # optimistic start helps reach goal early
        self.Q = np.full((n_states, n_actions), fill_value=self.cfg.optimistic_init, dtype=np.float32)

    def _schedule(self, start, end, ep, horizon):
        if horizon <= 0: return end
        frac = min(1.0, max(0.0, ep / horizon))
        return start + frac * (end - start)

    def epsilon(self, episode_idx:int) -> float:
        return float(self._schedule(self.cfg.epsilon_start, self.cfg.epsilon_end,
                                    episode_idx, self.cfg.epsilon_decay_episodes))

    def alpha(self, episode_idx:int) -> float:
        return float(self._schedule(self.cfg.alpha_start, self.cfg.alpha_end,
                                    episode_idx, self.cfg.alpha_decay_episodes))

    def select_action(self, state:int, eps:float) -> int:
        return _eps_greedy(self.Q[state], self.n_actions, eps, self.rng)

    def learn_episode(self, env, max_steps:int, episode_idx:int):

        s0 = _reset_obs(env, seed=None)
        eps = self.epsilon(episode_idx)

        states = [s0]
        actions = [self.select_action(s0, eps)]
        rewards = [0.0]  # align so rewards[t+1] exists

        T = float('inf')
        t = 0
        cum_reward = 0.0

        while True:
            if t < T:
                s_next, r, done = _step(env, actions[t])
                rewards.append(r)
                cum_reward += r
                states.append(s_next)

                if done or (t + 1) >= max_steps:
                    T = t + 1
                else:
                    actions.append(self.select_action(s_next, eps))

            tau = t - self.cfg.n + 1
            if tau >= 0:
                # n-step return with max bootstrap (Q-learning style)
                upper = (tau + self.cfg.n) if T == float('inf') else min(tau + self.cfg.n, int(T))
                G = 0.0
                for i in range(tau + 1, upper + 1):
                    G += (self.cfg.gamma ** (i - tau - 1)) * rewards[i]

                # bootstrapped term if we haven’t hit terminal before tau+n
                if tau + self.cfg.n < T:
                    s_tp_n = states[tau + self.cfg.n]
                    G += (self.cfg.gamma ** self.cfg.n) * np.max(self.Q[s_tp_n])

                s_tau = states[tau]
                a_tau = actions[tau]
                a = self.alpha(episode_idx)
                self.Q[s_tau, a_tau] += a * (G - self.Q[s_tau, a_tau])

            if tau == T - 1:
                break
            t += 1

        return float(cum_reward)
