"""
ALIAS = 'markham'

Pure Q-Learning trainer for the Dynamic GridWorld environment
with strong optimizations (Double Q-Learning default).

Required spec (kept intact):
  SEED = 123
  SIZE = 10
  EPISODES = 100
  FPS = 0  (no rendering)

Console output:
  - Per-episode return, moving average (window=10), epsilon
  - Final training summary line (spec-compliant)
  - Extra: Greedy post-training evaluation over 20 episodes
"""

import os
import sys
import random
import numpy as np
from typing import Tuple, List

# Local path: expect exam_env.py in same directory
sys.path.insert(0, os.path.dirname(__file__) or ".")

from exam_env import GridWorldEnv
from markham_model import QLearningAgent

# ------------------- Spec-required constants (do not change) -------------------
SEED = 123
SIZE = 10
EPISODES = 100
FPS = 0  # Not used here (no rendering), retained for spec compliance
# -------------------------------------------------------------------------------

# Tuned defaults for high performance on this GridWorld
GAMMA = 0.99

# Learning-rate schedule
ALPHA = 0.8
ALPHA_MIN = 0.05
ALPHA_DECAY = 0.998

# Exploration schedule
EPSILON = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.999

# Optimistic init
OPTIMISTIC_INIT = 5.0

# Use Double Q-Learning (recommended). Set to False for classic single-table Q-Learning.
USE_DOUBLE_Q = True

# Episode step cap: tighter than 4x to cut excess -1 penalties when the policy improves
def _episode_max_steps(size: int) -> int:
    return size * size * 3

def _moving_avg(xs: List[float], k: int = 10) -> float:
    if not xs:
        return 0.0
    k = min(k, len(xs))
    return float(np.mean(xs[-k:]))

def _evaluate_greedy(env: GridWorldEnv, agent: QLearningAgent, episodes: int = 20, max_steps: int = 300) -> float:
    """Run greedy evaluation (epsilon forced to zero) and return avg cumulative reward."""
    # Temporarily store and set epsilon to zero
    old_eps = agent.epsilon
    agent.epsilon = 0.0

    returns = []
    for _ in range(episodes):
        s = env.reset()  # do not reseed; we want generalization
        ep_ret = 0.0
        for _ in range(max_steps):
            a = agent.policy()[s]  # greedy action under learned Q
            s_next, r, terminated = env.step(a)
            ep_ret += r
            s = s_next
            if terminated:
                break
        returns.append(ep_ret)

    # Restore epsilon
    agent.epsilon = old_eps
    return float(np.mean(returns)) if returns else 0.0

def train_and_eval(
    seed: int = SEED,
    size: int = SIZE,
    episodes: int = EPISODES,
    render: bool = False
) -> Tuple[float, List[float]]:
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)

    env = GridWorldEnv(
        render_mode=("human" if render else None),
        size=size,
        seed=seed,
        max_barriers=20
    )

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        seed=seed,
        gamma=GAMMA,
        alpha=ALPHA,
        alpha_min=ALPHA_MIN,
        alpha_decay=ALPHA_DECAY,
        epsilon=EPSILON,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        optimistic_init=OPTIMISTIC_INIT,
        use_double_q=USE_DOUBLE_Q,
    )

    rewards = []
    max_steps = _episode_max_steps(size)

    # First episode: seed reset for reproducible starting dynamics
    state = env.reset(seed=seed)

    best_ma10 = -1e9

    for ep in range(episodes):
        if ep > 0:
            state = env.reset()

        ep_return = 0.0
        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated = env.step(action)

            agent.update(state, action, reward, next_state, bool(terminated))
            state = next_state
            ep_return += reward

            if terminated:
                break

        rewards.append(ep_return)
        ma10 = _moving_avg(rewards, 10)
        best_ma10 = max(best_ma10, ma10)

        print(f"[Ep {ep+1:03d}] Return={ep_return:7.2f} | MA(10)={ma10:7.2f} | BestMA10={best_ma10:7.2f} | ε={agent.epsilon:5.3f}")

    # Training summary required by your spec
    avg = float(np.mean(rewards)) if rewards else 0.0
    print(f"\nALIAS=markham | Episodes={episodes} | Average Cumulative Reward={avg:.2f}")

    # Optional extra: greedy evaluation to showcase the learned policy's quality
    greedy_avg = _evaluate_greedy(env, agent, episodes=20, max_steps=max_steps)
    print(f"Greedy Evaluation over 20 episodes: Avg Cumulative Reward = {greedy_avg:.2f}")

    env.close()
    return avg, rewards


if __name__ == "__main__":
    train_and_eval()
