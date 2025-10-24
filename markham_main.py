import os
import sys
import random
import numpy as np
from typing import Tuple, List
sys.path.insert(0, os.path.dirname(__file__) or ".")
from exam_env import GridWorldEnv
from markham_model import QLearningAgent

SEED = 123
SIZE = 10
EPISODES = 100
FPS = 0
GAMMA = 0.99
OPTIMISTIC_INIT = 12.0
USE_DOUBLE_Q = True
ALPHA = 0.8
ALPHA_MIN = 0.10
ALPHA_DECAY = 0.995
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.985


def _episode_max_steps(size: int) -> int:
    return size * size * 3

def _moving_avg(xs: List[float], k: int = 10) -> float:
    if not xs:
        return 0.0
    k = min(k, len(xs))
    return float(np.mean(xs[-k:]))

def train_and_eval(seed: int = SEED, size: int = SIZE, episodes: int = EPISODES, render: bool = False,
                   fps: int = FPS) -> Tuple[float, List[float]]:
    random.seed(seed)
    np.random.seed(seed)
    env = GridWorldEnv(render_mode=("human" if render and fps > 0 else None), size=size, seed=seed, max_barriers=20)
    env.metadata["render_fps"] = fps

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        seed=seed,
        gamma=GAMMA,
        alpha=ALPHA,
        epsilon=EPSILON,
        optimistic_init=OPTIMISTIC_INIT,
        use_double_q=USE_DOUBLE_Q,
    )

    rewards: List[float] = []
    max_steps = _episode_max_steps(size)
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
            env.q = agent.q_values()

            if terminated:
                break
        rewards.append(ep_return)
        agent.epsilon = max(agent.epsilon * EPSILON_DECAY, EPSILON_MIN)
        agent.alpha = max(agent.alpha * ALPHA_DECAY, ALPHA_MIN)

        ma10 = _moving_avg(rewards, 10)
        best_ma10 = max(best_ma10, ma10)

        print(
            f"[Ep {ep + 1:03d}] Return={ep_return:7.2f} | MA(10)={ma10:7.2f} | BestMA10={best_ma10:7.2f} | α={agent.alpha:.3f} | ε={agent.epsilon:.3f}")

    avg = float(np.mean(rewards)) if rewards else 0.0
    print(f"\n--- Training Complete ---")
    print(f"ALIAS=markham | Episodes={episodes} | Average Cumulative Reward={avg:.2f}")

    env.close()
    return avg, rewards

if __name__ == "__main__":
    train_and_eval()
