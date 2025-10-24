import os
import sys
import random
import numpy as np
from typing import Tuple, List

# Local import path
sys.path.insert(0, os.path.dirname(__file__) or ".")

# Environment contract:
#   reset(seed=None) -> observation
#   step(action) -> (observation, reward, terminated)
from exam_env import GridWorldEnv
from markham_model import QLearningAgent

# ---------------- required hyperparameters (DO NOT CHANGE) ----------------
SEED = 123
SIZE = 10
EPISODES = 100
FPS = 0  # 0 FPS means the environment will run as fast as possible, not rendering
# -------------------------------------------------------------------------

# --- Agent Hyperparameters ---
GAMMA = 0.99
OPTIMISTIC_INIT = 12.0  # Increased optimism to encourage deeper exploration initially
USE_DOUBLE_Q = True

# --- Decay Schedules (Episode-Based) ---
# Tweak 1: Slower decay for ALPHA, higher MIN
ALPHA = 0.8
ALPHA_MIN = 0.10  # Raised minimum learning rate to adapt to dynamic barriers
ALPHA_DECAY = 0.995  # Slower decay for better adaptation over 100 episodes

# Tweak 2: Faster decay for EPSILON, slightly higher MIN
EPSILON = 1.0
EPSILON_MIN = 0.05  # Slightly higher minimum exploration rate for dynamic environment
EPSILON_DECAY = 0.985  # Significantly faster decay to transition to exploitation quickly


def _episode_max_steps(size: int) -> int:
    """Sets a max steps limit to prevent infinite loops in poor policies."""
    # 3x the total number of states is a safe upper bound
    return size * size * 3


def _moving_avg(xs: List[float], k: int = 10) -> float:
    """Calculates the moving average over the last k elements."""
    if not xs:
        return 0.0
    k = min(k, len(xs))
    return float(np.mean(xs[-k:]))


def train_and_eval(seed: int = SEED, size: int = SIZE, episodes: int = EPISODES, render: bool = False,
                   fps: int = FPS) -> Tuple[float, List[float]]:
    """
    Main function to train the RL agent in the Dynamic GridWorld environment.
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Initialize environment, respecting the FPS setting
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

    # First episode: seed reset for deterministic starting dynamics
    state = env.reset(seed=seed)

    best_ma10 = -1e9

    for ep in range(episodes):
        if ep > 0:
            # Subsequent episodes reset without a specific seed, allowing for dynamic barriers
            state = env.reset()

        ep_return = 0.0
        for _ in range(max_steps):
            # 1. Select action
            action = agent.select_action(state)

            # 2. Take step
            next_state, reward, terminated = env.step(action)

            # 3. Update Q-table
            agent.update(state, action, reward, next_state, bool(terminated))
            state = next_state
            ep_return += reward

            # Update env.q for the built-in policy renderer overlay
            env.q = agent.q_values()

            if terminated:
                break

        # --- Episode End: Apply Decay and Log Metrics ---
        rewards.append(ep_return)

        # Apply epsilon decay (exploration decreases)
        agent.epsilon = max(agent.epsilon * EPSILON_DECAY, EPSILON_MIN)

        # Apply alpha decay (learning rate decreases)
        agent.alpha = max(agent.alpha * ALPHA_DECAY, ALPHA_MIN)

        ma10 = _moving_avg(rewards, 10)
        best_ma10 = max(best_ma10, ma10)

        print(
            f"[Ep {ep + 1:03d}] Return={ep_return:7.2f} | MA(10)={ma10:7.2f} | BestMA10={best_ma10:7.2f} | α={agent.alpha:.3f} | ε={agent.epsilon:.3f}")

    # Final summary line, calculating the requested average cumulative reward
    avg = float(np.mean(rewards)) if rewards else 0.0
    print(f"\n--- Training Complete ---")
    print(f"ALIAS=markham | Episodes={episodes} | Final Average Cumulative Reward={avg:.2f}")

    env.close()
    return avg, rewards


if __name__ == "__main__":
    train_and_eval()
