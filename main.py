import numpy as np
import random
from exam_env import GridWorldEnv
from model import NStepBootQLearningAgent, NStepBootConfig

SEED = 123
SIZE = 10
EPISODES = 100
FPS = 0

def train_and_evaluate():
    np.random.seed(SEED)
    random.seed(SEED)

    env = GridWorldEnv(render_mode=None, size=SIZE, seed=SEED)

    cfg = NStepBootConfig(
        n=10,
        alpha_start=0.20, alpha_end=0.05, alpha_decay_episodes=80,
        gamma=0.995,
        epsilon_start=0.80, epsilon_end=0.02, epsilon_decay_episodes=60,
        optimistic_init=1.0
    )
    agent = NStepBootQLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        config=cfg
    )

    returns = []
    for ep in range(1, EPISODES + 1):
        G = agent.learn_episode(env, max_steps=1000, episode_idx=ep)
        assert isinstance(G, (int, float)), f"Episode {ep} returned {G!r} instead of a number"
        returns.append(G)

    avg_return = float(np.mean(returns))
    print(f"[nstep_bootstrap_q] Average Cumulative Reward over {len(returns)} episodes: {avg_return:.2f}")
    return avg_return, returns

if __name__ == '__main__':
    train_and_evaluate()
