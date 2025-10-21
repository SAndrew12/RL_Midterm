import numpy as np
import matplotlib.pyplot as plt
from exam_env import GridWorldEnv
from model import MonteCarloAgent

SEED = 123
SIZE = 10
EPISODES = 100
FPS = 0


def train_monte_carlo():
    np.random.seed(SEED)

    env = GridWorldEnv(render_mode=None, size=SIZE, seed=SEED, max_barriers=20)
    agent = MonteCarloAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        epsilon=1.0,
        gamma=0.99,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    print(f"training {EPISODES} episodes, grid={SIZE}x{SIZE}, barriers=20")

    for episode in range(EPISODES):
        state = env.reset(seed=SEED + episode)
        terminated = False

        while not terminated:
            action = agent.select_action(state)
            next_state, reward, terminated = env.step(action)
            agent.store_transition(state, action, reward)
            state = next_state

        agent.update_q_values()
        env.q = agent.get_q_table()

        if (episode + 1) % 10 == 0:
            recent_steps = env.steps_per_episode[-10:] if len(env.steps_per_episode) >= 10 else env.steps_per_episode
            recent_rewards = env.rewards[-10:] if len(env.rewards) >= 10 else env.rewards
            print(
                f"ep {episode + 1}: eps={agent.epsilon:.3f}, steps={np.mean(recent_steps):.1f}, reward={np.mean(recent_rewards):.1f}")

    plot_results(env)
    test_agent(env, agent)
    env.close()

    return agent, env


def plot_results(env):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(env.steps_per_episode, alpha=0.5)
    ax1.plot(env.average_steps_per_episode, linewidth=2)
    ax1.set_xlabel('episode')
    ax1.set_ylabel('steps')
    ax1.grid(alpha=0.3)

    ax2.plot(env.rewards, alpha=0.5)
    if len(env.rewards) >= 10:
        ma = np.convolve(env.rewards, np.ones(10) / 10, mode='valid')
        ax2.plot(range(9, len(env.rewards)), ma, linewidth=2)
    ax2.set_xlabel('episode')
    ax2.set_ylabel('reward')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()


def test_agent(env, agent, n=3):
    test_env = GridWorldEnv(render_mode="human", size=SIZE, seed=SEED, max_barriers=20)
    test_env.q = agent.get_q_table()
    test_env.episode = env.episode

    for i in range(n):
        state = test_env.reset(seed=SEED + EPISODES + i)
        done = False
        steps = 0

        while not done and steps < 200:
            action = np.argmax(agent.q[state])
            state, reward, done = test_env.step(action)
            steps += 1

        print(f"test {i + 1}: {steps} steps, reward={test_env.cum_reward}")

    test_env.close()


if __name__ == "__main__":
    agent, env = train_monte_carlo()

    last20_steps = np.mean(env.steps_per_episode[-20:])
    last20_reward = np.mean(env.rewards[-20:])
    print(f"\nfinal stats: avg_steps={last20_steps:.1f}, avg_reward={last20_reward:.1f}")
    print(f"best: {min(env.steps_per_episode)} steps, {max(env.rewards):.1f} reward")