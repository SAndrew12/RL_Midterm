import numpy as np
import matplotlib.pyplot as plt
from exam_env import GridWorldEnv
from model import MonteCarloAgent

# Parameters
SEED = 123
SIZE = 10
EPISODES = 100
FPS = 0


def train_monte_carlo():
    """
    Train Monte Carlo agent on GridWorld environment
    """
    # Set random seed for reproducibility
    np.random.seed(SEED)

    # Initialize environment
    env = GridWorldEnv(render_mode=None, size=SIZE, seed=SEED, max_barriers=20)

    # Initialize Monte Carlo agent
    agent = MonteCarloAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        epsilon=1.0,  # Start with high exploration
        gamma=0.99,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    # Training loop
    print(f"Starting Monte Carlo training for {EPISODES} episodes...")
    print(f"Grid Size: {SIZE}x{SIZE}")
    print(f"Max Barriers: 20")
    print("-" * 60)

    for episode in range(EPISODES):
        # Reset environment
        state = env.reset(seed=SEED + episode)
        terminated = False

        # Run episode
        while not terminated:
            # Select action using epsilon-greedy policy
            action = agent.select_action(state)

            # Take action in environment
            next_state, reward, terminated = env.step(action)

            # Store transition
            agent.store_transition(state, action, reward)

            # Move to next state
            state = next_state

        # Update Q-values using Monte Carlo method after episode completion
        agent.update_q_values()

        # Update environment's Q-table for rendering
        env.q = agent.get_q_table()

        # Print progress
        if (episode + 1) % 10 == 0:
            avg_steps = np.mean(env.steps_per_episode[-10:]) if len(env.steps_per_episode) >= 10 else np.mean(
                env.steps_per_episode)
            avg_reward = np.mean(env.rewards[-10:]) if len(env.rewards) >= 10 else np.mean(env.rewards)
            print(f"Episode {episode + 1}/{EPISODES} | Epsilon: {agent.epsilon:.3f} | "
                  f"Avg Steps (last 10): {avg_steps:.2f} | Avg Reward (last 10): {avg_reward:.2f}")

    print("-" * 60)
    print("Training completed!")

    # Plot training metrics
    plot_training_metrics(env)

    # Test the trained agent with rendering
    print("\nTesting trained agent with visualization...")
    test_agent(env, agent)

    env.close()

    return agent, env


def plot_training_metrics(env):
    """
    Plot training metrics: steps per episode and rewards per episode
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot steps per episode
    ax1.plot(env.steps_per_episode, label='Steps per Episode', alpha=0.6)
    ax1.plot(env.average_steps_per_episode, label='Average Steps', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Steps')
    ax1.set_title('Steps per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot rewards per episode
    ax2.plot(env.rewards, label='Reward per Episode', alpha=0.6)
    # Calculate moving average for rewards
    window = 10
    if len(env.rewards) >= window:
        moving_avg = np.convolve(env.rewards, np.ones(window) / window, mode='valid')
        ax2.plot(range(window - 1, len(env.rewards)), moving_avg, label=f'{window}-Episode Moving Avg', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title('Rewards per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    print("\nTraining metrics saved as 'training_metrics.png'")
    plt.show()


def test_agent(env, agent, num_episodes=3):
    """
    Test the trained agent with rendering enabled
    """
    # Create new environment with rendering
    test_env = GridWorldEnv(render_mode="human", size=SIZE, seed=SEED, max_barriers=20)
    test_env.q = agent.get_q_table()

    # Copy the episode count to show proper barriers
    test_env.episode = env.episode

    for test_ep in range(num_episodes):
        state = test_env.reset(seed=SEED + EPISODES + test_ep)
        terminated = False
        steps = 0

        print(f"\nTest Episode {test_ep + 1}:")

        while not terminated and steps < 200:  # Max 200 steps to prevent infinite loops
            # Use greedy policy (no exploration)
            action = np.argmax(agent.q[state])
            next_state, reward, terminated = test_env.step(action)
            state = next_state
            steps += 1

        print(f"  Completed in {steps} steps with reward {test_env.cum_reward}")

    test_env.close()


if __name__ == "__main__":
    agent, env = train_monte_carlo()

    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Total Episodes: {EPISODES}")
    print(f"Average Steps (last 20 episodes): {np.mean(env.steps_per_episode[-20:]):.2f}")
    print(f"Average Reward (last 20 episodes): {np.mean(env.rewards[-20:]):.2f}")
    print(f"Best Episode Steps: {min(env.steps_per_episode)}")
    print(f"Best Episode Reward: {max(env.rewards):.2f}")