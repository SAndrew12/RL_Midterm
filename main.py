import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Add the directory containing env.py to the path
sys.path.insert(0, '/mnt/user-data/uploads')

from env import ATCGymEnv
from model_improved import ImprovedQLearningAgent, shape_reward


def train_agent(
        num_episodes: int = 5000,
        max_steps: int = 500,
        render: bool = False,
        save_interval: int = 500,
        checkpoint_dir: str = "checkpoints_improved",
        use_reward_shaping: bool = True,
        use_simplified_state: bool = True
):
    """
    Train improved Q-Learning agent on ATC environment.

    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        render: Whether to render environment
        save_interval: Save checkpoint every N episodes
        checkpoint_dir: Directory to save checkpoints
        use_reward_shaping: Whether to use reward shaping
        use_simplified_state: Whether to use simplified 5D state space
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize environment
    render_mode = "human" if render else None
    env = ATCGymEnv(
        render_mode=render_mode,
        continuous_actions=False,
        num_aircraft=1,
        max_steps=max_steps,
        timestep=5.0,
        random_spawn=True,
        difficulty="easy"
    )

    # Initialize improved agent
    agent = ImprovedQLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,
        use_simplified_state=use_simplified_state
    )

    # Training metrics
    episode_rewards = []
    episode_raw_rewards = []  # Track raw rewards separately
    episode_lengths = []
    success_count = 0
    moving_avg_window = 100

    # Best model tracking
    best_avg_reward = -float('inf')

    print("=" * 70)
    print(f"Starting IMPROVED Q-Learning Training on ATC Environment")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Action space: {env.action_space.n} discrete actions")
    print(f"Simplified state space: {use_simplified_state}")
    print(f"Reward shaping: {use_reward_shaping}")
    print(f"State space size estimate: ~{estimate_state_space(agent)}")
    print("=" * 70)
    print("\nKey Improvements:")
    print("  ✓ Simplified state space (9D → 5D)")
    print("  ✓ Coarser discretization for faster learning")
    print("  ✓ Optional reward shaping for better guidance")
    print("  ✓ Higher learning rate (0.2)")
    print("  ✓ Slower epsilon decay (0.998)")
    print("=" * 70)

    # Training loop
    for episode in range(num_episodes):
        observation, info = env.reset()
        state = agent.discretize_state(observation)

        episode_reward = 0
        episode_raw_reward = 0
        step = 0
        done = False

        # For reward shaping
        prev_dist = info['distance_to_faf']
        prev_alt_error = abs(info['altitude'] - 3000.0)

        while not done and step < max_steps:
            # Select action
            action = agent.get_action(state, explore=True)

            # Take action
            next_observation, raw_reward, terminated, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_observation)
            done = terminated or truncated

            # Apply reward shaping if enabled
            if use_reward_shaping:
                shaped_reward = shape_reward(raw_reward, info, prev_dist, prev_alt_error)
                training_reward = shaped_reward
            else:
                training_reward = raw_reward

            # Update Q-table
            agent.update(state, action, training_reward, next_state, done)

            # Update state and metrics
            state = next_state
            episode_reward += training_reward
            episode_raw_reward += raw_reward
            step += 1

            # Update previous values for reward shaping
            prev_dist = info['distance_to_faf']
            prev_alt_error = abs(info['altitude'] - 3000.0)

            if render:
                env.render()

        # Decay epsilon
        agent.decay_epsilon()

        # Track metrics
        episode_rewards.append(episode_reward)
        episode_raw_rewards.append(episode_raw_reward)
        episode_lengths.append(step)

        # Check success
        if info.get('faf_captured', False):
            success_count += 1

        # Print progress
        if (episode + 1) % 50 == 0:
            recent_window = min(moving_avg_window, episode + 1)
            avg_reward = np.mean(episode_rewards[-recent_window:])
            avg_raw_reward = np.mean(episode_raw_rewards[-recent_window:])
            avg_length = np.mean(episode_lengths[-recent_window:])
            success_rate = success_count / (episode + 1) * 100

            print(f"Ep {episode + 1:4d}/{num_episodes} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Raw: {episode_raw_reward:7.2f} | "
                  f"Avg: {avg_reward:7.2f} | "
                  f"AvgRaw: {avg_raw_reward:7.2f} | "
                  f"Len: {avg_length:5.1f} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"ε: {agent.epsilon:.4f} | "
                  f"Q-size: {len(agent.q_table)}")

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_path = os.path.join(checkpoint_dir, "q_learning_best.pkl")
                agent.save(best_path)

        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"q_learning_checkpoint_ep{episode + 1}.pkl"
            )
            agent.save(checkpoint_path)

    # Save final model
    final_path = os.path.join(checkpoint_dir, "q_learning_final.pkl")
    agent.save(final_path)

    env.close()

    # Plot training curves
    plot_training_results(
        episode_rewards,
        episode_raw_rewards,
        episode_lengths,
        success_count,
        num_episodes,
        use_reward_shaping
    )

    return agent, episode_rewards, episode_lengths


def estimate_state_space(agent: ImprovedQLearningAgent) -> str:
    """Estimate the size of the state space."""
    if agent.use_simplified_state:
        sizes = [len(bins) - 1 for bins in agent.state_bins.values()]
        total = np.prod(sizes)
        return f"{int(total):,} states (simplified)"
    else:
        sizes = [len(bins) - 1 for bins in agent.state_bins.values()]
        total = np.prod(sizes)
        return f"{int(total):,} states (full)"


def test_agent(
        checkpoint_path: str,
        num_episodes: int = 10,
        render: bool = True,
        max_steps: int = 500
):
    """Test trained Q-Learning agent."""
    render_mode = "human" if render else None
    env = ATCGymEnv(
        render_mode=render_mode,
        continuous_actions=False,
        num_aircraft=1,
        max_steps=max_steps,
        timestep=5.0,
        random_spawn=True,
        difficulty="easy"
    )

    # Load trained agent
    agent = ImprovedQLearningAgent(n_actions=env.action_space.n)
    agent.load(checkpoint_path)
    agent.epsilon = 0.0  # Pure exploitation

    print("=" * 70)
    print(f"Testing Improved Q-Learning Agent")
    print("=" * 70)
    print(f"Test episodes: {num_episodes}")
    print(f"Using simplified state: {agent.use_simplified_state}")
    print("=" * 70)

    test_rewards = []
    test_lengths = []
    successes = 0

    for episode in range(num_episodes):
        observation, info = env.reset()
        state = agent.discretize_state(observation)

        episode_reward = 0
        step = 0
        done = False

        print(f"\n--- Episode {episode + 1} ---")
        print(f"Initial distance to FAF: {info['distance_to_faf']:.2f} NM")
        print(f"Initial altitude: {int(info['altitude'])} ft")

        while not done and step < max_steps:
            action = agent.get_action(state, explore=False)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = agent.discretize_state(next_observation)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            step += 1

            if render:
                env.render()

            # Print progress every 100 steps
            if step % 100 == 0:
                print(f"  Step {step}: Dist={info['distance_to_faf']:.2f} NM, "
                      f"Alt={int(info['altitude'])} ft")

        test_rewards.append(episode_reward)
        test_lengths.append(step)

        if info.get('faf_captured', False):
            successes += 1
            print(f"✓ SUCCESS! FAF captured in {step} steps")
        else:
            print(f"✗ Failed - Episode ended")

        print(f"Total reward: {episode_reward:.2f}")
        print(f"Final distance to FAF: {info['distance_to_faf']:.2f} NM")
        print(f"Final altitude: {int(info['altitude'])} ft (target: 3000 ft)")

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Average Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"Average Steps: {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")
    print(f"Success Rate: {successes}/{num_episodes} ({successes / num_episodes * 100:.1f}%)")
    print(f"Best Reward: {max(test_rewards):.2f}")
    print(f"Worst Reward: {min(test_rewards):.2f}")
    print("=" * 70)

    env.close()

    return test_rewards, test_lengths, successes


def plot_training_results(
        episode_rewards: list,
        episode_raw_rewards: list,
        episode_lengths: list,
        success_count: int,
        num_episodes: int,
        use_reward_shaping: bool
):
    """Plot training results with comparison of shaped vs raw rewards."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    window = 100

    # Plot 1: Rewards comparison
    axes[0, 0].plot(episode_rewards, alpha=0.2, label='Shaped Reward' if use_reward_shaping else 'Reward', color='blue')
    if use_reward_shaping:
        axes[0, 0].plot(episode_raw_rewards, alpha=0.2, label='Raw Reward', color='orange')

    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
        axes[0, 0].plot(range(window - 1, len(episode_rewards)), moving_avg,
                        'b-', linewidth=2, label=f'Shaped MA-{window}')

        if use_reward_shaping:
            raw_moving_avg = np.convolve(episode_raw_rewards, np.ones(window) / window, mode='valid')
            axes[0, 0].plot(range(window - 1, len(episode_raw_rewards)), raw_moving_avg,
                            'r-', linewidth=2, label=f'Raw MA-{window}')

    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Training Rewards Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Episode Lengths
    axes[0, 1].plot(episode_lengths, alpha=0.3, label='Episode Length', color='green')

    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window) / window, mode='valid')
        axes[0, 1].plot(range(window - 1, len(episode_lengths)), moving_avg,
                        'darkgreen', linewidth=2, label=f'{window}-Episode MA')

    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Length Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Reward Distribution (Raw)
    axes[1, 0].hist(episode_raw_rewards, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[1, 0].axvline(np.mean(episode_raw_rewards), color='r', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(episode_raw_rewards):.2f}')
    axes[1, 0].set_xlabel('Total Raw Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Raw Reward Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Success Rate
    success_rate = success_count / num_episodes * 100
    colors = ['green' if success_count > 0 else 'orange', 'red']
    axes[1, 1].bar(['Success', 'Failure'],
                   [success_count, num_episodes - success_count],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Number of Episodes')
    axes[1, 1].set_title(f'Success Rate: {success_rate:.2f}%')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Add text annotation
    for i, (label, value) in enumerate([('Success', success_count),
                                        ('Failure', num_episodes - success_count)]):
        axes[1, 1].text(i, value, f'{value}\n({value / num_episodes * 100:.1f}%)',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"improved_training_results_{timestamp}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nTraining plot saved to: {filename}")

    plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Improved Q-Learning for ATC Environment')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Mode: train or test')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true',
                        help='Render environment')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_improved/q_learning_best.pkl',
                        help='Checkpoint path for testing')
    parser.add_argument('--save-interval', type=int, default=500,
                        help='Save checkpoint every N episodes')
    parser.add_argument('--no-reward-shaping', action='store_true',
                        help='Disable reward shaping')
    parser.add_argument('--full-state', action='store_true',
                        help='Use full 9D state space instead of simplified 5D')

    args = parser.parse_args()

    if args.mode == 'train':
        print("\n🚁 Starting IMPROVED Q-Learning Training\n")
        train_agent(
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            render=args.render,
            save_interval=args.save_interval,
            use_reward_shaping=not args.no_reward_shaping,
            use_simplified_state=not args.full_state
        )
    else:
        print("\n🚁 Testing Improved Q-Learning Agent\n")
        test_agent(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            render=args.render,
            max_steps=args.max_steps
        )


if __name__ == "__main__":
    main()