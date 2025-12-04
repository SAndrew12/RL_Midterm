"""
Main Training Script for Double DQN on ATC Environment

This script contains:
1. Training loop with experience collection and agent updates
2. Evaluation function for measuring performance
3. Visualization of training metrics
4. Command-line interface for training and testing
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import sys
import os
from datetime import datetime

# Import environment
sys.path.append('/mnt/user-data/uploads')
from env import ATCGymEnv

# Import our Double DQN agent
from model import DoubleDQNAgent


def train_double_dqn(
    env: ATCGymEnv,
    agent: DoubleDQNAgent,
    num_episodes: int = 1000,
    max_steps_per_episode: int = 500,
    eval_freq: int = 50,
    eval_episodes: int = 10,
    save_freq: int = 100,
    save_dir: str = '/home/claude',
    verbose: bool = True
):
    """
    Train Double DQN agent on ATC environment.
    
    Args:
        env: ATC Gym environment
        agent: Double DQN agent
        num_episodes: Total number of training episodes
        max_steps_per_episode: Maximum steps per episode
        eval_freq: Evaluate every N episodes
        eval_episodes: Number of episodes for evaluation
        save_freq: Save model every N episodes
        save_dir: Directory to save models and results
        verbose: Print detailed progress
        
    Returns:
        Dictionary containing training metrics
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    eval_rewards = []
    eval_episodes_list = []
    
    # Success metrics
    successful_episodes = []
    faf_capture_rate = []
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Total Episodes: {num_episodes}")
    print(f"Max Steps per Episode: {max_steps_per_episode}")
    print(f"Evaluation Frequency: Every {eval_freq} episodes")
    print(f"Save Frequency: Every {save_freq} episodes")
    print("=" * 70 + "\n")
    
    # Training loop with progress bar
    progress_bar = tqdm(range(num_episodes), desc="Training Progress")
    
    for episode in progress_bar:
        # Reset environment
        state, info = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        faf_captured = False
        
        # Episode loop
        for step in range(max_steps_per_episode):
            # Select action using epsilon-greedy
            action = agent.select_action(state, training=True)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Check if FAF was captured
            if info.get('faf_captured', False):
                faf_captured = True
            
            # Store experience in replay buffer
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train agent (if enough samples in buffer)
            loss = agent.train()
            episode_loss += loss
            
            # Update metrics
            episode_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Decay exploration rate
        agent.decay_epsilon()
        agent.episode_count = episode + 1
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        avg_loss = episode_loss / steps if steps > 0 else 0
        episode_losses.append(avg_loss)
        successful_episodes.append(1 if faf_captured else 0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'Reward': f'{episode_reward:.1f}',
            'Steps': steps,
            'ε': f'{agent.epsilon:.3f}',
            'Loss': f'{avg_loss:.4f}',
            'FAF': '✓' if faf_captured else '✗'
        })
        
        # Periodic evaluation
        if (episode + 1) % eval_freq == 0:
            eval_reward, eval_success_rate = evaluate_agent(env, agent, eval_episodes, verbose=False)
            eval_rewards.append(eval_reward)
            eval_episodes_list.append(episode + 1)
            
            # Compute success rate over last eval_freq episodes
            recent_success_rate = np.mean(successful_episodes[-eval_freq:]) * 100
            faf_capture_rate.append(recent_success_rate)
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"EVALUATION at Episode {episode + 1}")
                print(f"{'='*70}")
                print(f"Average Eval Reward: {eval_reward:.2f}")
                print(f"Eval Success Rate: {eval_success_rate:.1f}%")
                print(f"Training Success Rate (last {eval_freq} eps): {recent_success_rate:.1f}%")
                print(f"Current Epsilon: {agent.epsilon:.4f}")
                print(f"Buffer Size: {len(agent.replay_buffer)}")
                print(f"{'='*70}\n")
        
        # Save model checkpoint
        if (episode + 1) % save_freq == 0:
            save_path = os.path.join(save_dir, f'double_dqn_ep{episode + 1}.pth')
            agent.save(save_path)
    
    # Save final model
    final_save_path = os.path.join(save_dir, 'double_dqn_final.pth')
    agent.save(final_save_path)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Final Episode: {num_episodes}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print(f"Buffer Size: {len(agent.replay_buffer)}")
    print(f"Overall Success Rate: {np.mean(successful_episodes) * 100:.1f}%")
    print("=" * 70 + "\n")
    
    # Return all metrics
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_losses': episode_losses,
        'eval_rewards': eval_rewards,
        'eval_episodes': eval_episodes_list,
        'successful_episodes': successful_episodes,
        'faf_capture_rate': faf_capture_rate
    }


def evaluate_agent(
    env: ATCGymEnv,
    agent: DoubleDQNAgent,
    num_episodes: int = 10,
    render: bool = False,
    verbose: bool = True
) -> tuple:
    """
    Evaluate trained agent (greedy policy, no exploration).
    
    Args:
        env: ATC environment
        agent: Trained Double DQN agent
        num_episodes: Number of evaluation episodes
        render: Whether to render episodes
        verbose: Print episode details
        
    Returns:
        Tuple of (average_reward, success_rate)
    """
    total_rewards = []
    total_steps = []
    successful_landings = 0
    
    for ep in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Greedy action selection (no exploration)
            action = agent.select_action(state, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            if render:
                env.render()
        
        # Check if FAF was captured
        if info.get('faf_captured', False):
            successful_landings += 1
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        if verbose:
            status = "✓ SUCCESS" if info.get('faf_captured', False) else "✗ FAILED"
            print(f"Episode {ep + 1}: Reward = {episode_reward:.2f}, Steps = {steps}, {status}")
    
    avg_reward = np.mean(total_rewards)
    success_rate = (successful_landings / num_episodes) * 100
    
    if verbose:
        print(f"\nEvaluation Summary:")
        print(f"  Average Reward: {avg_reward:.2f} ± {np.std(total_rewards):.2f}")
        print(f"  Average Steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
        print(f"  Success Rate: {success_rate:.1f}% ({successful_landings}/{num_episodes})")
    
    return avg_reward, success_rate


def plot_training_results(metrics: dict, save_path: str = '/home/claude/training_results.png'):
    """
    Plot comprehensive training metrics.
    
    Args:
        metrics: Dictionary containing training metrics
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Double DQN Training Results - ATC Environment', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    episodes = range(len(metrics['episode_rewards']))
    ax.plot(episodes, metrics['episode_rewards'], alpha=0.4, color='blue', label='Episode Reward')
    
    # Moving average
    window = 50
    if len(metrics['episode_rewards']) >= window:
        moving_avg = np.convolve(metrics['episode_rewards'], np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(metrics['episode_rewards'])), moving_avg, 
                color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Episode Lengths
    ax = axes[0, 1]
    ax.plot(episodes, metrics['episode_lengths'], alpha=0.4, color='green')
    if len(metrics['episode_lengths']) >= window:
        moving_avg = np.convolve(metrics['episode_lengths'], np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(metrics['episode_lengths'])), moving_avg, 
                color='darkgreen', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Lengths')
    ax.grid(True, alpha=0.3)
    
    # 3. Training Loss
    ax = axes[0, 2]
    ax.plot(episodes, metrics['episode_losses'], alpha=0.4, color='orange')
    if len(metrics['episode_losses']) >= window:
        moving_avg = np.convolve(metrics['episode_losses'], np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(metrics['episode_losses'])), moving_avg, 
                color='darkorange', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss (TD Error)')
    ax.set_title('Training Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 4. Evaluation Rewards
    ax = axes[1, 0]
    if metrics['eval_rewards']:
        ax.plot(metrics['eval_episodes'], metrics['eval_rewards'], 
                marker='o', color='purple', linewidth=2, markersize=6)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Evaluation Reward')
        ax.set_title('Evaluation Performance')
        ax.grid(True, alpha=0.3)
    
    # 5. Success Rate
    ax = axes[1, 1]
    # Compute cumulative success rate
    cumulative_success = np.cumsum(metrics['successful_episodes']) / (np.arange(len(metrics['successful_episodes'])) + 1) * 100
    ax.plot(episodes, cumulative_success, color='teal', linewidth=2, label='Cumulative')
    
    # Plot rolling success rate
    if metrics['faf_capture_rate']:
        ax.plot(metrics['eval_episodes'], metrics['faf_capture_rate'], 
                marker='s', color='darkred', linewidth=2, markersize=6, label='Rolling (50 eps)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('FAF Capture Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Reward Distribution (last 100 episodes)
    ax = axes[1, 2]
    last_n = min(100, len(metrics['episode_rewards']))
    ax.hist(metrics['episode_rewards'][-last_n:], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(metrics['episode_rewards'][-last_n:]), color='red', 
               linestyle='--', linewidth=2, label=f'Mean: {np.mean(metrics["episode_rewards"][-last_n:]):.1f}')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Reward Distribution (Last {last_n} Episodes)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training plots saved to {save_path}")
    plt.close()


def test_agent(
    env: ATCGymEnv,
    agent: DoubleDQNAgent,
    num_episodes: int = 5,
    render: bool = True,
    save_video: bool = False
):
    """
    Test trained agent and optionally render/save episodes.
    
    Args:
        env: ATC environment
        agent: Trained agent
        num_episodes: Number of test episodes
        render: Whether to render
        save_video: Whether to save video (requires render_mode='rgb_array')
    """
    print("\n" + "=" * 70)
    print("TESTING AGENT")
    print("=" * 70)
    
    for ep in range(num_episodes):
        print(f"\nTest Episode {ep + 1}/{num_episodes}")
        print("-" * 70)
        
        state, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Get Q-values for visualization
            q_values = agent.get_q_values(state)
            action = agent.select_action(state, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Print step info
            if steps % 50 == 0:
                print(f"  Step {steps}: Action={action}, Reward={reward:.2f}, "
                      f"Distance to FAF={info.get('distance_to_faf', 0):.1f} NM")
            
            if render:
                env.render()
        
        # Episode summary
        status = "✓ SUCCESS" if info.get('faf_captured', False) else "✗ FAILED"
        print(f"\nEpisode {ep + 1} Complete: {status}")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Final Distance to FAF: {info.get('distance_to_faf', 0):.2f} NM")
        print(f"  Final Altitude: {info.get('altitude', 0):.0f} ft")
        print("-" * 70)
    
    print("\n" + "=" * 70)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Double DQN for ATC Environment')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'eval'],
                        help='Mode: train, test, or eval')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer capacity')
    parser.add_argument('--target-update', type=int, default=100, help='Target network update frequency')
    parser.add_argument('--eval-freq', type=int, default=50, help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=100, help='Model save frequency')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load model from')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Directory to save results')
    parser.add_argument('--render', action='store_true', help='Render environment during testing')
    parser.add_argument('--difficulty', type=str, default='medium', choices=['easy', 'medium', 'hard'],
                        help='Environment difficulty')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    
    # Create environment (discrete actions for Double DQN)
    render_mode = 'human' if args.render and args.mode == 'test' else None
    env = ATCGymEnv(
        render_mode=render_mode,
        continuous_actions=False,  # Use discrete actions for DQN
        num_aircraft=1,
        max_steps=args.max_steps,
        difficulty=args.difficulty
    )
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"\nEnvironment Info:")
    print(f"  State Dimension: {state_dim}")
    print(f"  Action Dimension: {action_dim}")
    print(f"  Difficulty: {args.difficulty}")
    
    # Create agent
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_size,
        target_update_freq=args.target_update,
        hidden_dims=[256, 256]
    )
    
    # Load model if specified
    if args.load_model:
        agent.load(args.load_model)
    
    # Execute based on mode
    if args.mode == 'train':
        # Train agent
        metrics = train_double_dqn(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            eval_freq=args.eval_freq,
            eval_episodes=10,
            save_freq=args.save_freq,
            save_dir=args.save_dir
        )
        
        # Plot results
        plot_path = os.path.join(args.save_dir, 'training_results.png')
        plot_training_results(metrics, save_path=plot_path)
        
    elif args.mode == 'test':
        # Test agent with visualization
        if args.load_model is None:
            print("Warning: No model loaded. Using untrained agent.")
        test_agent(env, agent, num_episodes=5, render=args.render)
        
    elif args.mode == 'eval':
        # Evaluate agent
        if args.load_model is None:
            print("Warning: No model loaded. Using untrained agent.")
        avg_reward, success_rate = evaluate_agent(env, agent, num_episodes=100, verbose=True)
        print(f"\nFinal Evaluation Results:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Success Rate: {success_rate:.1f}%")
    
    env.close()
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
