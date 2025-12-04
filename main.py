import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import os
from datetime import datetime

# Import environment
from env import ATCGymEnv

# Import models
from q_learn_model import ImprovedQLearningAgent, TrainingLogger, shape_reward
from ppo_model import ActorCritic
from dql_model import DoubleDQNAgent


# =============================================================================
# PPO Helper Functions
# =============================================================================

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation (GAE-Lambda)."""
    T = len(rewards)
    advantages = torch.zeros(T)
    next_value = 0.0
    gae = 0.0

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def ppo_update(model, optimizer, obs, actions, log_probs_old, returns, advantages,
               clip_ratio=0.2, vf_coef=0.5, ent_coef=0.01, train_epochs=10, batch_size=64):
    """Perform PPO policy and value function updates."""
    dataset_size = obs.shape[0]

    for _ in range(train_epochs):
        idx = np.arange(dataset_size)
        np.random.shuffle(idx)

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = idx[start:end]

            batch_obs = obs[batch_idx]
            batch_actions = actions[batch_idx]
            batch_old_log_probs = log_probs_old[batch_idx]
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]

            # Normalize advantages
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

            new_log_probs, entropy, values = model.evaluate_actions(batch_obs, batch_actions)
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values, batch_returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


# =============================================================================
# Training Functions
# =============================================================================

def train_q_learning(episodes=5000, max_steps=500, render=False,
                     save_path="models/q_learning_agent.pkl", log_dir="training_logs"):
    """Train Q-Learning agent on ATC environment."""
    print("\n" + "=" * 60)
    print("Training Q-Learning Agent")
    print("=" * 60)

    render_mode = "human" if render else None
    env = ATCGymEnv(render_mode=render_mode, continuous_actions=False, num_aircraft=1)

    agent = ImprovedQLearningAgent(
        n_actions=9,
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,
        use_simplified_state=True
    )

    logger = TrainingLogger(log_dir=log_dir, filename="q_learning_training.csv")

    cumulative_successes = 0
    all_rewards = []

    for episode in range(episodes):
        obs, info = env.reset()
        state = agent.discretize_state(obs)
        episode_reward = 0.0
        episode_raw_reward = 0.0
        prev_dist = None
        prev_alt_error = None

        for step in range(max_steps):
            action = agent.get_action(state, explore=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            shaped_reward = shape_reward(reward, info, prev_dist, prev_alt_error)
            prev_dist = info.get('distance_to_faf', 70.0)
            prev_alt_error = abs(info.get('altitude', 0) - 3000.0)

            next_state = agent.discretize_state(next_obs)
            agent.update(state, action, shaped_reward, next_state, done)

            episode_reward += shaped_reward
            episode_raw_reward += reward
            state = next_state

            if done:
                break

        success = info.get('distance_to_faf', 70.0) < 5.0
        if success:
            cumulative_successes += 1

        agent.decay_epsilon()
        all_rewards.append(episode_raw_reward)

        logger.log_episode(
            episode=episode,
            reward=episode_reward,
            raw_reward=episode_raw_reward,
            episode_length=step + 1,
            epsilon=agent.epsilon,
            q_table_size=len(agent.q_table),
            success=success,
            cumulative_successes=cumulative_successes,
            distance_to_faf=info.get('distance_to_faf', 70.0),
            final_altitude=info.get('altitude', 0)
        )

        if (episode + 1) % 100 == 0:
            success_rate = (cumulative_successes / (episode + 1)) * 100
            avg_reward = np.mean(all_rewards[-100:])
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Return: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Success Rate: {success_rate:.1f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    env.close()

    final_success_rate = (cumulative_successes / episodes) * 100
    final_avg_return = np.mean(all_rewards)

    print(f"\n{'=' * 60}")
    print(f"Q-Learning Training Complete!")
    print(f"{'=' * 60}")
    print(f"Average Return: {final_avg_return:.2f}")
    print(f"Success Rate: {final_success_rate:.1f}%")
    print(f"Fail Rate: {100 - final_success_rate:.1f}%")
    print(f"Model saved to: {save_path}")

    return agent, logger


def train_ppo(episodes=1000, render=False, save_path="models/ppo_agent.pt"):
    """Train PPO agent on ATC environment."""
    print("\n" + "=" * 60)
    print("Training PPO Agent")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    render_mode = "human" if render else None
    env = ATCGymEnv(render_mode=render_mode, continuous_actions=True, num_aircraft=1)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim, hidden_sizes=(128, 128)).to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)

    # Hyperparameters
    rollout_length = 2048
    gamma, lam = 0.99, 0.95
    clip_ratio, train_epochs, batch_size = 0.2, 10, 64

    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32)

    episode_return = 0.0
    episode_length = 0
    episode = 0
    all_returns = []

    # Calculate total timesteps based on episodes (approximate)
    max_steps_per_episode = 500
    total_timesteps = episodes * max_steps_per_episode
    timestep = 0

    while episode < episodes:
        obs_buf, actions_buf, log_probs_buf = [], [], []
        rewards_buf, values_buf, dones_buf = [], [], []

        for _ in range(rollout_length):
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)

            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor = model.act(obs_tensor)

            action = action_tensor.squeeze(0).cpu().numpy()
            action_clipped = np.clip(action, -1.0, 1.0)
            next_obs, reward, terminated, truncated, info = env.step(action_clipped)
            done = terminated or truncated

            obs_buf.append(obs.copy())
            actions_buf.append(action)
            log_probs_buf.append(log_prob_tensor.cpu().item())
            rewards_buf.append(reward)
            values_buf.append(value_tensor.cpu().item())
            dones_buf.append(float(done))

            episode_return += reward
            episode_length += 1
            timestep += 1

            obs = np.array(next_obs, dtype=np.float32)

            if done:
                obs, _ = env.reset()
                obs = np.array(obs, dtype=np.float32)
                all_returns.append(episode_return)

                if (episode + 1) % 50 == 0:
                    avg_return = np.mean(all_returns[-50:]) if len(all_returns) >= 50 else np.mean(all_returns)
                    print(f"Episode {episode + 1} | Return: {episode_return:.2f} | Avg(50): {avg_return:.2f}")

                episode_return = 0.0
                episode_length = 0
                episode += 1

                if episode >= episodes:
                    break

        if len(obs_buf) == 0:
            continue

        # Convert to tensors and update
        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(device)
        actions_t = torch.tensor(np.array(actions_buf), dtype=torch.float32).to(device)
        log_probs_t = torch.tensor(np.array(log_probs_buf), dtype=torch.float32).to(device)
        rewards_t = torch.tensor(np.array(rewards_buf), dtype=torch.float32).to(device)
        values_t = torch.tensor(np.array(values_buf), dtype=torch.float32).to(device)
        dones_t = torch.tensor(np.array(dones_buf), dtype=torch.float32).to(device)

        advantages, returns = compute_gae(rewards_t, values_t, dones_t, gamma=gamma, lam=lam)
        ppo_update(model, optimizer, obs_t, actions_t, log_probs_t, returns, advantages,
                   clip_ratio=clip_ratio, train_epochs=train_epochs, batch_size=batch_size)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    env.close()

    final_avg_return = np.mean(all_returns) if all_returns else 0

    print(f"\n{'=' * 60}")
    print(f"PPO Training Complete!")
    print(f"{'=' * 60}")
    print(f"Total Episodes: {len(all_returns)}")
    print(f"Average Return: {final_avg_return:.2f}")
    print(f"Model saved to: {save_path}")

    return model


def train_dqn(episodes=1000, max_steps=500, render=False, save_path="models/dqn_agent.pt"):
    """Train Double DQN agent on ATC environment."""
    print("\n" + "=" * 60)
    print("Training Double DQN Agent")
    print("=" * 60)

    render_mode = "human" if render else None
    env = ATCGymEnv(render_mode=render_mode, continuous_actions=False, num_aircraft=1)

    state_dim = env.observation_space.shape[0]

    agent = DoubleDQNAgent(
        state_dim=state_dim,
        action_dim=9,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=100000,
        batch_size=64,
        target_update_freq=100,
        hidden_dims=[256, 256]
    )

    episode_rewards = []

    for episode in range(episodes):
        obs, info = env.reset()
        state = np.array(obs, dtype=np.float32)
        episode_reward = 0.0
        losses = []

        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_obs, dtype=np.float32)

            agent.store_experience(state, action, reward, next_state, float(done))
            loss = agent.train()
            if loss > 0:
                losses.append(loss)

            episode_reward += reward
            state = next_state

            if done:
                break

        agent.decay_epsilon()
        agent.episode_count += 1
        episode_rewards.append(episode_reward)

        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_loss = np.mean(losses) if losses else 0
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Return: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    env.close()

    final_avg_return = np.mean(episode_rewards)

    print(f"\n{'=' * 60}")
    print(f"Double DQN Training Complete!")
    print(f"{'=' * 60}")
    print(f"Average Return: {final_avg_return:.2f}")
    print(f"Model saved to: {save_path}")

    return agent


# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_agent(algorithm, model_path, episodes=10, render=True):
    """Evaluate a trained agent."""
    print(f"\nEvaluating {algorithm.upper()} agent...")

    render_mode = "human" if render else None
    continuous = algorithm == 'ppo'
    env = ATCGymEnv(render_mode=render_mode, continuous_actions=continuous, num_aircraft=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if algorithm == 'q_learning':
        agent = ImprovedQLearningAgent(n_actions=9)
        agent.load(model_path)
        agent.epsilon = 0.0
    elif algorithm == 'ppo':
        obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        model = ActorCritic(obs_dim, act_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    elif algorithm == 'dqn':
        agent = DoubleDQNAgent(state_dim=env.observation_space.shape[0], action_dim=9)
        agent.load(model_path)
        agent.epsilon = 0.0

    total_rewards = []
    successes = 0

    for ep in range(episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            if algorithm == 'q_learning':
                state = agent.discretize_state(obs)
                action = agent.get_action(state, explore=False)
            elif algorithm == 'ppo':
                obs_tensor = torch.from_numpy(np.array(obs, dtype=np.float32)).to(device).unsqueeze(0)
                with torch.no_grad():
                    action_tensor, _, _ = model.act(obs_tensor)
                action = np.clip(action_tensor.squeeze(0).cpu().numpy(), -1.0, 1.0)
            else:
                action = agent.select_action(np.array(obs, dtype=np.float32), training=False)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)
        if info.get('distance_to_faf', 70.0) < 5.0:
            successes += 1
        print(f"Episode {ep + 1}: Return = {episode_reward:.2f}")

    env.close()

    print(f"\n{algorithm.upper()} Evaluation Results:")
    print(f"  Average Return: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")
    print(f"  Success Rate: {successes}/{episodes} ({100 * successes / episodes:.1f}%)")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified RL Training Script for Air Traffic Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_unified.py --algorithm q_learning --episodes 5000
  python main_unified.py --algorithm ppo --episodes 1000
  python main_unified.py --algorithm dqn --episodes 1000
  python main_unified.py --algorithm all
  python main_unified.py --evaluate --algorithm dqn --model-path models/dqn_agent.pt

Team Big Pie - Reinforcement Learning Final Project
        """
    )

    parser.add_argument('--algorithm', '-a', type=str,
                        choices=['q_learning', 'ppo', 'dqn', 'all'],
                        default='q_learning', help='Algorithm to train')
    parser.add_argument('--episodes', '-e', type=int, default=None,
                        help='Number of training episodes')
    parser.add_argument('--render', '-r', action='store_true',
                        help='Render environment during training')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate a trained model instead of training')
    parser.add_argument('--model-path', '-m', type=str, default=None,
                        help='Path to saved model (for evaluation)')
    parser.add_argument('--output-dir', '-o', type=str, default='models',
                        help='Output directory for saved models')

    args = parser.parse_args()

    # Set default episodes based on algorithm (matching original experiments)
    default_episodes = {
        'q_learning': 5000,
        'ppo': 1000,
        'dqn': 1000
    }

    if args.evaluate:
        if args.model_path is None:
            print("Error: --model-path required for evaluation")
            return
        evaluate_agent(args.algorithm, args.model_path, episodes=10, render=args.render)
        return

    print("\n" + "=" * 60)
    print("Air Traffic Control - Reinforcement Learning Training")
    print("Team Big Pie: Stephen, Dhwanil, Sayam")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.algorithm in ['q_learning', 'all']:
        eps = args.episodes or default_episodes['q_learning']
        train_q_learning(
            episodes=eps,
            render=args.render,
            save_path=os.path.join(args.output_dir, 'q_learning_agent.pkl')
        )

    if args.algorithm in ['ppo', 'all']:
        eps = args.episodes or default_episodes['ppo']
        train_ppo(
            episodes=eps,
            render=args.render,
            save_path=os.path.join(args.output_dir, 'ppo_agent.pt')
        )

    if args.algorithm in ['dqn', 'all']:
        eps = args.episodes or default_episodes['dqn']
        train_dqn(
            episodes=eps,
            render=args.render,
            save_path=os.path.join(args.output_dir, 'dqn_agent.pt')
        )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
