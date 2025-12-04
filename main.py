# main.py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from env import ATCGymEnv
from model import ActorCritic


def make_env():
    # Use your custom environment
    env = ATCGymEnv(
        render_mode="human",
        continuous_actions=True,  # PPO here assumes continuous actions
        num_aircraft=1,           # can adjust as you like
    )
    return env


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute GAE-Lambda advantage estimates and returns.
    Args (all are 1D tensors of length T):
        rewards, values, dones (0/1)
    Returns:
        advantages, returns
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    returns = torch.zeros(T)

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


def ppo_update(
    model,
    optimizer,
    obs,
    actions,
    log_probs_old,
    returns,
    advantages,
    clip_ratio=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    train_epochs=10,
    batch_size=64,
):
    """
    Perform PPO updates over collected rollout data.
    """
    dataset_size = obs.shape[0]
    for _ in range(train_epochs):
        # Shuffle indices
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

            # Normalize advantages for stability
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                batch_advantages.std() + 1e-8
            )

            # Evaluate current policy
            new_log_probs, entropy, values = model.evaluate_actions(
                batch_obs, batch_actions
            )

            # Ratio for clipping
            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            # PPO objective
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value_loss = nn.functional.mse_loss(values, batch_returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim, hidden_sizes=(128, 128)).to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)

    # PPO hyperparameters
    total_timesteps = 200_000
    rollout_length = 2048
    gamma = 0.99
    lam = 0.95
    clip_ratio = 0.2
    train_epochs = 10
    batch_size = 64

    obs, _ = env.reset()
    obs = np.array(obs, dtype=np.float32)

    timestep = 0
    episode_return = 0.0
    episode_length = 0
    episode = 0

    while timestep < total_timesteps:
        # Collect rollout
        obs_buf = []
        actions_buf = []
        log_probs_buf = []
        rewards_buf = []
        values_buf = []
        dones_buf = []

        for _ in range(rollout_length):
            obs_tensor = torch.from_numpy(obs).float().to(device).unsqueeze(0)

            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor = model.act(obs_tensor)

            action = action_tensor.squeeze(0).cpu().numpy()
            # Clip actions to env bounds (env expects [-1, 1] for each dim)
            action_clipped = np.clip(action, -1.0, 1.0)

            next_obs, reward, terminated, truncated, info = env.step(action_clipped)
            done = terminated or truncated

            # Store to buffer
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
                print(
                    f"Episode {episode}  "
                    f"Return: {episode_return:.2f}  "
                    f"Length: {episode_length}"
                )
                episode_return = 0.0
                episode_length = 0
                episode += 1

            if timestep >= total_timesteps:
                break

        # Convert buffers to tensors
        obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(np.array(actions_buf), dtype=torch.float32).to(device)
        log_probs_tensor = torch.tensor(np.array(log_probs_buf), dtype=torch.float32).to(device)
        rewards_tensor = torch.tensor(np.array(rewards_buf), dtype=torch.float32).to(device)
        values_tensor = torch.tensor(np.array(values_buf), dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(np.array(dones_buf), dtype=torch.float32).to(device)

        # Compute advantages and returns
        advantages, returns = compute_gae(
            rewards_tensor, values_tensor, dones_tensor, gamma=gamma, lam=lam
        )

        # PPO update
        ppo_update(
            model,
            optimizer,
            obs_tensor,
            actions_tensor,
            log_probs_tensor,
            returns,
            advantages,
            clip_ratio=clip_ratio,
            vf_coef=0.5,
            ent_coef=0.01,
            train_epochs=train_epochs,
            batch_size=batch_size,
        )

        # Optional: save model periodically
        # torch.save(model.state_dict(), "ppo_atc.pt")

    env.close()
    # Final save
    torch.save(model.state_dict(), "ppo_atc_final.pt")
    print("Training finished, model saved as ppo_atc_final.pt")


if __name__ == "__main__":
    main()
