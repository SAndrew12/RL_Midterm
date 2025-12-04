import numpy as np
import pickle
import csv
import os
from typing import Tuple, Optional, List
from collections import defaultdict
from datetime import datetime


class TrainingLogger:
    """
    CSV Logger for tracking Q-Learning training results.
    """

    def __init__(self, log_dir: str = "training_logs", filename: Optional[str] = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_results_{timestamp}.csv"

        self.filepath = os.path.join(log_dir, filename)
        self.episode_data: List[dict] = []
        self.moving_avg_window = 100

        self.headers = [
            'episode',
            'reward',
            'raw_reward',
            'episode_length',
            'epsilon',
            'q_table_size',
            'success',
            'cumulative_successes',
            'success_rate',
            'moving_avg_reward',
            'moving_avg_raw_reward',
            'distance_to_faf',
            'final_altitude',
            'timestamp'
        ]

        self._write_headers()
        print(f"Training logger initialized: {self.filepath}")

    def _write_headers(self):
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_episode(
            self,
            episode: int,
            reward: float,
            raw_reward: float,
            episode_length: int,
            epsilon: float,
            q_table_size: int,
            success: bool,
            cumulative_successes: int,
            distance_to_faf: float,
            final_altitude: float
    ):
        success_rate = (cumulative_successes / (episode + 1)) * 100

        self.episode_data.append({
            'reward': reward,
            'raw_reward': raw_reward
        })

        window = min(self.moving_avg_window, len(self.episode_data))
        recent_rewards = [d['reward'] for d in self.episode_data[-window:]]
        recent_raw_rewards = [d['raw_reward'] for d in self.episode_data[-window:]]
        moving_avg_reward = np.mean(recent_rewards)
        moving_avg_raw_reward = np.mean(recent_raw_rewards)

        row = [
            episode + 1,
            round(reward, 4),
            round(raw_reward, 4),
            episode_length,
            round(epsilon, 6),
            q_table_size,
            int(success),
            cumulative_successes,
            round(success_rate, 2),
            round(moving_avg_reward, 4),
            round(moving_avg_raw_reward, 4),
            round(distance_to_faf, 4),
            round(final_altitude, 2),
            datetime.now().isoformat()
        ]

        with open(self.filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def get_summary(self) -> dict:
        if not self.episode_data:
            return {}

        rewards = [d['reward'] for d in self.episode_data]
        raw_rewards = [d['raw_reward'] for d in self.episode_data]

        return {
            'total_episodes': len(self.episode_data),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'mean_raw_reward': np.mean(raw_rewards),
            'std_raw_reward': np.std(raw_rewards),
            'log_file': self.filepath
        }


class ImprovedQLearningAgent:
    """
    Improved Q-Learning Agent for ATC environment.
    """

    def __init__(
            self,
            n_actions: int = 9,
            learning_rate: float = 0.2,
            discount_factor: float = 0.95,
            epsilon: float = 1.0,
            epsilon_min: float = 0.05,
            epsilon_decay: float = 0.998,
            use_simplified_state: bool = True,
            state_bins: Optional[dict] = None
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_simplified_state = use_simplified_state

        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        if state_bins is None:
            if use_simplified_state:
                self.state_bins = self._simplified_state_bins()
            else:
                self.state_bins = self._default_state_bins()
        else:
            self.state_bins = state_bins

        self.training_episodes = 0
        self.total_steps = 0

    def _simplified_state_bins(self) -> dict:
        return {
            'dist_to_faf': np.array([0, 10, 20, 35, 50, 70]),
            'bearing_to_faf': np.array([-180, -90, -30, 30, 90, 180]),
            'altitude': np.array([0, 2000, 3500, 5000, 10000]),
            'heading_diff': np.array([-180, -60, -20, 20, 60, 180]),
            'vertical_speed': np.array([-2000, -500, 500, 2000]),
        }

    def _default_state_bins(self) -> dict:
        return {
            'x': np.linspace(-50, 50, 6),
            'y': np.linspace(-50, 50, 6),
            'altitude': np.linspace(0, 15000, 6),
            'heading': np.linspace(0, 360, 5),
            'speed': np.linspace(150, 250, 4),
            'dist_to_faf': np.linspace(0, 70, 6),
            'bearing_to_faf': np.linspace(-180, 180, 6),
            'heading_diff': np.linspace(-180, 180, 6),
            'vertical_speed': np.linspace(-2000, 2000, 4),
        }

    def discretize_state(self, observation: np.ndarray) -> Tuple[int, ...]:
        x = observation[0] * 50
        y = observation[1] * 50
        altitude = observation[2] * 15000
        heading = observation[3] * 360
        speed = observation[4] * 100 + 150
        dist_to_faf = observation[5] * 70
        bearing_to_faf = observation[6] * 360 - 180
        heading_diff = observation[7] * 180
        vertical_speed = observation[8] * 2000

        if self.use_simplified_state:
            state = (
                np.digitize(dist_to_faf, self.state_bins['dist_to_faf']),
                np.digitize(bearing_to_faf, self.state_bins['bearing_to_faf']),
                np.digitize(altitude, self.state_bins['altitude']),
                np.digitize(heading_diff, self.state_bins['heading_diff']),
                np.digitize(vertical_speed, self.state_bins['vertical_speed']),
            )
        else:
            state = (
                np.digitize(x, self.state_bins['x']),
                np.digitize(y, self.state_bins['y']),
                np.digitize(altitude, self.state_bins['altitude']),
                np.digitize(heading, self.state_bins['heading']),
                np.digitize(speed, self.state_bins['speed']),
                np.digitize(dist_to_faf, self.state_bins['dist_to_faf']),
                np.digitize(bearing_to_faf, self.state_bins['bearing_to_faf']),
                np.digitize(heading_diff, self.state_bins['heading_diff']),
                np.digitize(vertical_speed, self.state_bins['vertical_speed']),
            )

        return state

    def get_action(self, state: Tuple[int, ...], explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

    def update(
            self,
            state: Tuple[int, ...],
            action: int,
            reward: float,
            next_state: Tuple[int, ...],
            done: bool
    ):
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_next_q

        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)
        self.total_steps += 1

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_episodes += 1

    def save(self, filepath: str):
        data = {
            'q_table': dict(self.q_table),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'use_simplified_state': self.use_simplified_state,
            'state_bins': self.state_bins,
            'training_episodes': self.training_episodes,
            'total_steps': self.total_steps,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for state, q_values in data['q_table'].items():
            self.q_table[state] = q_values

        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.epsilon = data['epsilon']
        self.epsilon_min = data['epsilon_min']
        self.epsilon_decay = data['epsilon_decay']
        self.use_simplified_state = data['use_simplified_state']
        self.state_bins = data['state_bins']
        self.training_episodes = data['training_episodes']
        self.total_steps = data['total_steps']

        print(f"Agent loaded from {filepath}")
        print(f"Training episodes: {self.training_episodes}, Total steps: {self.total_steps}")
        print(f"Current epsilon: {self.epsilon:.4f}")
        print(f"Q-table size: {len(self.q_table)} states")
        print(f"Using simplified state: {self.use_simplified_state}")

    def get_stats(self) -> dict:
        return {
            'training_episodes': self.training_episodes,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'use_simplified_state': self.use_simplified_state,
        }


def shape_reward(
        raw_reward: float,
        info: dict,
        prev_dist: float,
        prev_alt_error: float
) -> float:
    shaped_reward = raw_reward

    current_dist = info['distance_to_faf']
    if prev_dist is not None:
        dist_improvement = prev_dist - current_dist
        shaped_reward += dist_improvement * 0.5

    target_altitude = 3000.0
    current_alt_error = abs(info['altitude'] - target_altitude)
    if prev_alt_error is not None:
        alt_improvement = prev_alt_error - current_alt_error
        shaped_reward += alt_improvement * 0.001

    if current_dist < 5.0 and current_alt_error < 500:
        shaped_reward += 1.0

    return shaped_reward


def export_training_summary(
        logger: TrainingLogger,
        agent: ImprovedQLearningAgent,
        output_path: Optional[str] = None
) -> str:
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(logger.log_dir, f"training_summary_{timestamp}.csv")

    summary = logger.get_summary()
    agent_stats = agent.get_stats()

    combined_stats = {
        **summary,
        **agent_stats,
        'learning_rate': agent.learning_rate,
        'discount_factor': agent.discount_factor,
        'epsilon_min': agent.epsilon_min,
        'epsilon_decay': agent.epsilon_decay
    }

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in combined_stats.items():
            writer.writerow([key, value])

    print(f"Training summary exported to: {output_path}")
    return output_path