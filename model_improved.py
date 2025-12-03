import numpy as np
import pickle
from typing import Tuple, Optional
from collections import defaultdict


class ImprovedQLearningAgent:
    """
    Improved Q-Learning Agent for ATC environment.

    Key improvements:
    1. Simplified state space (focus on most important features)
    2. Coarser discretization to speed up learning
    3. Better reward shaping option
    4. State aggregation for faster convergence
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
        """
        Initialize improved Q-Learning agent.

        Args:
            n_actions: Number of discrete actions
            learning_rate: Higher learning rate for faster updates
            discount_factor: Slightly lower to focus on near-term rewards
            epsilon: Initial exploration rate
            epsilon_min: Higher minimum to maintain exploration
            epsilon_decay: Slower decay to explore longer
            use_simplified_state: Use reduced state space (recommended)
            state_bins: Custom bin definitions
        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.use_simplified_state = use_simplified_state

        # Q-table
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        # State discretization bins
        if state_bins is None:
            if use_simplified_state:
                self.state_bins = self._simplified_state_bins()
            else:
                self.state_bins = self._default_state_bins()
        else:
            self.state_bins = state_bins

        # Statistics
        self.training_episodes = 0
        self.total_steps = 0

    def _simplified_state_bins(self) -> dict:
        """
        Simplified state space with fewer, more relevant features.

        Uses only 5 key features instead of 9:
        - Distance to FAF (most important)
        - Bearing to FAF (direction)
        - Altitude (height)
        - Heading difference (alignment)
        - Vertical speed (climb/descent rate)

        This reduces state space from ~10^8 to ~10^4 states!
        """
        return {
            'dist_to_faf': np.array([0, 10, 20, 35, 50, 70]),  # 5 bins: close, medium, far, very far
            'bearing_to_faf': np.array([-180, -90, -30, 30, 90, 180]),  # 5 bins: direction sectors
            'altitude': np.array([0, 2000, 3500, 5000, 10000]),  # 4 bins: low, target zone, high
            'heading_diff': np.array([-180, -60, -20, 20, 60, 180]),  # 5 bins: alignment zones
            'vertical_speed': np.array([-2000, -500, 500, 2000]),  # 3 bins: descending, level, climbing
        }

    def _default_state_bins(self) -> dict:
        """Coarser bins for full state space (still reduced from original)."""
        return {
            'x': np.linspace(-50, 50, 6),  # 5 bins (reduced from 10)
            'y': np.linspace(-50, 50, 6),  # 5 bins
            'altitude': np.linspace(0, 15000, 6),  # 5 bins
            'heading': np.linspace(0, 360, 5),  # 4 bins
            'speed': np.linspace(150, 250, 4),  # 3 bins
            'dist_to_faf': np.linspace(0, 70, 6),  # 5 bins
            'bearing_to_faf': np.linspace(-180, 180, 6),  # 5 bins
            'heading_diff': np.linspace(-180, 180, 6),  # 5 bins
            'vertical_speed': np.linspace(-2000, 2000, 4),  # 3 bins
        }

    def discretize_state(self, observation: np.ndarray) -> Tuple[int, ...]:
        """
        Convert continuous observation to discrete state.

        Args:
            observation: Normalized observation array [9 dimensions]

        Returns:
            Tuple of integers representing discretized state
        """
        # Denormalize observations
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
            # Use only 5 key features
            state = (
                np.digitize(dist_to_faf, self.state_bins['dist_to_faf']),
                np.digitize(bearing_to_faf, self.state_bins['bearing_to_faf']),
                np.digitize(altitude, self.state_bins['altitude']),
                np.digitize(heading_diff, self.state_bins['heading_diff']),
                np.digitize(vertical_speed, self.state_bins['vertical_speed']),
            )
        else:
            # Use full state space (with coarser bins)
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
        """Select action using epsilon-greedy policy."""
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        else:
            q_values = self.q_table[state]
            # Handle ties by random selection
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
        """Update Q-value using Q-learning."""
        current_q = self.q_table[state][action]

        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_next_q

        # Q-learning update
        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)
        self.total_steps += 1

    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.training_episodes += 1

    def save(self, filepath: str):
        """Save agent to file."""
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
        """Load agent from file."""
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
        """Get agent statistics."""
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
    """
    Add reward shaping to guide the agent.

    Adds intermediate rewards for:
    - Getting closer to FAF
    - Approaching target altitude
    - Aligning with runway heading

    Args:
        raw_reward: Original reward from environment
        info: Info dict from environment
        prev_dist: Previous distance to FAF
        prev_alt_error: Previous altitude error

    Returns:
        Shaped reward
    """
    shaped_reward = raw_reward

    # Distance-based shaping: reward for getting closer
    current_dist = info['distance_to_faf']
    if prev_dist is not None:
        dist_improvement = prev_dist - current_dist
        shaped_reward += dist_improvement * 0.5  # Scale factor

    # Altitude-based shaping: reward for approaching target altitude
    target_altitude = 3000.0
    current_alt_error = abs(info['altitude'] - target_altitude)
    if prev_alt_error is not None:
        alt_improvement = prev_alt_error - current_alt_error
        shaped_reward += alt_improvement * 0.001  # Scale factor (altitude in feet)

    # Bonus for being in "capture zone"
    if current_dist < 5.0 and current_alt_error < 500:
        shaped_reward += 1.0

    return shaped_reward