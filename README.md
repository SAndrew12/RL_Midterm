# Reinforcement Learning for Autonomous Air Traffic Control

**Team Big Pie: Stephen, Dhwanil, Sayam**  
**Guided by Prof. Tyler Wallet**  
George Washington University

## Overview

This project implements and compares three reinforcement learning algorithms for autonomous air traffic control guidance. The goal is to train agents capable of guiding aircraft to a runway while managing altitude and glideslope alignment.

| Algorithm | Avg Return | Success Rate | Episodes |
|-----------|------------|--------------|----------|
| Q-Learning | -71 | 3% | 5,000 |
| PPO | -48.02 | — | 1,000 |
| Double DQN | -161 | — | 1,000 |

## Installation

### Requirements

```bash
pip install numpy torch gymnasium
```

### File Structure

```
project/
├── env.py                 # Shared ATC simulation environment
├── q_learn_model.py       # Q-Learning implementation
├── ppo_model.py           # PPO Actor-Critic network  
├── dql_model.py           # Double DQN implementation
├── main_unified.py        # Unified training script (NEW)
├── models/                # Saved model checkpoints
└── training_logs/         # CSV training logs
```

## Quick Start

### Training

Train a single algorithm:

```bash
# Q-Learning (5000 episodes - matches original experiment)
python main_unified.py --algorithm q_learning --episodes 5000

# PPO (1000 episodes)
python main_unified.py --algorithm ppo --episodes 1000

# Double DQN (1000 episodes)
python main_unified.py --algorithm dqn --episodes 1000
```

Train all algorithms sequentially:

```bash
python main_unified.py --algorithm all
```

### Command Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--algorithm` | `-a` | Algorithm: `q_learning`, `ppo`, `dqn`, `all` | `q_learning` |
| `--episodes` | `-e` | Number of training episodes | 5000/1000/1000 |
| `--render` | `-r` | Render environment visually | False |
| `--evaluate` | | Evaluate trained model | False |
| `--model-path` | `-m` | Model path for evaluation | None |
| `--output-dir` | `-o` | Directory for saved models | `models` |

### Evaluation

Evaluate a trained model:

```bash
python main_unified.py --evaluate --algorithm q_learning --model-path models/q_learning_agent.pkl
python main_unified.py --evaluate --algorithm ppo --model-path models/ppo_agent.pt --render
python main_unified.py --evaluate --algorithm dqn --model-path models/dqn_agent.pt
```

## Algorithms

### Q-Learning (Tabular)

- **Action Space**: Discrete (9 actions)
- **State**: Discretized into bins for distance, bearing, altitude, heading error, vertical speed
- **Key Features**: 
  - Reward shaping for dense feedback
  - Epsilon-greedy exploration (1.0 → 0.05)
  - Success/failure rate tracking

**Hyperparameters:**
- Learning rate: 0.2
- Discount factor: 0.95
- Epsilon decay: 0.998

### PPO (Proximal Policy Optimization)

- **Action Space**: Continuous (3 dimensions)
- **Network**: Actor-Critic with 128-128 hidden layers
- **Key Features**:
  - Generalized Advantage Estimation (GAE)
  - Clipped surrogate objective
  - Entropy bonus for exploration

**Hyperparameters:**
- Learning rate: 3e-4
- Gamma: 0.99, Lambda: 0.95
- Clip ratio: 0.2

### Double DQN

- **Action Space**: Discrete (9 actions)
- **Network**: 256-256 hidden layers with LayerNorm
- **Key Features**:
  - Experience replay buffer (100K capacity)
  - Target network (updated every 100 steps)
  - Decoupled action selection/evaluation

**Hyperparameters:**
- Learning rate: 1e-4
- Gamma: 0.99
- Epsilon decay: 0.995

## Environment

The `ATCGymEnv` simulates aircraft approach guidance:

**State Space:**
- Aircraft position (x, y)
- Velocity
- Altitude
- Glideslope deviation
- Distance from runway

**Action Space:**
- Discrete: 9 combinations of heading/altitude changes
- Continuous: Direct heading and altitude rate control

**Rewards:**
- Positive: Progress toward runway, glideslope alignment
- Negative: Danger zones, hard turns, altitude errors, crashes

## Output Files

After training, models are saved to the `models/` directory:

| File | Description |
|------|-------------|
| `q_learning_agent.pkl` | Pickled Q-table and agent state |
| `ppo_agent.pt` | PyTorch model state dict |
| `dqn_agent.pt` | PyTorch checkpoint with optimizer state |

Training logs are saved as CSV files in `training_logs/`.

## Results Summary

Based on our experiments:

- **PPO achieved the best average return** (-48.02), suggesting continuous control is well-suited to this domain
- **Q-Learning achieved 3% success rate** over 5,000 episodes, demonstrating the challenge of this problem
- **Double DQN had the lowest return** (-161), likely due to insufficient training (1,000 vs 5,000 episodes)

## Future Improvements

1. **Extended training**: Run all algorithms for equivalent durations (5,000+ episodes)
2. **Curriculum learning**: Start with easier scenarios
3. **Enhanced reward shaping**: More informative intermediate rewards
4. **Multi-aircraft scenarios**: Extend to handle multiple aircraft

## Citation

If you use this code, please cite:

```
Team Big Pie (2024). Reinforcement Learning for Autonomous Air Traffic Control.
George Washington University, Reinforcement Learning Course.
Guided by Prof. Tyler Wallet.
```

## License

This project was developed for educational purposes as part of a Reinforcement Learning course.
