# Power Grid Management using Reinforcement Learning

## Project Overview
This project implements and compares two reinforcement learning approaches for power grid management:
- Actor-Critic (A2C)
- Proximal Policy Optimization (PPO)

The project uses the Grid2Op environment to simulate power grid operations and train agents to maintain reliable electricity flow while adhering to operational constraints.

## Features
- Implementation of A2C with improvements:
  - Observation space normalization
  - Curriculum learning
  - Graph Neural Networks (GNN)
  - Momentum-based optimization
- Implementation of PPO with improvements:
  - Multi-agent system
  - Hierarchical Reinforcement Learning
  - GNN feature extraction
  - Reduced action and observation spaces

## Key Results
- A2C with curriculum learning achieved stable convergence around 600 reward points
- GNN with momentum occasionally achieved rewards up to 800
- PPO implementation showed specialized capabilities through multi-agent system
- Both implementations demonstrated different strengths in managing complex power grid scenarios

## Environment Setup
This project uses the following environment:

# Environment: l2rpn_case14_sandbox
# Required Libraries:
- Grid2Op
- Stable-baselines3
- PyTorch
- Gymnasium (OpenAI Gym)

## Installation
1. Clone the repository
git clone https://github.com/TumiJourdan/RL-Ass-1/tree/main/Assignment

2. Install dependencies
pip install grid2op stable-baselines3 torch gymnasium

## Usage
The project includes multiple agent implementations that can be run independently:

### A2C Implementation
# Run base A2C agent
python run_a2c.py --mode base

# Run A2C with curriculum learning
python run_a2c.py --mode curriculum

# Run A2C with GNN
python run_a2c.py --mode gnn

### PPO Implementation
# Run base PPO agent
python run_ppo.py --mode base

# Run PPO with multi-agent system
python run_ppo.py --mode multi_agent

# Run PPO with hierarchical learning
python run_ppo.py --mode hierarchical

# Project 
The project is in the Assignment folder

## Training Details
- Training environment: l2rpn_case14_sandbox
- Reward structure combines L2RPN Reward and N1 Reward
- Observation space includes key metrics like rho, line status, topology vector, etc.
- Action space includes line operations, bus changes, and power dispatch controls

## Results and Metrics
- Training times vary by implementation:
  - Base A2C: 211 minutes
  - A2C with curriculum: 81 minutes
  - PPO implementations showed varying training times based on complexity
- Performance metrics focus on reward stability and convergence

## Contributors
- Tao Yuan (2332155)
- Shakeel Malagas (2424161)
- Tumi Jourdan (2180153)
- Dean Solomon (2347848)

## License
This project is part of a university assignment for the University of the Witwatersrand, Johannesburg.

## References
- [Grid2Op Documentation](https://grid2op.readthedocs.io/en/latest/)
- [Grid2Op GitHub Repository](https://github.com/rte-france/Grid2Op)
- Bengio, Y., et al. (2009). Curriculum learning.
- Gilmer, J., et al. (2017). Neural message passing for quantum chemistry.
