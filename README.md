# Edge Service Placement: Multi-Agent Reinforcement Learning

A comprehensive reinforcement learning framework for solving microservice placement problems in edge computing environments. This project implements multiple multi-agent approaches using both RLlib and Stable Baselines3 to optimize service deployment across edge sites while satisfying latency requirements and resource constraints.

## 🎯 Project Overview

This system addresses the challenge of placing microservices across distributed edge computing sites to minimize latency while maximizing service coverage. The problem is modeled as a multi-agent reinforcement learning task where each edge site acts as an autonomous agent making deployment decisions.

### Key Features

- **Multi-Agent RL**: Each edge site operates as an independent agent
- **Configurable State Sharing**: Full, partial, or no state sharing between agents
- **Multiple Observation Modes**: Centralized, local, or hybrid observation strategies  
- **Flexible Reward Structures**: Configurable balance between individual and shared rewards
- **Advanced Metrics**: Comprehensive tracking and visualization of training progress
- **Multiple Frameworks**: Support for both RLlib (Ray) and Stable Baselines3

## 🏗️ Architecture

### Environment Components

- **Edge Sites**: Physical locations where microservices can be deployed
- **Base Stations**: Service request origins with latency requirements
- **Microservices**: Individual service components to be placed
- **Service Chains**: Sequences of microservices that must work together
- **Network Graph**: Topology with latency weights between nodes

### Network Configuration

The project supports different network topologies depending on the use case:

#### 1. **RL Training Environment (Demo Setup)**
Used in all RL training scripts for simplified learning:
- **Base Stations**: 2 (b1, b2)
- **Edge Sites**: 3 (e1, e2, e3)
- **Ratio**: **0.67 base stations per edge site** (2:3 ratio)
- **Purpose**: Simplified topology for faster training and algorithm development

#### 2. **Physical Topology (Realistic Deployment)**
Defined in `topology.py` for real-world scenarios:
- **Base Stations**: 10 
- **Edge Sites**: 4 (E1-E4, located in NYC area)
- **Ratio**: **2.5 base stations per edge site** (10:4 ratio)
- **Purpose**: Realistic deployment with coverage redundancy and fault tolerance

#### Network Connectivity
- Each base station connects to User Plane Functions (UPFs)
- Edge sites connect to UPFs for service delivery
- Graph topology includes latency weights for path optimization
- Multiple connectivity paths ensure network resilience

### Agent Action Structure

Each agent (edge site) has a **MultiDiscrete action space** with 2 components:

#### Action Format: `[action_type, microservice_idx]`

**1. Action Type** (First Component)
- `0` = **HOLD/NO-OP** - Do nothing
- `1` = **DEPLOY** - Deploy a microservice  
- `2` = **EVICT** - Remove a microservice

**2. Microservice Index** (Second Component)
- Range: `0` to `S-1` (where S = number of microservices)
- Specifies which microservice to deploy/evict

#### Action Examples

For a system with 3 edge sites and 23 microservices:

| Agent | Action | Meaning |
|-------|--------|---------|
| `edge_0` | `[1, 5]` | Deploy microservice #5 on edge site 0 |
| `edge_1` | `[2, 12]` | Evict microservice #12 from edge site 1 |
| `edge_2` | `[0, 3]` | Do nothing (microservice index ignored) |

#### Reward Structure

- ✅ **Valid Deploy**: Microservice not already on edge → `+1.0` reward
- ❌ **Invalid Deploy**: Microservice already deployed → `-1.0` penalty
- ✅ **Valid Evict**: Microservice exists on edge → `+0.5` reward  
- ❌ **Invalid Evict**: No microservice to remove → `-1.0` penalty
- ❌ **Invalid Index**: Out of bounds → `-1.0` penalty + episode truncation
- 🤝 **Shared Reward**: Coverage bonus based on service chain satisfaction

### State Sharing Modes

#### 1. **Full State Sharing** (`state_sharing="full"`)
- All agents see complete global state
- Fully cooperative behavior
- Same rewards for all agents

#### 2. **Partial State Sharing** (`state_sharing="partial"`)  
- Agents see local state + limited global context
- Balanced individual/shared rewards
- Encourages coordination while maintaining autonomy

#### 3. **No State Sharing** (`state_sharing="none"`)
- Agents only see local information
- Individual rewards only
- Competitive/independent behavior

### Observation Modes

#### 1. **Centralized** (`observation_mode="centralized"`)
```
Observation = [
    All microservice placements (E×S),
    All edge resources (E×3), 
    All edge-to-base latencies (E×B),
    Service chain requirements
]
```

#### 2. **Local** (`observation_mode="local"`)
```
Observation = [
    Local microservices (S),
    Local resources (3),
    Local latencies (B),
    Service chain requirements
]
```

#### 3. **Hybrid** (`observation_mode="hybrid"`)
```
Observation = [
    Local microservices (S),
    Local resources (3), 
    Local latencies (B),
    Service chain requirements,
    Global coverage per base station (B),
    Global utilization per edge (E)
]
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install ray[rllib] stable-baselines3 gymnasium networkx numpy matplotlib
```

### Training with Stable Baselines3 (Recommended)

```bash
python rl_environment/train_multi_agent_sb3.py
```

This will:
- Train for 5 iterations with 10,000 timesteps each
- Save models, metrics, and visualizations
- Create detailed training progress plots
- Export configuration for reproducibility

### Training with RLlib

```bash
python rl_environment/train_edge_ppo.py
```

### Configuration Options

Edit the environment configuration in the training scripts:

```python
env_config = {
    'observation_mode': 'hybrid',    # 'centralized', 'local', 'hybrid'
    'state_sharing': 'partial',      # 'full', 'partial', 'none'  
    'reward_sharing': 0.7,           # 0.0 = individual, 1.0 = fully shared
}
```

## 📁 Project Structure

```
edge_service_placement/
├── rl_environment/
│   ├── train_multi_agent_sb3.py    # Enhanced SB3 implementation
│   ├── train_edge_ppo.py           # RLlib PPO implementation
│   └── env.py                      # Base environment
├── data/
│   └── services.json               # Microservice and chain definitions
├── algorithms/                     # Heuristic algorithms
├── training_output_*/              # Training results and logs
└── README.md
```

## 📊 Training Outputs

Each training run creates a timestamped directory containing:

- **Models**: `final_model.zip`, `model_iter_*.zip`
- **Metrics**: `training_metrics.json`, `training_progress.png`
- **Logs**: `monitor.csv`, `eval_monitor.csv`  
- **Configuration**: `env_config.json`
- **TensorBoard**: `tensorboard/` directory

### Visualizing Results

```bash
# View training progress
python -c "
import matplotlib.pyplot as plt
import json
with open('training_output_*/training_metrics.json') as f:
    data = json.load(f)
plt.plot(data['episode_rewards'])
plt.title('Training Progress')
plt.show()
"

# Launch TensorBoard
tensorboard --logdir training_output_*/tensorboard
```

## 🔬 Experimental Features

### Custom Metrics Tracking

The enhanced SB3 implementation includes:
- Episode-level performance metrics
- Resource utilization tracking  
- Coverage efficiency scoring
- Automatic plot generation

### Advanced Reward Engineering

- **Efficiency Bonus**: Rewards achieving coverage with fewer resources
- **Configurable Mixing**: Balance individual vs cooperative behavior
- **Dynamic Penalties**: Adaptive penalties for invalid actions

## 🎛️ Hyperparameter Tuning

Key parameters to experiment with:

```python
# PPO Configuration
learning_rate = 0.0003      # Learning rate
gamma = 0.99               # Discount factor  
gae_lambda = 0.95          # GAE parameter
clip_range = 0.2           # PPO clip range
ent_coef = 0.01           # Entropy bonus
vf_coef = 0.5             # Value function weight

# Environment Configuration  
reward_sharing = 0.7       # Cooperation level
max_steps = 1000          # Episode length
observation_mode = 'hybrid' # Information sharing
```

## 📈 Performance Metrics

The system tracks several key performance indicators:

- **Service Coverage**: Percentage of service chains satisfied
- **Resource Utilization**: Efficiency of microservice placement
- **Latency Compliance**: Meeting service chain latency requirements
- **Agent Coordination**: Degree of cooperative behavior
- **Training Stability**: Convergence and learning progress

## 🐛 Troubleshooting

### Common Issues

1. **Segmentation Faults with RLlib**
   - Use `num_rollout_workers=0` for single-threaded execution
   - Set environment variables: `OMP_NUM_THREADS=1`

2. **Memory Issues**
   - Reduce batch size and number of environments
   - Use smaller network architectures

3. **Slow Training**
   - Increase vectorized environments
   - Enable GPU acceleration if available

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1911.10635)
- [Edge Computing Service Placement](https://ieeexplore.ieee.org/document/8962182)

## 🏆 Acknowledgments

- Ray/RLlib team for the multi-agent RL framework
- Stable Baselines3 contributors for the robust RL implementations
- EdgeSim-py for edge computing simulation inspiration 