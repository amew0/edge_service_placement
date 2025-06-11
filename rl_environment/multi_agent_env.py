import gymnasium as gym
import numpy as np
from gymnasium import spaces
import networkx as nx
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import json
from typing import Dict, List, Tuple, Any

class MultiAgentEdgeEnv(MultiAgentEnv):
    """
    Multi-Agent Environment for Edge Service Placement
    Each edge site is an agent that makes placement decisions
    """
    def __init__(self, graph, service_chains, microservices, edge_sites, base_stations, max_steps=1000):
        super().__init__()
        self.G = graph
        self.service_chains = service_chains
        self.microservices = microservices
        self.edge_sites = edge_sites
        self.base_stations = base_stations
        self.B = len(base_stations)
        self.E = len(edge_sites)
        self.S = len(microservices)
        self.max_steps = max_steps
        
        # Create microservice name to index mapping
        self.ms_name_to_idx = {ms['name']: idx for idx, ms in enumerate(microservices)}
        
        # Convert service chains to use indices
        self.service_chains_idx = []
        for chain in service_chains:
            chain_idx = {
                'name': chain['name'],
                'microservices': [self.ms_name_to_idx[ms_name] for ms_name in chain['microservices']],
                'latency_requirement_ms': chain['latency_requirement_ms']
            }
            self.service_chains_idx.append(chain_idx)
        
        # Define action and observation spaces for each agent (edge site)
        self.action_space = spaces.Dict({
            f"edge_{i}": spaces.MultiDiscrete([3, self.S])  # [action_type, microservice_idx]
            for i in range(self.E)
        })
        
        # Observation space for each agent includes:
        # - Current microservices on the edge site
        # - Available resources
        # - Latency to base stations
        # - Service chain requirements
        obs_dim = self.S + 3 + self.B + len(self.service_chains_idx)
        self.observation_space = spaces.Dict({
            f"edge_{i}": spaces.Box(
                low=0,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
            for i in range(self.E)
        })
        
        # Initialize state
        self.server_microservices = np.zeros((self.E, self.S), dtype=int)
        self.step_count = 0
        self._compute_cost_matrix()
        
    def _compute_cost_matrix(self):
        """Compute cost matrix for service placement"""
        self.tau = np.full((self.S, self.E), fill_value=1e6)
        for si in range(self.S):
            for ei in range(self.E):
                self.tau[si, ei] = 1.0 + np.random.rand() * 0.1

    def _get_agent_observation(self, agent_id: int) -> np.ndarray:
        """Get observation for a specific agent (edge site)"""
        edge_idx = int(agent_id.split('_')[1])
        
        # Current microservices on this edge site
        current_ms = self.server_microservices[edge_idx]
        
        # Available resources (CPU, Memory, GPU)
        available_resources = np.array([
            self.microservices[edge_idx]['cpu'],
            self.microservices[edge_idx]['memory_mb'],
            1.0 if self.microservices[edge_idx]['gpu'] else 0.0
        ])
        
        # Latency to base stations
        latencies = np.array([
            nx.shortest_path_length(self.G, 
                                  source=self.edge_sites[edge_idx],
                                  target=bs,
                                  weight='latency',
                                  default=float('inf'))
            for bs in self.base_stations
        ])
        
        # Service chain requirements
        chain_requirements = np.array([
            chain['latency_requirement_ms']
            for chain in self.service_chains_idx
        ])
        
        return np.concatenate([
            current_ms,
            available_resources,
            latencies,
            chain_requirements
        ])

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.server_microservices.fill(0)
        self.step_count = 0
        
        # Return observations for all agents
        return {
            f"edge_{i}": self._get_agent_observation(f"edge_{i}")
            for i in range(self.E)
        }, {}

    def step(self, action_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], 
                                                               Dict[str, float], 
                                                               Dict[str, bool], 
                                                               Dict[str, bool], 
                                                               Dict[str, Any]]:
        """Step the environment"""
        self.step_count += 1
        rewards = {agent_id: 0.0 for agent_id in action_dict.keys()}
        terminated = {agent_id: False for agent_id in action_dict.keys()}
        truncated = {agent_id: False for agent_id in action_dict.keys()}
        infos = {agent_id: {} for agent_id in action_dict.keys()}
        
        # Process each agent's action
        for agent_id, action in action_dict.items():
            edge_idx = int(agent_id.split('_')[1])
            act_type, ms_idx = action
            
            # Validate action
            if not (0 <= ms_idx < self.S):
                rewards[agent_id] = -1
                truncated[agent_id] = True
                continue
                
            # Apply action
            if act_type == 1:  # deploy
                if self.server_microservices[edge_idx, ms_idx] == 0:
                    self.server_microservices[edge_idx, ms_idx] = 1
                    rewards[agent_id] += 1
                else:
                    rewards[agent_id] -= 1
            elif act_type == 2:  # evict
                if self.server_microservices[edge_idx, ms_idx] == 1:
                    self.server_microservices[edge_idx, ms_idx] = 0
                    rewards[agent_id] += 0.5
                else:
                    rewards[agent_id] -= 1
        
        # Calculate coverage reward
        coverage = self._service_latency()
        coverage_reward = coverage.sum() * 0.1
        
        # Add coverage reward to all agents
        for agent_id in action_dict.keys():
            rewards[agent_id] += coverage_reward
        
        # Check if episode is done
        if self.step_count >= self.max_steps:
            for agent_id in action_dict.keys():
                truncated[agent_id] = True
        
        # Get new observations
        observations = {
            f"edge_{i}": self._get_agent_observation(f"edge_{i}")
            for i in range(self.E)
        }
        
        return observations, rewards, terminated, truncated, infos

    def _service_latency(self):
        """Calculate service latency coverage"""
        coverage = np.zeros(self.B, dtype=int)
        for chain in self.service_chains_idx:
            sites = [np.where(self.server_microservices[:, ms_idx] > 0)[0] 
                    for ms_idx in chain['microservices']]
            if all(len(slist)>0 for slist in sites):
                for b_idx, bs in enumerate(self.base_stations):
                    total_lat = 0
                    for ms_idx, slist in zip(chain['microservices'], sites):
                        try:
                            edge_site = self.edge_sites[slist[0]]
                            dlat = nx.shortest_path_length(self.G, 
                                                         source=bs,
                                                         target=edge_site,
                                                         weight='latency')
                            total_lat += dlat
                        except nx.NetworkXNoPath:
                            total_lat += float('inf')
                    if total_lat <= chain['latency_requirement_ms']:
                        coverage[b_idx] += 1
        return coverage

# Example usage
if __name__ == "__main__":
    # Build sample graph
    G = nx.Graph()
    base_stations = ['b1', 'b2']
    edge_sites = ['e1', 'e2', 'e3']
    nodes = base_stations + edge_sites
    G.add_nodes_from(nodes)
    
    # Add latencies
    G.add_edge('b1','e1', latency=0.1)
    G.add_edge('b1','e2', latency=0.2)
    G.add_edge('b2','e2', latency=0.1)
    G.add_edge('b2','e3', latency=0.3)
    G.add_edge('e1','e2', latency=0.05)
    G.add_edge('e2','e3', latency=0.05)

    # Load services
    with open('data/services.json', 'r') as f:
        services = json.load(f)
    microservices = services['microservices']
    chains = services['service_chains']

    # Create environment
    env = MultiAgentEdgeEnv(G, chains, microservices, edge_sites, base_stations)
    
    # Example of using the environment
    obs, _ = env.reset()
    for _ in range(10):
        # Random actions for each agent
        actions = {
            f"edge_{i}": np.array([np.random.randint(3), np.random.randint(len(microservices))])
            for i in range(len(edge_sites))
        }
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"Rewards: {rewards}") 