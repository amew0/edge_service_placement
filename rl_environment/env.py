import gymnasium as gym
import numpy as np
from gymnasium import spaces
import networkx as nx
from stable_baselines3 import PPO
import json

class RLSPEnv(gym.Env):
    """
    Custom Environment for Reinforcement Learning
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, graph, service_chains, microservices, edge_sites, base_stations, max_steps=1000):
        super(RLSPEnv, self).__init__()
        self.G = graph
        self.service_chains = service_chains
        self.microservices = microservices
        self.edge_sites = edge_sites
        self.base_stations = base_stations
        self.B = len(base_stations)
        self.E = len(edge_sites)
        self.S = len(microservices)
        
        # Create microservice name to index mapping
        self.ms_name_to_idx = {ms['name']: idx for idx, ms in enumerate(microservices)}
        
        # Convert service chains to use indices instead of names
        self.service_chains_idx = []
        for chain in service_chains:
            chain_idx = {
                'name': chain['name'],
                'microservices': [self.ms_name_to_idx[ms_name] for ms_name in chain['microservices']],
                'latency_requirement_ms': chain['latency_requirement_ms']
            }
            self.service_chains_idx.append(chain_idx)
        
        # Define action and observation space
        # Action space: [action_type (0=hold, 1=deploy, 2=evict), edge_site_idx, microservice_idx]
        self.action_space = spaces.MultiDiscrete([3, self.E, self.S])
        # Observation space: number of chains per BS + number of microservices per site
        obs_high = np.array([len(service_chains)] * self.B + [sum(m['cpu'] for m in microservices)]* self.E)
        self.observation_space = spaces.Box(low=0, high=obs_high, shape=(self.B+self.E,), dtype=int)

        self.server_microservices = np.zeros((self.E, self.S), dtype=int)
        self.step_count = 0
        self.max_steps = max_steps
        self._compute_cost_matrix()
        self.reset()

    def _compute_cost_matrix(self):

        self.tau = np.full((self.S, self.E), fill_value=1e6)

        # For simplicity, use uniform compute cost + random communication cost
        for si in range(self.S):
            for ei in range(self.E):
                self.tau[si, ei] = 1.0 + np.random.rand() * 0.1
                # Add any other relevant costs here 
    def _service_latency(self):
        coverage = np.zeros(self.B, dtype=int)
        for chain in self.service_chains_idx:
            sites = [np.where(self.server_microservices[:, ms_idx] > 0)[0] for ms_idx in chain['microservices']]
            if all(len(slist)>0 for slist in sites):
                for b_idx, bs in enumerate(self.base_stations):
                    total_lat = 0
                    for ms_idx, slist in zip(chain['microservices'], sites):
                        try:
                            # Convert edge site index to edge site name
                            edge_site = self.edge_sites[slist[0]]
                            dlat = nx.shortest_path_length(self.G, source=bs, target=edge_site, weight='latency')
                            total_lat += dlat
                        except nx.NetworkXNoPath:
                            # If no path exists, use a large latency
                            total_lat += float('inf')
                    if total_lat <= chain['latency_requirement_ms']:
                        coverage[b_idx] += 1
        return coverage

    def _calc_deploy_cost(self):
        # cost proportional to total microservices deployed
        return self.server_microservices.sum()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.server_microservices.fill(0)
        self.step_count = 0
        obs = np.zeros(self.B+self.E, dtype=int)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        act_type, site_idx, ms_idx = action  # Note: changed order to match action space definition
        reward = 0
        terminated = False
        truncated = False
        
        # Validate indices
        if not (0 <= site_idx < self.E and 0 <= ms_idx < self.S):
            reward = -1
            truncated = True
            return self._get_obs(), reward, terminated, truncated, {"error": "Invalid indices"}
        
        # apply action validity
        if act_type == 1:  # deploy
            if self.server_microservices[site_idx, ms_idx] == 0:
                self.server_microservices[site_idx, ms_idx] = 1
                reward += 1  # small reward for valid deploy
            else:
                reward -= 1
        elif act_type == 2:  # evict
            if self.server_microservices[site_idx, ms_idx] == 1:
                self.server_microservices[site_idx, ms_idx] = 0
                reward += 0.5
            else:
                reward -= 1
        # hold gives no change
        
        # compute coverage reward
        coverage = self._service_latency()
        reward += coverage.sum() * 0.1
        
        # subtract cost
        cost = self._calc_deploy_cost()
        reward -= cost * 0.01
        
        if self.step_count >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """Helper method to get current observation"""
        coverage = self._service_latency()
        return np.concatenate([coverage, self.server_microservices.sum(axis=1)])

# Example instantiation and training
if __name__ == "__main__":
    # Build sample graph
    G = nx.Graph()
    base_stations = ['b1', 'b2']
    edge_sites = ['e1', 'e2', 'e3']
    nodes = base_stations + edge_sites
    G.add_nodes_from(nodes)
    # add latencies
    G.add_edge('b1','e1', latency=0.1)
    G.add_edge('b1','e2', latency=0.2)
    G.add_edge('b2','e2', latency=0.1)
    G.add_edge('b2','e3', latency=0.3)
    G.add_edge('e1','e2', latency=0.05)
    G.add_edge('e2','e3', latency=0.05)

    # dummy microservices and chains
    with open('data/services.json', 'r') as f:
        services = json.load(f)
    microservices = services['microservices']
    chains = services['service_chains']

    env = RLSPEnv(G, chains, microservices, edge_sites, base_stations)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_rlsp")
