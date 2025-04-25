import gym
import numpy as np
from gym import spaces
import networkx as nx
from stable_baselines3 import PPO
import json

class RLSPEnv(gym.Env):
    """
    Custom Environment for Reinforcement Learning
    """
    metadata = {'render.modes': ['human']}

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
        # Define action and observation space
        # Example: 3 actions between specified microservice and edge server (deploy, evict, hold)
        self.action_space = spaces.MultiDiscrete([3,self.E, self.S])
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
        # Compute for each BS how many chains accessible within latency
        coverage = np.zeros(self.B, dtype=int)
        # for each chain, check if all its microservices are placed
        for ci, chain in enumerate(self.service_chains):
            # find sites for each microservice in chain
            sites = [np.where(self.server_microservices[:, si] > 0)[0] for si in chain]
            if all(len(slist)>0 for slist in sites):
                # choose nearest site per microservice and sum latencies
                for b_idx, bs in enumerate(self.base_stations):
                    total_lat = 0
                    for si, slist in zip(chain, sites):
                        dlat = min(nx.shortest_path_length(self.G, source=bs, target=e, weight='latency') for e in slist)
                        total_lat += dlat
                    if total_lat <= chain.latency:
                        coverage[b_idx] += 1
        return coverage

    def _calc_deploy_cost(self):
        # cost proportional to total microservices deployed
        return self.server_microservices.sum()

    def reset(self):
        self.server_microservices.fill(0)
        self.step_count = 0
        obs = np.zeros(self.B+self.E, dtype=int)
        return obs

    def step(self, action):
        self.step_count += 1
        act_type, ms_idx, site_idx = action
        reward = 0
        done = False
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
        # build next obs
        obs = np.concatenate([coverage, self.server_microservices.sum(axis=1)])
        if self.step_count >= self.max_steps:
            done = True
        return obs, reward, done, {}


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
    # chain defined as list of microservice indices with latency
    
    chains = services['service_chains']


    env = RLSPEnv(G, chains, microservices, edge_sites, base_stations)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_rlsp")
