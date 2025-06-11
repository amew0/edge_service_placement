#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-edge micro-service placement with RLlib PPO (new API stack)
-----------------------------------------------------------------
* Keeps the Dict observation space ("edge_0", "edge_1", "edge_2"…)
* Uses a shared policy for all edge agents
"""

import os
# Set environment variables to avoid threading conflicts
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from typing import Dict, Any, Tuple, List

import gymnasium as gym
import networkx as nx
import numpy as np
import ray
from ray.tune.registry import register_env

# ------------- RLlib imports -----------------
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn as nn

class EdgePlacementModel(TorchModelV2, nn.Module):
    """Custom model for edge placement that handles flat vector observations."""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Determine observation dimension
        if isinstance(obs_space, gym.spaces.Dict):
            obs_dim = next(iter(obs_space.spaces.values())).shape[0]
        else:
            obs_dim = obs_space.shape[0]
        
        # Create shared network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Linear(256, num_outputs)
        
        # Value function head
        self.value_head = nn.Linear(256, 1)
        
        # Last value function output
        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        # Get observations for all edges
        obs = input_dict["obs"]
        
        # Process through shared network
        features = self.shared_net(obs)
        
        # Get policy logits
        logits = self.policy_head(features)
        
        # Store value function output
        self._cur_value = self.value_head(features).squeeze(1)
        
        return logits, state

    def value_function(self):
        return self._cur_value

class MultiAgentEdgeEnv(MultiAgentEnv):
    """
    Multi-Agent Environment for Edge Service Placement.
    Each edge site is an agent that deploys / evicts micro-services.
    """

    def __init__(
        self,
        graph,
        service_chains,
        microservices,
        edge_sites,
        base_stations,
        max_steps: int = 1_000,
    ):
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

        # map micro-service names → indices
        self.ms_name_to_idx = {
            ms["name"]: idx for idx, ms in enumerate(microservices)
        }

        # service chains using indices (faster later on)
        self.service_chains_idx: List[Dict[str, Any]] = []
        for chain in service_chains:
            self.service_chains_idx.append(
                {
                    "name": chain["name"],
                    "microservices": [
                        self.ms_name_to_idx[ms_name]
                        for ms_name in chain["microservices"]
                    ],
                    "latency_requirement_ms": chain["latency_requirement_ms"],
                }
            )

        # ---------- spaces ----------
        self.action_space = gym.spaces.Dict(
            {
                f"edge_{i}": gym.spaces.MultiDiscrete([3, self.S])
                for i in range(self.E)
            }
        )

        obs_dim = self.S + 3 + self.B + len(self.service_chains_idx)
        self.observation_space = gym.spaces.Dict(
            {
                f"edge_{i}": gym.spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(obs_dim,),
                    dtype=np.float32,
                )
                for i in range(self.E)
            }
        )

        # RLlib requires a set of agent IDs supported by the environment
        self._agent_ids = set(self.action_space.spaces.keys())

        # ---------- state ----------
        self.server_microservices = np.zeros((self.E, self.S), dtype=int)
        self.step_count = 0
        self._compute_cost_matrix()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _compute_cost_matrix(self):
        """Pre-compute (dummy) placement costs for each (service, edge)."""
        self.tau = 1.0 + 0.1 * np.random.rand(self.S, self.E)

    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        edge_idx = int(agent_id.split("_")[1])

        # services deployed on this edge site (0/1 flags)
        current_ms = self.server_microservices[edge_idx]

        # dummy resource availability (cpu, mem, gpu flag)
        avail = np.array(
            [
                self.microservices[edge_idx]["cpu"],
                self.microservices[edge_idx]["memory_mb"],
                1.0 if self.microservices[edge_idx]["gpu"] else 0.0,
            ],
            dtype=np.float32,
        )

        # latency from this edge site to every base station
        latencies = np.array(
            [
                self._get_shortest_path_length(
                    self.edge_sites[edge_idx],
                    bs
                )
                for bs in self.base_stations
            ],
            dtype=np.float32,
        )

        chain_requirements = np.array(
            [c["latency_requirement_ms"] for c in self.service_chains_idx],
            dtype=np.float32,
        )

        return np.concatenate(
            [current_ms.astype(np.float32), avail, latencies, chain_requirements]
        )

    def _get_shortest_path_length(self, source: str, target: str) -> float:
        """Get shortest path length between two nodes, returning inf if no path exists."""
        try:
            return nx.shortest_path_length(
                self.G,
                source=source,
                target=target,
                weight="latency"
            )
        except nx.NetworkXNoPath:
            return float("inf")

    def _service_latency(self) -> np.ndarray:
        """Return an array (#base_stations,) with service-latency coverage."""
        coverage = np.zeros(self.B, dtype=int)

        for chain in self.service_chains_idx:
            sites_per_ms = [
                np.where(self.server_microservices[:, ms] > 0)[0]
                for ms in chain["microservices"]
            ]
            if not all(len(s) > 0 for s in sites_per_ms):
                continue

            for b_idx, bs in enumerate(self.base_stations):
                total_lat = 0.0
                for ms_idx, slist in zip(
                    chain["microservices"], sites_per_ms
                ):
                    edge_site = self.edge_sites[slist[0]]
                    try:
                        total_lat += nx.shortest_path_length(
                            self.G,
                            source=bs,
                            target=edge_site,
                            weight="latency",
                        )
                    except nx.NetworkXNoPath:
                        total_lat = float("inf")
                        break
                if total_lat <= chain["latency_requirement_ms"]:
                    coverage[b_idx] += 1
        return coverage

    # ------------------------------------------------------------------
    # RLlib interface
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.server_microservices.fill(0)
        self.step_count = 0
        return (
            {
                f"edge_{i}": self._get_agent_observation(f"edge_{i}")
                for i in range(self.E)
            },
            {},
        )

    def step(
        self, action_dict: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        self.step_count += 1

        rewards = {aid: 0.0 for aid in action_dict}
        terminated = {aid: False for aid in action_dict}
        truncated = {aid: False for aid in action_dict}
        infos = {aid: {} for aid in action_dict}

        for agent_id, action in action_dict.items():
            edge_idx = int(agent_id.split("_")[1])
            act_type, ms_idx = map(int, action)

            if not (0 <= ms_idx < self.S):
                rewards[agent_id] = -1.0
                truncated[agent_id] = True
                continue

            if act_type == 1:  # deploy
                if self.server_microservices[edge_idx, ms_idx] == 0:
                    self.server_microservices[edge_idx, ms_idx] = 1
                    rewards[agent_id] += 1.0
                else:
                    rewards[agent_id] -= 1.0
            elif act_type == 2:  # evict
                if self.server_microservices[edge_idx, ms_idx] == 1:
                    self.server_microservices[edge_idx, ms_idx] = 0
                    rewards[agent_id] += 0.5
                else:
                    rewards[agent_id] -= 1.0

        # shared coverage reward
        coverage_reward = 0.1 * self._service_latency().sum()
        for aid in rewards:
            rewards[aid] += coverage_reward

        if self.step_count >= self.max_steps:
            for aid in truncated:
                truncated[aid] = True

        # RLlib requires '__all__' key for multi-agent environments
        terminated['__all__'] = all(terminated.values())
        truncated['__all__'] = all(truncated.values())

        observations = {
            f"edge_{i}": self._get_agent_observation(f"edge_{i}")
            for i in range(self.E)
        }
        return observations, rewards, terminated, truncated, infos


# --------------------------------------------------------------------------
# 2.  Build graph & load services
# --------------------------------------------------------------------------
def build_demo_graph():
    G = nx.Graph()
    base_stations = ["b1", "b2"]
    edge_sites = ["e1", "e2", "e3"]
    nodes = base_stations + edge_sites
    G.add_nodes_from(nodes)
    # sample latencies
    G.add_edge("b1", "e1", latency=0.1)
    G.add_edge("b1", "e2", latency=0.2)
    G.add_edge("b2", "e2", latency=0.1)
    G.add_edge("b2", "e3", latency=0.3)
    G.add_edge("e1", "e2", latency=0.05)
    G.add_edge("e2", "e3", latency=0.05)
    return G, base_stations, edge_sites


def load_services(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return data["microservices"], data["service_chains"]


# --------------------------------------------------------------------------
# 3.  RLlib set-up: policies, PPO config
# --------------------------------------------------------------------------
def main():
    ray.init(ignore_reinit_error=True)

    # Register custom model
    ModelCatalog.register_custom_model("edge_placement_model", EdgePlacementModel)

    # ---- environment factory ------------------------------------------------
    G, BS, ES = build_demo_graph()
    microservices, chains = load_services("data/services.json")

    def env_creator(env_config):
        return MultiAgentEdgeEnv(G, chains, microservices, ES, BS)

    # Register the environment
    register_env("edge_placement", env_creator)

    # sample env to grab spaces
    sample_env = env_creator({})

    # ---- multi-agent wiring --------------------------------------------------
    box_obs_space = sample_env.observation_space["edge_0"]
    box_act_space = sample_env.action_space["edge_0"]

    policies = {
        "shared_policy": (
            None,  # use default RLModule (TorchPolicy)
            box_obs_space,
            box_act_space,
            {},
        )
    }

    def policy_mapping_fn(agent_id, *_, **__):
        return "shared_policy"

    # ---- PPO config ----------------------------------------------------------
    cfg = (
        PPOConfig()
        .environment("edge_placement")  # Use registered environment name
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        .training(
            model={
                "custom_model": "edge_placement_model",
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "vf_share_layers": True,
            },
            lr=0.0003,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            train_batch_size=2000,  # Reduced batch size
            sgd_minibatch_size=64,  # Reduced minibatch size
            num_sgd_iter=5,  # Reduced iterations
        )
        .rollouts(
            num_rollout_workers=0,  # Use local worker only to avoid multiprocessing issues
            num_envs_per_worker=1,
        )
        .resources(num_gpus=0)
        .debugging(log_level="INFO")
    )

    algo = cfg.build()

    # ---- train a few iterations ---------------------------------------------
    for i in range(5):
        print(f"\nStarting iteration {i+1}/5")
        result = algo.train()
        
        # Extract metrics safely
        episode_reward_mean = result.get('env_runners', {}).get('episode_reward_mean', 
                                        result.get('episode_reward_mean', 0.0))
        
        print(f"Iter {i+1:02d} | reward mean = {episode_reward_mean:.3f}")
        
        # Print additional metrics if available
        if 'info' in result:
            learner_info = result['info'].get('learner', {}).get('default_policy', {})
            if learner_info:
                print(f"   | Policy loss = {learner_info.get('policy_loss', 'N/A')}")
                print(f"   | Value loss = {learner_info.get('vf_loss', 'N/A')}")
        
        print(f"Completed iteration {i+1}/5")

    ray.shutdown()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()