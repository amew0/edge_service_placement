import gymnasium as gym
import numpy as np
import networkx as nx
import json
from typing import Dict, Any, Tuple, List, Literal
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
import os
from datetime import datetime
import matplotlib.pyplot as plt


class TrainingMetricsCallback(BaseCallback):
    """Custom callback to log training metrics and create visualizations."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.coverage_rewards = []

    def _on_step(self) -> bool:
        # Log episode metrics when episode ends
        if "episode" in self.locals.get("infos", [{}])[0]:
            info = self.locals["infos"][0]["episode"]
            self.episode_rewards.append(info["r"])
            self.episode_lengths.append(info["l"])

        return True

    def _on_training_end(self) -> None:
        # Save training metrics
        if self.episode_rewards:
            self.save_metrics()

    def save_metrics(self):
        """Save training metrics and create plots."""
        output_dir = getattr(self.model, "output_dir", "training_output")

        # Save raw data
        metrics = {
            "episode_rewards": [float(r) for r in self.episode_rewards],
            "episode_lengths": [int(l) for l in self.episode_lengths],
        }

        with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Create plots
        if len(self.episode_rewards) > 0:
            self.plot_training_progress(output_dir)

    def plot_training_progress(self, output_dir):
        """Create training progress plots."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title("Episode Rewards Over Time")
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.grid(True)

        # Episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title("Episode Lengths Over Time")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Steps")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_progress.png"))
        plt.close()


class MultiAgentEdgeEnv(gym.Env):
    """
    Enhanced Multi-Agent Environment for Edge Service Placement.

    Features:
    - Configurable state sharing modes
    - Multiple observation strategies
    - Detailed metrics and logging
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        graph,
        service_chains,
        microservices,
        edge_sites,
        base_stations,
        max_steps: int = 1_000,
        render_mode: str | None = None,
        observation_mode: Literal["centralized", "local", "hybrid"] = "hybrid",
        state_sharing: Literal["full", "partial", "none"] = "partial",
        reward_sharing: float = 0.5,  # Weight for shared vs individual rewards
    ):
        super().__init__()
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self.state_sharing = state_sharing
        self.reward_sharing = reward_sharing

        self.G = graph
        self.service_chains = service_chains
        self.microservices = microservices
        self.edge_sites = edge_sites
        self.base_stations = base_stations

        self.B = len(base_stations)
        self.E = len(edge_sites)
        self.S = len(microservices)
        self.max_steps = max_steps

        # Performance metrics
        self.episode_metrics = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "coverage_achieved": 0,
            "resource_utilization": 0.0,
        }

        # map micro-service names → indices
        self.ms_name_to_idx = {ms["name"]: idx for idx, ms in enumerate(microservices)}

        # service chains using indices (faster later on)
        self.service_chains_idx: List[Dict[str, Any]] = []
        for chain in service_chains:
            self.service_chains_idx.append(
                {
                    "name": chain["name"],
                    "microservices": [self.ms_name_to_idx[ms_name] for ms_name in chain["microservices"]],
                    "latency_requirement_ms": chain["latency_requirement_ms"],
                }
            )

        # Setup observation and action spaces based on configuration
        self._setup_spaces()

        # ---------- state ----------
        self.server_microservices = np.zeros((self.E, self.S), dtype=int)
        self.step_count = 0
        self._compute_cost_matrix()

    def _setup_spaces(self):
        """Setup observation and action spaces based on configuration."""
        # Action space: [action_type, microservice_idx] for each edge
        self.action_space = gym.spaces.MultiDiscrete([3, self.S] * self.E)

        # Observation space depends on mode
        if self.observation_mode == "centralized":
            # Full global view for all agents
            obs_dim = self._get_centralized_obs_dim()
        elif self.observation_mode == "local":
            # Only local information for each agent
            obs_dim = self._get_local_obs_dim()
        else:  # hybrid
            # Local + limited global information
            obs_dim = self._get_hybrid_obs_dim()

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def _get_centralized_obs_dim(self) -> int:
        """Get observation dimension for centralized mode."""
        # All microservices on all edges + resources + latencies + chain requirements
        return (
            self.E * self.S  # microservice placement matrix
            + self.E * 3  # resources for all edges
            + self.E * self.B  # latencies from all edges to all base stations
            + len(self.service_chains_idx)
        )  # chain requirements

    def _get_local_obs_dim(self) -> int:
        """Get observation dimension for local mode."""
        # Only local edge information for each edge
        return (
            self.S  # local microservices
            + 3  # local resources
            + self.B  # local latencies
            + len(self.service_chains_idx)
        )  # chain requirements

    def _get_hybrid_obs_dim(self) -> int:
        """Get observation dimension for hybrid mode."""
        # Local info + global coverage info
        return (
            self.S  # local microservices
            + 3  # local resources
            + self.B  # local latencies
            + len(self.service_chains_idx)  # chain requirements
            + self.B  # global coverage per base station
            + self.E
        )  # global utilization per edge

    def _compute_cost_matrix(self):
        """Pre-compute (dummy) placement costs for each (service, edge)."""
        self.tau = 1.0 + 0.1 * np.random.rand(self.S, self.E)

    def _get_observation(self) -> np.ndarray:
        """Get observation based on current observation mode."""
        if self.observation_mode == "centralized":
            return self._get_centralized_observation()
        elif self.observation_mode == "local":
            return self._get_local_observation()
        else:  # hybrid
            return self._get_hybrid_observation()

    def _get_centralized_observation(self) -> np.ndarray:
        """Get centralized observation (full global state)."""
        # Flatten microservice placement matrix
        ms_placement = self.server_microservices.flatten()

        # All edge resources
        resources = []
        for i in range(self.E):
            resources.extend(
                [
                    self.microservices[i]["cpu"],
                    self.microservices[i]["memory_mb"],
                    1.0 if self.microservices[i]["gpu"] else 0.0,
                ]
            )

        # All edge latencies
        latencies = []
        for i in range(self.E):
            for bs in self.base_stations:
                latencies.append(self._get_shortest_path_length(self.edge_sites[i], bs))

        # Chain requirements
        chain_reqs = [c["latency_requirement_ms"] for c in self.service_chains_idx]

        return np.concatenate(
            [
                ms_placement.astype(np.float32),
                np.array(resources, dtype=np.float32),
                np.array(latencies, dtype=np.float32),
                np.array(chain_reqs, dtype=np.float32),
            ]
        )

    def _get_local_observation(self) -> np.ndarray:
        """Get local observation (only first edge's view for simplicity)."""
        # For SB3 single-agent approach, we use the first edge's perspective
        return self._get_agent_observation(0)

    def _get_hybrid_observation(self) -> np.ndarray:
        """Get hybrid observation (local + global context)."""
        # Local observation for first edge
        local_obs = self._get_agent_observation(0)

        # Global context
        coverage = self._service_latency()
        utilization = np.sum(self.server_microservices, axis=1) / self.S  # utilization per edge

        return np.concatenate([local_obs, coverage.astype(np.float32), utilization.astype(np.float32)])

    def _get_agent_observation(self, edge_idx: int) -> np.ndarray:
        """Get observation for a specific edge agent."""
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
            [self._get_shortest_path_length(self.edge_sites[edge_idx], bs) for bs in self.base_stations],
            dtype=np.float32,
        )

        chain_requirements = np.array(
            [c["latency_requirement_ms"] for c in self.service_chains_idx],
            dtype=np.float32,
        )

        return np.concatenate([current_ms.astype(np.float32), avail, latencies, chain_requirements])

    def _get_shortest_path_length(self, source: str, target: str) -> float:
        """Get shortest path length between two nodes, returning inf if no path exists."""
        try:
            return nx.shortest_path_length(self.G, source=source, target=target, weight="latency")
        except nx.NetworkXNoPath:
            return float("inf")

    def _service_latency(self) -> np.ndarray:
        """Return an array (#base_stations,) with service-latency coverage."""
        coverage = np.zeros(self.B, dtype=int)

        for chain in self.service_chains_idx:
            sites_per_ms = [np.where(self.server_microservices[:, ms] > 0)[0] for ms in chain["microservices"]]
            if not all(len(s) > 0 for s in sites_per_ms):
                continue

            for b_idx, bs in enumerate(self.base_stations):
                total_lat = 0.0
                for ms_idx, slist in zip(chain["microservices"], sites_per_ms):
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.server_microservices.fill(0)
        self.step_count = 0

        # Reset episode metrics
        self.episode_metrics = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "coverage_achieved": 0,
            "resource_utilization": 0.0,
        }

        return self._get_observation(), {}

    def step(self, action):
        self.step_count += 1

        # Reset episode metrics
        self.episode_metrics["total_deployments"] = 0
        self.episode_metrics["successful_deployments"] = 0

        # Parse actions for multiple edges
        actions = action.reshape(self.E, 2)

        individual_rewards = np.zeros(self.E)

        for edge_idx, (act_type, ms_idx) in enumerate(actions):
            self.episode_metrics["total_deployments"] += 1

            if not (0 <= ms_idx < self.S):
                individual_rewards[edge_idx] -= 2.0  # Penalty for invalid action
                continue

            if act_type == 1:  # deploy
                if self.server_microservices[edge_idx, ms_idx] == 0:
                    self.server_microservices[edge_idx, ms_idx] = 1
                    individual_rewards[edge_idx] += 1.0
                    self.episode_metrics["successful_deployments"] += 1
                else:
                    individual_rewards[edge_idx] -= 1.0  # Already deployed
            elif act_type == 2:  # evict
                if self.server_microservices[edge_idx, ms_idx] == 1:
                    self.server_microservices[edge_idx, ms_idx] = 0
                    individual_rewards[edge_idx] += 0.5
                    self.episode_metrics["successful_deployments"] += 1
                else:
                    individual_rewards[edge_idx] -= 0.5  # Nothing to evict

        # Compute shared coverage reward
        coverage = self._service_latency()
        coverage_reward = 0.1 * coverage.sum()
        self.episode_metrics["coverage_achieved"] = coverage.sum()

        # Compute resource utilization
        total_utilization = np.sum(self.server_microservices) / (self.E * self.S)
        self.episode_metrics["resource_utilization"] = total_utilization

        # Efficiency bonus (reward for achieving coverage with fewer resources)
        efficiency_bonus = 0.0
        if coverage.sum() > 0:
            efficiency_bonus = coverage.sum() / max(total_utilization * 10, 1.0)

        # Calculate final reward based on sharing configuration
        if self.state_sharing == "full":
            # Fully cooperative: same reward for all
            reward = coverage_reward + efficiency_bonus
        elif self.state_sharing == "none":
            # Individual rewards only
            reward = float(np.mean(individual_rewards))
        else:  # partial
            # Weighted combination
            individual_component = np.mean(individual_rewards)
            shared_component = coverage_reward + efficiency_bonus
            reward = (1 - self.reward_sharing) * individual_component + self.reward_sharing * shared_component

        terminated = False
        truncated = self.step_count >= self.max_steps

        # Enhanced info dictionary
        info = {
            "episode_metrics": self.episode_metrics.copy(),
            "coverage_per_bs": coverage.tolist(),
            "total_coverage": coverage.sum(),
            "resource_utilization": total_utilization,
            "efficiency_score": efficiency_bonus,
            "individual_rewards": individual_rewards.tolist(),
            "shared_reward": coverage_reward + efficiency_bonus,
        }

        return self._get_observation(), reward, terminated, truncated, info


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


def main():
    print("=== Enhanced Multi-Agent Edge Service Placement Training ===")
    print("Using Stable Baselines3 with configurable state sharing")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_output_sb3_enhanced_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Build environment
    G, BS, ES = build_demo_graph()
    microservices, chains = load_services("data/services.json")

    print(f"Environment setup:")
    print(f"  - Base stations: {len(BS)}")
    print(f"  - Edge sites: {len(ES)}")
    print(f"  - Microservices: {len(microservices)}")
    print(f"  - Service chains: {len(chains)}")

    # Environment configuration
    env_config = {
        "observation_mode": "hybrid",  # centralized, local, hybrid
        "state_sharing": "partial",  # full, partial, none
        "reward_sharing": 0.7,  # 0.0 = individual, 1.0 = fully shared
    }

    print(f"Environment configuration: {env_config}")

    def make_env():
        env = MultiAgentEdgeEnv(G, chains, microservices, ES, BS, render_mode=None, **env_config)
        return env

    # Create vectorized environments
    print("Creating vectorized environments...")
    env = make_vec_env(make_env, n_envs=4, monitor_dir=None)
    env = VecMonitor(env, filename=os.path.join(output_dir, "monitor.csv"))

    eval_env = make_vec_env(make_env, n_envs=1, monitor_dir=None)
    eval_env = VecMonitor(eval_env, filename=os.path.join(output_dir, "eval_monitor.csv"))

    # Create enhanced callbacks
    print("Setting up callbacks...")
    metrics_callback = TrainingMetricsCallback(verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=os.path.join(output_dir, "eval_logs"),
        eval_freq=100,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix="edge_placement_model",
    )

    # Enhanced PPO configuration
    print("Configuring PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Gradient clipping
        tensorboard_log=os.path.join(output_dir, "tensorboard"),
        policy_kwargs={"net_arch": [{"pi": [256, 256], "vf": [256, 256]}]},  # Separate networks
    )

    # Store output directory in model for callbacks
    model.output_dir = output_dir

    # Enhanced training loop
    print("Starting training...")
    total_timesteps = 10000
    for i in range(5):
        print(f"\n=== Training Iteration {i+1}/5 ===")

        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback, metrics_callback],
            reset_num_timesteps=False,
            progress_bar=True,
        )

        print(f"Completed iteration {i+1}/5")

        # Save intermediate model with detailed name
        model_path = os.path.join(
            output_dir,
            f"model_iter_{i+1}_{env_config['observation_mode']}_{env_config['state_sharing']}",
        )
        model.save(model_path)
        print(f"Model saved: {model_path}")

    # Save final model and configuration
    final_model_path = os.path.join(output_dir, "final_model")
    model.save(final_model_path)

    # Save environment configuration
    with open(os.path.join(output_dir, "env_config.json"), "w") as f:
        json.dump(env_config, f, indent=2)

    print(f"\n=== Training Complete ===")
    print(f"Final model saved: {final_model_path}")
    print(f"Configuration saved: {os.path.join(output_dir, 'env_config.json')}")
    print(f"Logs and metrics available in: {output_dir}")


if __name__ == "__main__":
    main()
