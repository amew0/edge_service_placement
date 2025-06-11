import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import networkx as nx
import json
from multi_agent_env import MultiAgentEdgeEnv
import os
from datetime import datetime

def env_creator(env_config):
    """Create and return the environment"""
    # Build network graph
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

    return MultiAgentEdgeEnv(G, chains, microservices, edge_sites, base_stations)

def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    # Register the environment
    register_env("edge_placement", env_creator)

    # Create output directory for checkpoints and logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"training_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Configure PPO with updated API
    config = (PPOConfig()
        .environment("edge_placement")
        .env_runners(
            num_env_runners=2,
            batch_mode="complete_episodes"
        )
        .training(
            train_batch_size=4000,
            lr=0.0003,
            gamma=0.99,
            lambda_=0.95,
            entropy_coeff=0.01,
            clip_param=0.2,
            num_epochs=10,
            minibatch_size=128,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
                "vf_share_layers": True
            }
        )
        .framework("torch")
        .debugging(log_level="INFO")
        .resources(num_gpus=0)
        .evaluation(
            evaluation_interval=5,
            evaluation_duration=10,
            evaluation_duration_unit="episodes"
        )
        .checkpointing(
            export_native_model_files=True
        )
    )

    # Build the algorithm
    algo = config.build()

    # Training loop
    best_reward = float('-inf')
    for i in range(100):  # Train for 100 iterations
        result = algo.train()
        print(f"\nIteration {i}")
        print(pretty_print(result))

        # Save best model
        mean_reward = result["episode_reward_mean"]
        if mean_reward > best_reward:
            best_reward = mean_reward
            checkpoint_path = algo.save(f"{output_dir}/best_model")
            print(f"New best model saved at {checkpoint_path}")

        # Save periodic checkpoint
        if i % 10 == 0:
            checkpoint_path = algo.save(f"{output_dir}/checkpoint_{i}")
            print(f"Checkpoint saved at {checkpoint_path}")

    # Save final model
    final_checkpoint = algo.save(f"{output_dir}/final_model")
    print(f"\nTraining completed. Final model saved at {final_checkpoint}")

    # Cleanup
    algo.stop()
    ray.shutdown()

if __name__ == "__main__":
    main()