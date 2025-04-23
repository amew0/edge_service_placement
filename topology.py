import numpy as np
import edge_sim_py as es # type: ignore
from gym import spaces
from edge_components import edge_components as ec # type: ignore
import networkx as nx
import matplotlib.pyplot as plt
import random
locations = [
    {
        "name": "Central Park, New York (South End)",
        "lat": 40.7690,
        "lon": -73.9817
    }
    ,
    {
        "name": "Columbus Circle, New York",
        "lat": 40.7681,
        "lon": -73.9819
    }
    ,
    {
        "name": "The Plaza Hotel, New York",
        "lat": 40.7644,
        "lon": -73.9747
    }
    ,
    {   
        "name": "Apple Store Fifth Avenue, New York",
        "lat": 40.7637,
        "lon": -73.9726
    }
]
def create_network():
    # Edge Sites (E1-E4)
    edge_sites = [
        ec.CustomEdgeServer(model_name="E1", coordinates=(locations[0]["lat"], locations[0]["lon"])),
        ec.CustomEdgeServer(model_name="E2", coordinates=(locations[1]["lat"], locations[1]["lon"])),
        ec.CustomEdgeServer(model_name="E3", coordinates=(locations[2]["lat"], locations[2]["lon"])),
        ec.CustomEdgeServer(model_name="E4", coordinates=(locations[3]["lat"], locations[3]["lon"]))
    ]

    # Base Stations (10 BS)
    base_stations = [ec.CustomBaseStation(obj_id=i) for i in range(10)]

    # User Equipment (50 drones)
    drones = [es.User(obj_id=i) for i in range(20)]
    # UPFs
    upfs = [ec.NetworkFunction(name=f"UPF_{i}", function_type="UPF", function_id=i) for i in range(4)]
    #print(len(upfs))
    for idx in range(len(base_stations)):
        upf_idx = random.randint(0, len(upfs) - 1)
        base_stations[idx].connect_to(upfs[upf_idx], latency=0.1, bandwidth=1000)
    for idx in range(len(edge_sites)):
        edge_sites[idx].connect_to(upfs[idx], latency=0.2, bandwidth=5000)

    return edge_sites, base_stations, drones, upfs



def visualize_topology(edge_sites, base_stations, drones, upfs):
    """
    Visualize the network topology using NetworkX and Matplotlib.
    :param edge_sites: List of edge servers.
    :param base_stations: List of base stations.
    :param drones: List of user equipment (drones).
    :param upf: The UPF network function.
    """
    G = nx.Graph()
    dashed_edges = []
    normal_edges = []
    # Add UPF nodes
    for idx in range(len(upfs)):
        G.add_node(upfs[idx].name, type=f"UPF_{idx}")
    # Add edge servers and connect them to UPF
    for edge in edge_sites:
        G.add_node(edge.model_name, type="EdgeServer")
        for connection in edge.connections:
            target = connection["target"]
            if isinstance(target, ec.NetworkFunction):
                print(f"Edge server {edge.model_name} connected to UPF {target.name}")
                normal_edges.append((edge.model_name, target.name))
                G.add_edge(edge.model_name, target.name, latency=connection["latency"], bandwidth=connection["bandwidth"])
    for i in range(len(edge_sites)):
        for j in range(len(edge_sites) - i):
            if i != j:
                dashed_edges.append((edge_sites[i].model_name, edge_sites[j].model_name))
                G.add_edge(edge_sites[i].model_name, edge_sites[j].model_name, color="green", weight=0.5)
    print(dashed_edges)
    # Add base stations and connect them to UPF
    for bs in base_stations:
        G.add_node(f"BS_{bs.id}", type="BaseStation")
        for connection in bs.connections:
            target = connection["target"]
            if isinstance(target, ec.NetworkFunction):
                normal_edges.append((f"BS_{bs.id}", target.name))
                G.add_edge(f"BS_{bs.id}", target.name, latency=connection["latency"], bandwidth=connection["bandwidth"])
    # Add drones and connect them to their respective base stations (example logic)
    for i, drone in enumerate(drones):
        bs_id = i % len(base_stations)  # Assign drones to base stations in a round-robin fashion
        G.add_node(f"Drone_{drone.id}", type="Drone")
        normal_edges.append((f"Drone_{drone.id}", f"BS_{bs_id}"))
        G.add_edge(f"Drone_{drone.id}", f"BS_{bs_id}", latency=0.05, bandwidth=500)

    # Draw the graph
    pos = nx.drawing.nx_agraph.graphviz_layout(G)  # Layout for visualization
    node_colors = [get_node_color(G.nodes[node]["type"]) for node in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)
    # Draw solid edges
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, style='solid', edge_color='black')

    # Draw dashed edges
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, style='dashed', edge_color='grey',width=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title("Network Topology")
    plt.show()

def get_node_color(node_type):
    """
    Get the color for a node based on its type.
    :param node_type: The type of the node (e.g., UPF, EdgeServer, BaseStation, Drone).
    :return: A color string.
    """
    colors = {
        "UPF": "red",
        "EdgeServer": "blue",
        "BaseStation": "green",
        "Drone": "orange"
    }
    return colors.get(node_type, "gray")

if __name__ == "__main__":
    # Example usage
    edge_sites, base_stations, drones, upfs = create_network()
    print("-----------------------------------------------------")
    print("Edge Sites:")
    for edge in edge_sites:
        print(f" - {edge.model_name} at {edge.coordinates}")
    print("Base Stations:")
    for bs in base_stations:
        print(f" - {bs.id}")
    print("Drones:")
    for drone in drones:
        print(f" - {drone.id}")
    for i in range(len(upfs)):
        print(f" - {upfs[i].name} (ID: {upfs[i].function_id})")
    # Example usage
    visualize_topology(edge_sites, base_stations, drones, upfs)
    