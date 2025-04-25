
"""
    G: networkx graph where nodes are the network components (UE+BS+ES) and edges are the links between them
    C: chain of microservices
    S: microservices
    R: latency requirements of the microservices (SLO)
"""
import json
from edge_components import edge_components as ec 

def wssp(service_chains, edge_sites):
    placement = {}
    # Step 1: Sort chains by latency (strictest first)
    sorted_chains = sorted(service_chains, key=lambda x: x["latency_requirement_ms"], reverse=True)
    for chain in sorted_chains:
        # Step 2: Find candidate edge sites
        P = []
        latency_req = chain["latency_requirement_ms"]
        microservices = chain["microservices"]
        memory_requirement = sum([m["memory_mb"] for m in microservices])
        for site in edge_sites:
            # Split chain and assign parts (Algorithm 3.2)
            if site["latency"] <= latency_req and site["available_resources"] >= memory_requirement:
                cost = score_function(site, chain)
                P.append((cost, site))
            else:
                split_sites = split_and_assign(chain, site)
                
                P.extend(split_sites)
        # Step 3: Score candidates and select minimal cover
        scores = [score_function(site, chain) for site in P]
        # Select the site with the best score
        selected_sites = greedy_set_cover(P, scores)
        placement[chain["name"]] = P
    return placement

def split_and_assign(chain, site):
    """
    Split the chain into smaller parts and assign them to the site
    """
    split_chains = []
    microservices = chain["microservices"]
    latency_req = chain["latency_requirement_ms"]
    memory_requirement = sum([m["memory_mb"] for m in microservices])
    
    # Split the chain into smaller parts
    for i in range(0, len(microservices), 2):
        split_chain = {
            "name": f"{chain['name']}_part_{i//2}",
            "microservices": microservices[i:i+2],
            "latency_requirement_ms": latency_req / 2,
            "memory_mb": memory_requirement / 2
        }
        split_chains.append(split_chain)
    
    return split_chains

def score_function(site, weights):
    return weights["dep"] * site["dep"] + weights["cpu"] * site["cpu"] + weights["storage"] * site["storage"] + weights["comm"] * site["comm"] + weights["update"] * site["update"]


if __name__ == "__main__":

    with open("service_chains.json", "r") as f:
        services = json.load(f)
    
    # Example usage
    service_chains = services["service_chains"]

    # Edge Sites (E1-E4)
    edge_sites = [
        ec.CustomEdgeServer(model_name="E1", coordinates=(locations[0]["lat"], locations[0]["lon"])),
        ec.CustomEdgeServer(model_name="E2", coordinates=(locations[1]["lat"], locations[1]["lon"])),
        ec.CustomEdgeServer(model_name="E3", coordinates=(locations[2]["lat"], locations[2]["lon"])),
        ec.CustomEdgeServer(model_name="E4", coordinates=(locations[3]["lat"], locations[3]["lon"]))
    ]

    result = wssp(service_chains, edge_sites)
    print(result)