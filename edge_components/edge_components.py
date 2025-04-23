import edge_sim_py as es # type: ignore



class NetworkFunction:
    def __init__(self, name, function_type, function_id):
        self.name = name
        self.function_type = function_type
        self.function_id = function_id
        self.connections = []
    """    
    def connect_to(self, other, type, latency, bandwidth):
        connection = {
            "target": other,
            "type": type, # if connection established with the edge server, type is N6, if it's base station, type is N3
            "latency": latency,
            "bandwidth": bandwidth
        }
        self.connections.append(connection)
    """
class CustomEdgeServer(es.EdgeServer):
    def __init__(self, model_name, coordinates):
        super().__init__(model_name=model_name, coordinates=coordinates)
        self.connections = []
    def connect_to(self, network_function, latency, bandwidth):
        # Custom logic for connecting to a network function
        connection = {
            "target": network_function,
            "latency": latency,
            "type": "N6",
            "bandwidth": bandwidth
        }
        self.connections.append(connection)
        print(f"Connected {self} to {network_function.name} with latency {latency} ms and bandwidth {bandwidth} Mbps")


class CustomBaseStation(es.BaseStation):
    def __init__(self, obj_id):
        super().__init__(obj_id=obj_id)
        self.connections = []
    def connect_to(self, network_function, latency, bandwidth):
        # Custom logic for connecting to a network function
        connection = {
            "target": network_function,
            "latency": latency,
            "type": "N3",
            "bandwidth": bandwidth
        }
        self.connections.append(connection)
        print(f"Connected {self} to {network_function.name} with latency {latency} ms and bandwidth {bandwidth} Mbps")