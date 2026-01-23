class InfluenceData:
    def __init__(self, config, num_nodes, G, node_costs, neighbors, scaling_factor):
        self.config = config
        self.num_nodes = num_nodes
        self.G = G
        self.node_costs = node_costs
        self.neighbors = neighbors
        self.scaling_factor = scaling_factor