import torch
from sparse_diffusion import utils
import networkx as nx

class PDMProjector:
    """Projection onto the feasible set using the PDM algorithm."""

    def __init__(self, cfg):
        self.grid_shape = cfg.dataset.grid_shape
        self.max_weight = cfg.general.max_weight
        self.four_neighborhood = cfg.general.four_neighborhood

    def project(self, sparse_sampled_data):
        print("Projecting sampled data onto feasible set using PDM constraints")
        # self._print_edge_list(sparse_sampled_data)
        if self.four_neighborhood:
            sparse_sampled_data = self._enforce_four_neighborhood(sparse_sampled_data)

        if self.max_weight:
            sparse_sampled_data = self._enforce_max_weight(sparse_sampled_data)

        print("Projected edge list:")
        # self._print_edge_list(sparse_sampled_data)
        return sparse_sampled_data
    
    def _enforce_four_neighborhood(self, sparse_sampled_data):
        # remove all edges that violate 4-neighborhood constraint
        # networkx internal indices
        grid_size = self.grid_shape[0] * self.grid_shape[1]
        node_indices = sparse_sampled_data.edge_index % grid_size

        # extract actual composite coordinates from node features (one-hot encoded)
        composite_coordinates_one_hot = sparse_sampled_data.node[node_indices]

        # convert one-hot encoding to actual coordinates
        # composite_coordinates = composite_coordinates_one_hot 
        composite_coordinates = torch.argmax(composite_coordinates_one_hot, dim=2)

        # four neighborhood <=> absolute distance is either 1 or grid_width (due to composite indexing)
        grid_width = self.grid_shape[0]
        absolute_coordinate_distance = torch.abs(composite_coordinates[0] - composite_coordinates[1])
        same_row = (composite_coordinates[0] // grid_width) == (composite_coordinates[1] // grid_width)
        valid_edge = ((absolute_coordinate_distance == 1) & same_row) | (absolute_coordinate_distance == grid_width)

        # keep only valid edges
        sparse_sampled_data.edge_index = sparse_sampled_data.edge_index[:, valid_edge]
        sparse_sampled_data.edge_attr = sparse_sampled_data.edge_attr[valid_edge]
        return sparse_sampled_data
    
    def _enforce_max_weight(self, sparse_sampled_data):
        # greedily remove heaviest edges until sum of all weights is below max_weight
        return sparse_sampled_data
    
    def _print_edge_list(self, sparse_sampled_data):
        graph = self.to_networkx(
            node=sparse_sampled_data.node.long().cpu().numpy(),
            edge_index=sparse_sampled_data.edge_index.long().cpu().numpy(),
            edge_attr=sparse_sampled_data.edge_attr.long().cpu().numpy(),
        )

        for u, v, data in graph.edges(data=True):
            print(f"Edge from {u} to {v} with attributes {data}")
            print(f"\t Node {u} features: {sparse_sampled_data.node[u].nonzero().item()}")
            print(f"\t Node {v} features: {sparse_sampled_data.node[v].nonzero().item()}")

    def to_networkx(self, node, edge_index, edge_attr):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        graph = nx.Graph()

        for i in range(len(node)):
            graph.add_node(i, number=i, symbol=node[i], color_val=node[i])

        for i, edge in enumerate(edge_index.T):
            edge_type = edge_attr[i]
            graph.add_edge(edge[0], edge[1], color=edge_type, weight=3 * edge_type)

        return graph
    
    
if __name__ == "__main__":
    sparse_sampled_data, _ = utils.SparsePlaceHolder.load("/home/nico/projects/sparse_sampled_data.pt")
    projector = PDMProjector(cfg=utils.AttrDict({
        "dataset": {
            "grid_shape": [8, 8],
        },
        "general": {
            "max_weight": 40,
            "four_neighborhood": True,
        }
    }))
    if sparse_sampled_data:
        projected_data = projector(sparse_sampled_data)
        projected_data.save("/home/nico/projects/sparse_sampled_data_projected.pt")