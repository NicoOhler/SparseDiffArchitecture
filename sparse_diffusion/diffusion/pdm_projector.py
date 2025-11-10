import torch
from torch_scatter import scatter_add

class PDMProjector:
    # projection onto feasible set C using PDM algorithm.
    def __init__(self, cfg):
        self.grid_shape = cfg.dataset.grid_shape
        self.max_weight = cfg.general.max_weight
        self.four_neighborhood = cfg.general.four_neighborhood

    def project(self, sample):
        if self.four_neighborhood:
            sample = self._enforce_four_neighborhood(sample)

        if self.max_weight:
            sample = self._enforce_max_weight(sample)

        return sample
    
    def _enforce_four_neighborhood(self, sample):
        # remove all edges that violate 4-neighborhood constraint
        # extract actual composite coordinates from node features (originally one-hot encoded)
        composite_coordinates_one_hot = sample.node[sample.edge_index]
        composite_coordinates = torch.argmax(composite_coordinates_one_hot, dim=2)

        # four neighborhood <=> absolute distance is either 1 or grid_width (due to composite indexing)
        grid_width = self.grid_shape[0]
        absolute_coordinate_distance = torch.abs(composite_coordinates[0] - composite_coordinates[1])
        same_row = (composite_coordinates[0] // grid_width) == (composite_coordinates[1] // grid_width)
        horizontal_edge = (absolute_coordinate_distance == 1) & same_row
        vertical_edge = (absolute_coordinate_distance == grid_width)
        valid_edge = horizontal_edge | vertical_edge

        # keep only valid edges
        sample.edge_index = sample.edge_index[:, valid_edge]
        sample.edge_attr = sample.edge_attr[valid_edge]
        return sample
    
    def _enforce_max_weight(self, sample):
        # // greedily remove heaviest edges until sum of all weights is below max_weight
        # drop random edges from overweight graphs until all graphs are below max weight
        # determine which edge belongs to which graph in the batch
        nodes_per_graph = self.grid_shape[0] * self.grid_shape[1]
        graph_index_per_edge = sample.edge_index[0] // nodes_per_graph
        num_graphs = graph_index_per_edge.max().item() + 1

        # get edge weights and compute total weight per graph
        edge_weights = torch.argmax(sample.edge_attr, dim=-1)
        total_weight_per_graph = scatter_add(edge_weights, graph_index_per_edge, dim=0, dim_size=num_graphs)
        total_weight_per_graph = total_weight_per_graph // 2 # since edges are counted twice

        # iteratively drop edges until below max weight for all graphs
        device = sample.edge_index.device
        valid_edges = torch.ones(sample.edge_index.size(1), dtype=torch.bool, device=device)
        while True:
            # determine which graphs exceed max weight
            overweight_graphs_indices = torch.where(total_weight_per_graph > self.max_weight)[0]
            if overweight_graphs_indices.numel() == 0:
                break

            for graph_idx in overweight_graphs_indices.tolist():
                # ? maybe drop heaviest edge instead of random and vectorize this
                # identify all remaining edges of the current graph (i.e. valid and part of graph)
                remaining_edges_of_graph = torch.where((graph_index_per_edge == graph_idx) & valid_edges)[0]
                num_remaining_edges = remaining_edges_of_graph.size(0) // 2 # since edges are counted twice
                
                # select a random edge (and its counterpart) to drop
                edge_to_drop = torch.randint(low=0, high=num_remaining_edges, size=(1,),device=device)[0]
                global_edge_index_to_drop = remaining_edges_of_graph[edge_to_drop]
                counterpart_edge_to_drop = edge_to_drop + num_remaining_edges 
                global_edge_index_to_drop_pair = remaining_edges_of_graph[counterpart_edge_to_drop]
                
                # drop the edge (i.e. update mask and total weight)
                valid_edges[global_edge_index_to_drop] = False
                valid_edges[global_edge_index_to_drop_pair] = False
                total_weight_per_graph[graph_idx] -= edge_weights[global_edge_index_to_drop]

        # keep only valid edges
        sample.edge_index = sample.edge_index[:, valid_edges]
        sample.edge_attr = sample.edge_attr[valid_edges]

        return sample