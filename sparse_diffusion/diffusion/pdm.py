from collections import defaultdict
import torch
from torch_scatter import scatter_add
from sparse_diffusion.datasets.dataset_generator import generate_edge_crossings_lookup_table
import networkx as nx
from collections import deque

class PDM:
    # projection onto feasible set C using PDM algorithm.
    def __init__(self, cfg):
        self.grid_shape = cfg.dataset.grid_shape
        self.connected = cfg.general.connected
        self.planar = cfg.general.planar
        if self.planar:
            self.lookup_table = generate_edge_crossings_lookup_table()
        self.max_weight = cfg.general.max_weight

    def detect_violations(self, sample):
        violations = {}
        num_graphs = sample.batch.max().item() + 1
        graph_index_by_edge = sample.edge_index[0] // (self.grid_shape[0] * self.grid_shape[1])
        edge_count_per_graph = torch.bincount(graph_index_by_edge, minlength=num_graphs)

        # remove duplicate edges per graph
        duplicate_edges_indices = self._detect_duplicate_edges(sample)
        duplicate_edges_per_graph = scatter_add(duplicate_edges_indices.to(torch.int), graph_index_by_edge, dim=0, dim_size=num_graphs)
        duplicate_edge_ratio_per_graph = duplicate_edges_per_graph.float() / edge_count_per_graph.float()
        violations['num_duplicate_edges'] = duplicate_edges_per_graph.tolist()
        violations['duplicate_edge_ratio'] = duplicate_edge_ratio_per_graph.tolist()

        # count illegal edges per graph
        illegal_edges = self._detect_illegal_edges(sample)
        illegal_edges_per_graph = scatter_add(illegal_edges.to(torch.int), graph_index_by_edge, dim=0, dim_size=num_graphs) 
        illegal_edges_per_graph = illegal_edges_per_graph // 2  # since edges are counted twice
        illegal_edge_ratio_per_graph = illegal_edges_per_graph.float() / edge_count_per_graph.float()
        violations['num_illegal_edges'] = illegal_edges_per_graph.tolist()
        violations['illegal_edge_ratio'] = illegal_edge_ratio_per_graph.tolist()

        # count edge crossings per graph
        if self.planar:
            edge_crossings_by_graph = self._detect_edge_crossings(sample)
            edge_crossings_by_graph = [len(edge_crossings_by_graph[i]) for i in range(num_graphs)]
            edge_crossing_ratio_per_graph = [edge_crossings_by_graph[i] / edge_count_per_graph[i].item() for i in range(num_graphs)]
            violations['num_edge_crossings'] = edge_crossings_by_graph
            violations['edge_crossing_ratio'] = edge_crossing_ratio_per_graph

        # count disconnected components per graph
        if self.connected:
            violations['disconnected_components'] = self._detect_disconnected_components(sample)
        
        # count excess weight per graph
        if self.max_weight:
            violations['excess_weight'] = self._detect_excess_weight(sample).tolist()

        violations_list = []
        for i in range(num_graphs):
            violation = {}
            for key in violations.keys():
                violation[key] = violations[key][i]
            violations_list.append(violation)

        return violations_list

    def project(self, sample):
        sample = self._enforce_legal_edges(sample)
        # if self.planar:
        #     sample = self._enforce_planarity(sample)
        if self.connected:
            sample = self._enforce_connectivity(sample)
        if self.max_weight:
            sample = self._enforce_max_weight(sample)
        return sample
    
    # detect constraint violations
    def _detect_duplicate_edges(self, sample):
        # reverse edges are allowed
        edges = set()
        edges_to_remove = []
        for edge in sample.edge_index.t().tolist():
            edge_as_tuple = tuple(edge)
            edges_to_remove.append(edge_as_tuple in edges)
            edges.add(edge_as_tuple)
        return torch.tensor(edges_to_remove, dtype=torch.bool, device=sample.edge_index.device)
    
    def _detect_illegal_edges(self, sample):
        # use distances to determine valid edges
        x, y = self._get_coordinates(sample)
        delta_y = torch.abs(y[0] - y[1])
        delta_x = torch.abs(x[0] - x[1])
        
        # eight neighborhood + knight moves
        legal_edges = (
            ((delta_x == 1) & (delta_y == 0)) |  
            ((delta_x <= 2) & (delta_y == 1)) | 
            ((delta_x == 1) & (delta_y == 2))
        )
        illegal_edges = ~legal_edges
        return illegal_edges
    
    def _detect_edge_crossings(self, sample):
        lookup = self.lookup_table
        edges_by_graph = self._get_edges_by_graph(sample)

        # detect crossings for each graph in the batch
        edge_crossings_by_graph = defaultdict(set)
        for i in range(len(edges_by_graph)):
            edges = edges_by_graph[i]
            edge_crossings = edge_crossings_by_graph[i]

            # iterate over all edges and check all potential crossings
            for edge in edges:
                start, end = edge
                idx = edges[edge]
                direction = (end[0] - start[0], end[1] - start[1])
                edges_to_check = lookup.get(direction, [])
                # check all potential crossing edges (created using precomputed offsets)
                for ((start_x_offset, start_y_offset), (end_x_offset, end_y_offset)) in edges_to_check:
                    crossing_start = (start[0] + start_x_offset, start[1] + start_y_offset)
                    crossing_end = (start[0] + end_x_offset, start[1] + end_y_offset)
                    crossing_edge = (crossing_start, crossing_end)

                    # ensure consistent ordering to prevent double counting
                    if crossing_start > crossing_end:
                        continue
                    if crossing_edge in edges:
                        crossing_idx = edges[crossing_edge]
                        if idx < crossing_idx:
                            edge_crossings.add((idx, crossing_idx))
                        else:
                            edge_crossings.add((crossing_idx, idx))

            # print edge crossings for debugging
            # for idx1, idx2 in edge_crossings:
            #     edge1 = sample.edge_index[:, idx1[0]].tolist()
            #     edge2 = sample.edge_index[:, idx2[0]].tolist()
            #     source1 = sample.node[edge1[0]].nonzero().item()
            #     target1 = sample.node[edge1[1]].nonzero().item()
            #     source2 = sample.node[edge2[0]].nonzero().item()
            #     target2 = sample.node[edge2[1]].nonzero().item()
            #     print(f"Graph {i}: Edge {edge1} crosses with Edge {edge2}")
            #     print(f"\t{edge1} connects {source1} to {target1}")
            #     print(f"\t{edge2} connects {source2} to {target2}")

        return edge_crossings_by_graph

    def _detect_disconnected_components(self, sample):
        edges_by_graph = self._get_edges_by_graph(sample)

        # count disconnected components for each graph in the batch
        disconnected_components_by_graph = []
        for i in range(len(edges_by_graph)):
            edges = edges_by_graph[i]
            # each edge consists of two indices/references to the global edge list
            G = nx.Graph()
            for edge in edges:
                start, end = edge
                G.add_edge(start, end)
            disconnected_components_by_graph.append(nx.number_connected_components(G) - 1)  

        return disconnected_components_by_graph

    def _detect_excess_weight(self, sample):
        # determine which edge belongs to which graph in the batch
        nodes_per_graph = self.grid_shape[0] * self.grid_shape[1]
        graph_index_by_edge = sample.edge_index[0] // nodes_per_graph
        num_graphs = graph_index_by_edge.max().item() + 1

        # get edge weights and compute total weight per graph
        edge_weights = torch.argmax(sample.edge_attr, dim=-1)
        total_weight_per_graph = scatter_add(edge_weights, graph_index_by_edge, dim=0, dim_size=num_graphs)
        total_weight_per_graph = total_weight_per_graph // 2 # since edges are counted twice
        excess_weight_per_graph = torch.clamp(total_weight_per_graph - self.max_weight, min=0)
        return excess_weight_per_graph

    # enforce constraints
    def _enforce_legal_edges(self, sample):
        illegal_edges = self._detect_illegal_edges(sample)
        return self._remove_edges(sample, illegal_edges)
    
    def _enforce_planarity(self, sample):
        edge_crossings_by_graph = self._detect_edge_crossings(sample)
        edges_to_remove = torch.zeros(sample.edge_index.size(1), dtype=torch.bool, device=sample.edge_index.device)
        for edge_crossings in edge_crossings_by_graph.values():
            for ((i1, i2), (i3, i4)) in edge_crossings:
                # skip if conflicting edge already removed
                if edges_to_remove[i3] and edges_to_remove[i4]:
                    continue
                # drop first edge of crossing pair
                edges_to_remove[i1] = True
                edges_to_remove[i2] = True
                # edges_to_remove[i3] = True
                # edges_to_remove[i4] = True
        return self._remove_edges(sample, edges_to_remove)
    
    def _enforce_connectivity(self, sample):
        edges_by_graph = self._get_edges_by_graph(sample)
        grid_width, grid_height = self.grid_shape
        max_weight = sample.edge_attr.size(1)
        device = sample.edge_index.device
    
        def get_neighbors(pos, existing_edges):
            r, c = pos
            directions = [
                (0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1), # 8-neighborhood
                (1,2), (1,-2), (-1,2), (-1,-2), (2,1), (2,-1), (-2,1), (-2,-1) # knight moves
            ]
            
            # iterate over all possible directions and yield valid neighbors
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < grid_height and 0 <= nc < grid_width):
                    continue

                # ensure that adding edge to neighbor does not create an edge crossing
                if self.planar:
                    direction = (dr, dc)
                    potential_crossers = self.lookup_table.get(direction, [])
                    
                    is_crossing = False
                    for ((s_x, s_y), (e_x, e_y)) in potential_crossers:
                        # check if specific edge that would cross this move exists
                        cross_start = (r + s_x, c + s_y)
                        cross_end = (r + e_x, c + e_y)
                        cross_edge = tuple(sorted((cross_start, cross_end)))
                        if cross_edge in existing_edges:
                            is_crossing = True
                            break
                    
                    # skip neighbor if it would create a crossing
                    if is_crossing:
                        continue

                yield (nr, nc)

        for graph_idx in range(len(edges_by_graph)):
            edges = edges_by_graph[graph_idx]
            edges_to_add = set()
            existing_edges = set()
            
            # each edge consists of two indices/references to the global edge list
            G = nx.Graph()
            for edge in edges:
                start, end = edge
                G.add_edge(start, end)
                existing_edges.add(tuple(sorted((start, end))))
                
            components = list(nx.connected_components(G))
            if len(components) <= 1:
                continue  # already connected

            # fast lookup for which node belongs to which component, maps (x,y) -> component_index
            node_to_comp_id = {}
            for i, nodes in enumerate(components):
                for node in nodes:
                    node_to_comp_id[node] = i

            # BFS with multiple sources (all component nodes)
            queue = deque()
            visited = {} # (x,y) -> (source_component_id, origin_node_in_comp)
            predecessors = {} # (x,y) -> parent_node (x_parent, y_parent), needed for backtracking paths
            connected_components = set() # track which components are already connected

            for component, nodes in enumerate(components):
                for node in nodes:
                    queue.append(node)
                    visited[node] = component
                    predecessors[node] = None # mark as root

            while queue:
                node = queue.popleft()
                component = visited[node]

                for neighbor in get_neighbors(node, existing_edges):
                    if neighbor in visited:
                        neighbor_component = visited[neighbor]
                        
                        # path found if new neighbor belongs to different component
                        if component != neighbor_component:
                            already_connected = component in connected_components and neighbor_component in connected_components
                            # skip if these components are already connected
                            if not already_connected:
                                connected_components.add(component)
                                connected_components.add(neighbor_component)
                                
                                # connect node and neighbor
                                bridge_edge = tuple(sorted((node, neighbor)))
                                edges_to_add.add(bridge_edge)
                                existing_edges.add(bridge_edge)
                                
                                # backtrack from node to its component root
                                current = node
                                while predecessors[current] is not None:
                                    parent = predecessors[current]
                                    path_edge = tuple(sorted((current, parent)))
                                    edges_to_add.add(path_edge)
                                    existing_edges.add(path_edge)
                                    current = parent
                                
                                # backtrack from neighbor to its component root
                                current = neighbor
                                while predecessors[current] is not None:
                                    parent = predecessors[current]
                                    path_edge = tuple(sorted((current, parent)))
                                    edges_to_add.add(path_edge)
                                    existing_edges.add(path_edge)
                                    current = parent

                    else:
                        # add node from empty space to current component
                        visited[neighbor] = component
                        predecessors[neighbor] = node 
                        queue.append(neighbor)

            # add new edges to sample
            for (start, end) in edges_to_add:
                start_coordinate = start[1] * grid_width + start[0]
                end_coordinate = end[1] * grid_width + end[0]
                
                # get node indices that correspond to coordinates
                same_start_node = start_coordinate == sample.node.argmax(dim=1)
                same_end_node = end_coordinate == sample.node.argmax(dim=1)
                same_graph = sample.batch == graph_idx
                start_idx = torch.nonzero((same_graph & same_start_node)).item()
                end_idx = torch.nonzero((same_graph & same_end_node)).item()

                # add edge to edge_index and edge_attr
                edge = torch.tensor([[start_idx], [end_idx]], device=device)
                reverse_edge = torch.tensor([[end_idx], [start_idx]], device=device)
                edge_attr = torch.zeros((1, max_weight), device=device)
                random_weight = torch.randint(1, max_weight, (1,), device=device)
                edge_attr[0, random_weight] = 1  
                sample.edge_index = torch.cat([sample.edge_index, edge, reverse_edge], dim=1)
                sample.edge_attr = torch.cat([sample.edge_attr, edge_attr, edge_attr], dim=0)
        return sample
    
    def _enforce_max_weight(self, sample):
        # // greedily remove heaviest edges until sum of all weights is below max_weight
        # drop random edges from overweight graphs until all graphs are below max weight
        # determine which edge belongs to which graph in the batch
        nodes_per_graph = self.grid_shape[0] * self.grid_shape[1]
        graph_index_by_edge = sample.edge_index[0] // nodes_per_graph
        num_graphs = graph_index_by_edge.max().item() + 1

        # get edge weights and compute total weight per graph
        edge_weights = torch.argmax(sample.edge_attr, dim=-1)
        total_weight_per_graph = scatter_add(edge_weights, graph_index_by_edge, dim=0, dim_size=num_graphs)
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
                remaining_edges_of_graph = torch.where((graph_index_by_edge == graph_idx) & valid_edges)[0]
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
    
    # helper functions
    def _remove_edges(self, sample, edges_to_remove):
        edges_to_keep = ~edges_to_remove
        sample.edge_index = sample.edge_index[:, edges_to_keep]
        sample.edge_attr = sample.edge_attr[edges_to_keep]
        return sample
    
    def _get_coordinates(self, sample):
        composite_coordinates_one_hot = sample.node[sample.edge_index]
        composite_coordinates = torch.argmax(composite_coordinates_one_hot, dim=2)
        # composite_coordinates = composite_coordinates_one_hot 
        grid_width = self.grid_shape[0]
        x = composite_coordinates % grid_width
        y = composite_coordinates // grid_width
        return x, y 
    
    def _get_edges_by_graph(self, sample):
        x, y = self._get_coordinates(sample)  
        nodes_per_graph = self.grid_shape[0] * self.grid_shape[1]
        graph_index_by_edge = sample.edge_index[0] // nodes_per_graph

        # store all edges in hash set for quick existence checking
        edges_by_graph = defaultdict(dict)
        for i in range(sample.edge_index.size(1)):
            start = (x[0][i].item(), y[0][i].item())
            end = (x[1][i].item(), y[1][i].item())
            edge = (start, end) if start < end else (end, start)
            graph_index = graph_index_by_edge[i].item()

            # store edge with its global indices (each edges appears twice)
            if edge not in edges_by_graph[graph_index]:
                edges_by_graph[graph_index][edge] = i
            else:
                old_index = edges_by_graph[graph_index][edge]
                edges_by_graph[graph_index][edge] = (old_index, i)
            
        return edges_by_graph

    
