import os
import shutil
import csv
import random

class RandomWalkGenerator:
    def __init__(self, cfg):
        self.grid_height, self.grid_width = cfg.dataset.grid_shape
        self.min_edges = cfg.dataset.min_edges
        self.max_edges = cfg.dataset.max_edges
        self.num_graphs = cfg.dataset.num_graphs
        self.weight_range = cfg.dataset.weight_range

    def _is_valid(self, x, y):
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    def _generate_connected_graph(self):
        num_edges = random.randint(self.min_edges, self.max_edges)
        edges = set()
        visited_nodes = set()
        start_x, start_y = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
        visited_nodes.add((start_x, start_y))
        
        # We use a growing frontier (stack) to ensure all paths are explored 
        # while maintaining connectivity.
        frontier = [(start_x, start_y)]
        edge_count = 0

        while frontier and edge_count < num_edges:
            # 2. Pick a node from the frontier to expand (can be DFS-like or random)
            node_to_expand_idx = random.randint(0, len(frontier) - 1)
            px, py = frontier.pop(node_to_expand_idx)
            
            # Possible neighbors (4-neighborhood: up, down, left, right)
            neighbors = []
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = px + dx, py + dy
                if self._is_valid(nx, ny):
                    neighbors.append((nx, ny))
            
            random.shuffle(neighbors)
            
            for nx, ny in neighbors:
                if edge_count >= num_edges:
                    break
                    
                # 3. If neighbor is unvisited, create an edge and add the neighbor to the frontier
                if (nx, ny) not in visited_nodes:
                    visited_nodes.add((nx, ny))
                    frontier.append((nx, ny))
                    weight = round(random.uniform(*self.weight_range), 1)
                    edges.add((px, py, nx, ny, weight))
                    edge_count += 1
        
        return list(edges)

    def generate(self, root):
        print(f"Generating {self.num_graphs} connected graphs...")
        
        graphs_dir = f"{root}/graphs"
        if os.path.exists(graphs_dir):
            shutil.rmtree(graphs_dir)
        os.makedirs(graphs_dir, exist_ok=True)

        for i in range(1, self.num_graphs + 1):
            filename = f"{graphs_dir}/graph_{i}.csv"
            
            edges_list = self._generate_connected_graph()
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['x1', 'y1', 'x2', 'y2', 'weight'])
                writer.writerows(edges_list)
        
        print(f"Generation complete. Files saved in {graphs_dir}")