import os
import shutil
import csv
import random
import tqdm

def get_dataset_generator(cfg):
    if cfg.dataset.type == 'shape':
        return ShapeDatasetGenerator(cfg)
    elif cfg.dataset.type == 'random_walk':
        return RandomWalkGenerator(cfg)
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset.type}")
        
class DatasetGeneratorBase:
    def __init__(self, cfg):
        self.type = cfg.dataset.type
        self.num_graphs = cfg.dataset.num_graphs
        self.weight_range = cfg.dataset.weight_range
        self.delete_existing = cfg.dataset.regenerate_dataset
        self.grid_height, self.grid_width = cfg.dataset.grid_shape

    def generate(self, path):
        graphs_dir = self.handle_existing_dataset(path)
        if not graphs_dir:
            return
        
        print(f"Generating {self.num_graphs} graphs of type {self.type}...")
        weight_distribution = []
        for i in tqdm.tqdm(range(1, self.num_graphs + 1)):
            edges_list = self._generate_graph()
            weight_sum = sum([edge[4] for edge in edges_list])
            weight_distribution.append(weight_sum)
            with open(f"{graphs_dir}/graph_{i}.csv", mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['x1', 'y1', 'x2', 'y2', 'weight'])
                writer.writerows(edges_list)
        
        print(f"Generation complete. Files saved in {graphs_dir}")
        min_weight = min(weight_distribution)
        max_weight = max(weight_distribution)
        avg_weight = sum(weight_distribution) / len(weight_distribution)
        print(f"Weight distribution across graphs - Min: {min_weight:.2f}, Max: {max_weight:.2f}, Avg: {avg_weight:.2f}")
    
    def handle_existing_dataset(self, path):
        graphs_dir = f"{path}/graphs"
        if os.path.exists(graphs_dir):
            if not self.delete_existing:
                print(f"Directory {graphs_dir} already exists. Skipping generation.")
                return None
            print(f"Directory {graphs_dir} already exists. Deleting existing dataset.")
            shutil.rmtree(path)
            
        os.makedirs(graphs_dir, exist_ok=True)
        return graphs_dir

    def _generate_graph(self):
        raise NotImplementedError("Subclasses should implement this method.")


class RandomWalkGenerator(DatasetGeneratorBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.min_edges = cfg.dataset.min_edges
        self.max_edges = cfg.dataset.max_edges
        self.directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),   # 4-neighborhood
            (1, 1), (1, -1), (-1, 1), (-1, -1), # diagonals
            (2, 1), (1, -2), (-2, -1), (-1, 2), 
            (1, 2), (2, -1), (-1, -2), (-2, 1)  # knight moves
        ]
        self.lookup_table = generate_edge_crossings_lookup_table()

    def _generate_graph(self):
        num_edges = random.randint(self.min_edges, self.max_edges)
        edges = set()
        visited_nodes = set()
        start_x, start_y = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
        visited_nodes.add((start_x, start_y))
        
        edge_count = 0
        while edge_count < num_edges:
            # pick a random node and a random direction
            source_node_x, source_node_y = random.choice(list(visited_nodes))
            direction_x, direction_y = random.choice(self.directions)
            
            # check if the new edge is valid
            target_node_x, target_node_y = source_node_x + direction_x, source_node_y + direction_y
            if not self._is_within_bounds(target_node_x, target_node_y):
                continue
            if self._causes_edge_crossing(source_node_x, source_node_y, direction_x, direction_y, edges):
                continue
            edge = (source_node_x, source_node_y, target_node_x, target_node_y)
            reverse_edge = (target_node_x, target_node_y, source_node_x, source_node_y)
            if edge in edges or reverse_edge in edges:
                continue
            
            # add the edge
            visited_nodes.add((target_node_x, target_node_y))
            edges.add(edge)
            edge_count += 1

        weighted_edges = []
        for (x1, y1, x2, y2) in edges:
            weight = round(random.uniform(*self.weight_range), 1)
            weighted_edges.append((x1, y1, x2, y2, weight))

        # assign weights to edges based on their lengths
        """
        weighted_edges = []
        weight_by_length = {
            1: 1.0,
            (2 ** 0.5): 2.0,
            (5 ** 0.5): 3.0
        }
        for (x1, y1, x2, y2) in edges:
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            weighted_edges.append((x1, y1, x2, y2, weight_by_length[length]))
        """

        return weighted_edges
   
    def _is_within_bounds(self, x, y):
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height
    
    def _causes_edge_crossing(self, start_x, start_y, direction_x, direction_y, edges):
        candidate_edges = self.lookup_table[(direction_x, direction_y)]
        for source_offset, target_offset in candidate_edges:
            # determine new edge positions
            source = (start_x + source_offset[0], start_y + source_offset[1])
            target = (start_x + target_offset[0], start_y + target_offset[1])
            edge = (source[0], source[1], target[0], target[1])
            reverse_edge = (target[0], target[1], source[0], source[1])
            if edge in edges or reverse_edge in edges:
                return True
        return False

class ShapeDatasetGenerator(DatasetGeneratorBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.choices = cfg.dataset.shape_types
        self.rectangle_max_width = int(self.grid_width * 0.8)
        self.rectangle_max_height = int(self.grid_height * 0.8)
        self.max_shapes_per_graph = self.grid_height * self.grid_width // 2

    def _generate_graph(self):
        # generate random shapes
        shapes = []
        num_shapes = random.randint(1, self.max_shapes_per_graph)
        weight = round(random.uniform(*self.weight_range), 1)
        for _ in range(num_shapes):
            shapes.extend(self._generate_shape(weight=weight))
        return shapes

    def _get_random_shape_type(self):
        return random.choice(self.choices)

    def _generate_shape(self, shape_type=None, weight=None):
        if shape_type is None:
            shape_type = self._get_random_shape_type()
        if weight is None:
            weight = round(random.uniform(*self.weight_range), 1)

        if shape_type == 'rectangle':
            return self._generate_rectangle(weight=weight)
        elif shape_type == 'triangle':
            return self._generate_triangle(weight=weight)
        return self._generate_line(weight=weight)
    
    def _generate_rectangle(self, weight=None):
        width = random.randint(1, self.rectangle_max_width)
        height = random.randint(1, self.rectangle_max_height)
        x_start = random.randint(0, self.grid_width - width - 1)
        y_start = random.randint(0, self.grid_height - height - 1)
        x_end = x_start + width
        y_end = y_start + height
        if weight is None:
            weight = round(random.uniform(*self.weight_range), 1)

        edges = []
        for x in range(x_start, x_end):
            edges.append((x, y_start, x + 1, y_start, weight))
            edges.append((x, y_end, x + 1, y_end, weight))
        for y in range(y_start, y_end):
            edges.append((x_start, y, x_start, y + 1, weight))
            edges.append((x_end, y, x_end, y + 1, weight))
        return edges
    
    """
    LINE_MIN_LENGTH = 3
    LINE_MAX_LENGTH = 5
    TRIANGLE_MAX_BASE = 6

    def _generate_line(self, weight=None):
        horizontal = random.choice([True, False])
        length = random.randint(LINE_MIN_LENGTH, LINE_MAX_LENGTH)
        start = random.randint(0, GRID_SIZE - length - 1)
        end = start + length - 1
        fixed = random.randint(0, GRID_SIZE - 1)
        if weight is None:
            weight = round(random.uniform(0, 3), 1)

        edges = []
        for i in range(start, end):
            if horizontal:
                edges.append((i, fixed, i + 1, fixed, weight))
            else:
                edges.append((fixed, i, fixed, i + 1, weight))
        return edges
    
    def _generate_triangle(self, weight=None):
        base = random.randint(1, TRIANGLE_MAX_BASE // 2) * 2  # ensure base is even
        height = base // 2
        x_start = random.randint(0, GRID_SIZE - base - 1)
        y_start = random.randint(0, GRID_SIZE - height - 1)
        if weight is None:
            weight = round(random.uniform(0, 3), 1)

        edges = []
        for i in range(base // 2):
            edges.append((x_start + i, y_start, x_start + i + 1, y_start, weight)) # left base
            edges.append((x_start + base - i - 1, y_start, x_start + base - i, y_start, weight)) # right base
            edges.append((x_start + i, y_start + i, x_start + i, y_start + i + 1, weight)) # left diagonal
            edges.append((x_start + base - i, y_start + i, x_start + base - i - 1, y_start + i + 1, weight)) # right diagonal
        return edges
    """

def generate_edge_crossings_lookup_table():
    # constants
    up = (0, 1)
    down = (0, -1)
    left = (-1, 0)
    right = (1, 0)

    up_right = (1, 1)
    up_left = (-1, 1)
    down_right = (1, -1)

    up_up_right = (1, 2)
    right_right_up = (2, 1)

    # manually determined edge crossings for base moves
    up_edge_crossings = [
        (up_right, left),
        (up_left, right)
    ]

    up_right_edge_crossings = [
        (up, right),
        (up, (2, 0)),       
        ((0, 2), right),   
        (up_left, right),
        (up, down_right)
    ]

    right_right_up_edge_crossings = [
        (up_right, right), # vertical part
        (up, right), 
        (up_right, (2, 0)), # two diagonals
        (up_left, right),
        (up, (2, 0)),
        (up_right, (3, 0)), # three knight moves (two right, one down)
        ((0, 2), right),
        (up, down_right),
        (up_right, (2, -1)), 
        ((1, 2), (2, 0)),    # four knight moves (one right, two down)
        (down, up_right),
        (right, (2, 2))      # two knight moves (one right, two up)
    ]

    # add reverses of edge crossings
    up_edge_crossings += [(b, a) for (a, b) in up_edge_crossings]
    up_right_edge_crossings += [(b, a) for (a, b) in up_right_edge_crossings]
    right_right_up_edge_crossings += [(b, a) for (a, b) in right_right_up_edge_crossings]

    # obtain remaining edge crossings via rotations and reflections
    def rot90(v):  return (v[1], -v[0])
    def rot180(v): return (-v[0], -v[1])
    def rot270(v): return (-v[1], v[0])
    def reflect_xy(v): return (v[1], v[0])  # reflection across y=x

    def transform_crossings(crossings, f):
        return [(f(a), f(b)) for (a, b) in crossings]

    def add_rotations(base_move, crossings, table):
        for f in [lambda x: x, rot90, rot180, rot270]:
            new_move = f(base_move)
            new_crossings = transform_crossings(crossings, f)
            table[new_move] = new_crossings

    # construct lookup table
    lookup_table = {}
    add_rotations(up, up_edge_crossings, lookup_table) # regular 4-neighborhood
    add_rotations(up_right, up_right_edge_crossings, lookup_table) # diagonals
    add_rotations(right_right_up, right_right_up_edge_crossings, lookup_table) # 2-horizontal + 1-vertical knight move 

    # derive 1-horizontal + 2-vertical knight moves via reflection 
    up_up_right = reflect_xy(right_right_up)
    up_up_right_edge_crossings = transform_crossings(right_right_up_edge_crossings, reflect_xy)
    add_rotations(up_up_right, up_up_right_edge_crossings, lookup_table)

    return lookup_table