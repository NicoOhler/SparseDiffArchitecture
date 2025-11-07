import os
import shutil
import random
import csv

class ShapeDatasetGenerator:
    def __init__(self, cfg):
        self.choices = cfg.dataset.shape_types
        self.grid_height, self.grid_width = cfg.dataset.grid_shape
        self.rectangle_max_width = int(self.grid_width * 0.8)
        self.rectangle_max_height = int(self.grid_height * 0.8)
        self.max_shapes_per_graph = self.grid_height * self.grid_width // 2
        self.num_graphs = cfg.dataset.num_graphs
        self.weight_range = cfg.dataset.weight_range

    def generate(self, root):
        # delete folder recursively if it exists
        if os.path.exists(root):
            shutil.rmtree(root)

        os.makedirs(f"{root}/graphs", exist_ok=True)
        for i in range(1, self.num_graphs + 1):
            filename = f"{root}/graphs/graph_{i}.csv"
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['x1', 'y1', 'x2', 'y2', 'weight'])

                # generate random shapes
                shapes = []
                num_shapes = random.randint(1, self.max_shapes_per_graph)
                weight = round(random.uniform(*self.weight_range), 1)
                for _ in range(num_shapes):
                    shapes.extend(self._generate_shape(weight=weight))

                # write edges without duplicates to csv
                seen_edges = set()
                for edge in shapes:
                    edge_key = (edge[0], edge[1], edge[2], edge[3])
                    if edge_key not in seen_edges:
                        writer.writerow(edge)
                        seen_edges.add(edge_key)       

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