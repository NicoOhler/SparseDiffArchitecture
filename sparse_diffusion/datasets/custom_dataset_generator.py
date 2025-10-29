import os
import shutil
import random
import csv

NUM_GRAPHS = 200
GRID_SIZE = 8
RECTANGLE_MAX_WIDTH = 3
RECTANGLE_MAX_HEIGHT = 2
LINE_MIN_LENGTH = 3
LINE_MAX_LENGTH = 5
TRIANGLE_MAX_BASE = 6
MAX_SHAPES_PER_GRAPH = 30

class CustomDatasetGenerator:
    def __call__(self, root):
        # delete folder recursively if it exists
        if os.path.exists(root):
            shutil.rmtree(root)

        os.makedirs(f"{root}/graphs", exist_ok=True)
        for i in range(1, NUM_GRAPHS + 1):
            filename = f"{root}/graphs/graph_{i}.csv"
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['x1', 'y1', 'x2', 'y2', 'weight'])

                # generate random shapes
                shapes = []
                num_shapes = random.randint(1, MAX_SHAPES_PER_GRAPH)
                shape_type = self._get_random_shape_type()
                for _ in range(num_shapes):
                    shapes.extend(self._generate_shape(shape_type=shape_type))

                # write edges without duplicates to csv
                seen_edges = set()
                for edge in shapes:
                    edge_key = (edge[0], edge[1], edge[2], edge[3])
                    if edge_key not in seen_edges:
                        writer.writerow(edge)
                        seen_edges.add(edge_key)       

    def _get_random_shape_type(self, choices=['rectangle', 'triangle', 'line']):
        return random.choice(choices)

    def _generate_shape(self, shape_type=None):
        if shape_type is None:
            shape_type = self._get_random_shape_type()
        if shape_type == 'rectangle':
            return self._generate_rectangle()
        elif shape_type == 'triangle':
            return self._generate_triangle()
        return self._generate_line()
    

    def _generate_line(self):
        horizontal = random.choice([True, False])
        length = random.randint(LINE_MIN_LENGTH, LINE_MAX_LENGTH)
        start = random.randint(0, GRID_SIZE - length - 1)
        end = start + length - 1
        fixed = random.randint(0, GRID_SIZE - 1)
        weight = round(random.uniform(0, 3), 1)
        edges = []
        for i in range(start, end):
            if horizontal:
                edges.append((i, fixed, i + 1, fixed, weight))
            else:
                edges.append((fixed, i, fixed, i + 1, weight))
        return edges
    
    def _generate_rectangle(self):
        width = random.randint(1, RECTANGLE_MAX_WIDTH)
        height = random.randint(1, RECTANGLE_MAX_HEIGHT)
        x_start = random.randint(0, GRID_SIZE - width - 1)
        y_start = random.randint(0, GRID_SIZE - height - 1)
        x_end = x_start + width
        y_end = y_start + height
        weight = round(random.uniform(0, 3), 1)
        edges = []
        for x in range(x_start, x_end):
            edges.append((x, y_start, x + 1, y_start, weight))
            edges.append((x, y_end, x + 1, y_end, weight))
        for y in range(y_start, y_end):
            edges.append((x_start, y, x_start, y + 1, weight))
            edges.append((x_end, y, x_end, y + 1, weight))
        return edges
    
    def _generate_triangle(self):
        base = random.randint(1, TRIANGLE_MAX_BASE // 2) * 2  # ensure base is even
        height = base // 2
        x_start = random.randint(0, GRID_SIZE - base - 1)
        y_start = random.randint(0, GRID_SIZE - height - 1)
        weight = round(random.uniform(0, 3), 1)
        edges = []

        for i in range(base // 2):
            edges.append((x_start + i, y_start, x_start + i + 1, y_start, weight)) # left base
            edges.append((x_start + base - i - 1, y_start, x_start + base - i, y_start, weight)) # right base
            edges.append((x_start + i, y_start + i, x_start + i, y_start + i + 1, weight)) # left diagonal
            edges.append((x_start + base - i, y_start + i, x_start + base - i - 1, y_start + i + 1, weight)) # right diagonal
        return edges