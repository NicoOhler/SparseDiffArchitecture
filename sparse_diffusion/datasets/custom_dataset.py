import os
import pathlib
import os.path as osp
import numpy as np
import csv
import torch
import math
import networkx as nx

import torch_geometric.utils
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, download_url
from hydra.utils import get_original_cwd

from sparse_diffusion.utils import PlaceHolder
from sparse_diffusion.datasets.abstract_dataset import (
    AbstractDataModule,
    AbstractDatasetInfos,
)
from sparse_diffusion.datasets.dataset_utils import (
    load_pickle,
    save_pickle,
    Statistics,
    to_list,
)
from sparse_diffusion.metrics.metrics_utils import (
    node_counts,
    atom_type_counts,
    edge_counts,
)

import networkx as nx
import matplotlib.pyplot as plt
import sparse_diffusion.datasets.random_walk_dataset_generator as dataset_generator

class CustomDataset(InMemoryDataset):
    def __init__(
        self,
        dataset_name,
        split,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        visualize=False,
        grid_shape=(8, 8),
    ):
        self.dataset_name = dataset_name

        self.split = split
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2

        self.grid_height, self.grid_width = grid_shape
        self.visualize = visualize
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            node_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
        )


    @property
    def raw_file_names(self):
        return ["train.pt", "val.pt", "test.pt"]
        # return ["train.pkl", "val.pkl", "test.pkl"]

    @property
    def split_file_name(self):
        return ["train.pt", "val.pt", "test.pt"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.split == "train":
            return [
                f"train.pt",
                f"train_n.pickle",
                f"train_node_types.npy",
                f"train_bond_types.npy",
            ]
        elif self.split == "val":
            return [
                f"val.pt",
                f"val_n.pickle",
                f"val_node_types.npy",
                f"val_bond_types.npy",
            ]
        else:
            return [
                f"test.pt",
                f"test_n.pickle",
                f"test_node_types.npy",
                f"test_bond_types.npy",
            ]

    def download(self):
        graphs_folder = osp.join(self.root, "graphs")
        if not osp.exists(graphs_folder):
            print(f"{graphs_folder} does not exist. Please store the graph files there.")
            raise FileNotFoundError

        graph_files = [f for f in os.listdir(graphs_folder) if f.endswith('.csv')]
        if not graph_files:
            raise FileNotFoundError(f"No graph files found in {graphs_folder}")
        
        adjs = []
        os.makedirs(osp.join(self.root, "graph_visualizations"), exist_ok=True)
        for file in graph_files:
            with open(osp.join(graphs_folder, file), 'r') as f:
                reader = csv.reader(f)
                reader.__next__()
                n = self.grid_width * self.grid_height
                adj = np.zeros((n, n))
                for row in reader:
                    # todo handle extra attributes
                    x1, y1, x2, y2, weight = int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4])
                    idx1 = self._get_index(x1, y1)
                    idx2 = self._get_index(x2, y2)
                    weight = self._discretize_weight(weight)
                    adj[idx1, idx2], adj[idx2, idx1] = weight, weight
                if self.visualize:
                    self.create_visualization(adj, file)      
                adjs.append(torch.from_numpy(adj))

        g_cpu = torch.Generator()
        g_cpu.manual_seed(1234)
        self.num_graphs = len(adjs)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        indices = torch.randperm(self.num_graphs, generator=g_cpu)
        print(f"Dataset sizes: train {train_len}, val {val_len}, test {test_len}")
        train_indices = indices[:train_len]
        val_indices = indices[train_len : train_len + val_len]
        test_indices = indices[train_len + val_len :]

        print(f"Train indices: {train_indices}")
        print(f"Val indices: {val_indices}")
        print(f"Test indices: {test_indices}")
        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append(adj)
            if i in val_indices:
                val_data.append(adj)
            if i in test_indices:
                test_data.append(adj)

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        raw_dataset = torch.load(os.path.join(self.raw_dir, "{}.pt".format(self.split)))
        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]

            # represent x, y coordinates as single composite node feature
            X = torch.arange(n, dtype=torch.long).unsqueeze(-1)
            
            # permutate nodes for permutation invariance
            random_order = torch.randperm(adj.shape[-1])
            adj = adj[random_order, :]
            adj = adj[:, random_order]
            X = X[random_order, :].squeeze()

            # include weights as edge attributes
            edge_index, edge_attr = torch_geometric.utils.dense_to_sparse(adj)
            n_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(
                x=X.long(), edge_index=edge_index, edge_attr=edge_attr.long(), n_nodes=n_nodes
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        num_nodes = node_counts(data_list) 
        node_type_distribution = atom_type_counts(data_list, num_classes=n) # single class to represent x/y coordinates
        edge_type_distribution = edge_counts(data_list, num_bond_types=4) # weights: 0, 1, 2, 3 (0 means no edge)

        torch.save(self.collate(data_list), self.processed_paths[0])
        save_pickle(num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], node_type_distribution)
        np.save(self.processed_paths[3], edge_type_distribution)

    def _get_index(self, x, y):
        assert x >= 0 and x < self.grid_width, "x coordinate out of bounds: {}".format(x)
        assert y >= 0 and y < self.grid_height, "y coordinate out of bounds: {}".format(y)
        return x * self.grid_width + y

    def _discretize_weight(self, weight):
        assert weight >= 0 and weight <= 3.0, "Weight out of bounds [0, 3]: {}".format(weight)
        weight = int(math.ceil(weight))
        return weight if weight > 0 else 1
    
    def create_visualization(self, adj, filename):
        graph = nx.from_numpy_array(adj)
        pos = {}
        for j, node_type in enumerate(graph.nodes(data=True)):
            composite_coord = node_type[0]
            x = composite_coord // self.grid_width
            y = composite_coord % self.grid_width
            pos[j] = (x, y)

        # color edges according to weights
        edge_weights = [graph.get_edge_data(u, v)["weight"] for u, v in graph.edges()]
        norm = plt.Normalize(vmin=1, vmax=3)
        edge_color = [plt.cm.viridis(norm(weight)) for weight in edge_weights]

        plt.figure()
        nx.draw(graph, pos, font_size=5, node_size=100, with_labels=False, node_color="grey", edge_color=edge_color)
        plt.tight_layout()
        plt.savefig(osp.join(self.root, "graph_visualizations", f"{filename[:-4]}.png"))
        plt.close("all")
    

class CustomDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_name = self.cfg.dataset.name
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(get_original_cwd()).parents[0]
        root_path = os.path.join(base_path, self.datadir)
        dataset_generator.RandomWalkGenerator(cfg).generate(root_path)

        datasets = {
            "train": CustomDataset(
                dataset_name=self.cfg.dataset.name,
                split="train",
                root=root_path,
                visualize=True,
            ),
            "val": CustomDataset(
                dataset_name=self.cfg.dataset.name,
                split="val",
                root=root_path,
            ),
            "test": CustomDataset(
                dataset_name=self.cfg.dataset.name,
                split="test",
                root=root_path,
            ),
        }

        self.statistics = {
            "train": datasets["train"].statistics,
            "val": datasets["val"].statistics,
            "test": datasets["test"].statistics,
        }

        super().__init__(cfg, datasets)
        super().prepare_dataloader()
        self.inner = self.train_dataset


class CustomDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule):
        self.is_molecular = False
        self.spectre = True
        self.use_charge = False
        self.dataset_name = datamodule.dataset_name
        self.node_types = datamodule.inner.statistics.node_types
        self.bond_types = datamodule.inner.statistics.bond_types
        super().complete_infos(
            datamodule.statistics, len(datamodule.inner.statistics.node_types)
        )
        self.input_dims = PlaceHolder(
            X=len(self.node_types), E=len(self.bond_types), y=0, charge=0
        )
        self.output_dims = PlaceHolder(
            X=len(self.node_types), E=len(self.bond_types), y=0, charge=0
        )
        self.statistics = {
            'train': datamodule.statistics['train'],
            'val': datamodule.statistics['val'],
            'test': datamodule.statistics['test']
        }

    def to_one_hot(self, data):
        data.x = F.one_hot(data.x, num_classes=self.num_node_types).float()
        data.charge = data.x.new_zeros((*data.x.shape[:-1], 0))
        if data.y is None:
            data.y = data.x.new_zeros((data.batch.max().item()+1, 0))
        data.edge_attr = F.one_hot(data.edge_attr, num_classes=self.num_edge_types).float()

        return data
