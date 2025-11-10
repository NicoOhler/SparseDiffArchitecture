import os

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
import rdkit.Chem
import imageio
import networkx as nx
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sparse_diffusion.metrics.molecular_metrics import SparseMolecule


class Visualizer:
    def __init__(self, dataset_infos, grid_shape=None):
        self.dataset_infos = dataset_infos
        self.grid_shape = grid_shape
        self.is_molecular = self.dataset_infos.is_molecular

        if self.is_molecular:
            self.remove_h = dataset_infos.remove_h

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

    def visualize_non_molecule(self, graph, path, pos=None, node_size=100, largest_component=False, custom_dataset=False):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = self._get_node_positions(graph) if custom_dataset else nx.spring_layout(graph, seed=0)

        if custom_dataset:
            vmin, vmax = None, None
            cmap = None
            node_color = "grey"
        else:
            # Set node colors based on the eigenvectors
            _, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
            vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
            m = max(np.abs(vmin), vmax)
            vmin, vmax = -m, m
            node_color = U[:, 1]
            cmap = plt.cm.coolwarm

        plt.figure()
        nx.draw(
            graph,
            pos,
            edgelist=sorted(graph.edges()),
            font_size=5,
            node_size=node_size,
            with_labels=False,
            node_color=node_color,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edge_color=self._get_edge_colors(graph) if custom_dataset else "grey",
            width=self._get_edge_widths(graph) if custom_dataset else 1
        )

        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def _get_node_positions(self, graph):
        # place the nodes according to their node features
        # node feature = composite index of x, y coordinates
        # idx = x * grid_width + y 
        grid_width, _ = self.grid_shape
        pos = {}
        for j, node_type in enumerate(graph.nodes(data=True)):
            composite_coord = node_type[1]["symbol"].item()
            x = composite_coord // grid_width
            y = composite_coord % grid_width
            pos[j] = (y, x)
        return pos

    def _get_edge_colors(self, graph):
        color_by_weight = {
            1: "salmon",
            2: "red",
            3: "darkred"
        }
        edge_weights = [graph.get_edge_data(u, v)["color"] for u, v in sorted(graph.edges())]
        if not len(edge_weights):
            return "grey"
        return [color_by_weight.get(weight, "grey") for weight in edge_weights]
    
    def _get_edge_widths(self, graph):
        width_by_weight = {
            1: 1.0,
            2: 2.0,
            3: 3.0
        }
        edge_weights = [graph.get_edge_data(u, v)["color"] for u, v in sorted(graph.edges())]
        print(f"Sum of edge weights: {sum(edge_weights)}")
        if not len(edge_weights):
            return None
        return [width_by_weight.get(weight, 1) for weight in edge_weights]
    
    def visualize(
        self, path: str, graphs: list, num_graphs_to_visualize: int, log="graph", local_rank=0
    ):
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        num_graphs = max(graphs.batch) + 1
        num_graphs_to_visualize = min(num_graphs_to_visualize, num_graphs)
        print(f"Visualizing {num_graphs_to_visualize} graphs out of {num_graphs}")

        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, "graph_{}_{}.png".format(i, local_rank))
            node_mask = graphs.batch == i
            edge_mask = graphs.batch[graphs.edge_index[0]] == i

            if self.is_molecular:
                # TODO: change graph lists to a list of PlaceHolders
                molecule = SparseMolecule(
                    node_types=graphs.node[node_mask].long(),
                    bond_index=graphs.edge_index[:, edge_mask].long() - graphs.ptr[i],
                    bond_types=graphs.edge_attr[edge_mask].long(),
                    atom_decoder=self.dataset_infos.atom_decoder,
                    charge=None,
                )
                mol = molecule.rdkit_mol
                try:
                    Draw.MolToFile(mol, file_path)
                except rdkit.Chem.KekulizeException:
                    print("Can't kekulize molecule")
            else:
                graph = self.to_networkx(
                    node=graphs.node[node_mask].long().cpu().numpy(),
                    edge_index=(graphs.edge_index[:, edge_mask].long() - graphs.ptr[i])
                    .cpu()
                    .numpy(),
                    edge_attr=graphs.edge_attr[edge_mask].long().cpu().numpy(),
                )

                custom_dataset = self.dataset_infos.dataset_name == "custom"
                self.visualize_non_molecule(graph, file_path, custom_dataset=custom_dataset)

            if wandb.run is not None and log is not None:
                if i < 3:
                    print(f"Saving {file_path} to wandb")
                wandb.log({log: [wandb.Image(file_path)]}, commit=False)

    def visualize_chain(self, chain_path, batch_id, chain, local_rank):
        node_list = chain.node_list
        edge_index_list = chain.edge_index_list
        edge_attr_list = chain.edge_attr_list
        batch = chain.batch
        ptr = chain.ptr

        keep_chain = int(chain.batch.max() + 1)
        custom_dataset = self.dataset_infos.dataset_name == "custom"

        for k in range(keep_chain):
            path = os.path.join(chain_path, f"graph_{batch_id + k}_{local_rank}")
            if not os.path.exists(path):
                os.makedirs(path)

            # get the list for molecules
            if self.is_molecular:
                mols = []
                for i in range(len(node_list)):
                    node_mask = batch == k
                    edge_mask = batch[edge_index_list[i][0]] == k
                    mol = SparseMolecule(
                        node_types=node_list[i][node_mask].long(),
                        bond_index=edge_index_list[i][:, edge_mask].long() - ptr[k],
                        bond_types=edge_attr_list[i][edge_mask].long(),
                        atom_decoder=self.dataset_infos.atom_decoder,
                        charge=None,
                    ).rdkit_mol
                    mols.append(mol)

                # find the coordinates of atoms in the final molecule
                final_molecule = mols[-1]
                AllChem.Compute2DCoords(final_molecule)
                coords = []
                for i, atom in enumerate(final_molecule.GetAtoms()):
                    positions = final_molecule.GetConformer().GetAtomPosition(i)
                    coords.append((positions.x, positions.y, positions.z))

                # align all the molecules
                for i, mol in enumerate(mols):
                    AllChem.Compute2DCoords(mol)
                    conf = mol.GetConformer()
                    for j, atom in enumerate(mol.GetAtoms()):
                        x, y, z = coords[j]
                        conf.SetAtomPosition(j, Point3D(x, y, z))

                save_paths = []
                num_frames = len(node_list)

                for frame in range(num_frames):
                    file_name = os.path.join(path, "frame_{}.png".format(frame))
                    Draw.MolToFile(
                        mols[frame], file_name, size=(300, 300), legend=f"Frame {frame}"
                    )
                    save_paths.append(file_name)

            else:
                # generate graphs for each intermediate sample
                graphs = []
                for i in range(len(node_list)):
                    node_mask = batch == k
                    edge_mask = batch[edge_index_list[i][0]] == k
                    graph = self.to_networkx(
                        node=node_list[i][node_mask].long().cpu().numpy(),
                        edge_index=(
                            edge_index_list[i][:, edge_mask].long() - ptr[k]
                        )
                        .cpu()
                        .numpy(),
                        edge_attr=edge_attr_list[i][edge_mask].long().cpu().numpy(),
                    )
                    graphs.append(graph)

                # visualize and save each frame
                save_paths = []
                num_frames = len(node_list)
                for frame in range(num_frames):
                    file_name = os.path.join(path, "frame_{}.png".format(frame))
                    self.visualize_non_molecule(graphs[frame], file_name, custom_dataset=custom_dataset)
                    save_paths.append(file_name)

            # create gif
            print("\r{}/{} complete".format(k + 1, keep_chain), end="", flush=True)
            imgs = [imageio.v3.imread(fn) for fn in save_paths]
            gif_path = os.path.join(
                os.path.dirname(path), "{}_{}.gif".format(path.split("/")[-1], local_rank)
            )
            imgs.extend([imgs[-1]] * 10)
            imageio.mimsave(gif_path, imgs, subrectangles=True, duration=200)
            if wandb.run is not None:
                wandb.log(
                    {"chain": [wandb.Video(gif_path, caption=gif_path, format="gif")]}, commit=False
                )
                print(f"Saving {gif_path} to wandb")
                wandb.log(
                    {"chain": wandb.Video(gif_path, fps=8, format="gif")}, commit=False
                )
