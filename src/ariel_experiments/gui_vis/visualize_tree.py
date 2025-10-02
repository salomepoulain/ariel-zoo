# Standard library
import operator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Third-party libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from rich.console import Console

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
SEED = 42

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)
DPI = 300


class FaceDirection(Enum):
    """used for spatial positioning for final drawing."""

    FRONT = 1
    LEFT = 2
    BOTTOM = 3
    RIGHT = 4
    TOP = 5
    BACK = 6


@dataclass
class NodeConfig:
    """template for node config poltting."""

    color: str
    shape: str
    label_color: str
    node_size: int = 3000
    linewidth: int = 2
    alpha: float = 1.0


@dataclass
class EdgeConfig:
    """template for edge config poltting."""

    width: int
    color: str
    style: str
    alpha: float


@dataclass
class VisualizationConfig:
    """config for plotting."""

    figure_size: tuple[int, int] = (20, 15)
    level_spacing: float = 4.0
    node_spacing: float = 6.0
    arrow_length: float = 2.5
    arrow_head_width: float = 0.35
    arrow_head_length: float = 0.2
    font_size_labels: int = 9
    font_size_edges: int = 12


# TODO: to be changed ??????
@dataclass
class NodeEdgeConfigs:
    """chosen values for the node and edge visualization using configs."""

    node_configs: dict[str, NodeConfig] = field(
        default_factory=lambda: {
            "CORE": NodeConfig("blue", "H", "white"),
            "BRICK": NodeConfig("green", "s", "white"),
            "HINGE": NodeConfig("white", "o", "black", linewidth=3),
            "NONE": NodeConfig("gray", "v", "white"),
            "UNKNOWN": NodeConfig("gray", "v", "white"),
        },
    )
    edge_configs: dict[str, EdgeConfig] = field(
        default_factory=lambda: {
            "TOP": EdgeConfig(6, "darkgray", "solid", 1.0),
            "BOTTOM": EdgeConfig(6, "darkgray", "dotted", 1.0),
            "LEFT": EdgeConfig(2, "darkgray", "solid", 0.8),
            "RIGHT": EdgeConfig(2, "darkgray", "solid", 0.8),
            "FRONT": EdgeConfig(3, "black", "solid", 1.0),
            "BACK": EdgeConfig(2, "darkgray", "solid", 0.8),
            "UNKNOWN": EdgeConfig(2, "darkgray", "solid", 0.8),
        },
    )


def _extract_spanning_tree(graph: nx.Graph, root: int) -> nx.DiGraph:
    """Extract spanning tree from graph using BFS with face ordering.

    Parameters
    ----------
    graph : nx.Graph
        Input graph
    root : int
        Root node for tree extraction

    Returns
    -------
    nx.DiGraph
        Directed spanning tree
    """
    actual_root: int = (
        next(iter(graph.nodes())) if root not in graph.nodes() else root
    )

    tree: nx.DiGraph = nx.DiGraph()
    visited: set[int] = {actual_root}
    queue: list[int] = [actual_root]
    tree.add_node(actual_root, **graph.nodes[actual_root])

    while queue:
        current: int = queue.pop(0)
        potential_children: list[tuple[int, int, dict[str, Any], int]] = []

        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                edge_data: dict[str, Any] = graph.get_edge_data(
                    current, neighbor,
                )
                face: str = edge_data.get("face", "UNKNOWN")
                angle: int = FaceDirection[face].value
                potential_children.append((current, neighbor, edge_data, angle))

        potential_children.sort(key=operator.itemgetter(3))

        for parent, child, edge_data, _ in potential_children:
            if child not in visited:
                visited.add(child)
                queue.append(child)
                tree.add_node(child, **graph.nodes[child])
                tree.add_edge(parent, child, **edge_data)

    return tree


def _position_level(
    tree: nx.DiGraph,
    nodes: list[int],
    pos: dict[int, tuple[float, float]],
    level: int,
    root: int,
    config: VisualizationConfig,
) -> dict[int, tuple[float, float]]:
    """Position nodes in a specific level.

    Parameters
    ----------
    tree : nx.DiGraph
        Tree structure
    nodes : list[int]
        Nodes to position at this level
    pos : dict[int, tuple[float, float]]
        Existing node positions
    level : int
        Current level number
    root : int
        Root node
    config : VisualizationConfig
        Visualization configuration

    Returns
    -------
    dict[int, tuple[float, float]]
        New positions for nodes at this level
    """
    parent_children: dict[int, list[int]] = {}
    for node in nodes:
        if (parents := list(tree.predecessors(node))):
            parent: int = parents[0]
            parent_children.setdefault(parent, []).append(node)

    for parent, children in parent_children.items():
        children_with_faces: list[tuple[int, int]] = [
            (
                child,
                FaceDirection[
                    (
                        tree.get_edge_data(parent, child, {}).get(
                            "face", "UNKNOWN",
                        )
                    )
                ].value,
            )
            for child in children
        ]
        children_with_faces.sort(key=operator.itemgetter(1))
        parent_children[parent] = [child for child, _ in children_with_faces]

    all_children: list[int] = [
        child for children in parent_children.values() for child in children
    ]

    parent_positions: list[tuple[float, int]] = sorted(
        [(pos[parent][0], parent) for parent in parent_children],
    )

    total_children: int = len(all_children)
    total_width: float = (
        (total_children - 1) * config.node_spacing if total_children > 1 else 0
    )

    current_x: float = -total_width / 2
    y_level: float = pos[root][1] - level * config.level_spacing

    level_pos: dict[int, tuple[float, float]] = {}
    for _, parent in parent_positions:
        for child in parent_children[parent]:
            level_pos[child] = (current_x, y_level)
            current_x += config.node_spacing

    return level_pos


def _create_tree_layout(
    tree: nx.DiGraph, root: int, config: VisualizationConfig,
) -> dict[int, tuple[float, float]]:
    """Create spatial layout for the tree.

    Parameters
    ----------
    tree : nx.DiGraph
        Input tree to layout
    root : int
        Root node
    config : VisualizationConfig
        Visualization configuration

    Returns
    -------
    dict[int, tuple[float, float]]
        Node positions as {node_id: (x, y)}
    """
    pos: dict[int, tuple[float, float]] = {}
    levels: dict[int, int] = nx.single_source_shortest_path_length(tree, root)

    level_nodes: dict[int, list[int]] = {}
    for node, level in levels.items():
        level_nodes.setdefault(level, []).append(node)

    pos[root] = (0, 0)

    for level in sorted(level_nodes.keys())[1:]:
        pos.update(
            _position_level(tree, level_nodes[level], pos, level, root, config),
        )

    return pos


def _create_node_labels(
    tree: nx.DiGraph,
) -> tuple[dict[int, str], dict[str, list[int]]]:
    """Create node labels and group nodes by type.

    Parameters
    ----------
    tree : nx.DiGraph
        Input tree

    Returns
    -------
    tuple[dict[int, str], dict[str, list[int]]]
        Node labels and nodes grouped by type
    """
    node_types: dict[int, str] = dict(nx.get_node_attributes(tree, "type"))
    node_rotations: dict[int, str] = dict(
        nx.get_node_attributes(tree, "rotation"),
    )
    node_configs = NodeEdgeConfigs().node_configs

    node_labels: dict[int, str] = {}
    nodes_by_type: dict[str, list[int]] = {
        node_type: [] for node_type in node_configs
    }

    for node in tree.nodes():
        node_type: str = node_types.get(node, "UNKNOWN")
        node_rotation: str = node_rotations.get(node, "UNKNOWN")

        if node_rotation != "UNKNOWN":
            node_rotation = node_rotation.replace("DEG_", "")

        node_labels[node] = f"{node}\n{node_type}\n{node_rotation}°"
        nodes_by_type[node_type].append(node)

    return node_labels, nodes_by_type


def _draw_rotation_arrows(
    tree: nx.DiGraph,
    pos: dict[int, tuple[float, float]],
    config: VisualizationConfig,
) -> None:
    """Draw rotation arrows for nodes.

    Parameters
    ----------
    tree : nx.DiGraph
        Tree structure
    pos : dict[int, tuple[float, float]]
        Node positions
    config : VisualizationConfig
        Visualization configuration
    """
    node_rotations: dict[int, str] = nx.get_node_attributes(tree, "rotation")  # type: ignore[misc]

    for node in tree.nodes():
        if node not in pos:
            continue

        x, y = pos[node]
        node_rotation: str = node_rotations.get(node, "UNKNOWN")

        if node_rotation in {"UNKNOWN", "DEG_0"}:
            continue

        angle_deg: float = float(node_rotation.replace("DEG_", ""))
        angle_rad: float = np.radians(angle_deg)

        dx: float = config.arrow_length * np.cos(angle_rad)
        dy: float = config.arrow_length * np.sin(angle_rad)
        plt.arrow(
            x - dx / 2,
            y - dy / 2,
            dx,
            dy,  # type: ignore[misc]
            head_width=config.arrow_head_width,
            head_length=config.arrow_head_length,
            fc="black",
            ec="black",
            alpha=0.9,
            zorder=1,
            linewidth=2,
        )


def _draw_nodes(
    tree: nx.DiGraph,
    pos: dict[int, tuple[float, float]],
    node_labels: dict[int, str],
    nodes_by_type: dict[str, list[int]],
    config: VisualizationConfig,
) -> None:
    """Draw all nodes grouped by type.

    Parameters
    ----------
    tree : nx.DiGraph
        Tree structure
    pos : dict[int, tuple[float, float]]
        Node positions
    node_labels : dict[int, str]
        Node label strings
    nodes_by_type : dict[str, list[int]]
        Nodes grouped by type
    config : VisualizationConfig
        Visualization configuration
    """
    node_configs = NodeEdgeConfigs().node_configs

    for node_type, node_config in node_configs.items():
        nodes: list[int] = nodes_by_type[node_type]
        if not nodes:
            continue

        nx.draw_networkx_nodes(  # type: ignore[misc]
            tree,
            pos,
            nodelist=nodes,
            node_color=node_config.color,
            node_shape=node_config.shape,
            node_size=node_config.node_size,
            edgecolors="black",
            linewidths=node_config.linewidth,
            alpha=node_config.alpha,
        )

        type_labels: dict[int, str] = {
            node: node_labels[node] for node in nodes
        }

        nx.draw_networkx_labels(  # type: ignore[misc]
            tree,
            pos,
            labels=type_labels,
            font_size=config.font_size_labels,
            font_weight="bold",
            font_color=node_config.label_color,
        )


def _draw_edges(tree: nx.DiGraph, pos: dict[int, tuple[float, float]]) -> None:
    """Draw all edges grouped by face type.

    Parameters
    ----------
    tree : nx.DiGraph
        Tree structure
    pos : dict[int, tuple[float, float]]
        Node positions
    """
    edge_configs = NodeEdgeConfigs().edge_configs

    edges_by_face: dict[str, list[tuple[int, int]]] = {}
    for u, v, data in tree.edges(data=True):
        face: str = data.get("face", "UNKNOWN")
        edges_by_face.setdefault(face, []).append((u, v))

    for face, edges in edges_by_face.items():
        if edges:
            edge_config: EdgeConfig = edge_configs.get(
                face, edge_configs["UNKNOWN"],
            )
            nx.draw_networkx_edges(  # type: ignore[misc]
                tree,
                pos,
                edgelist=edges,
                width=edge_config.width,
                edge_color=edge_config.color,
                style=edge_config.style,
                alpha=edge_config.alpha,
            )


def _draw_edge_labels(
    tree: nx.DiGraph,
    pos: dict[int, tuple[float, float]],
    config: VisualizationConfig,
) -> None:
    """Draw edge labels.

    Parameters
    ----------
    tree : nx.DiGraph
        Tree structure
    pos : dict[int, tuple[float, float]]
        Node positions
    config : VisualizationConfig
        Visualization configuration
    """
    edge_labels: dict[tuple[int, int], str] = {
        (u, v): data.get("face", "F?") for u, v, data in tree.edges(data=True)
    }
    nx.draw_networkx_edge_labels(  # type: ignore[misc]
        tree,
        pos,
        edge_labels=edge_labels,
        font_size=config.font_size_edges,
    )


def visualize_tree_from_graph(
    graph: nx.Graph,
    root: int = 0,
    *,
    title: str = "",
    save_file: Path | str | None = None,
    config: VisualizationConfig | None = None,
) -> None:
    """Visualize a robot graph as a tree structure.

    Parameters
    ----------
    graph : nx.Graph
        Input graph representing robot structure
    root : int, optional
        Root node for tree extraction, by default 0
    save_file : Path | str | None, optional
        Path to save the figure, by default None
    config : VisualizationConfig | None, optional
        Visualization configuration, by default None

    Returns
    -------
    nx.DiGraph
        Extracted spanning tree from the input graph
    """
    config = config or VisualizationConfig()

    tree: nx.DiGraph = _extract_spanning_tree(graph, root)
    pos: dict[int, tuple[float, float]] = _create_tree_layout(
        tree, root, config,
    )

    plt.figure(figsize=config.figure_size)  # type: ignore[misc]

    node_labels, nodes_by_type = _create_node_labels(tree)

    _draw_rotation_arrows(tree, pos, config)
    _draw_nodes(tree, pos, node_labels, nodes_by_type, config)
    _draw_edges(tree, pos)
    _draw_edge_labels(tree, pos, config)

    plt.tight_layout()
    # plt.axis("equal")  # type: ignore[misc]

    plt.title(title)

    if save_file:
        path = DATA / Path(save_file)
        console.log("saving file")
        plt.savefig(str(path), dpi=DPI)  # type: ignore[misc]

    plt.show()  # type: ignore[misc]


if __name__ == "__main__":
    from ariel_experiments.utils.initialize import generate_random_individual

    graph = generate_random_individual()
    visualize_tree_from_graph(graph, save_file="test.png")
