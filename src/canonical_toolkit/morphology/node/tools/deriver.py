from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..node import Node

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces


__all__ = [
    "canonicalize",
    "collect_subtrees",
    "collect_neighbourhoods",
]


# def _calc_highest_priority_child_face(
#     node: Node,
# ) -> ModuleFaces | None:
#     """Return the face where the child is attached with the highest priority."""
#     raise DeprecationWarning
#     if not node.has_children or len(node.config.radial_face_order) == 0:
#         return None

#     max_priority = 0
#     winning_face = None
#     for face, child in node.radial_children_items:
#         if child.full_priority > max_priority:
#             max_priority = child.full_priority
#             winning_face = face

#     return winning_face


def _calc_highest_priority_child_face(
    node: Node,
) -> ModuleFaces | None:
    """Return the face where the child is attached with the highest priority."""
    if not node.has_children or len(node.config.radial_face_order) == 0:
        return None

    items = [(face, child) for face, child in node.radial_children_items]

    if len(items) == 0:
        return None

    child_strings = [child.to_string() for face, child in items]
    n = len(child_strings)

    best_rotation = 0
    best_sequence = tuple(child_strings)

    for shift in range(1, n):
        sequence = tuple(child_strings[(shift + i) % n] for i in range(n))
        if sequence > best_sequence:
            best_sequence = sequence
            best_rotation = shift

    return items[best_rotation][0]


def _canonicalize_child_order(node: Node) -> None:
    """How many shifts are needed to get 1 face to a different face?"""
    highest_face: ModuleFaces | None = _calc_highest_priority_child_face(node)
    if highest_face is None or highest_face == node.priority_face:
        return

    shift = node.calc_shift_face_to_target(highest_face, node.priority_face)
    node.shift_visual_rotation(shift)


def _normalize_rotations(node: Node) -> Node:
    """Normalize node rotations to canonical form."""
    shift = node.internal_rotation // node.config.unique_rotation_amt
    node.shift_internal_rotation(shift)
    return node


def canonicalize(
    starting_node: Node,
    *,
    zero_root_angle: bool = True,
    child_order: bool = True,
    return_copy: bool = True,
) -> Node:
    """Canonicalize a node tree to a standardized form."""
    if return_copy:
        starting_node = starting_node.copy()

    starting_node.traverse_depth_first(
        visit_fn=[
            _normalize_rotations,
        ],
    )

    if child_order:
        _canonicalize_child_order(starting_node)

    if zero_root_angle:
        starting_node.rotate_amt(-starting_node.internal_rotation)

    return starting_node


def collect_subtrees(
    starting_node: Node,
    serializer_fn: Callable[[Node], Any] | None = None,
    *,
    canonicalized: bool = True,
) -> dict[int, list[str | nx.DiGraph[Any] | Node]]:
    """Collect all subtrees from a node tree."""
    node_to_process = (
        canonicalize(starting_node) if canonicalized else starting_node
    )
    list_dict = {}

    node_distance = _calculate_subtree_height(node_to_process)
    list_dict[node_distance] = []
    list_dict[node_distance].append(_serialize(node_to_process, serializer_fn))

    for child in node_to_process.axial_children:
        child_dict = collect_subtrees(child, serializer_fn)
        for radius, items in child_dict.items():
            list_dict.setdefault(radius, []).extend(items)

    for child in node_to_process.radial_children:
        child_dict = collect_subtrees(child, serializer_fn)
        for radius, items in child_dict.items():
            list_dict.setdefault(radius, []).extend(items)

    return list_dict


def _expand_downward(
    current: Node,
    canon_current: Node,
    distance_from_center: int,
    radius: int,
) -> None:
    """Expand neighborhood downward (toward children)."""
    if distance_from_center + 1 > radius or not current.has_children:
        canon_current.tree_tags["MAX_D"] = max(
            canon_current.tree_tags.get("MAX_D", 0),
            distance_from_center,
        )
        return

    distance_from_center += 1
    for face, child in current.children_items:
        canon_child = child.copy(empty=True, copy_children=False)
        canon_current[face] = canon_child

        hash_str = f"dist_{distance_from_center}_CHILD"
        canon_current.tree_tags.setdefault(hash_str, []).append(canon_child)

        _expand_downward(
            child,
            canon_child,
            distance_from_center,
            radius,
        )


def _expand_upward(
    current: Node,
    canon_current: Node,
    distance_from_center: int,
    radius: int,
) -> int | None:
    """Expand neighborhood upward (toward parents)."""
    if distance_from_center + 1 > radius or not current.parent:
        canon_current.tree_tags["MAX_D"] = max(
            canon_current.tree_tags.get("MAX_D", 0),
            distance_from_center,
        )
        return distance_from_center

    distance_from_center += 1
    parent = current.parent
    par_face = current.parent_attachment_face
    canon_parent = parent.copy(empty=True, copy_children=False)
    canon_parent[par_face] = canon_current

    hash_str: str = f"dist_{distance_from_center}_PARENT"
    canon_current.tree_tags[hash_str] = canon_parent

    if distance_from_center + 1 <= radius and parent.has_children:
        for face, sibling in parent.children_items:
            if sibling is current:
                continue
            canon_sibling = sibling.copy(empty=True, copy_children=False)
            canon_parent[face] = canon_sibling

            hash_str = f"dist_{distance_from_center + 1}_CHILD"
            canon_current.tree_tags.setdefault(hash_str, []).append(
                canon_sibling,
            )

            _expand_downward(
                sibling,
                canon_sibling,
                distance_from_center + 1,
                radius,
            )
    _expand_upward(
        parent,
        canon_parent,
        distance_from_center,
        radius,
    )
    return None


def _collect_node_neighbourhood(
    node: Node,
    radius: int,
) -> Node:
    """Collect the neighborhood around a single node up to given radius."""
    canon_center = node.copy(empty=True, copy_children=False)
    canon_center.tree_tags["CENTER"] = canon_center

    distance_from_center = 0

    _expand_downward(node, canon_center, distance_from_center, radius)
    _expand_upward(node, canon_center, distance_from_center, radius)

    while canon_center.parent:
        canon_center = canon_center.parent

    return canon_center


def _calculate_subtree_height(node: Node) -> int:
    """Calculate the height (max depth) of a subtree rooted at the given node.

    Height is defined as the number of edges in the longest path from the node
    to any leaf. A leaf node has height 0.
    """
    if not node.has_children:
        return 0

    child_heights = []
    for child in node.children:
        height = _calculate_subtree_height(child)
        child_heights.append(height + 1)

    return max(child_heights)


def _determine_max_distance(tree_root: Node) -> int:
    """Calculate the diameter (longest path) of a tree.

    Diameter is the longest path between any two nodes in the tree.
    """
    max_diameter = 0

    def _calc_diameter_at_node(node: Node) -> None:
        nonlocal max_diameter

        if not node.has_children:
            return

        # Calculate heights of all children using the extracted function
        child_heights = []
        for child in node.children:
            height = _calculate_subtree_height(child)
            child_heights.append(height + 1)

        child_heights.sort(reverse=True)

        # Calculate diameter through this node
        if len(child_heights) >= 2:
            diameter_through_node = child_heights[0] + child_heights[1]
        elif len(child_heights) == 1:
            diameter_through_node = child_heights[0]
        else:
            diameter_through_node = 0

        max_diameter = max(max_diameter, diameter_through_node)

        # Recursively check all children
        for child in node.children:
            _calc_diameter_at_node(child)

    _calc_diameter_at_node(tree_root)
    return max_diameter


def collect_neighbourhoods(
    starting_node: Node,
    serializer_fn: Callable[[Node], Any] | None = None,
    *,
    min_radius: int = 0,  # TODO
    use_node_max_radius: bool = True,
    tree_max_radius: int | None = None,
    canonicalized: bool = True,
    do_radius_prefix: bool = True,
    hash_prefix: str | None = None,
    # empty_value: str | None = None
) -> dict[int, list[Any]]:
    """
    Collect neighborhoods around each node in the tree, strating at starting_node

    use_node_max_radius:
        If True, uses each node's own local max radius (no node neighbourhood duplicates).
        If False, uses (global) tree_max_radius (can result in neighbourhood duplicates per node).
    tree_max_radius:
        If an int is given (e.g., 10), use it as a STATIC global radius.
        If None, CALCULATE a DYNAMIC global radius from the tree.
        (Only used when use_node_max_radius=False).
    """
    if tree_max_radius is None:
        tree_max_radius = _determine_max_distance(starting_node)

    max_distance_per_node = []
    result_per_radius = {r: [] for r in range(tree_max_radius + 1)}

    for starting_node in starting_node:
        # Collect the full neighborhood once, up to an arbitrary limit
        neighborhood = _collect_node_neighbourhood(
            starting_node,
            tree_max_radius,
        )
        node_max_dist = neighborhood.tree_tags["MAX_D"]

        max_distance_per_node.append(node_max_dist)

        if use_node_max_radius:
            loop_max_dist = min(node_max_dist, tree_max_radius)
        else:
            loop_max_dist = tree_max_radius

        for radius in range(loop_max_dist, -1, -1):
            if radius == 0:
                root = neighborhood.tree_tags["CENTER"]
                root.detatch_children()
            else:
                parent_key = f"dist_{radius}_PARENT"
                root = neighborhood.tree_tags.get(parent_key, neighborhood)
                _prune_neighborhood_to_radius(root, radius)

            serialized = _serialize(
                root,
                serializer_fn,
                canonicalized=canonicalized,
                return_copy=False,
            )

            if isinstance(serialized, str):
                if do_radius_prefix:
                    serialized = f"r{radius}__" + serialized

                if hash_prefix:
                    serialized = hash_prefix + serialized


            result_per_radius[radius].append(serialized)

    return result_per_radius


def _prune_neighborhood_to_radius(
    neighborhood: Node,
    radius: int,
) -> None:
    """Prune neighborhood tree to only include nodes within given radius."""
    if radius == 0:
        return

    # Get children at current radius boundary
    current_level_key = f"dist_{radius}_CHILD"
    current_level_children = neighborhood.tree_tags.get(current_level_key)

    # Get children at next level (siblings beyond radius)
    next_level_key = f"dist_{radius + 1}_CHILD"
    next_level_children = neighborhood.tree_tags.get(next_level_key, [])

    # Remove descendants from current level boundary children (fast)
    if current_level_children:
        for child in current_level_children:
            child.detatch_children()
            if child in next_level_children:
                next_level_children.remove(child)

    # Detach siblings not connected to previous child
    siblings = neighborhood.tree_tags.get(next_level_key)
    if siblings:
        for sibling in siblings:
            sibling.detatch_from_parent()  # this returns the parent


def _serialize(
    root: Node,
    serializer_fn: Callable[[Node], Any] | None = None,
    *,
    canonicalized: bool = True,
    return_copy: bool = True,
) -> Node | Any:
    """Serialize a node tree using the provided serializer function."""
    if canonicalized:
        root = canonicalize(root, return_copy=return_copy)

    return serializer_fn(root) if serializer_fn else root


if __name__ == "__main__":
    from rich.console import Console

    import canonical_toolkit as ctk
    from ariel_experiments.gui_vis.view_mujoco import view
    from ariel_experiments.gui_vis.visualize_tree import (
        visualize_tree_from_graph,
    )
    from ariel_experiments.utils.initialize import (
        generate_random_individual,
    )

    from .serializer import (
        to_graph,
        to_string,
    )

    console = Console()

    # # * CANONICALIZATION -----

    # robot = ctk.from_string('B[r(H2)]B2B2B5')

    # robot.canonicalize()

    # print(robot.to_string())

    # print(robot.priority_face)

    # view(robot.to_graph(), with_viewer=True)

    # * SUBTREES -----

    # individual = generate_random_individual(seed=41)

    # visualize_tree_from_graph(individual)

    # canonicalnode: Node = ctk.from_graph(individual, auto_id=True)

    # subtree_dict = ctk.collect_subtrees(canonicalnode)

    # console.print(subtree_dict)

    # * NEIGHBOURHOOD -----

    individual = generate_random_individual(num_modules=30, seed=42)

    img = view(individual, return_img=True)
    img.show()

    max_radius = 10

    canonicalnode: Node = ctk.from_graph(individual, auto_id=True)

    console.print(canonicalnode.to_string())

    img = view(canonicalnode.to_graph(), return_img=True)
    img.show()

    neighbours_new = collect_neighbourhoods(
        canonicalnode,
        to_string,
        use_node_max_radius=True,
        tree_max_radius=None,
    )

    img = view(canonicalnode.to_graph(), return_img=True)
    img.show()

    console.print(canonicalnode.to_string())

    console.print(neighbours_new)
