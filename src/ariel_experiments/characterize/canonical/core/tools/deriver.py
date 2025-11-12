from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ariel_experiments.characterize.canonical.core.node import CanonicalizableNode

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
    from ariel_experiments.characterize.canonical.core.node import (
        CanonicalizableNode,
    )


class TreeDeriver:
    _ARB_MAX_RADIUS = 50

    @staticmethod
    def _calc_highest_priority_child_face(
        node: CanonicalizableNode,
    ) -> ModuleFaces | None:
        """Returns the face where the child is attached with the higest priority."""
        if not node.has_children or len(node.config.radial_face_order) == 0:
            return None

        max_priority = 0
        winning_face = None
        for face, child in node.radial_children_items:
            if child.full_priority > max_priority:
                max_priority = child.full_priority
                winning_face = face

        return winning_face

    @classmethod
    def _canonicalize_child_order(cls, node: CanonicalizableNode) -> None:
        """How many shifts are needed to get 1 face to a different face?"""
        highest_face: ModuleFaces | None = (
            cls._calc_highest_priority_child_face(node)
        )
        if highest_face is None or highest_face == node.priority_face:
            return

        shift = node.calc_shift_face_to_target(highest_face, node.priority_face)
        node.shift_visual_rotation(shift)

    @staticmethod
    def _normalize_rotations(node: CanonicalizableNode) -> CanonicalizableNode:
        shift = node.internal_rotation // node.config.unique_rotation_amt
        node.shift_internal_rotation(shift)
        return node

    @classmethod
    def canonicalize(
        cls,
        node: CanonicalizableNode,
        *,
        zero_root_angle: bool = True,
        child_order: bool = True,
        return_copy: bool = True,
    ) -> CanonicalizableNode:
        if return_copy:
            node = node.copy()

        if child_order:
            cls._canonicalize_child_order(node)

        node.traverse_depth_first(
            visit_fn=[
                cls._normalize_rotations,
            ],
        )

        if zero_root_angle:
            node.rotate_amt(-node.internal_rotation)

        return node

    @classmethod
    def collect_subtrees(
        cls,
        node: CanonicalizableNode,
        serializer_fn: Callable[[CanonicalizableNode], Any] | None = None,
        *,
        canonicalized: bool = True,
    ) -> dict[int, list[str | nx.DiGraph[Any] | CanonicalizableNode]]:
        node_to_process = cls.canonicalize(node) if canonicalized else node
        list_dict = {}

        node_distance = cls._calculate_subtree_height(node_to_process)
        list_dict[node_distance] = []
        list_dict[node_distance].append(cls._serialize(node_to_process, serializer_fn))

        for child in node_to_process.axial_children:
            child_dict = cls.collect_subtrees(child, serializer_fn)
            for radius, items in child_dict.items():
                list_dict.setdefault(radius, []).extend(items)

        for child in node_to_process.radial_children:
            child_dict = cls.collect_subtrees(child, serializer_fn)
            for radius, items in child_dict.items():
                list_dict.setdefault(radius, []).extend(items)

        return list_dict

    @classmethod
    def _expand_downward(
        cls,
        current: CanonicalizableNode,
        canon_current: CanonicalizableNode,
        distance_from_center: int,
        radius: int,
    ) -> None:
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

            cls._expand_downward(
                child,
                canon_child,
                distance_from_center,
                radius,
            )

    @classmethod
    def _expand_upward(
        cls,
        current: CanonicalizableNode,
        canon_current: CanonicalizableNode,
        distance_from_center: int,
        radius: int,
    ) -> int | None:
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

                cls._expand_downward(
                    sibling,
                    canon_sibling,
                    distance_from_center + 1,
                    radius,
                )
        cls._expand_upward(
            parent,
            canon_parent,
            distance_from_center,
            radius,
        )
        return None

    @classmethod
    def _collect_node_neighbourhood(
        cls,
        node: CanonicalizableNode,
        radius: int,
    ) -> CanonicalizableNode:
        canon_center = node.copy(empty=True, copy_children=False)
        canon_center.tree_tags["CENTER"] = canon_center

        distance_from_center = 0

        cls._expand_downward(node, canon_center, distance_from_center, radius)
        cls._expand_upward(node, canon_center, distance_from_center, radius)

        while canon_center.parent:
            canon_center = canon_center.parent

        return canon_center

    # @classmethod
    # def collect_tree_neighbourhoods_old(
    #     cls,
    #     tree: CanonicalizableNode,
    #     serializer_fn: Callable[[CanonicalizableNode], Any] | None = None,
    #     *,
    #     max_radius: int = 10,
    #     canonicalized: bool = True,
    # ) -> dict[int, list[Any]]:
    #     result_per_radius = {r: [] for r in range(max_radius)}

    #     for node in tree:
    #         for radius in range(max_radius):
    #             neighborhood = cls._collect_node_neighbourhood(node, radius)

    #             if canonicalized:
    #                 neighborhood = cls.canonicalize(
    #                     neighborhood,
    #                     return_copy=False,
    #                 )

    #             serialized = cls._serialize(
    #                 neighborhood,
    #                 serializer_fn,
    #                 canonicalized=canonicalized,
    #                 return_copy=False,
    #             )
    #             result_per_radius[radius].append(serialized)

    #     return result_per_radius

    @staticmethod
    def _calculate_subtree_height(node: CanonicalizableNode) -> int:
        """Calculate the height (max depth) of a subtree rooted at the given node.

        Height is defined as the number of edges in the longest path from the node
        to any leaf. A leaf node has height 0.
        """
        if not node.has_children:
            return 0

        child_heights = []
        for child in node.children:
            height = TreeDeriver._calculate_subtree_height(child)
            child_heights.append(height + 1)

        return max(child_heights)

    @classmethod
    def _determine_max_distance(
        cls,
        tree_root: CanonicalizableNode,
    ) -> int:
        """Calculate the diameter (longest path) of a tree.

        Diameter is the longest path between any two nodes in the tree.
        """
        max_diameter = 0

        def _calc_diameter_at_node(node: CanonicalizableNode) -> None:
            nonlocal max_diameter

            if not node.has_children:
                return

            # Calculate heights of all children using the extracted function
            child_heights = []
            for child in node.children:
                height = cls._calculate_subtree_height(child)
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

    @classmethod
    def collect_neighbourhoods(
        cls,
        tree: CanonicalizableNode,
        serializer_fn: Callable[[CanonicalizableNode], Any] | None = None,
        *,
        use_node_max_radius: bool = False,
        tree_max_radius: int | None = None,
        canonicalized: bool = True,
    ) -> dict[int, list[Any]]:
        """
        use_node_max_radius:
            If True, uses each node's own local max radius (no node neighbourhood duplicates).
            If False, uses (global) tree_max_radius (can result in neighbourhood duplicates per node).
        tree_max_radius:
            If an int is given (e.g., 10), use it as a STATIC global radius.
            If None, CALCULATE a DYNAMIC global radius from the tree.
            (Only used when use_node_max_radius=False).
        """
        if tree_max_radius is None:
            tree_max_radius = cls._determine_max_distance(tree)

        max_distance_per_node = []
        result_per_radius = {r: [] for r in range(tree_max_radius + 1)}

        for node in tree:
            # Collect the full neighborhood once, up to an arbitrary limit
            neighborhood = cls._collect_node_neighbourhood(
                node,
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
                    cls._prune_neighborhood_to_radius(root, radius)

                serialized = cls._serialize(
                    root,
                    serializer_fn,
                    canonicalized=canonicalized,
                    return_copy=False,
                )
                result_per_radius[radius].append(serialized)


        return result_per_radius
        # if use_node_max_radius


        # if not use_node_max_radius:
        #     final_max_radius = (
        #         max(max_distance_per_node)
        #         if tree_max_radius is None
        #         else tree_max_radius
        #     )
        # elif not max_distance_per_node:
        #     tree_max_radius = -1
        # else:
        #     tree_max_radius = max(max_distance_per_node)

        # for radius in range(final_max_radius + 1, tree_max_radius):
        #     if radius in result_per_radius:
        #         result_per_radius.pop(radius)



    @classmethod
    def _prune_neighborhood_to_radius(
        cls,
        neighborhood: CanonicalizableNode,
        radius: int,
    ) -> None:
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
                sibling.detatch_from_parent()

    @classmethod
    def _serialize(
        cls,
        root: CanonicalizableNode,
        serializer_fn: Callable[[CanonicalizableNode], Any] | None = None,
        *,
        canonicalized: bool = True,
        return_copy: bool = True,
    ) -> CanonicalizableNode | Any:
        if canonicalized:
            root = cls.canonicalize(root, return_copy=return_copy)

        return serializer_fn(root) if serializer_fn else root


if __name__ == "__main__":
    from rich.console import Console

    from ariel_experiments.characterize.canonical.core.toolkit import (
        CanonicalToolKit as ctk,
    )
    from ariel_experiments.characterize.canonical.core.tools.serializer import (
        TreeSerializer,
    )
    from ariel_experiments.utils.initialize import (
        generate_random_individual,
    )

    from ariel_experiments.gui_vis.visualize_tree import visualize_tree_from_graph

    from ariel_experiments.gui_vis.view_mujoco import view

    console = Console()

    # * CANONICALIZATION -----

    robot = ctk.from_string('B[r(H2)]B2B2B5')

    robot.canonicalize()

    print(robot.to_string())

    print(robot.priority_face)

    view(robot.to_graph(), with_viewer=True)



    # * SUBTREES -----


    # individual = generate_random_individual(seed=41)

    # visualize_tree_from_graph(individual)

    # canonicalnode: CanonicalizableNode = ctk.from_graph(individual, auto_id=True)

    # subtree_dict = ctk.collect_subtrees(canonicalnode)

    # console.print(subtree_dict)


    # * NEIGHBOURHOOD -----

    # individual = generate_random_individual(num_modules=10, seed=41)

    # max_radius = 10

    # canonicalnode: CanonicalizableNode = ctk.from_graph(individual, auto_id=True)


    # # neighbours = TreeDeriver.collect_tree_neighbourhoods_old(
    # #     canonicalnode,
    # #     TreeSerializer.to_string,
    # #     max_radius=max_radius,
    # # )

    # # console.print(neighbours)

    # # console.rule()

    # neighbours_new = TreeDeriver.collect_neighbourhoods(
    #     canonicalnode,
    #     TreeSerializer.to_string,
    #     use_node_max_radius=False,
    #     tree_max_radius=None
    # )

    # console.print(neighbours_new)
