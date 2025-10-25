from __future__ import annotations 

from typing import Any, Literal

import networkx as nx

from ariel_experiments.characterize.canonical.canonical_node import (
    CanonicalNode,
)
from ariel_experiments.characterize.canonical.tree_exporter import TreeExporter


class TreeProcessor:
    """
    Contains stateless algorithms to transform a node tree into its
    canonical (standard) form.
    """

    @staticmethod
    def _canonicalize_child_order(node: CanonicalNode) -> None:
        if not node.radial_list:
            return

        max_priority = 0
        max_index = 0
        for i, child in enumerate(node.radial_list):
            if child:
                priority = child.full_priority
                if priority > max_priority:
                    max_priority = priority
                    max_index = i
        
        if max_priority == 0 or max_index == 0:
            return
        
        num_rotations = len(node.radial_list) - max_index
        node.rotation += num_rotations * node.config.max_allowed_rotations

    @staticmethod
    def _undo_node_rotation(node: CanonicalNode) -> CanonicalNode:
        # modifies in place
        swap_amt = node.rotation // node.config.max_allowed_rotations
        if swap_amt == 0:
            return node

        # Update this node's rotation
        node.rotation -= node.config.max_allowed_rotations * swap_amt

        axial_adjustment = swap_amt * node.config.max_allowed_rotations
        for child in node.axial_children:
            child.rotation += axial_adjustment

        # Handle radial children only if they exist
        if len(node.radial_list) > 0:
            shift = (node.config.radial_shift * swap_amt) % len(
                node.radial_list
            )
            radial_adjustments_to_apply = []
            for i, child in enumerate(node.radial_list):
                if child:
                    original_pos = (i - shift) % len(node.radial_list)
                    total_adjustment = 0
                    for step in range(swap_amt):
                        idx = (
                            original_pos + step * node.config.radial_shift
                        ) % len(
                            node.radial_list,
                        )
                        total_adjustment += node.config.radial_adjustments[idx]
                    radial_adjustments_to_apply.append((
                        child,
                        total_adjustment,
                    ))

            node.radial_list[:] = (
                node.radial_list[shift:] + node.radial_list[:shift]
            )
            for child, total_adjustment in radial_adjustments_to_apply:
                child.rotation += total_adjustment

        return node

    @classmethod
    def canonicalize(
        cls,
        node: CanonicalNode,
        *,
        zero_root_angle: bool = True,
        child_order: bool = True,
        return_copy: bool = True,
    ) -> CanonicalNode:
        if return_copy:
            node = node.copy_subtree()

        if child_order:
            cls._canonicalize_child_order(node)

        node.traverse_depth_first(
            visit_fn=[
                cls._undo_node_rotation,
            ],
        )

        if zero_root_angle:
            node.rotation = 0

        return node

    @classmethod
    def collect_subtrees(
        cls,
        node: CanonicalNode,
        output_type: Literal["graph", "string", "node"] = "graph",
        *,
        canonicalized: bool = True,
    ) -> list[nx.DiGraph[Any] | str]:
        node_to_process = cls.canonicalize(node) if canonicalized else node

        list_items = []

        # 1. Add the current node's full tree
        if output_type == "graph":
            list_items.append(TreeExporter.to_graph(node_to_process))
        elif output_type == "string":
            list_items.append(TreeExporter.to_string(node_to_process))
        elif output_type == "node":
            list_items.append(cls)

        for c in node_to_process.axial_list:
            if c is not None:
                list_items.extend(
                    cls.collect_subtrees(c, output_type=output_type),
                )
        for c in node_to_process.radial_list:
            if c is not None:
                list_items.extend(
                    cls.collect_subtrees(c, output_type=output_type),
                )

        return list_items

    @staticmethod
    def collect_node_neighbourhood(
        node: "CanonicalNode", radius: int = 5
    ) -> "CanonicalNode":
        def _expand_downward(
            current: CanonicalNode,
            canon_current: CanonicalNode,
            distance_from_center: int,
            radius: int,
        ) -> None:
            if distance_from_center >= radius or not current.has_children:
                return
            for face, child in current.children:
                canon_child = child.copy_node()
                canon_current[face] = canon_child
                _expand_downward(
                    child, canon_child, distance_from_center + 1, radius
                )

        def _expand_upward(
            current: CanonicalNode,
            canon_current: CanonicalNode,
            distance_from_center: int,
            radius: int,
        ) -> CanonicalNode:
            if distance_from_center >= radius or not current.parent:
                return canon_current

            parent = current.parent
            par_face = current.parent_attachment_face
            canon_parent = parent.copy_node()
            canon_current.attach_parent(par_face, canon_parent)

            if distance_from_center + 2 <= radius and parent.has_children:
                for face, sibling in parent.children:
                    if sibling is current:
                        continue
                    canon_sibling = sibling.copy_node()
                    canon_parent[face] = canon_sibling
                    _expand_downward(
                        sibling, canon_sibling, distance_from_center + 2, radius
                    )

            return _expand_upward(
                parent, canon_parent, distance_from_center + 1, radius
            )

        canon_center = node.copy_node()
        distance_from_center = 0
        _expand_downward(node, canon_center, distance_from_center, radius)
        return _expand_upward(node, canon_center, distance_from_center, radius)
    
    #todo move his out so no dependency on the exporter!!!
    @classmethod
    def collect_full_tree_neighbourhoods(
        cls,
        tree: CanonicalNode,
        output_type: Literal["string", "graph", "node"] = "string",
        *,
        max_radius: int = 5,
        canonicalized: bool = True,
    ) -> dict[int, list[Literal["string", "graph", "node"]]]:
        result_per_radius = {r: [] for r in range(max_radius)}

        for node in tree:
            for radius in range(max_radius):
                neighborhood = cls.collect_node_neighbourhood(node, radius)
                if canonicalized:
                    neighborhood = cls.canonicalize(
                        neighborhood, return_copy=False
                    )

                match output_type:
                    case "string":
                        result_per_radius[radius].append(
                            TreeExporter.to_string(neighborhood),
                        )
                    case "graph":
                        result_per_radius[radius].append(
                            TreeExporter.to_graph(neighborhood),
                        )
                    case "node":
                        result_per_radius[radius].append(neighborhood)

        return result_per_radius
