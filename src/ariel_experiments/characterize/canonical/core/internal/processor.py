from __future__ import annotations

from typing import Any, Literal

import networkx as nx

from ariel_experiments.characterize.canonical.core.node import (
    CanonicalizableNode,
)
from ariel_experiments.characterize.canonical.core.internal.serializer import TreeSerializer #TODO: make not dependent on this


class TreeProcessor:
    """
    Contains stateless algorithms to transform a node tree into its
    canonical (standard) form.
    """

    @staticmethod
    def _canonicalize_child_order(node: CanonicalizableNode) -> None:
        highest_face = node.highest_priority_child_face

        if highest_face is None or highest_face == node.priority_face:
            return

        shift = node.calc_shift_face_to_target(highest_face, node.priority_face)
        node.shift_radial_children_local(shift)

    @staticmethod
    def _normalize_rotations(node: CanonicalizableNode) -> CanonicalizableNode:
        shift = node.global_rotation_state
        node.shift_radial_children_local(shift)
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
            node.local_rotation_state = 0

        return node

    #todo make it not dependend on the serializer
    @classmethod
    def collect_subtrees(
        cls,
        node: CanonicalizableNode,
        output_type: Literal["graph", "string", "node"] = "graph",
        *,
        canonicalized: bool = True,
    ) -> list[str | nx.DiGraph[Any] | CanonicalizableNode]:
        node_to_process = cls.canonicalize(node) if canonicalized else node

        list_items = []

        # 1. Add the current node's full tree
        if output_type == "graph":
            list_items.append(TreeSerializer.to_graph(node_to_process))
        elif output_type == "string":
            list_items.append(TreeSerializer.to_string(node_to_process))
        elif output_type == "node":
            list_items.append(cls)

        for c in node_to_process.axial_children:
            list_items.extend(
                cls.collect_subtrees(c, output_type=output_type),
            )

        for c in node_to_process.radial_children:
            list_items.extend(
                cls.collect_subtrees(c, output_type=output_type),
            )

        return list_items

    # @staticmethod
    # def _collect_node_neighbourhood(
    #     node: "CanonicalizableNode", radius: int = 5
    # ) -> "CanonicalizableNode":
    #     def _expand_downward(
    #         current: CanonicalizableNode,
    #         canon_current: CanonicalizableNode,
    #         distance_from_center: int,
    #         radius: int,
    #     ) -> None:
    #         if distance_from_center >= radius or not current.has_children:
    #             return
    #         for face, child in current.children_items:
    #             canon_child = child.copy_node_single()
    #             canon_current[face] = canon_child
    #             _expand_downward(
    #                 child, canon_child, distance_from_center + 1, radius
    #             )

    #     def _expand_upward(
    #         current: CanonicalizableNode,
    #         canon_current: CanonicalizableNode,
    #         distance_from_center: int,
    #         radius: int,
    #     ) -> CanonicalizableNode:
    #         if distance_from_center >= radius or not current.parent:
    #             return canon_current

    #         parent = current.parent
    #         par_face = current.parent_attachment_face
    #         canon_parent = parent.copy_node_single()
    #         canon_current.attach_parent(par_face, canon_parent)

    #         if distance_from_center + 2 <= radius and parent.has_children:
    #             for face, sibling in parent.children_items:
    #                 if sibling is current:
    #                     continue
    #                 canon_sibling = sibling.copy_node_single()
    #                 canon_parent[face] = canon_sibling
    #                 _expand_downward(
    #                     sibling, canon_sibling, distance_from_center + 2, radius
    #                 )

    #         return _expand_upward(
    #             parent, canon_parent, distance_from_center + 1, radius
    #         )

    #     canon_center = node.copy_node_single()
    #     distance_from_center = 0
    #     _expand_downward(node, canon_center, distance_from_center, radius)
    #     return _expand_upward(node, canon_center, distance_from_center, radius)


    # @classmethod
    # def collect_full_tree_neighbourhoods(
    #     cls,
    #     tree: CanonicalizableNode,
    #     output_type: Literal["string", "graph", "node"] = "string",
    #     *,
    #     max_radius: int = 5,
    #     canonicalized: bool = True,
    # ) -> dict[int, list[Literal["string", "graph", "node"]]]:
    #     result_per_radius = {r: [] for r in range(max_radius)}

    #     for node in tree:
    #         for radius in range(max_radius):
    #             neighborhood: CanonicalizableNode = cls._collect_node_neighbourhood(node, radius)
    #             if canonicalized:
    #                 neighborhood = cls.canonicalize(
    #                     neighborhood, return_copy=False
    #                 )

    #             match output_type:
    #                 case "string":
    #                     result_per_radius[radius].append(
    #                         TreeSerializer.to_string(neighborhood),
    #                     )
    #                 case "graph":
    #                     result_per_radius[radius].append(
    #                         TreeSerializer.to_graph(neighborhood),
    #                     )
    #                 case "node":
    #                     result_per_radius[radius].append(neighborhood)

    #     return result_per_radius


    @staticmethod
    def _collect_node_neighbourhood(
        node: "CanonicalizableNode", radius: int = 5
    ) -> "CanonicalizableNode":
        def _expand_downward(
            current: CanonicalizableNode,
            canon_current: CanonicalizableNode,
            distance_from_center: int,
            radius: int,
        ) -> None:
            if distance_from_center >= radius or not current.has_children:
                return
            for face, child in current.children_items:
                canon_child = child.copy_node_single()
                canon_current[face] = canon_child
                _expand_downward(
                    child, canon_child, distance_from_center + 1, radius
                )

        def _expand_upward(
            current: CanonicalizableNode,
            canon_current: CanonicalizableNode,
            distance_from_center: int,
            radius: int,
        ) -> CanonicalizableNode:
            if distance_from_center >= radius or not current.parent:
                return canon_current

            parent = current.parent
            par_face = current.parent_attachment_face
            canon_parent = parent.copy_node_single()
            canon_current.attach_parent(par_face, canon_parent)

            if distance_from_center + 2 <= radius and parent.has_children:
                for face, sibling in parent.children_items:
                    if sibling is current:
                        continue
                    canon_sibling = sibling.copy_node_single()
                    canon_parent[face] = canon_sibling
                    _expand_downward(
                        sibling, canon_sibling, distance_from_center + 2, radius
                    )

            return _expand_upward(
                parent, canon_parent, distance_from_center + 1, radius
            )

        canon_center = node.copy_node_single()
        distance_from_center = 0
        _expand_downward(node, canon_center, distance_from_center, radius)
        return _expand_upward(node, canon_center, distance_from_center, radius)


    #todo make it not dependend on the serializer
    @classmethod
    def collect_tree_neighbourhoods(
        cls,
        tree: CanonicalizableNode,
        output_type: Literal["string", "graph", "node"] = "string",
        *,
        max_radius: int = 5,
        canonicalized: bool = True,
    ) -> dict[int, list[str | nx.DiGraph[Any] | CanonicalizableNode]]:
        result_per_radius = {r: [] for r in range(max_radius)}

        for node in tree:
            # for radius in range(max_radius):
            neighborhood: CanonicalizableNode = cls._collect_node_neighbourhood(node, max_radius)
            # for radius in range(max_radius - 1, max_radius):
            for i in range(max_radius):
                # add useful tags to quickly travel 1 parent down, and remove corresponding children,
                # end then canonicalize without (copying?)
                # reduces the complexity grately!!!!

                if canonicalized:
                    neighborhood = cls.canonicalize(
                        neighborhood, return_copy=False
                    )

            match output_type:
                case "string":
                    result_per_radius[radius].append(
                        TreeSerializer.to_string(neighborhood),
                    )
                case "graph":
                    result_per_radius[radius].append(
                        TreeSerializer.to_graph(neighborhood),
                    )
                case "node":
                    result_per_radius[radius].append(neighborhood)

        return result_per_radius
