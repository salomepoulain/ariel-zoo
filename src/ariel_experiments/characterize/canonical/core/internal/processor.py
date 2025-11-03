from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import networkx as nx

    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
    from ariel_experiments.characterize.canonical.core.node import (
        CanonicalizableNode,
    )


class TreeProcessor:
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
            node.internal_rotation = 0

        return node

    @classmethod
    def collect_subtrees(
        cls,
        node: CanonicalizableNode,
        serializer_fn: Callable[[CanonicalizableNode], Any] | None = None,
        *,
        canonicalized: bool = True,
    ) -> list[str | nx.DiGraph[Any] | CanonicalizableNode]:
        node_to_process = cls.canonicalize(node) if canonicalized else node
        list_items = []
        list_items.append(cls._serialize(node_to_process, serializer_fn))

        for child in node_to_process.axial_children:
            list_items.extend(
                cls.collect_subtrees(child, serializer_fn),
            )

        for child in node_to_process.radial_children:
            list_items.extend(
                cls.collect_subtrees(child, serializer_fn),
            )

        return list_items

    @classmethod
    def _expand_downward(
        cls,
        current: CanonicalizableNode,
        canon_current: CanonicalizableNode,
        distance_from_center: int,
        radius: int,
    ) -> None:
        if distance_from_center + 1 > radius or not current.has_children:
            return

        distance_from_center += 1
        for face, child in current.children_items:
            canon_child = child.copy(empty=True, copy_children=False)
            canon_current[face] = canon_child

            hash_str = f"r_{radius}_dist_{distance_from_center}_CHILD"
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
    ) -> CanonicalizableNode:
        if distance_from_center + 1 > radius or not current.parent:
            return canon_current

        distance_from_center += 1
        parent = current.parent
        par_face = current.parent_attachment_face
        canon_parent = parent.copy(empty=True, copy_children=False)
        canon_current.attach_parent(par_face, canon_parent)

        hash_str: str = f"r_{radius}_dist_{distance_from_center}_PARENT"
        canon_current.tree_tags[hash_str] = canon_parent

        if distance_from_center + 1 <= radius and parent.has_children:
            for face, sibling in parent.children_items:
                if sibling is current:
                    continue
                canon_sibling = sibling.copy(empty=True, copy_children=False)
                canon_parent[face] = canon_sibling

                hash_str = f"r_{radius}_dist_{distance_from_center + 1}_CHILD"
                canon_current.tree_tags.setdefault(hash_str, []).append(
                    canon_sibling,
                )

                cls._expand_downward(
                    sibling,
                    canon_sibling,
                    distance_from_center + 1,
                    radius,
                )

        return cls._expand_upward(
            parent,
            canon_parent,
            distance_from_center,
            radius,
        )

    @classmethod
    def _collect_node_neighbourhood(
        cls,
        node: CanonicalizableNode,
        radius: int = 5,
    ) -> CanonicalizableNode:
        canon_center = node.copy(empty=True, copy_children=False)
        canon_center.tree_tags["CENTER"] = canon_center

        distance_from_center = 0

        cls._expand_downward(node, canon_center, distance_from_center, radius)
        return cls._expand_upward(
            node,
            canon_center,
            distance_from_center,
            radius,
        )

    @classmethod
    def collect_tree_neighbourhoods_old(
        cls,
        tree: CanonicalizableNode,
        serializer_fn: Callable[[CanonicalizableNode], Any] | None = None,
        *,
        max_radius: int = 10,
        canonicalized: bool = True,
    ) -> dict[int, list[Any]]:
        result_per_radius = {r: [] for r in range(max_radius)}

        for node in tree:
            for radius in range(max_radius):
                neighborhood: CanonicalizableNode = (
                    cls._collect_node_neighbourhood(node, radius)
                )

                if canonicalized:
                    neighborhood = cls.canonicalize(
                        neighborhood,
                        return_copy=False,
                    )

                serialized = cls._serialize(
                    neighborhood,
                    serializer_fn,
                    canonicalized=canonicalized,
                    return_copy=False,
                )
                result_per_radius[radius].append(serialized)

        return result_per_radius

    @classmethod
    def collect_neighbourhoods(
        cls,
        tree: CanonicalizableNode,
        serializer_fn: Callable[[CanonicalizableNode], Any] | None = None,
        *,
        max_radius: int = 10,
        canonicalized: bool = True,
    ) -> dict[int, list[Any]]:
        result_per_radius = {r: [] for r in range(max_radius)}

        for node in tree:
            neighborhood = cls._collect_node_neighbourhood(node, max_radius - 1)

            for radius in range(max_radius - 1, -1, -1):
                if radius == 0:
                    root = neighborhood.tree_tags["CENTER"]
                    root.detatch_children()
                else:
                    parent_key = f"r_{max_radius - 1}_dist_{radius}_PARENT"
                    root = neighborhood.tree_tags.get(parent_key, neighborhood)
                    cls._prune_neighborhood_to_radius(root, radius, max_radius)

                serialized = cls._serialize(
                    root,
                    serializer_fn,
                    canonicalized=canonicalized,
                    return_copy=False,
                )
                result_per_radius[radius].append(serialized)

        return result_per_radius

    @classmethod
    def _prune_neighborhood_to_radius(
        cls,
        neighborhood: CanonicalizableNode,
        radius: int,
        max_radius: int,
    ) -> None:
        if radius == 0:
            return

        # Get children at current radius boundary
        current_level_key = f"r_{max_radius - 1}_dist_{radius}_CHILD"
        current_level_children = neighborhood.tree_tags.get(current_level_key)

        # Get children at next level (siblings beyond radius)
        next_level_key = f"r_{max_radius - 1}_dist_{radius + 1}_CHILD"
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
                sibling.detatch_parent()

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

    from ariel_experiments.characterize.canonical.core.internal.serializer import (
        TreeSerializer,
    )
    from ariel_experiments.characterize.canonical.core.toolkit import (
        CanonicalToolKit as ctk,
    )
    from ariel_experiments.utils.initialize import (
        generate_random_individual,
    )

    console = Console()

    individual = generate_random_individual(seed=41)

    max_radius = 10

    canonicalnode = ctk.from_graph(individual, auto_id=True)
    neighbours = TreeProcessor.collect_tree_neighbourhoods_old(
        canonicalnode,
        TreeSerializer.to_string,
        max_radius=max_radius,
    )

    neighbours_new = TreeProcessor.collect_tree_neighbourhoods(
        canonicalnode,
        TreeSerializer.to_string,
        max_radius=max_radius,
    )

    console.print(neighbours)

    console.rule()

    console.print(neighbours_new)
