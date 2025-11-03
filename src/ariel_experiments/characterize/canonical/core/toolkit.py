from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Literal

from ariel_experiments.characterize.canonical.core.internal.factory import (
    TreeFactory,
)
from ariel_experiments.characterize.canonical.core.internal.processor import (
    TreeProcessor,
)
from ariel_experiments.characterize.canonical.core.internal.serializer import (
    TreeSerializer,
)

if TYPE_CHECKING:
    import networkx as nx

    from ariel_experiments.characterize.canonical.core.node import (
        CanonicalizableNode,
    )


class CanonicalToolKit:
    """Unified API for tree operations."""

    create_root = TreeFactory.create_root
    node = TreeFactory.node
    brick = TreeFactory.brick
    hinge = TreeFactory.hinge

    from_graph = TreeFactory.from_graph
    from_string = TreeFactory.from_string

    to_graph = TreeSerializer.to_graph
    to_string = TreeSerializer.to_string

    canonicalize = TreeProcessor.canonicalize

    @classmethod
    def collect_subtrees(
        cls,
        node: CanonicalizableNode,
        output_type: Literal["string", "graph", "node"] = "string",
    ) -> list[str | nx.DiGraph[Any] | CanonicalizableNode]:
        match output_type:
            case "string":
                return TreeProcessor.collect_subtrees(node, cls.to_string)
            case "node":
                return TreeProcessor.collect_subtrees(node, cls.to_graph)
            case "graph":
                return TreeProcessor.collect_subtrees(node)

    @classmethod
    def collect_neighbours(
        cls,
        node: CanonicalizableNode,
        *,
        max_radius: int = 10,
        output_type: Literal["string", "graph", "node"] = "string",
    ) -> dict[int, list[str | nx.DiGraph[Any] | CanonicalizableNode]]:
        match output_type:
            case "string":
                return TreeProcessor.collect_neighbourhoods(
                    node, cls.to_string, max_radius=max_radius,
                )
            case "node":
                return TreeProcessor.collect_neighbourhoods(
                    node, cls.to_graph, max_radius=max_radius,
                )
            case "graph":
                return TreeProcessor.collect_neighbourhoods(
                    node, max_radius=max_radius,
                )

    @classmethod
    def to_canonical_string(
        cls,
        node: CanonicalizableNode,
        *,
        zero_root_angle: bool = True,
        child_order: bool = True,
    ) -> str:
        canonical_tree = cls.canonicalize(
            node,
            zero_root_angle=zero_root_angle,
            child_order=child_order,
        )

        return cls.to_string(canonical_tree)

    @classmethod
    def to_canonical_graph(
        cls,
        node: CanonicalizableNode,
        *,
        zero_root_angle: bool = True,
        child_order: bool = True,
    ) -> nx.DiGraph[Any]:
        canonical_tree = cls.canonicalize(
            node,
            zero_root_angle=zero_root_angle,
            child_order=child_order,
        )

        return cls.to_graph(canonical_tree)

    @classmethod
    def tanimoto(
        cls,
        node1: CanonicalizableNode,
        node2: CanonicalizableNode,
        *,
        max_radius: int = 5,
        analysis_type: Literal["set", "count"] = "count",
    ) -> dict[int, float]:
        dict1 = cls.collect_neighbours(node1, max_radius=max_radius)
        dict2 = cls.collect_neighbours(node2, max_radius=max_radius)

        return cls.tanimoto_all_radii(dict1, dict2, analysis_type)

    # TODO: dont like this here, move it to new analysis helper thing?
    @classmethod
    def _tanimoto_strings_set(
        cls,
        fp1_dict: dict[int, list[str]],
        fp2_dict: dict[int, list[str]],
        radius: int,
    ):
        strings1 = set(fp1_dict[radius])
        strings2 = set(fp2_dict[radius])

        intersection = len(strings1 & strings2)
        union = len(strings1 | strings2)

        if union == 0:
            return 0.0

        return intersection / union

    # TODO: dont like this here, move it to new analysis helper thing?
    @classmethod
    def _tanimoto_strings_with_counts(
        cls,
        fp1_dict: dict[int, list[str]],
        fp2_dict: dict[int, list[str]],
        radius: int,
    ):
        """Tanimoto with fragment counts (bag of words approach)."""
        counts1 = Counter(fp1_dict[radius])
        counts2 = Counter(fp2_dict[radius])

        intersection = sum((counts1 & counts2).values())
        union = sum((counts1 | counts2).values())

        if union == 0:
            return 0.0

        return intersection / union

    # TODO: dont like this here, move it to new analysis helper thing?
    @classmethod
    def tanimoto_all_radii(
        cls,
        fp1_dict: dict[int, list[str]],
        fp2_dict: dict[int, list[str]],
        analysis_type: Literal["set", "count"] = "count",
    ) -> dict[int, float]:
        """Calculate Tanimoto for each radius level."""
        results = {}

        if analysis_type == "count":
            analyzer = cls._tanimoto_strings_with_counts
        else:
            analyzer = cls._tanimoto_strings_set

        for radius in fp1_dict:
            if radius in fp2_dict:
                results[radius] = analyzer(fp1_dict, fp2_dict, radius)
        return results
