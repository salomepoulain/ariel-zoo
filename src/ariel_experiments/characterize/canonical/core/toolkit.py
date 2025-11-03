from __future__ import annotations

import networkx as nx
from typing import Any, Literal
from collections import Counter

from ariel_experiments.characterize.canonical.core.node import (
    CanonicalizableNode,
)
from ariel_experiments.characterize.canonical.core.internal.serializer import TreeSerializer
from ariel_experiments.characterize.canonical.core.internal.factory import TreeFactory
from ariel_experiments.characterize.canonical.core.internal.processor import (
    TreeProcessor,
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
    collect_subtrees = TreeProcessor.collect_subtrees
    collect_neighbours = TreeProcessor.collect_tree_neighbourhoods

    @classmethod
    def to_canonical_string(
        cls,
        node: CanonicalizableNode,
        *,
        zero_root_angle: bool = True,
        child_order: bool = True,
    ) -> str:
        canonical_tree = cls.canonicalize(
            node, zero_root_angle=zero_root_angle, child_order=child_order
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
            node, zero_root_angle=zero_root_angle, child_order=child_order
        )

        return cls.to_graph(canonical_tree)

    @classmethod
    def tanimoto(cls, node1: CanonicalizableNode, node2: CanonicalizableNode, *, max_radius: int = 5, analysis_type: Literal["set", "count"] = "count") -> dict[int, float]:
        raise DeprecationWarning("heavily WIP")

        dict1 = cls.collect_tree_neighbourhoods(node1, max_radius=max_radius)
        dict2 = cls.collect_tree_neighbourhoods(node2, max_radius=max_radius)

        return cls.tanimoto_all_radii(dict1, dict2, analysis_type)

    # TODO: dont like this here, move it to new analysis helper thing?
    @classmethod
    def _tanimoto_strings_set(cls, fp1_dict: dict[int, list[str]], fp2_dict: dict[int, list[str]], radius: int):
        strings1 = set(fp1_dict[radius])
        strings2 = set(fp2_dict[radius])

        intersection = len(strings1 & strings2)
        union = len(strings1 | strings2)

        if union == 0:
            return 0.0

        return intersection / union

    # TODO: dont like this here, move it to new analysis helper thing?
    @classmethod
    def _tanimoto_strings_with_counts(cls, fp1_dict: dict[int, list[str]], fp2_dict: dict[int, list[str]], radius: int):
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
    def tanimoto_all_radii(cls, fp1_dict: dict[int, list[str]], fp2_dict: dict[int, list[str]], analysis_type: Literal["set", "count"] = "count") -> dict[int, float]:
        """Calculate Tanimoto for each radius level."""
        results = {}

        if analysis_type == "count":
            analyzer = cls._tanimoto_strings_with_counts
        else:
            analyzer = cls._tanimoto_strings_set

        for radius in fp1_dict.keys():
            if radius in fp2_dict:
                results[radius] = analyzer(fp1_dict, fp2_dict, radius)
        return results
