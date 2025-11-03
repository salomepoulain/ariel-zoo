import networkx as nx
from _typeshed import Incomplete
from ariel_experiments.characterize.canonical.core.node import CanonicalizableNode as CanonicalizableNode
from collections.abc import Callable as Callable
from typing import Any

console: Incomplete

class TreeProcessor:
    @classmethod
    def canonicalize(cls, node: CanonicalizableNode, *, zero_root_angle: bool = True, child_order: bool = True, return_copy: bool = True) -> CanonicalizableNode: ...
    @classmethod
    def collect_subtrees(cls, node: CanonicalizableNode, serializer_fn: Callable[[CanonicalizableNode], Any] | None = None, *, canonicalized: bool = True) -> list[str | nx.DiGraph[Any] | CanonicalizableNode]: ...
    @classmethod
    def collect_tree_neighbourhoods_old(cls, tree: CanonicalizableNode, serializer_fn: Callable[[CanonicalizableNode], Any] | None = None, *, max_radius: int = 10, canonicalized: bool = True) -> dict[int, list[Any]]: ...
    @classmethod
    def collect_tree_neighbourhoods(cls, tree: CanonicalizableNode, serializer_fn: Callable[[CanonicalizableNode], Any] | None = None, *, max_radius: int = 10, canonicalized: bool = True) -> dict[int, list[Any]]: ...
