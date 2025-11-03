from __future__ import annotations
from dataclasses import dataclass



from ariel_experiments.characterize.canonical.core.node import (
    CanonicalizableNode,
)

from dataclasses import field


@dataclass(slots=True)
class TreeCursor:
    root: CanonicalizableNode = field(init=True)
    current: CanonicalizableNode = root

    # hop
    # show tree
    # find by id
