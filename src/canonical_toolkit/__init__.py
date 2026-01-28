"""
          _____                _____                    _____
         /\    \              /\    \                  /\    \
        /::\    \            /::\    \                /::\____\
       /::::\    \           \:::\    \              /:::/    /
      /::::::\    \           \:::\    \            /:::/    /
     /:::/\:::\    \           \:::\    \          /:::/    /
    /:::/  \:::\    \           \:::\    \        /:::/____/
   /:::/    \:::\    \          /::::\    \      /::::\    \
  /:::/    / \:::\    \        /::::::\    \    /::::::\____\________
 /:::/    /   \:::\    \      /:::/\:::\    \  /:::/\:::::::::::\    \
/:::/____/     \:::\____\    /:::/  \:::\____\/:::/  |:::::::::::\____\
\:::\    \      \::/    /   /:::/    \::/    /\::/   |::|~~~|~~~~~
 \:::\    \      \/____/   /:::/    / \/____/  \/____|::|   |
  \:::\    \              /:::/    /                 |::|   |
   \:::\    \            /:::/    /                  |::|   |
    \:::\    \           \::/    /                   |::|   |
     \:::\    \           \/____/                    |::|   |
      \:::\    \                                     |::|   |
       \:::\____\                                    \::|   |
        \::/    /                                     \:|   |
         \/____/                                       \|___|

CanonicalToolkit - Morphological analysis toolkit for evolutionary robotics.

This toolkit provides:
- Generic matrix infrastructure (base.matrix)
- Morphological analysis tools (morphology.node, morphology.similarity, morphology.visual)

Quick start:
    # Import quickly [easiest]
    import canonical_toolkit as ctk

    # Import packages
    from canonical_toolkit import base, morphology

    # Or import directly from subpackages
    from canonical_toolkit.base.matrix import MatrixInstance
    from canonical_toolkit.morphology.node import Node
    from canonical_toolkit.morphology.similarity import SimilarityMatrix

Author Salomé Poulain 19/01/2026
"""


from .base import *
from .morphology import *
from .utils import *


__version__ = "0.1.0"
