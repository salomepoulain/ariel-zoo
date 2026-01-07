from typing import Any, Protocol

class SortableHashable(Protocol):
    """Protocol for objects that are both Hashable and Sortable (Comparable)."""
    
    def __hash__(self) -> int: ...
    
    def __lt__(self, other: Any) -> bool: ...
    
    def __eq__(self, other: Any) -> bool: ...
