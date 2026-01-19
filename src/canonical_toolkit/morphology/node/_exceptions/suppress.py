from __future__ import annotations

raise DeprecationWarning


import sys
from typing import TYPE_CHECKING


from .exceptions import (
    ChildNotFoundError,
    FaceNotFoundError,
)

if TYPE_CHECKING:
    from types import TracebackType


def suppress_face_errors() -> None:
    """Install custom exception handler to suppress tracebacks for FaceNotFoundError."""
    original_hook = sys.excepthook

    def custom_hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is ChildNotFoundError or exc_type is FaceNotFoundError:
            exc_value.print_rich()
        else:
            original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = custom_hook
