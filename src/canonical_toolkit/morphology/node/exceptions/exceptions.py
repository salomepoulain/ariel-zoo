from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
    from ..node import (
        Node,
    )

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class _BaseFaceError(Exception):
    """Base class for face-related errors with rich formatting."""

    def __init__(
        self,
        face: str,
        node: Node,
    ) -> None:
        """Create error with formatted message showing available faces."""
        self.face = face
        self.node = node
        super().__init__(self._format_message())

    def _format_face_line(self, face: ModuleFaces) -> str:
        """Format a single face line with proper alignment."""
        child = self.node.get(face.name)  # type: ignore[misc]
        full_name = face.name.lower()
        face_display = f"'[bold][underline]{full_name[:2]}[/bold][/underline]{full_name[2:]}'"
        # Calculate padding based on plain text length
        plain_face = f"'{full_name}'"
        padding = " " * max(0, 20 - len(plain_face))

        if child:
            content = f"{child.module_type.name[0]}{child.internal_rotation}"
        else:
            content = "[black dim](empty)[/black dim]"

        return f"  {face_display}{padding} → {content}"

    def _get_header_line(self) -> str:
        """Override this to provide the error-specific header line."""
        raise NotImplementedError

    def _get_error_title(self) -> str:
        """Override this to provide the error-specific title."""
        raise NotImplementedError

    def _get_error_subtitle(self) -> str:
        """Override this to provide the error-specific subtitle."""
        raise NotImplementedError

    def _format_message(self) -> str:
        """Format a helpful error message with radial/axial face breakdown."""
        lines = [self._get_header_line()]

        # Show radial faces
        if self.node.config.radial_face_order:
            lines.append("[steel_blue1][Radial faces][/steel_blue1]")
            lines.extend(
                self._format_face_line(face)
                for face in self.node.config.radial_face_order
            )

        # Show axial faces
        if self.node.config.axial_face_order:
            lines.append("\n[steel_blue1]<Axial faces>[/steel_blue1]")
            lines.extend(
                self._format_face_line(face)
                for face in self.node.config.axial_face_order
            )

        return "\n".join(lines)

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""

    def print_rich(self) -> None:
        """Print this error with rich formatting."""
        console = Console(stderr=True)

        title = Text(self._get_error_title(), style="blue")
        msg = Text.from_markup(
            f"[bold red]{self._get_error_subtitle()}[/bold red]\n\n{self._format_message()}",
        )
        panel = Panel(
            msg,
            title=title,
            border_style="blue",
            padding=(1, 2),
        )
        console.print(panel)


class FaceNotFoundError(_BaseFaceError, KeyError):
    """Raised when accessing a child at an invalid face (no traceback shown)."""

    @property
    def invalid_face(self) -> str:
        """Alias for backward compatibility."""
        return self.face

    def _get_header_line(self) -> str:
        return f"'{self.face}' is not a valid face on {self.node.module_type.name}.\n"

    def _get_error_title(self) -> str:
        return "FaceNotFoundError"

    def _get_error_subtitle(self) -> str:
        return "Invalid face access"


class ChildNotFoundError(_BaseFaceError, LookupError):
    """Raised when accessing a child at an empty face (no traceback shown)."""

    @property
    def empty_face(self) -> str:
        """Alias for backward compatibility."""
        return self.face

    def _get_header_line(self) -> str:
        return f"Face '{self.face}' on {self.node.module_type.name} is empty.\n"

    def _get_error_title(self) -> str:
        return "ChildNotFoundError"

    def _get_error_subtitle(self) -> str:
        return "Empty face access"
