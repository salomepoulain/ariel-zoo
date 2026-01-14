"""
Base package - Generic, reusable infrastructure.

This package contains domain-agnostic building blocks that can be used
across different applications (similarity analysis, feature extraction, etc.).
"""

from . import matrix

__all__ = ["matrix"]
