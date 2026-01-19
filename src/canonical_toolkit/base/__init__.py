"""
Base package - Generic, reusable infrastructure.

This package contains domain-agnostic building blocks that can be used
across different applications (similarity analysis, feature extraction, etc.).
"""

# Not importing matrix package by default, as similarity Matrix should be the preffered chooice. 

from .plotters import *
