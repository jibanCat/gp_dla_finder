"""Catalogue and result I/O.

Kept out of the inference core: every reader and writer here depends on an
optional package, and the numerical core must keep needing nothing but NumPy and
SciPy.
"""
