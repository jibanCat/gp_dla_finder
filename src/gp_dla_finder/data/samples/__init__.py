"""Packaged quasi-Monte-Carlo absorber sample grids.

Each grid is an ``.npz`` of QMC samples plus a ``.json`` recording the prior, the
QMC construction, the generating environment, per-array checksums, and — crucially
— whether the grid's identity with a deployed production grid has been verified.
See :mod:`gp_dla_finder.samples`.
"""
