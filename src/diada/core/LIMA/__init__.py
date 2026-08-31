"""
This directory is the Python port of LIMA (from the sibling LIMA_py repo),
the algorithm behind DIADA's soundness scoring -- an in-process replacement
for DIADA-0.8.jar. See main.py's runLima() for the entry point, and
diada.core.diada_tool for where it's actually wired in.

Layout mirrors the original LIMA_py repo exactly (just nested one level
deeper): LIMA/ (the lattice/sampler/scheduler algorithm), data/ (dataset +
predicate-group representations), util/, npstruct/, and main.py (runLima).
Every absolute import inside these files was rewritten from e.g.
`from LIMA.LIMA import LIMA` to `from diada.core.LIMA.LIMA.LIMA import LIMA`
so they resolve correctly as part of the diada package; nothing else about
the algorithm was changed.
"""

from .main import runLima

__all__ = ["runLima"]
