from typing import Any, TypeVar, Union

import numpy as np
import numpy.typing as npt

_T = TypeVar("_T", bound=np.generic)

# Something usable as a single value (or a homogeneous array of values) of
# numpy scalar type `_T`: a real ndarray, a numpy scalar instance, or a
# plain python int - mirroring how numpy indexing/assignment already
# accepts either an int or an array wherever a value or index is expected.
#
# Usage: subscript with whichever TypeVar the calling module uses, e.g.
# `ArrayLike[T]` inside a `Generic[T]` class - the free variable above gets
# substituted positionally, it does not need to be the same TypeVar object.
ArrayLike = Union[npt.NDArray[_T], _T, int]

# Same idea, specialized for indices: a single int, a numpy integer scalar,
# or an integer array - i.e. anything numpy itself accepts wherever it wants
# an index (`arr[i]` vs `arr[np.array([i, j, k])]`).
IndexLike = ArrayLike[np.integer[Any]]
