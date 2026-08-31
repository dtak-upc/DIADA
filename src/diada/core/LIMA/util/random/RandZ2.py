import numpy as np
import numpy.typing as npt


class RandZ2:
    """Generates random pairs of distinct integers."""

    def __init__(self, k: int = 64):
        # target number of pairs per grid cell, used by uniformGrid
        self.k = k

    def uniform(self, n: int, range_: int) -> npt.NDArray[np.int32]:
        """Return an (n, 2) int32 array where each row is two different
        random integers in [0, range_).

        `range_` must fit in int32 (<= 2**31 - 1); `np.random.randint`
        raises if it doesn't. Elements are stored as int32 since we only
        ever need row indices, not the rest of int64's range.
        """
        pairs = np.random.randint(0, range_, size=(n, 2), dtype=np.int32)
        equal = pairs[:, 0] == pairs[:, 1]
        # +1 stays within int32 (it's at most range_, which already fits by
        # construction) before the modulo brings it back into [0, range_)
        pairs[equal, 1] = (pairs[equal, 1] + 1) % range_
        return pairs

    def uniformGrid(self, n: int, range_: int) -> npt.NDArray[np.int32]:
        """Same as `uniform`, but reordered for cache-friendly access.

        The [0, range_) x [0, range_) domain is subdivided into a grid with
        roughly `self.k` pairs per cell (so ~n/k cells total, arranged in a
        sqrt(n/k) x sqrt(n/k) layout). Pairs are then sorted by cell (x
        first, then y), and by their original values within a cell, so
        pairs that land near each other in space end up near each other in
        the output too.
        """
        pairs = self.uniform(n, range_)

        num_cells = max(1, n // self.k)
        divisions = max(1, int(round(np.sqrt(num_cells))))

        cell_x = (pairs[:, 0].astype(np.int64) * divisions) // range_
        cell_y = (pairs[:, 1].astype(np.int64) * divisions) // range_

        # lexsort's last key is the most significant: cell_x, then cell_y,
        # then the original x, then the original y as final tie-breakers
        order = np.lexsort((pairs[:, 1], pairs[:, 0], cell_y, cell_x))
        return pairs[order]
