"""
java.util.SplittableRandom (JDK 8-17), bit for bit: the generator DIADA-0.8.jar draws its pairs of
rows with. Checked against Java 11 on 200,000 nextInt calls, including bounds where Java rejects
draws often.
"""

from __future__ import annotations

import numpy as np

GOLDEN_GAMMA = np.uint64(0x9E3779B97F4A7C15)
MIX1, MIX2 = np.uint64(0x62A9D9ED799705F5), np.uint64(0xCB24D0A5C88C35B3)
CHUNK = 1 << 22  # draws generated at once


class JavaSplittableRandom:
    """new SplittableRandom(seed) and its nextInt(bound). Draw k (from 1) is mix32(seed + k * gamma),
    so the generator is a counter and draws can be computed in vectorised chunks."""

    def __init__(self, seed: int):
        self.seed = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
        self.k = 0

    def _draws(self, count: int) -> np.ndarray:
        """the next `count` nextInt() values, as their 32-bit patterns (uint64 holding a uint32)"""
        with np.errstate(over="ignore"):
            z = self.seed + np.arange(self.k + 1, self.k + count + 1, dtype=np.uint64) * GOLDEN_GAMMA
            z = (z ^ (z >> np.uint64(33))) * MIX1
            z = ((z ^ (z >> np.uint64(28))) * MIX2) >> np.uint64(32)
        self.k += count
        return z

    def ints(self, bounds: np.ndarray) -> np.ndarray:
        """nextInt(bounds[0]), nextInt(bounds[1]), ... in order, as Java computes them"""
        out = np.empty(len(bounds), dtype=np.int64)
        done = 0
        while done < len(bounds):
            todo = bounds[done:done + CHUNK]
            r = self._draws(len(todo)).astype(np.int64)
            m = todo - 1
            power = (todo & m) == 0
            u = r >> 1  # r >>> 1
            val = np.where(power, r & m, u % todo)
            bad = np.flatnonzero(~power & (u + m - val >= 1 << 31))  # u + m - r < 0 in int arithmetic
            if len(bad) == 0:
                out[done:done + len(todo)] = val
                done += len(todo)
                continue
            j = bad[0]
            out[done:done + j] = val[:j]
            self.k -= len(todo) - (j + 1)  # give back the draws after the rejected one
            bound = int(todo[j])
            while True:  # the rejection loop: u = mix32(nextSeed()) >>> 1
                u1 = int(self._draws(1)[0]) >> 1
                v = u1 % bound
                if u1 + bound - 1 - v < 1 << 31:
                    break
            out[done + j] = v
            done += j + 1
        return out

    def pairs(self, count: int, size: int) -> tuple:
        """CSVDataset.sample: x = nextInt(size), y = nextInt(size - 1), y += 1 if y >= x"""
        bounds = np.empty(2 * count, dtype=np.int64)
        bounds[0::2], bounds[1::2] = size, size - 1
        v = self.ints(bounds)
        x, y = v[0::2], v[1::2]
        y = y + (y >= x)
        return x.astype(np.int32), y.astype(np.int32)
