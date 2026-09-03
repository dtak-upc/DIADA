"""
Column-soundness scoring: how related every pair of columns in a table is.

Was benchmarks_generation/diada.py. Two engines are available:

  - "jar" (default): the original DIADA-0.8.jar, invoked via `java -jar`
    (requires a Java runtime on PATH). Deterministic -- see below.
  - "lima": the in-process Python port of LIMA under diada.core.LIMA (moved
    here from the sibling LIMA_py repo -- see
    src/diada/core/LIMA/__init__.py). This is the algorithm DIADA-0.8.jar
    itself implements (in Java); needs no JVM/Java install, but is a
    randomized approximation of the jar's output -- see below -- so it's
    opt-in rather than the default. Pass engine="lima" explicitly to use it.

IMPORTANT -- the two engines are NOT numerically interchangeable:

  - LIMA is a randomized sampling algorithm with no fixed seed anywhere in
    the port (same as the upstream LIMA_py repo), so two calls to
    invoke_diada_lima on the same data return DIFFERENT relatedness values
    and a different -- though heavily overlapping -- set of "sound" pairs
    each time. The jar, by contrast, was empirically deterministic in
    testing (identical output across repeated runs on the same input).
  - The two engines' soundness scores live on different scales. LIMA's
    numbers are z-scores from its internal significance test (it only
    reports edges that already cleared its own significance bar, so its
    minimum reported value sits right at that bar); the jar's numbers are a
    differently-scaled statistic computed over a much larger candidate set,
    most of which never gets close to DEFAULT_THRESHOLD=4 in
    dataset_builder.py. Don't compare raw soundness values across engines,
    and be aware that DEFAULT_THRESHOLD's effect on each engine's output
    differs (it does much more filtering on the jar's output than on
    LIMA's, which is already mostly pre-filtered by construction).
  - Measured on four base_datasets (num_buckets=10, threshold=4; jar was
    deterministic across repeated runs in every case, so these are jar vs.
    a single LIMA run):

        dataset       rows x cols   jar time  lima time   jar >thr4  lima >thr4  jaccard  lima subset-of-jar
        students      4424 x 32     1.4s      3.7s        252        176         ~0.45    98.9%
        pendigits     10992 x 17    0.9s      0.7s        210         42         0.20     100%
        diamonds      53940 x 10    1.8s      0.5s         30         22         0.73     100%
        mice_protein  1080 x 81     20.4s     12.3s      2847        266         0.09     100%

    Two consistent takeaways: (1) LIMA is *always* a near-total subset of
    the jar's raw (unfiltered) relations (>=98.9% across all four) -- it
    never contradicts the jar, it's just far more conservative about what
    it reports as "sound" -- and (2) that conservatism gets much more
    pronounced as column count grows (mice_protein, 81 columns: LIMA finds
    266 vs. the jar's 2847). Whether LIMA reports similar timing to the jar
    varies by dataset -- it was faster on 3 of 4 here (notably on
    mice_protein, the slowest case for both: 12.3s vs. 20.4s) but slower on
    students -- no simple rule found yet linking speed to rows/columns.

Both engines share `_label_columns` (unchanged: same type-labelling scheme,
same bucketing rules) and return the same soundness triple format:
["column_1", "column_2", "soundness"], one row per pair with soundness > 0,
column names restored to their original (un-typed, un-bucketed) form.

Compared to the original benchmarks_generation/diada.py, three deliberate
differences carried over from the earlier jar-only version of this module
(now only relevant to the "jar" engine):

  1. temp_data.csv / output.txt live inside a tempfile.TemporaryDirectory()
     instead of the process's cwd, so this is safe to call from anywhere
     (or concurrently) instead of assuming a single fixed working directory.
  2. `invoke_diada` returns the soundness DataFrame directly instead of
     writing it to a hardcoded `../data/output/output_{file_name}.csv` path.
     Writing that file (for the reproducibility sweep) is now the caller's
     job -- see experiments/generate_benchmarks.py.
  3. A non-zero return code from the jar now raises, instead of only being
     printed while execution silently continued with whatever (possibly
     stale-looking, possibly missing) output.txt happened to exist.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .. import paths
from .LIMA import runLima

KURTOSIS_SENSIBILITY = 2
DEFAULT_APPROX = 0.000001  # LIMA's scheduler-threshold parameter; same value the jar was always called with


def invoke_diada(df: pd.DataFrame, num_buckets: int, engine: str = "jar", jar_path: Optional[Path] = None,
                  approx: float = DEFAULT_APPROX, verbose: bool = True) -> pd.DataFrame:
    """
    Score every pair of columns in `df` for soundness and return the result
    as a DataFrame of ["column_1", "column_2", "soundness"] (one row per
    pair with soundness > 0).

    Parameters
    ----------
    df          : table to analyze (every column, including the target, is
                  passed through -- soundness relations to the target are
                  exactly what downstream cleaning uses to keep it)
    num_buckets : 0 = label every column as String; >0 = infer
                  Integer/Double/String per column; >1 additionally buckets
                  every numeric column into `num_buckets` bins before typing
                  it as String (see _label_columns)
    engine      : "jar" (default) -- the original DIADA-0.8.jar via subprocess
                  (needs Java); or "lima" -- the in-process Python port,
                  opt-in only (randomized approximation -- not numerically
                  interchangeable, see module docstring).
    jar_path    : only used when engine="jar". Path to DIADA-0.8.jar
                  (default: paths.JAR_PATH)
    approx      : only used when engine="lima". LIMA's scheduler threshold
                  (default: the same 0.000001 the jar was always invoked
                  with, so results are comparable)
    verbose     : print progress/timing, matching the original script's output

    Returns
    -------
    DataFrame with columns ["column_1", "column_2", "soundness"], one row per
    pair with soundness > 0, restricted to their original (un-typed,
    un-bucketed) column names.
    """
    if engine == "lima":
        return invoke_diada_lima(df, num_buckets, approx=approx, verbose=verbose)
    elif engine == "jar":
        return invoke_diada_jar(df, num_buckets, jar_path=jar_path, verbose=verbose)
    else:
        raise ValueError(f"engine must be 'lima' or 'jar', got {engine!r}")


def invoke_diada_lima(df: pd.DataFrame, num_buckets: int, approx: float = DEFAULT_APPROX, verbose: bool = True) -> pd.DataFrame:
    """Score `df` using the in-process LIMA port (diada.core.LIMA.runLima).
    See `invoke_diada` for the parameters shared with the jar engine."""
    if verbose:
        print("Executing DIADA (engine=lima)")
    start_time = time.time()

    data = _label_columns(df.copy(), num_buckets)
    col_names = data.columns.tolist()  # index i <-> LIMA's predicate-group id i, same convention as the jar's column id

    with tempfile.TemporaryDirectory(prefix="diada_run_") as tmp_dir:
        input_path = Path(tmp_dir) / "temp_data.csv"
        data.to_csv(input_path, index=False)
        matrix = runLima(str(input_path), approx)

    soundness_df = _relatedness_matrix_to_soundness_df(matrix, col_names)

    if verbose:
        print("Augmentation time ->", time.time() - start_time)

    return soundness_df


def invoke_diada_jar(df: pd.DataFrame, num_buckets: int, jar_path: Optional[Path] = None, verbose: bool = True) -> pd.DataFrame:
    """Score `df` using the original DIADA-0.8.jar via subprocess (requires
    Java on PATH). See `invoke_diada` for the parameters shared with the
    "lima" engine. Kept for comparison against the "lima" engine and as a
    fallback if Java is available but something about the LIMA port is ever
    in doubt."""
    jar_path = Path(jar_path) if jar_path is not None else paths.JAR_PATH
    if not jar_path.exists():
        raise FileNotFoundError(f"DIADA jar not found at {jar_path}")

    if verbose:
        print("Executing DIADA (engine=jar)")
    start_time = time.time()

    data = _label_columns(df.copy(), num_buckets)
    col_names = data.columns.tolist()  # index i <-> the jar's column id i

    with tempfile.TemporaryDirectory(prefix="diada_run_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        input_path = tmp_dir / "temp_data.csv"
        output_path = tmp_dir / "output.txt"

        data.to_csv(input_path, index=False)

        cmd = ["java", "-jar", str(jar_path), str(input_path), "0.000001", f"{len(data)}"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_dir)
        if result.returncode != 0:
            raise RuntimeError(
                f"DIADA jar exited with code {result.returncode}\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        rel_df = pd.read_csv(output_path, sep=r"\s+", header=None, names=["col_id_1", "col_id_2", "soundness"])

    rel_df = rel_df[rel_df["soundness"] > 0]
    rel_df["column_1"] = rel_df["col_id_1"].map(lambda i: col_names[i])
    rel_df["column_2"] = rel_df["col_id_2"].map(lambda i: col_names[i])

    soundness_df = rel_df[["column_1", "column_2", "soundness"]].copy()
    for suffix in (r"\(String\)", r"\(Integer\)", r"\(Double\)"):
        soundness_df = soundness_df.replace(suffix, "", regex=True)

    if verbose:
        print("Augmentation time ->", time.time() - start_time)

    return soundness_df.reset_index(drop=True)


def _relatedness_matrix_to_soundness_df(matrix: np.ndarray, col_names: List[str]) -> pd.DataFrame:
    """Convert LIMA's dense NxN relatedness matrix (matrix[i, j] = soundness
    signal for column j given column i; not necessarily symmetric) into the
    same sparse ["column_1", "column_2", "soundness"] triple format the jar
    engine produces, with type-suffix stripping applied the same way."""
    i_idx, j_idx = np.nonzero(matrix > 0)
    soundness_df = pd.DataFrame({
        "column_1": [col_names[i] for i in i_idx],
        "column_2": [col_names[j] for j in j_idx],
        "soundness": matrix[i_idx, j_idx],
    })
    for suffix in (r"\(String\)", r"\(Integer\)", r"\(Double\)"):
        soundness_df = soundness_df.replace(suffix, "", regex=True)
    return soundness_df.reset_index(drop=True)


def _label_columns(data: pd.DataFrame, num_buckets: int) -> pd.DataFrame:
    """Rename columns with DIADA's `name(Type)` convention, optionally bucketing numerics."""
    if num_buckets == 0:  # Option 1: all columns labelled as String
        data.columns = [f"{col}(String)" for col in data.columns]
        return data

    if num_buckets <= 0:
        return data

    # Option 2: label float/integer columns as such.
    # IMPORTANT: a numerical column is labelled as string if it has fewer than 30 unique values.
    for col in data.columns:  # Convert floats that are actually integers, so typing doesn't mislabel them
        if pd.api.types.is_float_dtype(data[col]):
            if data[col].dropna().apply(float.is_integer).all():
                data[col] = data[col].astype("Int64")  # Nullable integer type

    def infer_type(col: pd.Series) -> str:
        if pd.api.types.is_float_dtype(col):
            return "(Double)"
        elif pd.api.types.is_integer_dtype(col):
            return "(Integer)" if col.nunique() > 30 else "(String)"
        else:
            return "(String)"

    data.columns = [f"{col}{infer_type(data[col])}" for col in data.columns]

    if num_buckets > 1:  # Option 3: bucket all numerical columns
        for col in data.columns:
            if "(Integer)" in col or "(Double)" in col:
                kurtosis = data[col].kurt()
                heavy_tailed = abs(kurtosis) > KURTOSIS_SENSIBILITY
                if heavy_tailed:  # Quantile (equi-depth) bins
                    data[col] = pd.qcut(data[col], q=num_buckets, duplicates="drop")
                else:  # Standard width bins
                    data[col] = pd.cut(data[col], bins=num_buckets)
        data.columns = [col.replace("(Integer)", "(String)").replace("(Double)", "(String)") for col in data.columns]

    return data
