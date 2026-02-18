"""
roadrunner.grid
~~~~~~~~~~~~~~
Parallel parameter-grid runner.
"""

import os
from functools import partial

import pandas as pd

from .config import (
    TEFFS_K,
    LOGGS_CGS,
    R_PLANETS_Rj,
    SEMI_MAJOR_AU,
    PHASE_DEG,
    LAM_GRID,
    REFLECT_THRESHOLD,
)
from .system import SystemParams
from .bands import evaluate_case, COLUMNS

# Try to import tqdm; fall back to a no-op wrapper
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it

import concurrent.futures


def run_grid_parallel(
    teffs=TEFFS_K,
    loggs=LOGGS_CGS,
    rps=R_PLANETS_Rj,
    as_au=SEMI_MAJOR_AU,
    phases=PHASE_DEG,
    thresh=REFLECT_THRESHOLD,
    max_workers=None,
):
    """
    Run ``evaluate_case`` over a full parameter grid using threads.

    Returns
    -------
    pd.DataFrame  with columns ``COLUMNS``.
    """
    cases = [
        SystemParams(Teff, logg, Rp, a, ph)
        for Teff in teffs
        for logg in loggs
        for Rp in rps
        for a in as_au
        for ph in phases
    ]

    max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1)
    func = partial(evaluate_case, lam_grid_um=LAM_GRID,
                   thresh=thresh, do_plots=False)

    out = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for df in tqdm(ex.map(func, cases), total=len(cases),
                       desc="Running grid"):
            if df is not None and not df.empty:
                out.append(df)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=COLUMNS)
