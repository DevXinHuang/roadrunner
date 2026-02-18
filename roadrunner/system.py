"""
roadrunner.system
~~~~~~~~~~~~~~~~
SystemParams dataclass and SLGRID atmosphere-file resolution.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    T_STAR_K,
    R_STAR_Rsun,
    ATM_NLAYERS,
    SLGRID_PT_DIR,
    SLGRID_CLD_DIR,
    SLGRID_FILES_BY_TEFF,
    FALLBACK_TEFF_MAP,
)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class SystemParams:
    """Physical parameters for one exoplanet system."""
    teff_k:      float
    logg_cgs:    float
    rj:          float          # planet radius [R_Jup]
    a_au:        float          # semi-major axis [AU]
    phase_deg:   float          # phase angle [deg]
    tstar_k:     float = T_STAR_K
    rstar_rsun:  float = R_STAR_Rsun
    pt_file:     str   = None   # override: absolute or basename PT file
    cld_file:    str   = None   # override: absolute or basename CLD file


# ---------------------------------------------------------------------------
# SLGRID file resolution
# ---------------------------------------------------------------------------

def _resolve_slgrid_file(spec, base_dir):
    """Return absolute path from *spec* (may be None, basename, or abs)."""
    if spec is None:
        return None
    if os.path.isabs(spec):
        return spec
    return os.path.join(base_dir, spec)


def resolve_slgrid_files(sys: SystemParams):
    """
    Return ``(pt_path, cld_path)`` for a given ``SystemParams``.

    Resolution order:
    1. Explicit ``sys.pt_file`` / ``sys.cld_file`` overrides.
    2. ``SLGRID_FILES_BY_TEFF[teff]`` lookup.
    3. ``FALLBACK_TEFF_MAP[teff]`` → try step 2 again.

    Raises ``FileNotFoundError`` if files cannot be located.
    """
    teff_key = int(round(sys.teff_k))
    pt_file  = sys.pt_file
    cld_file = sys.cld_file

    # Step 2 — Teff-based lookup
    if pt_file is None or cld_file is None:
        spec = SLGRID_FILES_BY_TEFF.get(teff_key)
        if spec:
            pt_file  = pt_file  or spec.get("pt")
            cld_file = cld_file or spec.get("cld")

    # Step 3 — fallback
    if (pt_file is None or cld_file is None) and teff_key in FALLBACK_TEFF_MAP:
        fb = FALLBACK_TEFF_MAP[teff_key]
        spec = SLGRID_FILES_BY_TEFF.get(fb)
        if spec:
            pt_file  = pt_file  or spec.get("pt")
            cld_file = cld_file or spec.get("cld")
            print(f"⚠ No SLGRID files for Teff={teff_key}K; "
                  f"falling back to Teff={fb}K files.")

    pt_path  = _resolve_slgrid_file(pt_file,  SLGRID_PT_DIR)
    cld_path = _resolve_slgrid_file(cld_file, SLGRID_CLD_DIR)

    if not pt_path or not cld_path:
        raise FileNotFoundError(
            f"Missing SLGRID PT/CLD file for Teff={teff_key}K. "
            f"Set SLGRID_FILES_BY_TEFF[{teff_key}] or "
            f"SystemParams(pt_file=..., cld_file=...)."
        )
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PT file not found: {pt_path}")
    if not os.path.exists(cld_path):
        raise FileNotFoundError(f"CLD file not found: {cld_path}")

    return pt_path, cld_path


# ---------------------------------------------------------------------------
# Fallback atmosphere builder (kept for compatibility)
# ---------------------------------------------------------------------------

def build_simple_atmosphere(teff_k: float, logg_cgs: float,
                            nlayer: int = ATM_NLAYERS) -> pd.DataFrame:
    """Simple H₂/He atmosphere with a power-law T–P profile.

    .. deprecated:: This is a placeholder; prefer SLGRID PT files.
    """
    p_bar = np.logspace(-6, 2, nlayer)
    temperature = np.clip(
        teff_k * (p_bar / p_bar.mean()) ** 0.02,
        0.5 * teff_k,
        1.5 * teff_k,
    )
    return pd.DataFrame({
        "pressure":    p_bar,
        "temperature": temperature,
        "H2":          np.full(nlayer, 0.85),
        "He":          np.full(nlayer, 0.15),
    })
