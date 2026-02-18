"""
roadrunner.config
~~~~~~~~~~~~~~~~
All configuration constants, environment setup, and parameter grids for
the Roman CGI reflected-light analysis pipeline.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Environment — set BEFORE any PICASO import
# ---------------------------------------------------------------------------
_PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Parent timestep/ directory (where picaso/ and SLGRID data live)
_TIMESTEP_ROOT = os.environ.get(
    "TIMESTEP_ROOT",
    os.path.dirname(_PROJECT_ROOT),
)

os.environ.setdefault(
    "picaso_refdata",
    os.path.join(_TIMESTEP_ROOT, "picaso", "reference"),
)
os.environ.setdefault(
    "PYSYN_CDBS",
    os.path.join(_TIMESTEP_ROOT, "picaso", "reference", "synphot3"),
)

# Threading
_nthreads = str(max(1, (os.cpu_count() or 2) - 1))
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ.setdefault(_var, _nthreads)

# ---------------------------------------------------------------------------
# Stellar defaults
# ---------------------------------------------------------------------------
T_STAR_K: float = 5778.0
R_STAR_Rsun: float = 1.0

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------
TEFFS_K        = [400, 500, 800, 1000, 1200, 1500]
LOGGS_CGS      = [3.0, 3.5, 4.0]
R_PLANETS_Rj   = [1.0, 1.2]
SEMI_MAJOR_AU  = [5.0, 10.0, 20.0]
PHASE_DEG      = [60.0]  # single phase angle

# ---------------------------------------------------------------------------
# Roman CGI band definitions  (wavelength in µm)
# ---------------------------------------------------------------------------
CGI_BANDS = {
    "CGI-1": (0.546, 0.604),
    "CGI-2": (0.610, 0.710),
    "CGI-3": (0.675, 0.785),
    "CGI-4": (0.783, 0.867),
}

# ---------------------------------------------------------------------------
# Radiative-transfer settings
# ---------------------------------------------------------------------------
REFLECT_NUM_GANGLE = 4
REFLECT_NUM_TANGLE = 4
THERMAL_NUM_GANGLE = 8
THERMAL_NUM_TANGLE = 1
ATM_NLAYERS        = 61   # match PICASO default; jupiter_cld expects (nlevel-1)=60

# ---------------------------------------------------------------------------
# Threshold & wavelength grid
# ---------------------------------------------------------------------------
REFLECT_THRESHOLD = 0.10                        # 10 %
LAM_GRID          = np.linspace(0.3, 1.0, 1200) # µm

# ---------------------------------------------------------------------------
# SLGRID data paths
# ---------------------------------------------------------------------------
SLGRID_BASE    = os.environ.get(
    "SLGRID_BASE_DIR",
    os.path.join(_TIMESTEP_ROOT, "2_9_2026"),
)
SLGRID_PT_DIR  = os.path.join(SLGRID_BASE, "SLGRID Climate Files")
SLGRID_CLD_DIR = os.path.join(SLGRID_BASE, "SLGRID Cloud Files")

# Per-Teff manual file assignments
SLGRID_FILES_BY_TEFF = {
    1000: {
        "pt":  "SLGRID_T1000_g31_m+000_CO100_fsed3_full.pt",
        "cld": "SLGRID_T1000_g31_m+000_CO100_fsed3_picaso.cld",
    },
    1500: {
        "pt":  "SLGRID_T1500_g31_m+000_CO100_fsed3_full.pt",
        "cld": "SLGRID_T1500_g31_m+000_CO100_fsed3_picaso.cld",
    },
    500: {
        "pt":  None,   # not available for this Teff
        "cld": "SLGRID_T500_g31_m+000_CO100_fsed0.3_frac50_picaso.cld",
    },
}

# Fallback: if a Teff is missing, use another Teff's files
FALLBACK_TEFF_MAP = {
    500: 1000,
}

# ---------------------------------------------------------------------------
# PICASO availability  (lazy — imported here so every module can check)
# ---------------------------------------------------------------------------
HAVE_PICASO = False
try:
    from picaso import justdoit as jdi   # noqa: F401
    from picaso import justplotit as jpi  # noqa: F401
    from picaso.fluxes import blackbody   # noqa: F401
    HAVE_PICASO = True
except Exception:
    jdi = None
    jpi = None
    blackbody = None
