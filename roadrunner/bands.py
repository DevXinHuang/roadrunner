"""
roadrunner.bands
~~~~~~~~~~~~~~~
Band-level metrics and the main single-case evaluation pipeline.
"""

import numpy as np
import pandas as pd

from .config import (
    CGI_BANDS,
    LAM_GRID,
    REFLECT_THRESHOLD,
    HAVE_PICASO,
    jpi,
)
from .physics import top_hat, trapz_band, frac_reflected
from .system import SystemParams
from .runner import run_picaso_once, extract_planet_fluxes
from .plotting import plot_spectra_with_bb, plot_band_bars

# ---------------------------------------------------------------------------
# Pre-computed band filters
# ---------------------------------------------------------------------------
BANDS = {name: top_hat(LAM_GRID, lo, hi) for name, (lo, hi) in CGI_BANDS.items()}

# DataFrame columns
COLUMNS = [
    "T_eff", "logg", "R_p_Rj", "a_AU", "phase_deg",
    "band", "f_reflect", "Fp_ref_band", "Fp_th_band", "decision",
]

# ---------------------------------------------------------------------------
# Per-band metrics
# ---------------------------------------------------------------------------

def band_metrics(lam_um, fp_ref, fp_th, bands_dict, thresh=REFLECT_THRESHOLD):
    """
    Compute reflected fraction for each band.

    Returns
    -------
    list of (name, f, Fp_ref_band, Fp_th_band, decision)
    """
    rows = []
    for name, Tband in bands_dict.items():
        f          = frac_reflected(lam_um, fp_ref, fp_th, Tband)
        Fp_ref_band = trapz_band(lam_um, fp_ref, Tband)
        Fp_th_band  = trapz_band(lam_um, fp_th, Tband)
        decision    = (f >= thresh) if np.isfinite(f) else False
        rows.append((name, f, Fp_ref_band, Fp_th_band, decision))
    return rows


# ---------------------------------------------------------------------------
# Full single-case evaluation
# ---------------------------------------------------------------------------

def evaluate_case(sys: SystemParams, lam_grid_um=LAM_GRID,
                  thresh=REFLECT_THRESHOLD, do_plots=False):
    """
    Run PICASO for one system → extract fluxes → compute band metrics.

    Parameters
    ----------
    sys : SystemParams
    lam_grid_um : array
    thresh : float
    do_plots : bool
        If True, produce spectra + bar-chart plots.

    Returns
    -------
    pd.DataFrame  with columns ``COLUMNS``.
    """
    if not HAVE_PICASO:
        raise RuntimeError("PICASO is required")

    import matplotlib.pyplot as plt

    # 1. run PICASO
    out_ref, out_em = run_picaso_once(sys, lam_grid_um)

    # 2. extract absolute planet fluxes
    lam, fp_ref, fp_th = extract_planet_fluxes(
        out_ref, out_em, lam_grid_um, sys,
    )

    # 3. band metrics
    rows = band_metrics(lam, fp_ref, fp_th, BANDS, thresh=thresh)

    # 4. optional plots
    if do_plots:
        suffix = (f" (Teff={sys.teff_k}K, a={sys.a_au}AU, "
                  f"α={sys.phase_deg}°)")
        plot_spectra_with_bb(lam, fp_ref, fp_th, sys, title_suffix=suffix)
        plot_band_bars(rows, title_suffix=suffix)

        # PICASO full-output diagnostics (heatmap of opacities)
        if "full_output" in out_ref:
            try:
                jpi.heatmap_taus(out_ref)
                plt.show()
            except Exception:
                pass
            try:
                jpi.output_notebook()
                jpi.show(jpi.cloud(out_ref["full_output"]))
                jpi.show(jpi.mixing_ratio(out_ref["full_output"]))
            except Exception as e:
                print("Cloud profile plot skipped:", e)

    # 5. build DataFrame
    recs = []
    for name, f, Fp_ref, Fp_th, decision in rows:
        recs.append({
            "T_eff":      sys.teff_k,
            "logg":       sys.logg_cgs,
            "R_p_Rj":     sys.rj,
            "a_AU":       sys.a_au,
            "phase_deg":  sys.phase_deg,
            "band":       name,
            "f_reflect":  float(f) if np.isfinite(f) else np.nan,
            "Fp_ref_band": float(Fp_ref),
            "Fp_th_band":  float(Fp_th),
            "decision":   bool(decision),
        })
    return pd.DataFrame(recs)
