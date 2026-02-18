"""
roadrunner.runner
~~~~~~~~~~~~~~~~
PICASO execution: run reflected + thermal spectra and extract absolute
planet fluxes with correct unit conversions.
"""

import os

import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from astropy.constants import R_jup, R_sun, au

from .config import (
    HAVE_PICASO,
    jdi,
    blackbody,
    REFLECT_NUM_GANGLE,
    REFLECT_NUM_TANGLE,
    THERMAL_NUM_GANGLE,
    THERMAL_NUM_TANGLE,
)
from .system import SystemParams, resolve_slgrid_files

# ---------------------------------------------------------------------------
# Run PICASO (reflected + thermal)
# ---------------------------------------------------------------------------

def run_picaso_once(sys: SystemParams, lam_grid_um: np.ndarray):
    """
    Run PICASO for both reflected and thermal spectra.

    Parameters
    ----------
    sys : SystemParams
        Planet / star system configuration.
    lam_grid_um : array
        Wavelength grid in µm (used only to define the opacity range).

    Returns
    -------
    out_ref, out_em : dict
        PICASO output dictionaries for reflected and thermal calculations.
    """
    assert HAVE_PICASO, "PICASO is required"

    opa = jdi.opannection(wave_range=[0.3, 1.0])

    case = jdi.inputs()
    g_cgs = 10 ** sys.logg_cgs
    case.gravity(
        gravity=g_cgs, gravity_unit=u.cm / u.s**2,
        radius=sys.rj, radius_unit=u.R_jup,
    )
    case.star(
        opa, temp=sys.tstar_k, metal=0, logg=4.44,
        radius=sys.rstar_rsun, radius_unit=u.R_sun,
        semi_major=sys.a_au, semi_major_unit=u.AU,
    )

    # SLGRID atmosphere + clouds
    pt_path, cld_path = resolve_slgrid_files(sys)
    case.atmosphere(filename=pt_path, delim_whitespace=True)
    case.clouds(filename=cld_path, delim_whitespace=True)
    print(f"✓ Using SLGRID PT:  {os.path.basename(pt_path)}")
    print(f"✓ Using SLGRID CLD: {os.path.basename(cld_path)}")

    # Reflected at requested phase  (full_output for diagnostics)
    case.phase_angle(
        np.deg2rad(sys.phase_deg),
        num_gangle=REFLECT_NUM_GANGLE,
        num_tangle=REFLECT_NUM_TANGLE,
    )
    out_ref = case.spectrum(opa, calculation="reflected",
                            as_dict=True, full_output=True)

    # Thermal at zero phase (1-D thermal must be phase=0)
    case.phase_angle(
        0.0,
        num_gangle=THERMAL_NUM_GANGLE,
        num_tangle=THERMAL_NUM_TANGLE,
    )
    out_em = case.spectrum(opa, calculation="thermal", as_dict=True)

    return out_ref, out_em


# ---------------------------------------------------------------------------
# Extract absolute fluxes
# ---------------------------------------------------------------------------

def extract_planet_fluxes(out_ref: dict, out_em: dict,
                          lam_grid_um: np.ndarray,
                          sys: SystemParams):
    """
    Extract absolute planet fluxes (reflected + thermal) from PICASO
    output dicts.

    Returns
    -------
    lam_um, fp_reflected, fp_thermal : ndarray
        All on *lam_grid_um* in erg s⁻¹ cm⁻² µm⁻¹.
    """
    # --- reflected ---
    wno_ref    = out_ref["wavenumber"]
    lam_cm_ref = 1.0 / wno_ref
    wl_ref_um  = lam_cm_ref * 1e4

    fpfs_data = out_ref.get("fpfs_reflected", None)
    if isinstance(fpfs_data, np.ndarray):
        fpfs_ref = fpfs_data
    else:
        albedo   = out_ref["albedo"]
        Rp_cm    = sys.rj * R_jup.value
        a_cm     = sys.a_au * au.value
        fpfs_ref = albedo * (Rp_cm / a_cm) ** 2

    Fs_per_cm     = np.pi * np.squeeze(blackbody(sys.tstar_k, lam_cm_ref))
    Fp_ref_per_cm = fpfs_ref * Fs_per_cm       # erg/cm²/s/cm
    fp_ref_raw    = Fp_ref_per_cm * 1e-4        # → per µm

    fp_reflected = interp1d(
        wl_ref_um, fp_ref_raw, bounds_error=False, fill_value=0.0,
    )(lam_grid_um)

    # --- thermal ---
    wno_em    = out_em["wavenumber"]
    lam_cm_em = 1.0 / wno_em
    wl_em_um  = lam_cm_em * 1e4

    fp_th_raw   = out_em["thermal"]           # erg/cm²/s/cm
    fp_th_per_um = fp_th_raw * 1e-4           # → per µm
    fp_thermal  = interp1d(
        wl_em_um, fp_th_per_um, bounds_error=False, fill_value=0.0,
    )(lam_grid_um)

    # clean NaNs
    fp_reflected = np.nan_to_num(fp_reflected, nan=0.0, posinf=0.0, neginf=0.0)
    fp_thermal   = np.nan_to_num(fp_thermal,   nan=0.0, posinf=0.0, neginf=0.0)

    return lam_grid_um, fp_reflected, fp_thermal
