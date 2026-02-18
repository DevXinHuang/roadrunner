# ROADRUNNER

**R**oman **O**bservational **A**lbedo & **D**irect-imaging **R**etrieval for **U**nified **N**ear-infrared & optical **N**oise & **E**mission **R**atios

A modular pipeline for evaluating reflected and thermal contributions in Roman CGI bands, using PICASO + Virga.

## Quick Start

```python
from roadrunner import SystemParams, evaluate_case, run_grid_parallel

# Single case
case = SystemParams(teff_k=1000, logg_cgs=3.5, rj=1.0,
                    a_au=10.0, phase_deg=60.0)
df = evaluate_case(case, do_plots=True)

# Full parameter grid
df_all = run_grid_parallel()
df_all.to_csv("results.csv", index=False)
```

## Package Structure

```
roadrunner/
├── __init__.py     # Public API
├── config.py       # Environment, constants, SLGRID paths
├── physics.py      # Planck function, bandpass filters, unit conversions
├── system.py       # SystemParams dataclass, SLGRID file resolver
├── runner.py       # PICASO reflected + thermal execution
├── bands.py        # Band metrics, evaluate_case() pipeline
├── plotting.py     # Spectra plots, bar charts, heatmaps
└── grid.py         # Parallel parameter sweep
```

## Driver Notebook

Open `run_roadrunner.ipynb` in Jupyter with the **PICASO** kernel to run the full analysis:
- 4 validation cases (A–D) covering different temperature/distance regimes
- Optional full parameter grid sweep
- Summary visualisations (histograms, heatmaps)

## Requirements

- Python 3.10+
- [PICASO](https://natashabatalha.github.io/picaso/) + Virga
- NumPy, Pandas, Matplotlib, SciPy, Astropy
- SLGRID climate/cloud data files (see `config.py` for path setup)

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TIMESTEP_ROOT` | parent of this project | Path to the `timestep/` directory containing `picaso/` and SLGRID data |
| `SLGRID_BASE_DIR` | `$TIMESTEP_ROOT/2_9_2026` | Path to SLGRID data directory |
| `picaso_refdata` | `$TIMESTEP_ROOT/picaso/reference` | PICASO reference data |
| `PYSYN_CDBS` | `$TIMESTEP_ROOT/picaso/reference/synphot3` | Pysynphot CDBS data |
