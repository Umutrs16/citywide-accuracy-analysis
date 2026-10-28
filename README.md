# Citywide accuracy analysis

Analysis code for the study of land-surface temperature accuracy, heat-adaptation priorities, and cooling-site selection in Urumqi, with a transfer check in Yining.

This repository contains the analysis scripts, the saved JSON summaries, and the CBC logs for the zero-gap location solves. It does not contain the manuscript or the document builder. Source imagery, population rasters, the point tables, and the fitted model are not included. The saved summaries and solver logs can be checked from these files. The remote-sensing analysis cannot be repeated from this repository alone.

The manuscript cites the tag `v1.0-submission`: https://github.com/Umutrs16/citywide-accuracy-analysis/tree/v1.0-submission

## Check the saved results

From the repository root:

```bash
python analysis/verify_reported_numbers.py
```

The script reads `analysis/results`. It reprints the GHSL and WorldPop median coverage gaps, checks that every subset-A solve is marked proven, and checks that each named CBC log contains `Result - Optimal solution found` together with the stored objective. It also reprints the nearest-point mean Jaccard similarity. It does not refit the temperature model and does not resolve the location model.

## Layout

- `analysis/` holds `analysis.py` and `revision.py` through `revision5.py`.
- `src/` holds the model settings, spatial cross-validation, and covering helpers used by those scripts.
- `config/settings.yaml` holds the model hyperparameters and the 2020 census totals.
- `analysis/results/` holds the JSON summaries, `cbc_logs/`, and the supplementary scenario tables. In `Supplementary_Table_S1_scenario_gaps.csv` the CBC objective is unscaled raster population. The coverage columns are census-scaled. `Supplementary_Table_S2_cbc_logs.csv` lists the logs in numeric file order.

## Inputs that stay outside this repository

Landsat, Sentinel, ERA5, GHSL, WorldPop, NASADEM, and OpenStreetMap products stay with their publishers. Re-running the scripts requires the project point tables, rasters, and fitted model. Do not re-run `revision2.py`; its saved JSON is the record of that run.

## Solver record

The zero-gap solves used CBC 2.10.3 through PuLP with `gapRel=0` and `gapAbs=0`. A solve is treated as proven only when the log contains `Result - Optimal solution found`. Census scaling uses 4,054,369, separately for each population raster. Residual fields use NumPy seed 21. The recall intervals use seed 11.
