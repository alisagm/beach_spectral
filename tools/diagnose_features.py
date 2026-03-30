# diagnose_features.py
# Run from beach_spectral/: python diagnose_features.py 1995
import sys
import pandas as pd
import numpy as np
from pathlib import Path

year = sys.argv[1] if len(sys.argv) > 1 else "1995"
feat_path = Path(f"OUTPUT/{year}/features_{year}.parquet")
prof_path = Path(f"OUTPUT/{year}/profiles_{year}.parquet")

feat = pd.read_parquet(feat_path)
prof = pd.read_parquet(prof_path)

print(f"\n=== {year} diagnostic ===")
print(f"Feature rows: {len(feat):,}  |  Profile rows: {len(prof):,}")

records = []
for tid, grp in prof.groupby("transect_id"):
    fgrp = feat[feat["transect_id"] == tid]

    d = grp["distance"].values
    nir = grp["nir"].values

    # Distance spacing
    diffs = np.diff(np.sort(d))
    spacing_mean = diffs.mean() if len(diffs) else np.nan
    spacing_std  = diffs.std()  if len(diffs) else np.nan
    spacing_min  = diffs.min()  if len(diffs) else np.nan

    # NIR stats
    nir_nan_pct = np.isnan(nir).mean() * 100

    # Derivative stats
    if len(fgrp) and "nir_d1_smooth" in fgrp.columns:
        d1 = fgrp["nir_d1_smooth"].values
        d1_finite = d1[np.isfinite(d1)]
        d1_max_abs = np.abs(d1_finite).max() if len(d1_finite) else np.nan
        d1_nan_pct = (np.isnan(d1).mean()) * 100
    else:
        d1_max_abs = np.nan
        d1_nan_pct = 100.0

    records.append({
        "transect_id": tid,
        "n_pts": len(grp),
        "dist_min": d.min(),
        "dist_max": d.max(),
        "spacing_mean": spacing_mean,
        "spacing_std": spacing_std,
        "spacing_min": spacing_min,
        "nir_nan_pct": nir_nan_pct,
        "d1_max_abs": d1_max_abs,
        "d1_nan_pct": d1_nan_pct,
    })

df = pd.DataFrame(records)

# Flag suspicious transects
flag_extreme = df[df["d1_max_abs"] > 1000].sort_values("d1_max_abs", ascending=False)
flag_flat    = df[df["d1_nan_pct"] > 80]
flag_spacing = df[df["spacing_std"] > 0.5]   # inconsistent spacing

print(f"\n--- Mode B: extreme |d1| > 1000 ({len(flag_extreme)} transects) ---")
print(flag_extreme[["transect_id","n_pts","spacing_mean","spacing_min","d1_max_abs"]].head(10).to_string(index=False))

print(f"\n--- Mode A: >80% NaN in nir_d1_smooth ({len(flag_flat)} transects) ---")
print(flag_flat[["transect_id","n_pts","nir_nan_pct","d1_nan_pct"]].head(10).to_string(index=False))

print(f"\n--- Irregular spacing (std > 0.5m, {len(flag_spacing)} transects) ---")
print(flag_spacing[["transect_id","n_pts","spacing_mean","spacing_std","spacing_min"]].head(10).to_string(index=False))

print(f"\n--- Overall distance spacing (all transects) ---")
print(df["spacing_mean"].describe().round(4))