"""
Utility functions for loading and reading spectral band configurations.
"""

import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical band-mode string constants
# These replace the deprecated BAND_CONFIG_* constants from utils/data_io.py.
# All transition modules and callers should import from here.
# ---------------------------------------------------------------------------
 
BAND_MODE_4BAND = "4band"  # 4-band RGBN imagery (NIR + blue both available)
BAND_MODE_CIR   = "cir"   # 3-band CIR imagery  (NIR available, blue absent)
BAND_MODE_RGB   = "rgb"   # 3-band RGB imagery  (no NIR)

# ---------------------------------------------------------------------------
# Band configuration helpers
# ---------------------------------------------------------------------------

def load_band_config(band_config_path: Path, year: str) -> dict:
    """
    Load the band configuration entry for a single year from band_config.json.

    Args:
        band_config_path: Path to band_config.json
        year: Year string, e.g. "2016"

    Returns:
        Dict with keys: format, nir_band, otsu_threshold, band_count

    Raises:
        KeyError: If year is not present in the config file
        FileNotFoundError: If band_config.json does not exist
    """
    with open(band_config_path) as f:
        full_config = json.load(f)

    if year not in full_config:
        raise KeyError(
            f"Year '{year}' not found in {band_config_path}. "
            f"Available years: {sorted(full_config.keys())}"
        )

    return full_config[year]


def resolve_band_indices(year_config: dict) -> Dict[str, Optional[int]]:
    """
    Resolve 1-based band indices for red, green, blue, and nir from a year config entry.

    For CIR and RGBNIR, the non-NIR bands are assigned in ascending index order:
        CIR    (3-band): non-NIR bands → [red, green]
        RGBNIR (4-band): non-NIR bands → [red, green, blue]

    This matches the standard sensor conventions that assign_band_labels.py was
    calibrated against (physics-invariant ranking: Red > Green). If your dataset
    deviates from this ordering, add a manual override in band_config.json and
    extend this function.

    Args:
        year_config: Single year entry from band_config.json

    Returns:
        Dict with keys 'red', 'green', 'blue', 'nir', values are 1-based int or None
    """
    fmt = year_config["format"].upper().replace("-", "")
    nir_band = year_config.get("nir_band")  # 1-based, None for RGB
    band_count = year_config["band_count"]

    if fmt == "RGB":
        return {"red": 1, "green": 2, "blue": 3, "nir": None}

    elif fmt == "CIR":
        # Two non-NIR bands; assign in ascending index order → [red, green]
        non_nir = sorted(i for i in range(1, band_count + 1) if i != nir_band)
        if len(non_nir) != 2:
            raise ValueError(
                f"CIR format expects 3 bands total, got {band_count} "
                f"(nir_band={nir_band})"
            )
        return {"red": non_nir[0], "green": non_nir[1], "blue": None, "nir": nir_band}

    elif fmt in ("RGBNIR", "RGBN", "4BAND"):
        # Three non-NIR bands; assign in ascending index order → [red, green, blue]
        non_nir = sorted(i for i in range(1, band_count + 1) if i != nir_band)
        if len(non_nir) != 3:
            raise ValueError(
                f"RGBNIR format expects 4 bands total, got {band_count} "
                f"(nir_band={nir_band})"
            )
        return {
            "red": non_nir[0],
            "green": non_nir[1],
            "blue": non_nir[2],
            "nir": nir_band,
        }

    else:
        raise ValueError(
            f"Unrecognised format '{year_config['format']}'. "
            "Expected one of: RGB, CIR, RGBNIR, RGBN, 4BAND."
        )
    
def band_mode_from_indices(band_indices: dict) -> str:
    """
    Derive band mode string from resolved band indices.

    Returns:
        '4band' if NIR and blue are both present
        'cir'   if NIR is present but blue is not
        'rgb'   if NIR is absent
    """
    if band_indices["nir"] is not None and band_indices["blue"] is not None:
        return "4band"
    elif band_indices["nir"] is not None:
        return "cir"
    else:
        return "rgb"
    
def detect_band_mode_from_features(features: pd.DataFrame) -> str:
    """
    Derive band mode from a computed features DataFrame.
 
    compute.py splits outputs into two Parquets: raw band values go to
    profiles_{year}.parquet, while features_{year}.parquet contains only
    derived columns.  The raw 'nir' and 'blue' columns are therefore
    absent from the features DataFrame — band availability must be inferred
    from the smoothed first-derivative columns instead:
 
        nir_d1_smooth  → present & non-null  iff  year has NIR band
        blue_d1_smooth → present & non-null  iff  year has blue band
 
    Used as a fallback inside TransitionDetector when band_mode is not
    supplied by the caller.  Prefer passing band_mode explicitly from the
    interpret layer where the year config is already known.
 
    Args:
        features: DataFrame with spectral feature columns produced by
                  compute.py / features.py.
 
    Returns:
        One of BAND_MODE_4BAND, BAND_MODE_CIR, BAND_MODE_RGB.
    """
    has_nir  = ("nir_d1_smooth"  in features.columns
                and not features["nir_d1_smooth"].isna().all())
    has_blue = ("blue_d1_smooth" in features.columns
                and not features["blue_d1_smooth"].isna().all())
 
    if has_nir and has_blue:
        return BAND_MODE_4BAND
    elif has_nir:
        return BAND_MODE_CIR
    else:
        return BAND_MODE_RGB