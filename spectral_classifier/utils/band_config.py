"""
Utility functions for loading and reading spectral band configurations.
"""

import json
from pathlib import Path
from typing import Dict, Optional

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