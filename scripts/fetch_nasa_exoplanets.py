"""
scripts/fetch_nasa_exoplanets.py

Downloads planetary system parameters from NASA Exoplanet Archive TAP service.
Pulls both Jupiter and Earth mass estimates to ensure terrestrial systems (like TRAPPIST-1)
are captured.
"""

from pathlib import Path

import pandas as pd
import requests

NASA_TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "pl_name",
    "hostname",
    "sy_snum",
    "sy_pnum",
    "pl_orbsmax",  # semi-major axis (AU)
    "pl_orbper",  # orbital period (days)
    "pl_bmassj",  # mass in Jupiter masses
    "pl_bmasse",  # mass in Earth masses
    "pl_rade",  # radius in Earth radii
    "pl_orbeccen",  # eccentricity
    "pl_orbincl",  # inclination (deg)
    "pl_orblper",  # argument of periastron (deg)
    "st_mass",  # host star mass (solar masses)
]


def query_nasa_exoplanet_archive() -> pd.DataFrame | None:
    query = (
        f"SELECT {', '.join(COLUMNS)} "
        f"FROM ps "
        f"WHERE default_flag=1 "
        f"AND pl_orbsmax IS NOT NULL "
        f"AND (pl_bmassj IS NOT NULL OR pl_bmasse IS NOT NULL) "
        f"AND st_mass IS NOT NULL"
    )

    print(f"Querying NASA Exoplanet Archive:\n{query}\n")
    params = {"query": query, "format": "csv"}

    try:
        response = requests.get(NASA_TAP_URL, params=params, timeout=60)
        response.raise_for_status()

        csv_path = RAW_DATA_DIR / "exoplanets_raw.csv"
        csv_path.write_text(response.text)

        df = pd.read_csv(csv_path)
        print(
            f"✅ Downloaded {len(df)} planets across {df['hostname'].nunique()} systems -> {csv_path}"
        )
        return df

    except Exception as e:
        print(f"❌ Error querying NASA TAP: {e}")
        return None


if __name__ == "__main__":
    query_nasa_exoplanet_archive()
