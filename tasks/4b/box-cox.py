# boxcox_transform_only.py
# pip install pandas numpy scipy

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from scipy import stats
from scipy.stats import skew

# ========= CONFIG =========
CSV_PATH = "datasets/new_cleaned_dataset7.csv"   # endre sti om nødvendig
OUTPUT_CSV = "datasets/boxcox_transformed.csv"   # original + _bc-kolonner
PARAMS_JSON = "tasks/4b/boxcox_params.json"       # lagrer lambda/shift per kolonne
EPS_SHIFT = 1e-6                                  # liten positiv buffer for å sikre > 0
TARGET_COLS = [
    "generation fossil gas",
    "generation hydro pumped storage consumption",
    "generation solar",
]
SUFFIX = "_bc"                                    # suffix for nye kolonner
# =========================

def boxcox_one_series(s: pd.Series, eps=1e-6):
    """
    Tar en pd.Series (kan inneholde NaN), returnerer:
      - y: pd.Series (samme index) med Box–Cox-transformerte verdier (NaN beholdes)
      - lam: lambda fra Box–Cox (float)
      - shift: hvor mye vi skjøv verdiene for å sikre >0 (float)
    """
    mask = s.notna()
    x = s[mask].astype(float)
    if x.empty:
        return s.copy(), None, 0.0

    # sikre strengt positive verdier (krav for Box–Cox)
    minv = float(x.min())
    shift = 0.0
    if minv <= 0:
        shift = -(minv) + eps
    xp = x + shift  # nå > 0

    # Box–Cox på gyldige verdier
    # NB: kan feile for konstante serier — håndter under
    try:
        y_vals, lam = boxcox(xp.values)  # ndarray
        y = pd.Series(np.nan, index=s.index)
        y.loc[mask.index[mask]] = y_vals
        return y, float(lam), float(shift)
    except Exception:
        # Konstante eller numerisk sårbare: returner uendret (med bare shift),
        # men marker lambda=None
        y = pd.Series(np.nan, index=s.index)
        y.loc[mask.index[mask]] = xp.values  # bare shifted
        return y, None, float(shift)

def main():
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise SystemExit(f"Fant ikke CSV: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    df_out = df.copy()

    # Sjekk at alle målkolonnene finnes
    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Mangler kolonner i CSV: {missing}")

    params = {}  # {col: {"lambda": lam, "shift": shift}}

    for col in TARGET_COLS:
        s = df[col]
        # Hvis kolonnen er identisk (konstant) vil Box–Cox ikke være definert -> håndteres i helper
        y, lam, shift = boxcox_one_series(s, eps=EPS_SHIFT)

        # Legg som ny kolonne med suffix
        df_out[col + SUFFIX] = y

        params[col] = {"lambda": lam, "shift": shift}

        print(f"[OK] Box–Cox på '{col}': lambda={lam}, shift={shift}")

    # Lagre resultater
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    Path(PARAMS_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(PARAMS_JSON).write_text(json.dumps(params, indent=2))

    print("\n=== FERDIG ===")
    print(f"Skrev transformert data til: {Path(OUTPUT_CSV).resolve()}")
    print(f"Skrev parametere til:        {Path(PARAMS_JSON).resolve()}")

if __name__ == "__main__":
    main()
