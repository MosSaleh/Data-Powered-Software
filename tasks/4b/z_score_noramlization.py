
from pathlib import Path
import json
import numpy as np
import pandas as pd

# ==== CONFIG ====
CSV_PATH = "datasets/boxcox_transformed.csv"     # sett din filsti
OUTPUT_ZSCORES = "datasets/zscores_table.csv"    # kun z-scorede verdier
PARAMS_JSON = "tasks/4b/zscore_params.json"      # lagrer mean/std per kolonne (nyttig videre)
# ===============

def compute_zscores(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    """
    Returnerer:
      - Z (DataFrame): kun numeriske kolonner i z-score-form (samme kolonnenavn)
      - params (dict): {col: {"mean": μ, "std": σ}} brukt i skaleringen
      - cols (list): liste over kolonnene som ble skalert
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        raise SystemExit("Ingen numeriske kolonner funnet.")

    Z = pd.DataFrame(index=df.index)
    params = {}
    for col in num_cols:
        x = pd.to_numeric(df[col], errors="coerce").astype(float)
        mu = float(x.mean())
        sd = float(x.std(ddof=0))  # standard praksis i scalere
        if not np.isfinite(sd) or np.isclose(sd, 0.0):
            # konstant kolonne → alle z blir 0
            sd = 1.0
        Z[col] = (x - mu) / sd
        params[col] = {"mean": mu, "std": sd}
    return Z, params, num_cols

def main():
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise SystemExit(f"Fant ikke CSV: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    Z, params, cols = compute_zscores(df)
    Z.to_csv(OUTPUT_ZSCORES, index=False)

    Path(PARAMS_JSON).write_text(json.dumps(params, indent=2))

    print("\n=== Z-SCORES GENERERT ===")
    print(f"Antall numeriske kolonner: {len(cols)}")
    print("Eksempelkolonner:", cols[: min(5, len(cols))])
    print(f"Skrev z-score-tabell: {Path(OUTPUT_ZSCORES).resolve()}")
    print(f"Skrev parametere:    {Path(PARAMS_JSON).resolve()}")

if __name__ == "__main__":
    main()
