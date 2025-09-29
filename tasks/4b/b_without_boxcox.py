from pathlib import Path
import json
import numpy as np
import pandas as pd

# ========= CONFIG =========
CSV_PATH = "datasets/new_cleaned_dataset7.csv"
OUTPUT_ZSCORED = "datasets/feature_scaled_zscore.csv"  # ny tabell med KUN z-scorede verdier
PARAMS_JSON = "tasks/4/zscore_params.json"            # lagrer mean/std per kolonne
# =========================

def zscore_scale_numeric_only(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
    """
    Returnerer:
      - z_df: DataFrame med KUN numeriske kolonner i z-score-form (samme kolonnenavn)
      - params: {col: {"mean": μ, "std": σ}}
      - cols: liste over kolonnene som ble skalert
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        raise SystemExit("Ingen numeriske kolonner å skalere.")

    z_df = pd.DataFrame(index=df.index)
    params = {}
    for col in num_cols:
        x = df[col].astype(float)
        mu = float(x.mean())
        sd = float(x.std(ddof=0))  # ddof=0 er vanlig i standard scalere
        if not np.isfinite(sd) or np.isclose(sd, 0.0):
            # Konstant kolonne: unngå deling på 0
            sd = 1.0
        z_df[col] = (x - mu) / sd
        params[col] = {"mean": mu, "std": sd}
    return z_df, params, num_cols

def main():
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise SystemExit(f"Fant ikke CSV: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    # Z-score på alle numeriske kolonner (NaN beholdes som NaN)
    z_df, params, cols = zscore_scale_numeric_only(df)

    # Lagre tabellen med z-scorede verdier (kun numeriske features)
    z_df.to_csv(OUTPUT_ZSCORED, index=False)

    # Lagre parametre (nyttig når du senere vil bruke samme μ/σ på testsett)
    Path(PARAMS_JSON).write_text(json.dumps(params, indent=2))

    # Oppsummering
    print("\n=== Z-SCORE SCALING FULLFØRT ===")
    print(f"Antall numeriske kolonner skalert: {len(cols)}")
    print("Eksempelkolonner:", cols[: min(5, len(cols))])
    print(f"Wrote: {Path(OUTPUT_ZSCORED).resolve()}")
    print(f"Wrote params: {Path(PARAMS_JSON).resolve()}")

if __name__ == "__main__":
    main()
