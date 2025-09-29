from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import boxcox

# ========= CONFIG =========
CSV_PATH = "datasets/new_cleaned_dataset7.csv"
OUTPUT_ZSCORED = "datasets/feature_scaled_zscore.csv"   # ny tabell med KUN z-scorede verdier
PARAMS_JSON = "tasks/4/zscore_params.json"              # lagrer Box-Cox + Z-score-parametre
DUMMY_MAX_UNIQUE = 2                                    # ≤2 unike verdier tolkes som dummy
EPS_SHIFT = 1e-6                                        # liten positiv shift for å sikre >0
# =========================

def boxcox_then_zscore_numeric_only(df: pd.DataFrame):
    """
    For hver numerisk kolonne:
      1) (Hvis ikke dummy) Box–Cox med automatisk lambda. Hvis min<=0, shift til >0 først.
      2) Z-score på den (ev. transformerte) kolonnen.
    Returnerer:
      - z_df: DataFrame med KUN numeriske kolonner i z-score-form (samme kolonnenavn)
      - params: {col: {"boxcox": {"lambda": lam, "shift": shift, "skipped": bool},
                       "zscore": {"mean": mu, "std": sd}}}
      - cols: liste over kolonnene som ble skalert
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        raise SystemExit("Ingen numeriske kolonner å skalere.")

    z_df = pd.DataFrame(index=df.index)
    params = {}

    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")  # bevar NaN
        mask = s.notna()
        x = s[mask].astype(float)

        # init param-logging
        params[col] = {"boxcox": {"lambda": None, "shift": 0.0, "skipped": False},
                       "zscore": {"mean": None, "std": None}}

        # Dummy/konstant? → hopp Box–Cox, Z-score direkte
        nunique = x.nunique(dropna=True)
        if nunique <= DUMMY_MAX_UNIQUE or np.isclose(x.std(ddof=0), 0.0) or len(x) < 2:
            # Z-score direkte på originalen
            mu = float(x.mean())
            sd = float(x.std(ddof=0))
            if not np.isfinite(sd) or np.isclose(sd, 0.0):
                sd = 1.0
            z = pd.Series(np.nan, index=s.index)
            z[mask] = (x - mu) / sd

            z_df[col] = z
            params[col]["boxcox"]["skipped"] = True
            params[col]["zscore"]["mean"] = mu
            params[col]["zscore"]["std"] = sd
            continue

        # --- Box–Cox: sikre >0 via shift om nødvendig ---
        shift = 0.0
        minv = float(x.min())
        if minv <= 0:
            shift = -(minv) + EPS_SHIFT  # så min blir > 0
        xp = x + shift

        # Box–Cox krever alle > 0 og ingen NaN (vi har maskert NaN)
        try:
            y, lam = boxcox(xp.values)  # ndarray
        except Exception as e:
            # Fallback: hvis det mot formodning feiler, gjør log1p og marker skip
            y = np.log1p(x.values - minv + EPS_SHIFT)  # garanter >0 før log1p
            lam = None
            params[col]["boxcox"]["skipped"] = True

        params[col]["boxcox"]["lambda"] = None if lam is None else float(lam)
        params[col]["boxcox"]["shift"] = float(shift)

        # Sett den transformer­te serien tilbake på samme index (bevar NaN)
        t = pd.Series(np.nan, index=s.index)
        t[mask] = y

        # --- Z-score på transformert kolonne ---
        mu = float(np.nanmean(t))
        sd = float(np.nanstd(t, ddof=0))
        if not np.isfinite(sd) or np.isclose(sd, 0.0):
            sd = 1.0

        z = (t - mu) / sd
        z_df[col] = z

        params[col]["zscore"]["mean"] = mu
        params[col]["zscore"]["std"] = sd

    return z_df, params, num_cols

def main():
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise SystemExit(f"Fant ikke CSV: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    # Box–Cox -> Z-score på alle numeriske kolonner (NaN beholdes som NaN)
    z_df, params, cols = boxcox_then_zscore_numeric_only(df)

    # Lagre KUN z-scorede verdier (samme kolonnenavn)
    z_df.to_csv(OUTPUT_ZSCORED, index=False)

    # Lagre parametre (Box–Cox shift/λ + Z-score μ/σ)
    Path(PARAMS_JSON).write_text(json.dumps(params, indent=2))

    # Oppsummering
    print("\n=== BOX–COX → Z-SCORE FULLFØRT ===")
    print(f"Antall numeriske kolonner skalert: {len(cols)}")
    print("Eksempelkolonner:", cols[: min(5, len(cols))])
    print(f"Wrote: {Path(OUTPUT_ZSCORED).resolve()}")
    print(f"Wrote params: {Path(PARAMS_JSON).resolve()}")

if __name__ == "__main__":
    main()
