from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# ========= CONFIG =========
CSV_PATH = "datasets/new_cleaned_dataset7.csv"
PLOTS_DIR = "tasks/4b/normality_plots"
SAVE_PLOTS = True                         # sett False hvis du ikke vil lagre figurer
SHOW_PLOTS = False                        # sett True hvis du vil åpne vinduer (kan være tregt)
MIN_POINTS = 20                           # hopp plott for svært korte serier
# =========================

def safe_name(name: str, maxlen: int = 80) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:maxlen].strip("_")

def freedman_diaconis_bins(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 10
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr <= 0:
        return min(50, max(10, int(np.sqrt(n))))
    bw = 2 * iqr * (n ** (-1/3))
    if bw <= 0:
        return 40
    return max(int(np.ceil((x.max() - x.min()) / bw)), 10)

def main():
    csv_path = Path(CSV_PATH)
    if not csv_path.exists():
        raise SystemExit(f"Fant ikke CSV: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        raise SystemExit("Ingen numeriske kolonner funnet i datasettet.")

    plots_dir = Path(PLOTS_DIR)
    if SAVE_PLOTS:
        plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== NORMALITETSSJEKK PER KOLONNE ===")
    for col in num_cols:
        s = df[col]                      # rører ikke originalen
        s_valid = pd.to_numeric(s, errors="coerce").dropna().astype(float)

        # beregninger (uten å endre df)
        if len(s_valid) >= 2:
            sk = float(stats.skew(s_valid, nan_policy="omit"))
            ku = float(stats.kurtosis(s_valid, fisher=True, nan_policy="omit"))
        else:
            sk, ku = np.nan, np.nan

        print(f"\nKolonne: {col}")
        print(f"  Antall gyldige verdier (til beregning/plot): {len(s_valid)}")
        print(f"  Skew: {sk}")
        print(f"  Kurtosis (excess): {ku}")

        # plott (hopper over for korte/konstante serier)
        if len(s_valid) < MIN_POINTS or np.isclose(s_valid.std(ddof=1), 0.0):
            print("  (Hopper over plott: for få punkter eller null variasjon.)")
            continue

        mu, sd = s_valid.mean(), s_valid.std(ddof=1)
        xgrid = np.linspace(s_valid.min(), s_valid.max(), 400)
        bins = freedman_diaconis_bins(s_valid.values)

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))

        # Histogram + normal-kurve (til sammenligning)
        ax[0].hist(s_valid, bins=bins, density=True, alpha=0.7)
        if sd > 0:
            ax[0].plot(xgrid, stats.norm.pdf(xgrid, mu, sd), linestyle="--")
        ax[0].set_title(f"{col} — Histogram vs Normal")

        # Q–Q-plot
        sm.ProbPlot(s_valid, fit=True).qqplot(line="45", ax=ax[1])
        ax[1].set_title(f"{col} — Q–Q")

        plt.tight_layout()

        if SAVE_PLOTS:
            fname = plots_dir / f"{safe_name(col)}_hist_qq.png"
            fig.savefig(fname, dpi=140)
            print(f"  Figurer lagret: {fname}")

        if SHOW_PLOTS:
            plt.show()
        plt.close(fig)

    print("\nFerdig. (Ingen rensing gjort: missing/outliers er ikke håndtert i datasettet.)")

if __name__ == "__main__":
    main()
