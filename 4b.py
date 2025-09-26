import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL

"""
Spørsmål:
1. Hvorfor får jeg ikke graf for alle kolonnene? Mistenker at kolonnene har for få datapunkter
"""

# Checking if the data is normally distributed
# ---

df = pd.read_csv("datasets/energy_dataset.csv") # tilpass sti når vi flytter 4b-fila

# 1) Tidsstempel i UTC, sett som index
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
df = df.set_index("time").sort_index()

# 2) Time-raster + aggreger, så fylle hull tidsmessig # HANDLING MISSING VALUES
df = df.resample("h").mean(numeric_only=True)
df = df.interpolate(method="time", limit_direction="both")

# 3) Riktig kolonnenavn (med mellomrom i denne CSV-en)
target_col = "total load actual"
y = df[target_col].dropna()

# 4) STL (period=24 for timeserie)
stl = STL(y, period=24, robust=True).fit()
df["resid"] = stl.resid



# --- 4) Velg numeriske kolonner (nå er num_cols definert) ---
num_cols = df.select_dtypes(include=np.number).columns
col_id_map = {col: i for i, col in enumerate(num_cols, start=1)}


# skew and kurtosis
# skew = "skjevhet" - is the graph tilted to the left (neg skew) or right (pos skew)?
# kurtosis = sier noe om hvor spiss toppen i fordelinga er
# |skew| < 0.5 og |kurtosis_excess| < 1  => ofte "greit nok" nær normal
summary = []
summary_to_use_later = {}
for col in num_cols:
    s = df[col].dropna() # HANDLING MISSING VALUES: REMOVING THEM
    if len(s) < 20:  # for få punkt til å si noe fornuftig - droppe??
        continue
    summary.append({
        "col_id": col_id_map[col],
        "feature": col,
        "n": len(s),
        "mean": s.mean(),
        "std": s.std(), # standardavvik
        "skew": stats.skew(s, nan_policy="omit"),
        "kurtosis_excess": stats.kurtosis(s, fisher=True, nan_policy="omit")
    })
    summary_to_use_later[col_id_map[col]] = [s.mean(), s.std()]
screen = pd.DataFrame(summary).sort_values("feature")
print(screen)



# 2) Normalitetstest (ikke stol blindt på p-verdi når n er stort!)
# Bruk D’Agostino K^2 (passer greit for mange n). 
# For svært store n: test på en tilfeldig delmengde, f.eks. 5000 punkter.
test_results = []
for col in num_cols:
    s = df[col].dropna() # FJERNER MISSING VALUES
    if len(s) < 20: # fjerne?
        continue
    x = s.sample(n=min(5000, len(s)), random_state=42)  # sub-sample
    k2, p = stats.normaltest(x)  # H0: normalfordelt
    test_results.append({"feature": col, "K2": k2, "p_value": p})
tests = pd.DataFrame(test_results).sort_values("p_value", ascending=False)
print("\n=== NORMALITETSTEST ===")
print(tests)

# --- 7) Plott: histogram + Q–Q for ALLE numeriske kolonner (inkl. resid) ---

def freedman_diaconis_bins(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return 10
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr == 0:
        return min(50, max(10, int(np.sqrt(len(x)))))  # fallback
    bin_width = 2 * iqr * (len(x) ** (-1/3))
    if bin_width <= 0:
        return 40
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return max(bins, 10)

# Valgfritt: lagre figurer i en mappe
# import os; os.makedirs("plots", exist_ok=True)

for col in num_cols:
    s = df[col].dropna().astype(float)

    # hopp over små/konstante serier (Q–Q krever spredning)
    if len(s) < 20 or np.isclose(s.std(ddof=1), 0.0):
        print(f"⚠️ Hopper over '{col}' (for få/konstant).")
        continue

    mu, sd = s.mean(), s.std(ddof=1)
    xgrid = np.linspace(s.min(), s.max(), 400)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram med normal-overlay
    bins = freedman_diaconis_bins(s)
    ax[0].hist(s, bins=bins, density=True, alpha=0.7)
    # Normal(μ,σ) – bare hvis sd>0
    if sd > 0:
        ax[0].plot(xgrid, stats.norm.pdf(xgrid, mu, sd), linestyle="--")
    
    col_id = col_id_map[col]
    ax[0].set_title(f"[{col_id}] {col}") # colID = et nummer jeg gir kolonnen, col = navnet på kolonnen i datasettet

    # Q–Q-plot
    sm.ProbPlot(s, fit=True).qqplot(line="45", ax=ax[1])
    ax[1].set_title(f"{col}")

    plt.tight_layout()
    plt.show()

    # Valgfritt: lagre figur
    # fig.savefig(f"plots/{col}_hist_qq.png", dpi=150)
    plt.close(fig)



# Skriver en metode for standardization / Z-score normalization
colIDs_for_standardization = [1, 6, 12, 17, 25, 26, 27, 28]

def standardization(colID): # antar at jeg skal standardisere alle verdiene i kolonna?
    # hente ut verdiene i kolonns
    # iterer gjennom kolonna
    # for hver verdi i kolonna
        # beregne standardization = z-value
    # lage et nytt datasett med stadardized values?
    # hva skal jeg med de standardiserte verdiene?
    return 0

# Skriver en metode for min-max scaling
