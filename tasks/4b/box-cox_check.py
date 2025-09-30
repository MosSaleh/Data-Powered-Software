import pandas as pd
import numpy as np
from scipy.stats import skew

# ---- konfig ----
CSV_PATH = "datasets/boxcox_transformed.csv"
COLS = [
    "generation fossil gas",
    "generation hydro pumped storage consumption",
    "generation solar",
]
SUFFIX = "_bc"
THRESH = 1.0  # krav: |skew| < 1
# ---------------

df = pd.read_csv(CSV_PATH)

rows = []
for col in COLS:
    bc_col = col + SUFFIX
    
    x_after  = pd.to_numeric(df[bc_col], errors="coerce").dropna().astype(float)

    sk_after = np.nan
    if len(x_after) >= 2 and not np.isclose(x_after.std(ddof=1), 0.0):
        sk_after = float(skew(x_after, nan_policy="omit"))

    ok = (np.isfinite(sk_after) and abs(sk_after) < THRESH)
    rows.append({
        "feature": col,
        "skew_after_boxcox": sk_after,
    })

res = pd.DataFrame(rows)
print(res.to_string(index=False))
