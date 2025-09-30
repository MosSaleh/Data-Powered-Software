import numpy as np
import pandas as pd

# File that verifies that the data in feature_scaled_zscore actually is normally distributed
# Data is normally distributed => mean = 0, std = 1

dfz = pd.read_csv("datasets/feature_scaled_zscore.csv")  # filen du skrev ut
num_cols = dfz.select_dtypes(include=np.number).columns

check = []
for c in num_cols:
    s = dfz[c].dropna().astype(float)
    check.append({
        "col": c,
        "mean": round(s.mean(), 5),
        "std": round(s.std(), 5),
    })
pd.DataFrame(check)

# Printing mean and std for each column => should be 0 and 1 if column is normally distributed
for i in check:
    print(i)
