import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('energy_dataset.csv')

# First, let's examine the missing data pattern
def analyze_missing_data(df):
    missing_info = df.isnull().sum()
    missing_percent = (missing_info / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing_info,
        'Missing_Percentage': missing_percent
    })
    return missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

print("Missing Data Analysis:")
print(analyze_missing_data(df))
"""
Methods to handle missing values

Legger inn gjennomsnitt av noen dager før og noen dager etter
"""