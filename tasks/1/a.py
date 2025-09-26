import pandas as pd
import numpy as np

energy_df = pd.read_csv("datasets/energy_dataset.csv")
weather_df = pd.read_csv("datasets/weather_features.csv")

# Printing the first columns of data of both datasets
print("The first columns of energy_df")
print(energy_df.head())
print("The first columns of weather_df")
print(weather_df.head())

# Getting quick stats about the data in both data sets
print("Stats - energy_df")
print(energy_df.describe())
print("Stats - weather_df")
print(weather_df.describe())

print("Info - energy_df")
print(energy_df.info())
print(energy_df.dtypes)
print("Info - weather_df")
print(weather_df.info())
print(weather_df.dtypes)

# Part B: Data Quality
print("\nMissing Values:")
print(energy_df.isnull().sum())

print("\nMissing Values Percentage:")
print((energy_df.isnull().sum() / len(energy_df)) * 100)

# For categorical columns
categorical_cols = energy_df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\nUnique values in {col}:")
    print(energy_df[col].value_counts())

# Outlier detection (example for numerical columns)
numerical_cols = energy_df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    Q1 = energy_df[col].quantile(0.25)
    Q3 = energy_df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = energy_df[(energy_df[col] < Q1 - 1.5*IQR) | (energy_df[col] > Q3 + 1.5*IQR)]
    print(f"Outliers in {col}: {len(outliers)}")

summary_df = pd.DataFrame({
    'Column': energy_df.columns,
    'Data_Type': energy_df.dtypes,
    'Missing_Values': energy_df.isnull().sum(),
    'Missing_Percentage': (energy_df.isnull().sum() / len(energy_df)) * 100,
    'Unique_Values': [energy_df[col].nunique() for col in energy_df.columns]
})
print("\nData Quality Summary:")
print(summary_df)

