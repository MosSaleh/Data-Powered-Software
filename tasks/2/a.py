# Handling missing values for the "energy_dataset.csv" dataset
import pandas as pd
import numpy as np
import os as os

"""
Justification for missing value handling approach:

This function uses a systematic approach to handle missing values based on:
1. Percentage of missing values
2. Data type (categorical vs numerical)
3. Statistical properties of the data

Decision framework:
- 50% missing: Drop column (unreliable for analysis)
- Categorical data: Mode imputation (preserves category distribution)(if any)
- Numerical < 5% missing: Mean imputation (maintains distribution)
- Numerical > 5% missing: Median imputation (robust to outliers)

Could use a technique to compute mean of closest neighbors instead of global mean/median for better accuracy. KNN - K-Nearest Neighbors imputation


!!!
Will move to a KNN imputer for missing.
"""


energy_df = pd.read_csv("datasets/energy_dataset.csv")


def missing_handler(df):
    df_cleaned = df.copy()
    columns_to_drop = []
    fill_values = {}

    # First pass: identify what to do with each column
    for col in df_cleaned.columns:
        missing_count = df_cleaned[col].isnull().sum()
        missing_pct = (missing_count / len(df_cleaned)) * 100

        if missing_count == 0:
            continue

        print(f"\nHandling {col}: {missing_count} missing ({missing_pct:.1f}%)")

        if missing_pct > 50:
            print(f"  -> Dropping column (too much missing data)")
            columns_to_drop.append(col)

        elif df_cleaned[col].dtype == "object":  # Categorical
            mode_val = df_cleaned[col].mode()[0]
            fill_values[col] = mode_val
            print(f"  -> Will fill with mode: {mode_val}")

        else:  # Numerical
            if missing_pct < 5:
                mean_val = df_cleaned[col].mean()
                fill_values[col] = mean_val
                print(f"  -> Will fill with mean: {mean_val:.2f}")
            else:
                median_val = df_cleaned[col].median()
                fill_values[col] = median_val
                print(f"  -> Will fill with median: {median_val:.2f}")

    # Apply changes
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)
    df_cleaned = df_cleaned.fillna(value=fill_values)

    return df_cleaned


# Apply missing handling
energy_df_cleaned = missing_handler(energy_df)


# Compare before and after
print("BEFORE handling missing values:")
print(energy_df.isnull().sum().sum(), "total missing values")
print("\n")
print("\nAFTER handling missing values:")
print(energy_df_cleaned.isnull().sum().sum(), "total missing values")

# Detailed comparison
comparison = pd.DataFrame(
    {
        "Original_Missing": energy_df.isnull().sum(),
        "Cleaned_Missing": [
            (
                energy_df_cleaned[col].isnull().sum()
                if col in energy_df_cleaned.columns
                else "Column Removed"
            )
            for col in energy_df.columns
        ],
        "Filled_Mean": [
            (
                energy_df_cleaned[col].mean()
                if col in energy_df_cleaned.columns
                and energy_df_cleaned[col].dtype != "object"
                else "N/A"
            )
            for col in energy_df.columns
        ],
    }
)
print("\nDetailed Comparison:")
print(comparison[comparison["Original_Missing"] > 0])

# Save the cleaned dataset
output_dir = "datasets"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, "energy_dataset_cleaned.csv")
energy_df_cleaned.to_csv(output_file, index=False)