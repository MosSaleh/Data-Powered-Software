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

        # Categorical
        elif df_cleaned[col].dtype == 'object':  
            mode_val = df_cleaned[col].mode()[0]
            fill_values[col] = mode_val
            print(f"  -> Will fill with mode: {mode_val}")

        # Numerical    
        else:  
            if missing_pct < 6:
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

def daily_window_missing_handler(df, days_window=7):
    """
    Uses same time from EVERY day within a window (past and future) to calculate mean
    """
    df_cleaned7 = df.copy()
    columns_to_drop = []
    
    # Convert time column to datetime
    df_cleaned7['time'] = pd.to_datetime(df_cleaned7['time'], utc=True)

    for col in df_cleaned7.columns:
        if col == 'time':
            continue

        missing_count = df_cleaned7[col].isnull().sum()
        missing_pct = (missing_count / len(df_cleaned7)) * 100

        if missing_count == 0:
            continue
            
        print(f"\nHandling {col}: {missing_count} missing ({missing_pct:.1f}%)")
        
        if missing_pct > 50:
            print(f"  -> Dropping column (too much missing data)")
            columns_to_drop.append(col)
        elif df_cleaned7[col].dtype == 'object':
            mode_val = df_cleaned7[col].mode()[0]
            df_cleaned7[col].fillna(mode_val, inplace=True)
            print(f"  -> Filled with mode: {mode_val}")
        else:
            filled_count = 0
            fallback_count = 0

            for idx in df_cleaned7[df_cleaned7[col].isnull()].index:
                current_time = df_cleaned7.loc[idx, 'time']
                current_hour_minute = current_time.time()  # Get just hour:minute
                
                # Create time window (e.g., 7 days before and after)
                start_date = current_time.date() - pd.Timedelta(days=days_window)
                end_date = current_time.date() + pd.Timedelta(days=days_window)
                
                # Find ALL days within window that have same time and non-null values
                same_time_mask = (
                    (df_cleaned7['time'].dt.time == current_hour_minute) &  # Same hour:minute
                    (df_cleaned7['time'].dt.date >= start_date) &           # Within date range
                    (df_cleaned7['time'].dt.date <= end_date) &             # Within date range
                    (df_cleaned7['time'].dt.date != current_time.date()) &  # Exclude current day
                    (df_cleaned7[col].notna())                              # Non-null values
                )

                same_time_values = df_cleaned7[same_time_mask][col]

                if len(same_time_values) > 0:
                    # Use mean of ALL available same-time values from different days
                    fill_value = same_time_values.mean()
                    df_cleaned7.loc[idx, col] = fill_value
                    filled_count += 1
                    print(f"    Used {len(same_time_values)} values from different days")
                else:
                    # Fallback to original strategy
                    if missing_pct < 6:
                        fill_value = df_cleaned7[col].mean()
                    else:
                        fill_value = df_cleaned7[col].median()
                    df_cleaned7.loc[idx, col] = fill_value
                    fallback_count += 1
            
            print(f"  -> Filled {filled_count} values using daily window mean")
            if fallback_count > 0:
                print(f"  -> Fallback: {fallback_count} values")
    
    df_cleaned7 = df_cleaned7.drop(columns=columns_to_drop)
    return df_cleaned7








# Apply missing handling
energy_df_cleaned = missing_handler(energy_df)
energy_df_cleaned.to_csv("datasets/new_cleaned_dataset.csv", index=False)

# Compare before and after
print("BEFORE handling missing values:")
print(energy_df.isnull().sum().sum(), "total missing values")
print("\n")
print("\nAFTER handling missing values:")
print(energy_df_cleaned.isnull().sum().sum(), "total missing values")

# Detailed comparison
comparison = pd.DataFrame({
    "Original_Missing": energy_df.isnull().sum(),
    "Cleaned_Missing": [energy_df_cleaned[col].isnull().sum() if col in energy_df_cleaned.columns else 'Column Removed' for col in energy_df.columns],
    "Filled_Mean": [energy_df_cleaned[col].mean() if col in energy_df_cleaned.columns and energy_df_cleaned[col].dtype != 'object' else 'N/A' for col in energy_df.columns]
})
print("\nDetailed Comparison:")
print(comparison[comparison["Original_Missing"] > 0])


# Apply missing handling
energy_df_cleaned7 = daily_window_missing_handler(energy_df)
energy_df_cleaned7.to_csv("datasets/new_cleaned_dataset7.csv", index=False)

# Compare before and after
print("BEFORE handling missing values:")
print(energy_df.isnull().sum().sum(), "total missing values")
print("\n")
print("\nAFTER handling missing values:")
print(energy_df_cleaned7.isnull().sum().sum(), "total missing values")

# Detailed comparison
comparison = pd.DataFrame({
    "Original_Missing": energy_df.isnull().sum(),
    "Cleaned_Missing": [energy_df_cleaned7[col].isnull().sum() if col in energy_df_cleaned7.columns else 'Column Removed' for col in energy_df.columns],
    "Filled_Mean": [energy_df_cleaned7[col].mean() if col in energy_df_cleaned7.columns and energy_df_cleaned7[col].dtype != 'object' else 'N/A' for col in energy_df.columns]
})
print("\nDetailed Comparison:")
print(comparison[comparison["Original_Missing"] > 0])

# Save the cleaned dataset
output_dir = "datasets"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = os.path.join(output_dir, "energy_dataset_cleaned.csv")
energy_df_cleaned.to_csv(output_file, index=False)