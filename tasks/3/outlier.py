import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the cleaned energy dataset (assuming it comes from Question 2: Data Cleaning)
energy_df = pd.read_csv("datasets/energy_dataset_cleaned.csv")

print("=" * 60)
print("QUESTION 3: HANDLING OUTLIERS")
print("Note: This analysis uses cleaned datasets from Question 2")
print("Method: IQR (Interquartile Range) method for outlier detection")
print("=" * 60)


# Function to detect outliers using IQR method
def detect_outliers_iqr(df, column):
    """Detect outliers using Interquartile Range (IQR) method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_indices = outliers.index.tolist()

    return {
        "outlier_count": len(outliers),
        "outlier_indices": outlier_indices,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "outlier_values": (
            df.loc[outlier_indices, column].values if len(outlier_indices) > 0 else []
        ),
    }


# Function to visualize outliers
def visualize_outliers(df, column, title="Energy Dataset Outlier Analysis"):
    """Create visualizations for outlier analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{title} - {column}", fontsize=16)

    # Box plot
    axes[0, 0].boxplot(df[column].dropna())
    axes[0, 0].set_title("Box Plot")
    axes[0, 0].set_ylabel("Value")

    # Histogram
    axes[0, 1].hist(df[column].dropna(), bins=50, edgecolor="black")
    axes[0, 1].set_title("Histogram")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].set_ylabel("Frequency")

    # Q-Q plot
    stats.probplot(df[column].dropna(), dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot")

    # Scatter plot (index vs value)
    axes[1, 1].scatter(df.index, df[column], alpha=0.6)
    axes[1, 1].set_title("Time Series Plot")
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Value")

    plt.tight_layout()
    plt.show()


# Select columns for outlier analysis from energy dataset
print("\n1. SELECTING COLUMNS FOR OUTLIER ANALYSIS")
print("-" * 45)

# Analyze ALL numerical columns (excluding time column)
energy_columns_to_analyze = [
    col
    for col in energy_df.columns
    if col != "time" and energy_df[col].dtype in ["float64", "int64"]
]

print(f"All numerical columns selected for analysis:")
for i, col in enumerate(energy_columns_to_analyze, 1):
    print(f"  {i:2d}. {col}")

print(f"\nDataset shape: {energy_df.shape}")
print(
    f"Analyzing {len(energy_columns_to_analyze)} out of {len(energy_df.columns)} total columns"
)

# 2. DETECT OUTLIERS USING IQR METHOD
print("\n2. OUTLIER DETECTION USING IQR METHOD")
print("-" * 40)

print("\nENERGY DATASET - IQR METHOD:")
print("Using IQR method: Outliers are values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR")

energy_iqr_results = {}
for col in energy_columns_to_analyze:
    if col in energy_df.columns:
        result = detect_outliers_iqr(energy_df, col)
        energy_iqr_results[col] = result

        print(f"\n{col}:")
        print(
            f"  Outliers detected: {result['outlier_count']} ({result['outlier_count']/len(energy_df)*100:.2f}%)"
        )
        print(
            f"  Normal range: {result['lower_bound']:.2f} to {result['upper_bound']:.2f}"
        )
        print(
            f"  Q1: {result['Q1']:.2f}, Q3: {result['Q3']:.2f}, IQR: {result['IQR']:.2f}"
        )

        if result["outlier_count"] > 0:
            print(f"  Min outlier: {min(result['outlier_values']):.2f}")
            print(f"  Max outlier: {max(result['outlier_values']):.2f}")
    else:
        print(f"\nColumn '{col}' not found in dataset")

# 3. VISUALIZE OUTLIERS USING BOX PLOTS (MOST INTUITIVE FOR IQR)
print("\n3. VISUALIZING OUTLIERS WITH BOX PLOTS")
print("-" * 40)
print("Box plots are the most intuitive visualization for IQR method:")
print("• Box shows Q1 to Q3 (IQR)")
print("• Whiskers extend to 1.5 × IQR from box edges")
print("• Points beyond whiskers = outliers")

# Show top 12 columns with most outliers in a 3x4 grid
outlier_counts = [
    (col, energy_iqr_results[col]["outlier_count"])
    for col in energy_columns_to_analyze
    if col in energy_df.columns
]
top_12_outlier_cols = sorted(outlier_counts, key=lambda x: x[1], reverse=True)[:12]

print(f"\nShowing box plots for top 12 columns with most outliers:")
for col, count in top_12_outlier_cols:
    print(f"  {col}: {count} outliers")

# Create 3x4 grid for top 12 columns
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle(
    "Energy Dataset - Box Plots for Top 12 Columns with Most Outliers", fontsize=16
)

for i, (col, outlier_count) in enumerate(top_12_outlier_cols):
    if i < 12:  # Safety check
        row = i // 4
        col_idx = i % 4
        current_ax = axes[row, col_idx]

        # Get the data for this column
        data = energy_df[col].dropna()  # Remove NaN values
        iqr_result = energy_iqr_results[col]

        # Create box plot with better styling
        bp = current_ax.boxplot(
            data,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7, linewidth=2),
            whiskerprops=dict(linewidth=2),
            capprops=dict(linewidth=2),
            medianprops=dict(color="darkblue", linewidth=2),
            flierprops=dict(
                marker="o",
                markerfacecolor="red",
                markeredgecolor="darkred",
                markersize=4,
                alpha=0.8,
            ),
        )

        # Set title with outlier count
        current_ax.set_title(
            f"{col}\n{outlier_count} outliers ({outlier_count/len(energy_df)*100:.1f}%)",
            fontsize=10,
            pad=10,
        )
        current_ax.set_ylabel("Value", fontsize=9)
        current_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        current_ax.grid(True, alpha=0.3)

        # Add statistics in a more readable way
        stats_text = (
            f"Min: {data.min():.0f}\nMax: {data.max():.0f}\nMedian: {data.median():.0f}"
        )
        current_ax.text(
            0.02,
            0.98,
            stats_text,
            transform=current_ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )

# Hide empty subplots if we have fewer than 12 columns
for i in range(len(top_12_outlier_cols), 12):
    row = i // 4
    col_idx = i % 4
    axes[row, col_idx].set_visible(False)

plt.tight_layout()
plt.show()

# Create a more detailed view for the top 5 columns
print(f"\nDetailed analysis for top 5 columns with most outliers:")

top_5_cols = top_12_outlier_cols[:5]
fig, axes = plt.subplots(1, 5, figsize=(25, 6))
fig.suptitle("Detailed Box Plots - Top 5 Columns with Most Outliers", fontsize=16)

for i, (col, outlier_count) in enumerate(top_5_cols):
    data = energy_df[col].dropna()
    iqr_result = energy_iqr_results[col]

    # Create detailed box plot
    bp = axes[i].boxplot(
        data,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.8, linewidth=2),
        whiskerprops=dict(linewidth=2, color="darkblue"),
        capprops=dict(linewidth=2, color="darkblue"),
        medianprops=dict(color="red", linewidth=3),
        flierprops=dict(
            marker="o",
            markerfacecolor="red",
            markeredgecolor="darkred",
            markersize=6,
            alpha=0.9,
        ),
    )

    # Add comprehensive statistics
    axes[i].set_title(f"{col}\n{outlier_count} outliers", fontsize=12, pad=15)
    axes[i].set_ylabel("Value" if i == 0 else "", fontsize=11)
    axes[i].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axes[i].grid(True, alpha=0.4)

    # Print detailed statistics
    print(f"\n{col}:")
    print(f"  Total values: {len(data)}")
    print(f"  Outliers: {outlier_count} ({outlier_count/len(data)*100:.2f}%)")
    print(f"  Range: {data.min():.2f} to {data.max():.2f}")
    print(f"  Q1: {iqr_result['Q1']:.2f}, Q3: {iqr_result['Q3']:.2f}")
    print(f"  IQR: {iqr_result['IQR']:.2f}")
    print(
        f"  Outlier bounds: [{iqr_result['lower_bound']:.2f}, {iqr_result['upper_bound']:.2f}]"
    )

plt.tight_layout()
plt.show()

print("\nBox plot interpretation:")
print("• Blue box = Interquartile Range (Q1 to Q3)")
print("• Red line in box = Median")
print("• Whiskers = Extend to 1.5 × IQR from box edges")
print("• Red dots = Outliers (values beyond whiskers)")
print("• This visualization directly shows what the IQR method detects!")

# 4. COMPREHENSIVE SUMMARY TABLE
print("\n4. COMPREHENSIVE OUTLIER SUMMARY TABLE (IQR METHOD)")
print("-" * 55)

outlier_summary = []
for col in energy_columns_to_analyze:
    if col in energy_df.columns:
        iqr_data = energy_iqr_results[col]

        outlier_summary.append(
            {
                "Column": col,
                "Total_Values": len(energy_df[col]),
                "IQR_Outliers": iqr_data["outlier_count"],
                "IQR_Percentage": f"{iqr_data['outlier_count']/len(energy_df)*100:.2f}%",
                "Q1": f"{iqr_data['Q1']:.2f}",
                "Q3": f"{iqr_data['Q3']:.2f}",
                "IQR": f"{iqr_data['IQR']:.2f}",
                "Lower_Bound": f"{iqr_data['lower_bound']:.2f}",
                "Upper_Bound": f"{iqr_data['upper_bound']:.2f}",
                "Min_Value": f"{energy_df[col].min():.2f}",
                "Max_Value": f"{energy_df[col].max():.2f}",
            }
        )

outlier_summary_df = pd.DataFrame(outlier_summary)
print("\nComplete Outlier Summary:")
print(outlier_summary_df.to_string(index=False))

# Show columns with highest outlier percentages
print(f"\nTop 10 Columns with Highest Outlier Rates (IQR Method):")
top_outlier_rates = outlier_summary_df.sort_values(
    "IQR_Outliers", ascending=False
).head(10)
print(
    top_outlier_rates[["Column", "IQR_Outliers", "IQR_Percentage"]].to_string(
        index=False
    )
)

# 5. OUTLIER TREATMENT DECISIONS AND JUSTIFICATIONS
print("\n5. OUTLIER TREATMENT DECISIONS AND JUSTIFICATIONS")
print("-" * 50)


# Helper functions
def cap_outliers(df, column, lower_bound, upper_bound):
    """Cap outliers to the bounds"""
    df_capped = df.copy()
    df_capped[column] = np.clip(df_capped[column], lower_bound, upper_bound)
    return df_capped


def remove_outliers(df, outlier_indices):
    """Remove outlier rows"""
    return df.drop(outlier_indices)


# Treatment decisions with justifications for ALL columns
print("\nTreatment Strategy:")
print("• CAP: Bound values to IQR limits (physical/technical constraints)")
print("• REMOVE: Delete outlier rows (likely data errors or extreme events)")

treatment_decisions = {}

# Define treatment strategies based on column types
for col in energy_columns_to_analyze:
    if col in energy_df.columns:
        col_lower = col.lower()

        if "generation" in col_lower:
            treatment_decisions[col] = "CAP"  # Generation bounded by physical capacity
        elif "price" in col_lower:
            treatment_decisions[col] = (
                "REMOVE"  # Price spikes might be errors or extreme events
            )
        elif "load" in col_lower:
            treatment_decisions[col] = "CAP"  # Load bounded by grid capacity
        elif "forecast" in col_lower:
            treatment_decisions[col] = (
                "CAP"  # Forecasts should be bounded by realistic values
            )
        else:
            # Default strategy for other columns
            treatment_decisions[col] = "CAP"  # Default to capping for energy data

# Group decisions by treatment type for better organization
cap_columns = []
remove_columns = []

for col, treatment in treatment_decisions.items():
    if treatment == "CAP":
        cap_columns.append(col)
    elif treatment == "REMOVE":
        remove_columns.append(col)

print(f"\nColumns to be CAPPED ({len(cap_columns)} columns):")
for col in cap_columns:
    outlier_count = energy_iqr_results[col]["outlier_count"]
    print(f"  • {col}: {outlier_count} outliers")

print(f"\nColumns to have outlier rows REMOVED ({len(remove_columns)} columns):")
for col in remove_columns:
    outlier_count = energy_iqr_results[col]["outlier_count"]
    print(f"  • {col}: {outlier_count} outlier rows will be deleted")

print("\nJUSTIFICATIONS:")
print("\n1. CAPPING JUSTIFICATION:")
print("   Applied to: Generation, Load, and Forecast columns")
print("   Reasoning:")
print("   • Generation values are physically bounded by plant/equipment capacity")
print("   • Load values are constrained by grid infrastructure limits")
print("   • Forecasts should stay within realistic operational ranges")
print("   • Extreme values likely represent measurement errors or temporary anomalies")
print("   • Capping preserves data points while constraining to realistic bounds")

print("\n2. REMOVAL JUSTIFICATION:")
print("   Applied to: Price columns")
print("   Reasoning:")
print("   • Extreme price spikes often indicate market manipulation or data errors")
print("   • Price outliers can severely skew economic analysis")
print("   • Removing entire rows with price outliers prevents contamination")
print("   • Price data quality is critical for accurate market modeling")

# 6. IMPLEMENT OUTLIER TREATMENT
print("\n6. IMPLEMENTING OUTLIER TREATMENT")
print("-" * 35)

# Create copy for treatment
energy_df_treated = energy_df.copy()
print(f"Starting with dataset: {len(energy_df_treated)} rows")

# Track treatment results
outliers_removed_rows = set()  # Track rows to be removed
outliers_capped_count = 0
columns_capped = []
columns_with_removals = []

print("\nApplying treatments:")

# First, handle CAPPING for all relevant columns
for col in energy_columns_to_analyze:
    if col in energy_df.columns and col in treatment_decisions:
        treatment = treatment_decisions[col]

        if treatment == "CAP":
            bounds = energy_iqr_results[col]
            original_outliers = bounds["outlier_count"]

            # Apply capping
            energy_df_treated = cap_outliers(
                energy_df_treated, col, bounds["lower_bound"], bounds["upper_bound"]
            )
            outliers_capped_count += original_outliers
            columns_capped.append(col)

            print(
                f"  ✓ {col}: Capped {original_outliers} outliers to [{bounds['lower_bound']:.2f}, {bounds['upper_bound']:.2f}]"
            )

# Then, collect all rows that need to be removed
for col in energy_columns_to_analyze:
    if col in energy_df.columns and col in treatment_decisions:
        treatment = treatment_decisions[col]

        if treatment == "REMOVE":
            outlier_indices = energy_iqr_results[col]["outlier_indices"]
            outliers_removed_rows.update(
                outlier_indices
            )  # Add to set (removes duplicates)
            columns_with_removals.append((col, len(outlier_indices)))

            print(f"  ✓ {col}: Marked {len(outlier_indices)} rows for removal")

# Remove all marked rows at once
if outliers_removed_rows:
    energy_df_treated = energy_df_treated.drop(list(outliers_removed_rows))
    print(f"\n  → Removed {len(outliers_removed_rows)} total rows with price outliers")

print(f"\nTreatment Summary:")
print(f"  • Columns capped: {len(columns_capped)}")
print(f"  • Individual outlier values capped: {outliers_capped_count}")
print(f"  • Columns with row removal: {len(columns_with_removals)}")
print(f"  • Total rows removed: {len(outliers_removed_rows)}")
print(f"  • Final dataset size: {len(energy_df_treated)} rows (was {len(energy_df)})")

# Show before/after comparison for a few key columns
print(f"\nBEFORE/AFTER COMPARISON (sample columns):")
sample_cols = (
    ["generation fossil gas", "generation nuclear", "price actual"]
    if all(
        col in energy_df.columns
        for col in ["generation fossil gas", "generation nuclear", "price actual"]
    )
    else list(treatment_decisions.keys())[:3]
)

for col in sample_cols:
    if col in energy_df.columns:
        before_outliers = energy_iqr_results[col]["outlier_count"]

        # Recalculate outliers in treated data
        after_result = detect_outliers_iqr(energy_df_treated, col)
        after_outliers = after_result["outlier_count"]

        print(f"  {col}:")
        print(f"    Before: {before_outliers} outliers")
        print(f"    After: {after_outliers} outliers")
        print(f"    Reduction: {before_outliers - after_outliers} outliers eliminated")

# 7. TREATMENT SUMMARY
print("\n7. TREATMENT SUMMARY")
print("-" * 25)

print(f"\nOriginal dataset size:")
print(f"  Energy: {len(energy_df)} rows")

print(f"\nTreated dataset size:")
print(f"  Energy: {len(energy_df_treated)} rows")

print(f"\nTreatment applied:")
print(f"  Outliers capped: {outliers_capped_count}")
print(f"  Outlier rows removed: {len(outliers_removed_rows)}")
print(f"  Rows retained: {len(energy_df_treated)/len(energy_df)*100:.2f}%")

print("\n" + "=" * 60)
print("OUTLIER HANDLING COMPLETE")
print("=" * 60)

# Optional: Save treated dataset
# energy_df_treated.to_csv('datasets/energy_dataset_outliers_treated.csv', index=False)
# print("\nTreated dataset saved to 'energy_dataset_outliers_treated.csv'")
