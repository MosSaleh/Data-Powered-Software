# Simple Data Splitting - 80-20 Split
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data_80_20(df, target_column=None, random_state=42):
    """
    Split dataset into 80% training and 20% testing sets.

    Parameters:
    df: pandas DataFrame - the dataset to split
    target_column: str - name of target column (if any)
    random_state: int - for reproducible results

    Returns:
    X_train, X_test, y_train, y_test (if target_column provided)
    or X_train, X_test (if no target_column)
    """
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    else:
        X_train, X_test = train_test_split(df, test_size=0.2, random_state=random_state)
        return X_train, X_test


def save_splits(X_train, X_test, y_train=None, y_test=None, target_column=None):
    """Save train and test splits to CSV files."""

    # Combine features and target if target exists
    if y_train is not None and y_test is not None:
        train_df = X_train.copy()
        train_df[target_column] = y_train
        test_df = X_test.copy()
        test_df[target_column] = y_test
    else:
        train_df = X_train
        test_df = X_test

    # Save to CSV
    train_df.to_csv("datasets/train_set.csv", index=False)
    test_df.to_csv("datasets/test_set.csv", index=False)

    print(f"Training set: {len(train_df)} rows saved to datasets/train_set.csv")
    print(f"Test set: {len(test_df)} rows saved to datasets/test_set.csv")


# Load and split the cleaned energy dataset
if __name__ == "__main__":
    # Load cleaned dataset
    energy_df = pd.read_csv("datasets/energy_dataset_cleaned.csv")

    # Split the data (80-20)
    # Change target_column to your actual target variable if needed
    target_column = None  # e.g., "price_actual" or whatever your target is

    if target_column:
        X_train, X_test, y_train, y_test = split_data_80_20(energy_df, target_column)
        save_splits(X_train, X_test, y_train, y_test, target_column)
    else:
        X_train, X_test = split_data_80_20(energy_df)
        save_splits(X_train, X_test)

    print("Data splitting complete.")
