import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the weather features CSV
df = pd.read_csv('./datasets/weather_features.csv')

# Columns to inspect
cat_columns = ['city_name', 'weather_main', 'weather_description', 'weather_icon']

# Create sets for each categorical column
categories = {col: set(df[col].unique()) for col in cat_columns}
for col, cats in categories.items():
    print(f"{col}: {cats}\n")

# Create a copy of the dataframe for processing
df_processed = df.copy()

# One-hot encoding for 'city_name' and 'weather_main'
one_hot_columns = ['city_name', 'weather_main']
for col in one_hot_columns:
    # Create one-hot encoded columns
    one_hot = pd.get_dummies(df_processed[col], prefix=col)
    # Drop the original column and add the one-hot encoded columns
    df_processed = df_processed.drop(columns=[col])
    df_processed = pd.concat([df_processed, one_hot], axis=1)

# Label encoding for 'weather_description' and 'weather_icon'
label_encode_columns = ['weather_description', 'weather_icon']
label_encoders = {}  # Store encoders in case you need to decode later

for col in label_encode_columns:
    le = LabelEncoder()
    # Fit and transform the column (this creates labels from 0 to n-1)
    df_processed[col] = le.fit_transform(df_processed[col])
    # If you want labels from 1 to n instead of 0 to n-1, add 1
    df_processed[col] = df_processed[col] + 1
    # Store the encoder for potential future use
    label_encoders[col] = le

# Convert all boolean columns to int (0/1)
for col in df_processed.columns:
    if df_processed[col].dtype == bool:
        df_processed[col] = df_processed[col].astype(int)

# Save the processed dataframe to a new CSV file
df_processed.to_csv('./datasets/weather_numeric.csv', index=False)

print("Processing complete!")
print(f"Original shape: {df.shape}")
print(f"Processed shape: {df_processed.shape}")

# Show the mapping for label encoded columns
print("\nLabel encoding mappings (original -> encoded):")
for col in label_encode_columns:
    le = label_encoders[col]
    mapping = dict(zip(le.classes_, range(1, len(le.classes_) + 1)))
    print(f"{col}: {mapping}")

# Display first few rows of processed data
print("\nFirst 5 rows of processed data:")
print(df_processed.head())
