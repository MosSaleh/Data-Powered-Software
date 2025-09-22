import pandas as pd
import numpy as np

energy_df = pd.read_csv("datasets/energy_dataset.csv")
weather_df = pd.read_csv("datasets/weather_features.csv")

print(energy_df.head())
print(weather_df.head())

