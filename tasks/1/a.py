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
