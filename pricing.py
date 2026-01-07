import pandas as pd
import numpy as np
import sklearn as skl
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = "/"

# Load the latest version
df = pd.read_csv('house_prices.csv')
#check df data types
print(df.info())
#print("First 5 records:", df.head())

