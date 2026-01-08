import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Reload to ensure a clean slate
df = pd.read_csv("house_prices.csv")
original_len = len(df)

# --- 1. CLEANING & FORMATTING ---
# Clean non-numeric characters and convert to float
df['Carpet Area'] = df['Carpet Area'].astype(str).str.replace(r"[^\d\.\-]", "", regex=True).replace('', np.nan).astype(float)
df['Super Area'] = df['Super Area'].astype(str).str.replace(r"[^\d\.\-]", "", regex=True).replace('', np.nan).astype(float)
df['Price (in rupees)'] = pd.to_numeric(df['Price (in rupees)'].astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors='coerce')

# --- 2. CROSS-IMPUTATION (The "Smart Fill") ---
# Instead of mean, calculate missing Super Area from Carpet Area (approx 1.25x)
df['Super Area'] = df['Super Area'].fillna(df['Carpet Area'] * 1.25)
# Calculate missing Carpet Area from Super Area (approx 0.8x)
df['Carpet Area'] = df['Carpet Area'].fillna(df['Super Area'] / 1.25)

# --- 3. PRUNING ---
# Drop rows that represent the 44% hole in your data
df.dropna(subset=['Super Area', 'Price (in rupees)'], inplace=True)
print(f"Data reduced from {original_len} to {len(df)} rows (Dropped rows with no size info).")

# --- 4. FEATURE ENGINEERING ---
# Location Frequency Encoding (recalculated on clean data)
df['location_freq'] = df.groupby('location')['location'].transform('count') / len(df)

# One-Hot Encoding for Furnishing
df = pd.get_dummies(df, columns=['Furnishing'], drop_first=True)

# Clean Bath/Balcony (Fill missing with 0)
for col in ['Bathroom', 'Balcony']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors='coerce').fillna(0)

# --- 5. MODEL PREPARATION ---
# Use Log Transformation on Price to normalize the target
y = np.log1p(df['Price (in rupees)'])

# Define Features: Note we excluded 'Amount' (leakage) and 'Carpet Area' (collinear)
feature_cols = ['Super Area', 'Bathroom', 'Balcony', 'location_freq'] + [c for c in df.columns if 'Furnishing_' in c]
X = df[feature_cols]

# --- 6. TRAINING ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# --- 7. EVALUATION ---
y_pred = model.predict(X_test)
print(f"\nNew R2 Score: {r2_score(y_test, y_pred):.4f}")