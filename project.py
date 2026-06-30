# =========================
# IMPORTS
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# 1. LOAD DATA
# =========================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(url, header=None)

columns = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors",
           "body-style","drive-wheels","engine-location","wheel-base","length","width",
           "height","curb-weight","engine-type","num-of-cylinders","engine-size",
           "fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm",
           "city-mpg","highway-mpg","price"]
df.columns = columns

# =========================
# 2. CREATE CORRUPTED DATASET
# =========================
df_corrupted = df.copy()

# Convert engine-size to string to allow insertion of 'unknown'
df_corrupted["engine-size"] = df_corrupted["engine-size"].astype(str)

np.random.seed(42)
df_corrupted.loc[df_corrupted.sample(5).index, "engine-size"] = "unknown"
df_corrupted.loc[df_corrupted.sample(5).index, "price"] = "error"

df_corrupted.to_csv("car_price_corrupted.csv", index=False)
print(" Corrupted dataset saved as car_price_corrupted.csv")

# =========================
# 3. VISUALIZATION BEFORE CLEANING
# =========================
plt.figure(figsize=(10,5))
sns.histplot(pd.to_numeric(df_corrupted["price"], errors='coerce'), bins=30, kde=True)
plt.title("Price Distribution Before Cleaning")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =========================
# 4. CLEANING
# =========================
df_cleaned = df_corrupted.copy()
num_cols = ["normalized-losses","wheel-base","length","width","height","curb-weight",
            "engine-size","bore","stroke","compression-ratio","horsepower","peak-rpm",
            "city-mpg","highway-mpg","price"]

# Convert numeric columns and fill missing values with median
for col in num_cols:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

# Clip extreme price values
df_cleaned["price"] = df_cleaned["price"].clip(lower=1000, upper=df_cleaned["price"].quantile(0.99))

# Encode categorical columns
cat_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
df_cleaned = pd.get_dummies(df_cleaned, columns=cat_cols, drop_first=True)

df_cleaned.to_csv("car_price_cleaned.csv", index=False)
print(" Cleaned dataset saved as car_price_cleaned.csv")

# =========================
# 5. VISUALIZATION AFTER CLEANING
# =========================
plt.figure(figsize=(10,5))
sns.histplot(df_cleaned["price"], bins=30, kde=True)
plt.title("Price Distribution After Cleaning")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =========================
# 6. PREPROCESSING
# =========================
target = "price"
features = [col for col in df_cleaned.columns if col != target]
X = df_cleaned[features]
y = df_cleaned[target]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save train/test sets
pd.DataFrame(X_train_scaled, columns=features).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=features).to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("Train/Test sets saved successfully")

# =========================
# 7. LINEAR REGRESSION MODEL
# =========================
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# =========================
# 8. EVALUATION
# =========================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# =========================
# 9. SIMPLE LINEAR REGRESSION PLOT
# =========================
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)  # diagonal line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Price")
plt.tight_layout()
plt.show()


print("\n Linear Regression Evaluation:")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
     



