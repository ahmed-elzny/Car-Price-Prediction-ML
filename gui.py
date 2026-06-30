# =========================
# CAR PRICE PROJECT - FULL GUI SINGLE PAGE
# =========================
# Save as: car_price_gui_singlepage.py
# Run: streamlit run car_price_gui_singlepage.py
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import streamlit as st

st.set_page_config(page_title="Car Price Prediction", layout="wide")

st.title("📊 Car Price Prediction Project (Single Page GUI)")

# =========================
# Load Dataset
# =========================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(url, header=None)
columns = ["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors",
           "body-style","drive-wheels","engine-location","wheel-base","length","width",
           "height","curb-weight","engine-type","num-of-cylinders","engine-size",
           "fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm",
           "city-mpg","highway-mpg","price"]
df.columns = columns

st.subheader("Dataset Overview")
st.write(df.head(10))
st.write("Shape:", df.shape)

# =========================
# Corruption
# =========================
df_corrupted = df.copy()
np.random.seed(42)
df_corrupted.loc[df_corrupted.sample(5).index, "engine-size"] = "unknown"
df_corrupted.loc[df_corrupted.sample(5).index, "price"] = "error"

st.subheader("Corrupted Dataset Example")
st.write(df_corrupted.head(10))

st.write("Distribution of 'price' before cleaning:")
plt.figure(figsize=(10,5))
sns.histplot(pd.to_numeric(df_corrupted["price"], errors='coerce'), bins=30, kde=True)
plt.xlabel("Price")
plt.ylabel("Count")
plt.title("Price Distribution - Corrupted")
st.pyplot(plt.gcf())

# =========================
# Cleaning
# =========================
df_cleaned = df_corrupted.copy()
num_cols = ["normalized-losses","wheel-base","length","width","height","curb-weight",
            "engine-size","bore","stroke","compression-ratio","horsepower","peak-rpm",
            "city-mpg","highway-mpg","price"]
for col in num_cols:
    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
df_cleaned["price"] = df_cleaned["price"].clip(lower=1000, upper=df_cleaned["price"].quantile(0.99))
cat_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()
df_cleaned = pd.get_dummies(df_cleaned, columns=cat_cols, drop_first=True)

st.subheader("Cleaned Dataset")
st.write(df_cleaned.head(10))

st.write("Distribution of 'price' after cleaning:")
plt.figure(figsize=(10,5))
sns.histplot(df_cleaned["price"], bins=30, kde=True)
plt.xlabel("Price")
plt.ylabel("Count")
plt.title("Price Distribution - Cleaned")
st.pyplot(plt.gcf())

# =========================
# Preprocessing & Split
# =========================
target = "price"
features = [col for col in df_cleaned.columns if col != target]
X = df_cleaned[features]
y = df_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.subheader("Train/Test Split")
st.write("X_train shape:", X_train.shape, "| X_test shape:", X_test.shape)
st.write("y_train shape:", y_train.shape, "| y_test shape:", y_test.shape)

# =========================
# Model & Evaluation
# =========================
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("Linear Regression Evaluation")
st.write(f"R² Score: {r2:.4f}")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")

coeff_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_}).sort_values(by="Coefficient", key=abs, ascending=False)
st.write("Top 15 Feature Impact on Price")
st.write(coeff_df.head(15))

plt.figure(figsize=(12,8))
sns.barplot(x="Coefficient", y="Feature", data=coeff_df.head(15), palette="coolwarm")
plt.title("Top 15 Feature Impact")
st.pyplot(plt.gcf())

# =========================
# Visualization
# =========================
st.subheader("Feature Visualization")
feature_to_plot = st.selectbox("Select Feature to Visualize", df_cleaned.columns.tolist())
plot_type = st.radio("Select Plot Type", ["Histogram", "Boxplot", "KDE"])

plt.figure(figsize=(10,5))
if plot_type == "Histogram":
    sns.histplot(df_cleaned[feature_to_plot], bins=30, kde=True)
elif plot_type == "Boxplot":
    sns.boxplot(y=df_cleaned[feature_to_plot])
elif plot_type == "KDE":
    sns.kdeplot(df_cleaned[feature_to_plot], fill=True)
plt.title(f"{feature_to_plot} ({plot_type})")
plt.tight_layout()
st.pyplot(plt.gcf())

# =========================
# Interactive Prediction
# =========================
st.subheader("Predict Car Price")
st.write("Adjust the sliders to modify feature values and predict the price:")

numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
input_dict = {}
for feature in numeric_features[:15]:  # limit to first 15 numeric features
    min_val = float(X_train[feature].min())
    max_val = float(X_train[feature].max())
    default_val = float(X_train[feature].median())
    input_dict[feature] = st.slider(feature, min_val, max_val, default_val)

input_full = {col: 0 for col in X_train.columns}
for k, v in input_dict.items():
    input_full[k] = v
input_df = pd.DataFrame([input_full], columns=X_train.columns)

input_scaled = scaler.transform(input_df)
pred_price = model.predict(input_scaled)

st.subheader("Predicted Car Price")
st.write(f"${pred_price[0]:,.2f}")



