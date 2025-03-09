import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the Dataset from Local File
file_path = r"C:\Users\Administrator\Downloads\BostonHousing.csv"  # Ensure the dataset is in the working directory
data = pd.read_csv(r"C:\Users\Administrator\Downloads\BostonHousing.csv")

# Step 2: Feature Selection
X = data.drop(columns=['medv'])  # 'medv' is the target variable
y = np.log1p(data['medv'])  # Applying log transformation to target variable

# Selecting top 9 features
X_selected = X.iloc[:, :9]  # Ensuring we have exactly 9 features
feature_names = [
    "Crime Rate", "Industrial Zone", "Non-Retail Acres", "Charles River Proximity",
    "Nitrogen Oxides Concentration", "Average Rooms per Dwelling", "Age of Property",
    "Distance to Employment Centers", "Accessibility to Highways"
]  # User-friendly feature names

# Step 3: Data Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Step 4: Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Step 5: Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Step 6: Model Training with Ridge Regression
ridge = Ridge()
ridge.fit(X_train, y_train)

def main():
    st.title("Boston Housing Price Prediction")
    st.write("This app predicts the median value of owner-occupied homes using Ridge Regression.")
    
    st.header("Make a Prediction")
    user_input = []
    
    for i in range(9):  # Ensure we collect inputs for 9 selected features
        min_val = float(X_selected.iloc[:, i].min())
        max_val = float(X_selected.iloc[:, i].max())
        default_val = float(X_selected.iloc[:, i].mean())
        user_input.append(st.slider(f"{feature_names[i]}", min_value=min_val, max_value=max_val, value=default_val))
    
    if st.button("Predict Price"):
        input_array = np.array(user_input).reshape(1, -1)
        input_array = poly.transform(scaler.transform(input_array))  # Apply transformations
        prediction = ridge.predict(input_array)
        st.success(f"Predicted Price: ${np.expm1(prediction[0]):.2f}")
    
    st.header("Model Performance")
    y_pred = ridge.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"R-squared Score: {r2:.4f}")

if __name__ == "__main__":
    main()
