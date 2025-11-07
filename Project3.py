import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv(r"C:\Users\kasha\OneDrive\Desktop\Python\HousePriceData.csv")

df["Area_sqft"] = df["Area_sqft"].fillna(df["Area_sqft"].mean())
df["Bedrooms"] = df["Bedrooms"].fillna(df["Bedrooms"].mean())
df["Bathrooms"] = df["Bathrooms"].fillna(df["Bathrooms"].mean())
df["Location"] = df["Location"].fillna(df["Location"].mode()[0])

print("\n One hot encoding \n")
encoded_df = pd.get_dummies(df, columns=["Location"], dtype=int)
print(encoded_df, "\n")

print("\n Scaling Numerical Features \n")
scaler = StandardScaler()
scaled_values = scaler.fit_transform(encoded_df[["Area_sqft", "Bedrooms", "Bathrooms"]])

encoded_df[["Area_sqft_scaled", "Bedrooms_scaled", "Bathrooms_scaled"]] = scaled_values
print("\n DataFrame after adding scaled values:\n")
print(encoded_df.head())

X = encoded_df[["Area_sqft_scaled", "Bedrooms_scaled", "Bathrooms_scaled"]]
y = encoded_df["Price"]

print("\n Train test split \n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)

print("\n Linear Regression Model \n")
model = LinearRegression()
model.fit(X_train, y_train)

print("\n --- Predict House Price ---")
area = float(input("Enter area in sqft: "))
bedrooms = float(input("Enter number of bedrooms: "))
bathrooms = float(input("Enter number of bathrooms: "))

input_df = pd.DataFrame([[area, bedrooms, bathrooms]], columns=["Area_sqft", "Bedrooms", "Bathrooms"])
input_scaled = scaler.transform(input_df)

predicted_price = model.predict(input_scaled)
print(f"\n Predicted House Price: {predicted_price[0]:.2f}")

print("\n Model Evaluation \n")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"MAE (Mean Absolute Error): {mae:.3f}")
print(f"MSE (Mean Squared Error): {mse:.3f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.3f}")
