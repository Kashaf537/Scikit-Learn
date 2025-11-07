import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\kasha\OneDrive\Desktop\Python\Mental_Health_and_Social_Media_Balance_Dataset.csv")

scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

numeric_cols = ["Daily_Screen_Time(hrs)", "Stress_Level(1-10)", "Days_Without_Social_Media", "Happiness_Index(1-10)"]


df_standard = df.copy()
df_standard[numeric_cols] = scaler_standard.fit_transform(df_standard[numeric_cols])

df_minmax = df.copy()
df_minmax[numeric_cols] = scaler_minmax.fit_transform(df_minmax[numeric_cols])

print("\n--- Standard Scaled Data (first 5 rows) ---")
print(df_standard.head())
print("\n--- Min-Max Scaled Data (first 5 rows) ---")
print(df_minmax.head())

X = df_standard[["Daily_Screen_Time(hrs)"]]
y = df_standard[["Stress_Level(1-10)"]]

model = LinearRegression()
model.fit(X, y)

screen_time = float(input("\nEnter the number of hours of screen time: "))

input_df = pd.DataFrame([[screen_time, 0, 0, 0]], columns=numeric_cols)
screen_time_scaled = scaler_standard.transform(input_df)[0][0]

pred_input = pd.DataFrame([[screen_time_scaled]], columns=["Daily_Screen_Time(hrs)"])
pred_stress_scaled = model.predict(pred_input)[0][0]

pred_stress_original = scaler_standard.inverse_transform([[0, pred_stress_scaled, 0, 0]])[0][1]

print(f"\nPredicted Stress Level (Scaled): {pred_stress_scaled:.3f}")
print(f"Predicted Stress Level (Original Scale): {pred_stress_original:.2f}")

y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation:")
print("MAE (Mean Absolute Error):", mae)
print("MSE (Mean Squared Error):", mse)
print("RMSE (Root Mean Squared Error):", rmse)

df_gender = pd.get_dummies(df["Gender"], prefix="Gender")
print("\nAfter applying One Hot Encoding:\n", df_gender.head())

X1 = df_gender
y1 = df["Social_Media_Platform"]

model1 = LogisticRegression(max_iter=200)
model1.fit(X1, y1)

gender_input = input("\nEnter gender (Male/Female/Other): ").capitalize()
if gender_input in df["Gender"].unique():
    encoded_input = pd.get_dummies([gender_input], prefix="Gender").reindex(columns=df_gender.columns, fill_value=0)
    result = model1.predict(encoded_input)[0]
    print(f"\nPredicted most likely social media platform for {gender_input}: {result}")
else:
    print("Invalid gender input!")

X2 = df_standard[["Days_Without_Social_Media"]]
y2 = df_standard[["Happiness_Index(1-10)"]]

model2 = LinearRegression()
model2.fit(X2, y2)

days = float(input("\nEnter number of days without social media: "))

input_days_df = pd.DataFrame([[0, 0, days, 0]], columns=numeric_cols)
days_scaled = scaler_standard.transform(input_days_df)[0][2]

pred_days_df = pd.DataFrame([[days_scaled]], columns=["Days_Without_Social_Media"])
pred_happiness_scaled = model2.predict(pred_days_df)[0][0]

pred_happiness_original = scaler_standard.inverse_transform([[0, 0, 0, pred_happiness_scaled]])[0][3]

print(f"\nPredicted Happiness (Scaled): {pred_happiness_scaled:.3f}")
print(f"Predicted Happiness (Original Scale): {pred_happiness_original:.2f}")

df["Gender"].value_counts().plot(kind="pie", autopct='%1.1f%%', figsize=(5,5))
plt.title("Gender Distribution")
plt.ylabel("")  
plt.show()
