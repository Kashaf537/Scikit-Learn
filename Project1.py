import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv(r"C:\Users\kasha\OneDrive\Desktop\Python\data_science_student_marks.csv")

X = df[["age"]]            
y = df[["python_marks"]] 


model = LinearRegression()
model.fit(X, y)

age = float(input("Enter the age: "))
predicted_marks = model.predict([[age]]) 

print(f"\nBased on your age {age}, you may score around {predicted_marks[0][0]:.2f} marks.")

y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation:")
print("MAE (Mean Absolute Error):", mae)
print("MSE (Mean Squared Error):", mse)
print("RMSE (Root Mean Squared Error):", rmse)
