import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv(r"D:\SpamDetection\ScreenTime vs MentalWellness.csv")
data = data.drop(columns=["user_id", "Unnamed: 15"], errors="ignore")

features = ["leisure_screen_hours", "sleep_hours", "productivity_0_100", "mental_wellness_index_0_100"]
X = data[features]

y = data["stress_level_0_10"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

pred_stress = model.predict(X_test)

mae = mean_absolute_error(y_test, pred_stress)
rmse = np.sqrt(mean_squared_error(y_test, pred_stress))

print("===== MODEL PERFORMANCE =====")
print("MAE  (Mean Absolute Error):", round(mae, 2))
print("RMSE (Root Mean Square Error):", round(rmse, 2))

importances = model.feature_importances_
feature_names = X.columns

print("\n===== FACTORS AFFECTING MOOD (Feature Importance) =====")
for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"{name}: {round(imp*100, 2)} %")

plt.barh(feature_names, importances)
plt.xlabel("Importance Score")
plt.title("Factors Affecting Mood (Random Forest)")
plt.show()

def stress_to_mood(stress):
    if stress <= 3:
        return "Good"
    elif stress <= 6:
        return "Neutral"
    else:
        return "Stressed"

pred_mood = [stress_to_mood(s) for s in pred_stress]

print("\n===== Sample Predicted Moods =====")
for i in range(5):
    print(f"Student {i+1}: Stress={round(pred_stress[i],2)} → Mood={pred_mood[i]}")

new_user = pd.DataFrame({
    "leisure_screen_hours": [6],
    "sleep_hours": [4],
    "productivity_0_100": [50],
    "mental_wellness_index_0_100": [40]
})

new_user_scaled = scaler.transform(new_user)
new_stress = model.predict(new_user_scaled)[0]
new_mood = stress_to_mood(new_stress)

print("\nNEW USER PREDICTION:")
print("Predicted Stress Level:", round(new_stress,2))
print("Predicted Mood:", new_mood)
