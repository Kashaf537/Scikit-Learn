import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score

# ================= LOAD DATA =================
data = pd.read_csv("study.csv", sep='\t')  # Tab-separated CSV
data.columns = data.columns.str.strip()  # Remove extra spaces

# ================= ENCODE CATEGORICAL =================
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])
data["InternetAccess"] = le.fit_transform(data["InternetAccess"])

# ================= PREDICT FINAL MARKS =================
X = data.drop("FinalScore", axis=1)
y = data["FinalScore"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train regression model
reg = LinearRegression()
reg.fit(X_train_scaled, y_train)

# Take only first 5 test cases for prediction
X_test_small = X_test.head(5)
X_test_small_scaled = scaler.transform(X_test_small)
predicted_marks = reg.predict(X_test_small_scaled)

print("===== Predicted Marks for 5 Test Students =====")
for i, mark in enumerate(predicted_marks, 1):
    print(f"Student {i}: {mark:.2f}")

# ================= CLASSIFY RISK =================
# Risk: 1 = High Risk (<70), 0 = Low Risk (>=70)
data["Risk"] = data["FinalScore"].apply(lambda x: 1 if x < 70 else 0)

X_risk = data.drop(["FinalScore", "Risk"], axis=1)
y_risk = data["Risk"]

X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)
clf = LogisticRegression()
clf.fit(X_train_risk, y_train_risk)

# Take first 5 test cases for risk classification
X_test_risk_small = X_test_risk.head(5)
pred_risk = clf.predict(X_test_risk_small)

print("\n===== Risk Classification for 5 Test Students =====")
for i, risk in enumerate(pred_risk, 1):
    status = "High Risk" if risk == 1 else "Low Risk"
    print(f"Student {i}: {status}")

# ================= RECOMMENDATION FUNCTION =================
def recommend(study_hours, attendance):
    recommendations = []
    if study_hours < 3:
        recommendations.append("Increase study hours")
    if attendance < 75:
        recommendations.append("Improve attendance")
    recommendations.append("Use online courses and revision schedule")
    return recommendations

# ================= RECOMMENDATIONS FOR 5 TEST STUDENTS =================
print("\n===== Recommendations for 5 Test Students =====")
for i, row in X_test_small.iterrows():
    recs = recommend(row["StudyHours"], row["Attendance"])
    print(f"Student {i+1} recommendations: {', '.join(recs)}")
