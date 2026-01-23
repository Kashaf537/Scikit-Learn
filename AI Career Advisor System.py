import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt


df = pd.read_csv(r'D:\SpamDetection\ai_job_dataset.csv')  

print(df.head())
print(df.info())

exp_map = {'EN':1, 'MI':2, 'SE':3, 'EX':4}  
df['experience_level_encoded'] = df['experience_level'].map(exp_map)
df['employment_type_encoded'] = LabelEncoder().fit_transform(df['employment_type'])
size_map = {'S':1, 'M':2, 'L':3}  
df['company_size_encoded'] = df['company_size'].map(size_map)
edu_map = {'High School':1, 'Bachelor':2, 'Master':3, 'PhD':4}  
df['education_encoded'] = df['education_required'].map(edu_map)
df['industry_encoded'] = LabelEncoder().fit_transform(df['industry'])
df['company_location_encoded'] = LabelEncoder().fit_transform(df['company_location'])
df['employee_residence_encoded'] = LabelEncoder().fit_transform(df['employee_residence'])


df['required_skills'] = df['required_skills'].fillna('')
df['skills_list'] = df['required_skills'].apply(lambda x: [skill.strip() for skill in x.split(',')])
mlb = MultiLabelBinarizer()
skills_encoded = pd.DataFrame(mlb.fit_transform(df['skills_list']), columns=mlb.classes_)
df = pd.concat([df, skills_encoded], axis=1)

numeric_cols = ['years_experience', 'remote_ratio', 'job_description_length', 'benefits_score', 'salary_usd']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

encoded_cols = ['experience_level_encoded', 'employment_type_encoded', 'company_size_encoded',
                'education_encoded', 'industry_encoded', 'company_location_encoded', 'employee_residence_encoded']
df[encoded_cols] = df[encoded_cols].fillna(0)

career_map = {
    'AI Research Scientist':'AI Engineer',
    'AI Software Engineer':'AI Engineer',
    'Data Scientist':'Data Scientist',
    'Business Analyst':'Business Analyst',
    'Web Developer':'Web Developer',
    'Software Engineer':'Software Engineer',
    'System Admin':'System Admin'
}
df['career_category'] = df['job_title'].map(career_map)
df = df.dropna(subset=['career_category'])

scaler = StandardScaler()
df[['years_experience', 'remote_ratio', 'job_description_length', 'benefits_score']] = scaler.fit_transform(
    df[['years_experience', 'remote_ratio', 'job_description_length', 'benefits_score']]
)

reg_features = ['experience_level_encoded', 'employment_type_encoded', 'company_size_encoded',
                'education_encoded', 'industry_encoded', 'company_location_encoded', 'employee_residence_encoded',
                'years_experience', 'remote_ratio', 'job_description_length', 'benefits_score'] + list(mlb.classes_)

X_reg = df[reg_features]
y_reg = df['salary_usd']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(n_estimators=200, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)

print("=== Salary Prediction Results ===")
print("RMSE:", rmse)
print("R2 Score:", r2)

clf_features = ['experience_level_encoded', 'employment_type_encoded', 'company_size_encoded',
                'education_encoded', 'industry_encoded', 'company_location_encoded', 'employee_residence_encoded',
                'years_experience', 'remote_ratio'] + list(mlb.classes_)

X_clf = df[clf_features]
y_clf = df['career_category']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf_model = RandomForestClassifier(n_estimators=200, random_state=42)
clf_model.fit(X_train_clf, y_train_clf)
y_pred_clf = clf_model.predict(X_test_clf)

print("\n=== Career Classification Results ===")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("F1 Score:", f1_score(y_test_clf, y_pred_clf, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(y_test_clf, y_pred_clf))

cluster_features = ['salary_usd', 'years_experience', 'experience_level_encoded'] + list(mlb.classes_)
X_cluster = df[cluster_features]

kmeans = KMeans(n_clusters=3, random_state=42)
df['career_cluster'] = kmeans.fit_predict(X_cluster)

cluster_labels = {0:'Low Salary Potential', 1:'Medium Salary Potential', 2:'High Salary Potential'}
df['career_cluster_label'] = df['career_cluster'].map(cluster_labels)

sns.scatterplot(data=df, x='years_experience', y='salary_usd', hue='career_cluster_label')
plt.title('Career Clusters by Experience and Salary')
plt.show()


def predict_career_advice(input_data):
    """
    Predict salary, career, and career cluster for a new candidate.

    User only needs to provide:
        - Important numeric/categorical features
        - Optional key skills (e.g., 'Python', 'Machine Learning')

    All other features will be automatically filled with 0 or median values.
    """

    X_input = pd.DataFrame([[0.0]*len(reg_features)], columns=reg_features, dtype=float)

    for key, value in input_data.items():
        if key in X_input.columns:
            X_input.at[0, key] = value

    numeric_cols = ['years_experience', 'remote_ratio', 'job_description_length', 'benefits_score']
    for col in numeric_cols:
        if col in X_input.columns and pd.isna(X_input.at[0, col]):
            X_input.at[0, col] = 0.0

    X_input[numeric_cols] = scaler.transform(X_input[numeric_cols])

    pred_salary = reg_model.predict(X_input)[0]

    X_clf_input = X_input[clf_features]
    pred_career = clf_model.predict(X_clf_input)[0]

    X_cluster_input = pd.DataFrame([[0.0]*len(cluster_features)], columns=cluster_features, dtype=float)

    for col in cluster_features:
        if col in X_input.columns:
            X_cluster_input[col] = X_input[col]

    X_cluster_input['salary_usd'] = pred_salary

    pred_cluster = kmeans.predict(X_cluster_input)[0]
    cluster_label = cluster_labels[pred_cluster]

    return {
        'Predicted Salary': pred_salary,
        'Recommended Career': pred_career,
        'Career Cluster': cluster_label
    }

example_input = {
    'experience_level_encoded': 3,
    'education_encoded': 3,
    'company_size_encoded': 2,
    'years_experience': 1.0,
    'remote_ratio': 0.5,
    'benefits_score': 0.7,
    'Python': 1,
    'Machine Learning': 1
}

advice = predict_career_advice(example_input)

print("\n=== Career Advice for New Candidate ===")
print(f"Predicted Salary (USD): {advice['Predicted Salary']:.2f}")
print(f"Recommended Career: {advice['Recommended Career']}")
print(f"Career Cluster: {advice['Career Cluster']}")
