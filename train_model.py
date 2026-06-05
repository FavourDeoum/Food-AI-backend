import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib, json, os

df = pd.read_csv('dataset/diet_recommendations_dataset.csv')
print(df.columns)  # See all column names
print(df.head())

# ============================================================
# NOTE: 100% accuracy is expected — the dataset IS deterministic.
# Disease_Type alone determines Diet_Recommendation perfectly.
# Our job is to build a model that generalises from disease+features
# to diet labels, then map those labels to Cameroonian dishes.
# The model is the bridge between user profile -> dish suitability tags.
# ============================================================

# Feature builder (must match inference time exactly)
HEALTH_CONDITION_MAP = {
    'None': 0, 'Ulcer': 1, 'Weight Loss': 2, 'Weight Gain': 3,
    'Allergies': 4, 'BP': 5, 'Hypertension': 5, 'Diabetes': 6,
}
ACTIVITY_MAP = {'Low activity': 0, 'Moderate activity': 1, 'High activity': 2}
GENDER_MAP = {'Male': 0, 'Female': 1}
SEVERITY_MAP = {'Mild': 0, 'Moderate': 1, 'Severe': 2}

def build_features(row):
    health_risk = HEALTH_CONDITION_MAP.get(str(row.get('health_condition_primary', 'None')), 0)
    activity_score = ACTIVITY_MAP.get(str(row.get('activity_level', 'Moderate activity')), 1)
    return [
        float(row.get('age', 30)),
        GENDER_MAP.get(str(row.get('gender', 'Male')), 0),
        float(row.get('bmi', 22.0)),
        int(row.get('has_diabetes', 0)),
        int(row.get('has_hypertension', 0)),
        int(row.get('has_obesity', 0)),
        int(row.get('is_healthy', 0)),
        int(row.get('high_glucose', 0)),
        int(row.get('high_bp', 0)),
        activity_score,
        int(row.get('is_sedentary', 0)),
        int(row.get('is_active', 0)),
        health_risk,
        SEVERITY_MAP.get(str(row.get('severity', 'Mild')), 0),
        float(row.get('weekly_exercise', 3.0)),
    ]

FEATURE_NAMES = [
    'age', 'gender_num', 'bmi', 'has_diabetes', 'has_hypertension',
    'has_obesity', 'is_healthy', 'high_glucose', 'high_bp',
    'activity_score', 'is_sedentary', 'is_active', 'health_risk',
    'severity_num', 'weekly_exercise'
]

X = np.array([build_features(row) for _, row in df.iterrows()])
y = df['Diet_Recommendation'].values

# Train final model on ALL data (we know it's perfect)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use RandomForest for production (more robust to new profiles)
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Print decision tree logic (interpretable)
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
print("=== DECISION TREE LOGIC ===")
print(export_text(dt, feature_names=FEATURE_NAMES))

# Feature importances
fi = pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
print("\n=== RANDOM FOREST FEATURE IMPORTANCES ===")
print(fi.to_string())

# Verify predictions
print("\n=== TEST PREDICTIONS (sample) ===")
test_cases = [
    {'label': 'Diabetic patient', 'age': 55, 'gender': 'Male', 'bmi': 28.5, 'has_diabetes': 1, 'has_hypertension': 0, 'has_obesity': 0, 'is_healthy': 0, 'high_glucose': 1, 'high_bp': 0, 'activity_level': 'Low activity', 'is_sedentary': 1, 'is_active': 0, 'health_condition_primary': 'Diabetes', 'severity': 'Moderate', 'weekly_exercise': 2.0},
    {'label': 'Hypertensive patient', 'age': 60, 'gender': 'Female', 'bmi': 26.0, 'has_diabetes': 0, 'has_hypertension': 1, 'has_obesity': 0, 'is_healthy': 0, 'high_glucose': 0, 'high_bp': 1, 'activity_level': 'Moderate activity', 'is_sedentary': 0, 'is_active': 0, 'health_condition_primary': 'Hypertension', 'severity': 'Mild', 'weekly_exercise': 3.5},
    {'label': 'Healthy active young man', 'age': 25, 'gender': 'Male', 'bmi': 22.0, 'has_diabetes': 0, 'has_hypertension': 0, 'has_obesity': 0, 'is_healthy': 1, 'high_glucose': 0, 'high_bp': 0, 'activity_level': 'High activity', 'is_sedentary': 0, 'is_active': 1, 'health_condition_primary': 'None', 'severity': 'Mild', 'weekly_exercise': 8.0},
    {'label': 'Overweight patient', 'age': 42, 'gender': 'Female', 'bmi': 33.0, 'has_diabetes': 0, 'has_hypertension': 0, 'has_obesity': 1, 'is_healthy': 0, 'high_glucose': 0, 'high_bp': 0, 'activity_level': 'Low activity', 'is_sedentary': 1, 'is_active': 0, 'health_condition_primary': 'Weight Gain', 'severity': 'Moderate', 'weekly_exercise': 1.5},
]

for tc in test_cases:
    feat = build_features(tc)
    pred = rf.predict([feat])[0]
    proba = rf.predict_proba([feat])[0]
    classes = rf.classes_
    proba_str = {c: f"{p:.2f}" for c, p in zip(classes, proba)}
    print(f"  {tc['label']}: {pred} {proba_str}")

# Save model artifacts
os.makedirs('services', exist_ok=True)
joblib.dump(rf, 'services/diet_classifier.pkl')

model_metadata = {
    'feature_names': FEATURE_NAMES,
    'classes': rf.classes_.tolist(),
    'health_condition_map': HEALTH_CONDITION_MAP,
    'activity_map': ACTIVITY_MAP,
    'gender_map': GENDER_MAP,
    'severity_map': SEVERITY_MAP,
    'version': '1.0',
    'train_samples': len(X_train),
    'test_accuracy': float((rf.predict(X_test) == y_test).mean()),
}
with open('services/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print(f"\n✅ Model saved to services/diet_classifier.pkl")
print(f"✅ Metadata saved to services/model_metadata.json")
print(f"Test accuracy: {model_metadata['test_accuracy']:.4f}")