import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, precision_score
import joblib, json, os

# 1. Load your dataset
df = pd.read_csv('dataset/diet_recommendations_dataset.csv')

# ============================================================
# PRODUCTION-READY FEATURE BUILDER (Matches Frontend Forms)
# ============================================================
GENDER_MAP = {'Male': 0, 'Female': 1, 'Other': 2}
ACTIVITY_MAP = {'Low activity': 0, 'Moderate activity': 1, 'High activity': 2, 'Low': 0, 'Moderate': 1, 'High': 2}

def build_features_from_user_profile(row):
    """
    Transforms raw dataset rows OR frontend user profiles into 
    the exact same normalized feature vector for training/inference.
    """
    # Age & BMI normalization
    age = float(row.get('Age', row.get('age', 30)))
    gender = GENDER_MAP.get(str(row.get('Gender', row.get('gender', 'Male'))), 0)
    bmi = float(row.get('BMI', row.get('bmi', 22.0)))
    
    # Extract the health condition string (handles raw dataset or frontend key)
    condition = str(row.get('Disease_Type', row.get('health_information', 'None'))).lower().strip()
    
    # 1-Hot encoding user form choices explicitly
    has_high_bp = 1 if any(x in condition for x in ['bp', 'hypertension', 'high blood']) else 0
    has_diabetes = 1 if any(x in condition for x in ['diabetes', 'diabetic', 'high sugar']) else 0
    has_ulcer = 1 if 'ulcer' in condition else 0
    has_yeast = 1 if 'yeast' in condition else 0
    has_allergies = 1 if any(x in condition for x in ['allergies', 'allergy']) else 0
    weight_loss = 1 if 'weight loss' in condition else 0
    weight_gain = 1 if 'weight gain' in condition else 0
    is_healthy = 1 if condition in ['none', 'healthy', 'normal'] else 0
    
    # Context-aware blood pressure backups from raw dataset markers
    if not has_high_bp:
        bp_val = row.get('Blood_Pressure_mmHg', 120)
        if isinstance(bp_val, str) and '/' in bp_val:
            try: has_high_bp = 1 if int(bp_val.split('/')[0]) >= 130 else 0
            except: pass
        elif isinstance(bp_val, (int, float)) and bp_val >= 130:
            has_high_bp = 1

    # Context-aware glucose backups from raw dataset markers
    if not has_diabetes:
        glucose = float(row.get('Glucose_mg/dL', 90))
        if glucose > 125: has_diabetes = 1

    # Activity mapping
    activity_str = str(row.get('Physical_Activity_Level', row.get('activity_level', 'Moderate')))
    activity_score = ACTIVITY_MAP.get(activity_str, 1)
    
    return [
        age, gender, bmi, has_high_bp, has_diabetes, has_ulcer, 
        has_yeast, has_allergies, weight_loss, weight_gain, is_healthy, activity_score
    ]

FEATURE_NAMES = [
    'age', 'gender_num', 'bmi', 'has_high_bp', 'has_diabetes', 'has_ulcer',
    'has_yeast', 'has_allergies', 'weight_loss', 'weight_gain', 'is_healthy', 'activity_score'
]

# 2. Process Dataset Rows through the Frontend-aligned pipeline
print("Transforming dataset into frontend-compatible features...")
X = np.array([build_features_from_user_profile(row) for _, row in df.iterrows()])
y = df['Diet_Recommendation'].values

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================
# HYPERPARAMETER TUNING & REGULARIZATION (Focus: Precision)
# ============================================================
print("Tuning hyperparameters via GridSearchCV...")

param_grid_rf = {
    'n_estimators': [100, 150],
    'max_depth': [5, 7, 10],            # Constrains depth (Regularization against overfitting)
    'min_samples_split': [5, 10],       # Requires more samples to split a node (Regularization)
    'class_weight': ['balanced']        # Prevents minority class starvation
}

param_grid_dt = {
    'max_depth': [5, 7, 10],
    'min_samples_split': [5, 10],
    'class_weight': ['balanced']
}

param_grid_gb = {
    'n_estimators': [100, 150],
    'max_depth': [5, 7, 10],
    'min_samples_split': [5, 10],
}

# Optimize for precision_macro so recommendations don't misclassify sensitive health needs
grid_search_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid_dt,
    scoring='precision_macro',
    cv=5
)
grid_search_dt.fit(X_train, y_train)
best_dt = grid_search_dt.best_estimator_


grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid_rf,
    scoring='precision_macro',
    cv=5
)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_


grid_search_gb = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid_gb,
    scoring='precision_macro',
    cv=5
)
grid_search_gb.fit(X_train, y_train)
best_gb = grid_search_gb.best_estimator_

print(f"Best RF Parameters: {grid_search_rf.best_params_}")
print(f"Best DT Parameters: {grid_search_dt.best_params_}")
print(f"Best GB Parameters: {grid_search_gb.best_params_}")

# 3. Evaluation
unique_classes = sorted(list(set(y)))

y_pred_dt = best_dt.predict(X_test)
print("\n=== DECISION TREE CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred_dt, target_names=unique_classes))

y_pred_rf = best_rf.predict(X_test)
print("\n=== RANDOM FOREST CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred_rf, target_names=unique_classes))

y_pred_gb = best_gb.predict(X_test)
print("\n=== GRADIENT BOOSTING CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred_gb, target_names=unique_classes))

# Feature importances
fi = pd.Series(best_rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
print("\n=== UPDATED FEATURE IMPORTANCES ===")
print(fi.to_string())

# ============================================================
# VISUALIZATION GENERATION
# ============================================================
plt.figure(figsize=(14, 5))

# Plot 1: Feature Importance
plt.subplot(1, 2, 1)
sns.barplot(x=fi.values, y=fi.index, hue=fi.index, palette='viridis', legend=False)
plt.title('How the Model Values Your User Form Fields')
plt.xlabel('Importance Weight')

# Plot 2: Precision Heatmap (Confusion Matrix)
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=unique_classes, yticklabels=unique_classes)
plt.title('Diet Recommendation Confusion Matrix (Random Forest)')
plt.ylabel('True Target Diet')
plt.xlabel('Predicted Diet Assignment')

plt.tight_layout()
os.makedirs('services', exist_ok=True)
plt.savefig('services/model_evaluation_metrics.png')
print("\n📊 Evaluation graphics saved to services/model_evaluation_metrics.png")
plt.show()

# 4. Save Production Artifacts
joblib.dump(best_rf, 'services/diet_classifier.pkl')

model_metadata = {
    'feature_names': FEATURE_NAMES,
    'classes': best_rf.classes_.tolist(),
    'version': '2.0-production',
    'test_precision_macro': float(precision_score(y_test, y_pred_rf, average='macro'))
}
with open('services/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("\n✅ Production model ready for backend API deployment!")