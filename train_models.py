import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('data/mental_health_dataset_numeric.csv')

# Replace invalid strings with numeric
df.replace({'Yes': 1, 'No': 0}, inplace=True)

# Rule-Based Label Generation
def rule_based_anxiety(row):
    risk_count = 0
    if row['stress_level'] >= 6: risk_count += 1
    if row['academic_pressure'] >= 6: risk_count += 1
    if row['financial_stress'] >= 6: risk_count += 1
    if row['social_interactions'] <= 2: risk_count += 1
    if row['negative_thoughts'] >= 6: risk_count += 1
    if row['exercise'] == 0: risk_count += 1
    if row['screen_time'] >= 6: risk_count += 1
    if row['sleep_hours'] <= 5: risk_count += 1
    if row['optimism_level'] <= 4: risk_count += 1
    if row['support_system'] <= 4: risk_count += 1
    if row['relationship_issues'] == 1: risk_count += 1
    return 1 if risk_count >= 6 else 0

def rule_based_depression(row):
    risk_count = 0
    if row['self_esteem'] <= 5: risk_count += 1
    if row['negative_thoughts'] >= 4: risk_count += 1
    if row['sleep_hours'] <= 5: risk_count += 1
    if row['alone_time'] >= 4: risk_count += 1
    if row['diet_quality'] == 0: risk_count += 1
    if row['exercise'] == 0: risk_count += 1
    if row['social_interactions'] <= 4: risk_count += 1
    if row['financial_stress'] >= 5: risk_count += 1
    if row['news_consumption'] >= 4: risk_count += 1
    if row['support_system'] <= 4: risk_count += 1
    if row['optimism_level'] <= 4: risk_count += 1
    if row['academic_pressure'] >= 4: risk_count += 1
    return 1 if risk_count >= 6 else 0

# Apply Rule-Based Labels
df['anxiety_level'] = df.apply(rule_based_anxiety, axis=1)
df['depression_level'] = df.apply(rule_based_depression, axis=1)

# Features
features = [
    'sleep_hours', 'study_hours', 'social_interactions', 'exercise',
    'screen_time', 'diet_quality', 'stress_level', 'academic_pressure',
    'financial_stress', 'relationship_issues', 'self_esteem',
    'negative_thoughts', 'optimism_level', 'social_media_usage',
    'news_consumption', 'support_system', 'alone_time'
]

# Targets
target_anxiety = 'anxiety_level'
target_depression = 'depression_level'

X = df[features]
y_anx = df[target_anxiety]
y_dep = df[target_depression]

# Train-test split
X_train_anx, X_test_anx, y_train_anx, y_test_anx = train_test_split(X, y_anx, test_size=0.2, random_state=42)
X_train_dep, X_test_dep, y_train_dep, y_test_dep = train_test_split(X, y_dep, test_size=0.2, random_state=42)

# Train Random Forest Models
model_anx = RandomForestClassifier(n_estimators=100, random_state=42)
model_anx.fit(X_train_anx, y_train_anx)

model_dep = RandomForestClassifier(n_estimators=100, random_state=42)
model_dep.fit(X_train_dep, y_train_dep)

# Predictions
pred_anx = model_anx.predict(X_test_anx)
pred_dep = model_dep.predict(X_test_dep)

# Evaluation & Print Accuracy + Reports
print("🔍 Random Forest - Anxiety Prediction Accuracy: {:.2f}%".format(accuracy_score(y_test_anx, pred_anx)*100))
print(classification_report(y_test_anx, pred_anx))

print("🔍 Random Forest - Depression Prediction Accuracy: {:.2f}%".format(accuracy_score(y_test_dep, pred_dep)*100))
print(classification_report(y_test_dep, pred_dep))

# Plotting functions
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_feature_importance(model, features, title):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances[indices], y=[features[i] for i in indices])
    plt.title(f'Feature Importance - {title}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def plot_roc_curve(model, X_test, y_test, title):
    y_score = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {title}')
    plt.legend(loc="lower right")
    plt.show()

# Plot results for Anxiety
plot_confusion_matrix(y_test_anx, pred_anx, "Anxiety")
plot_feature_importance(model_anx, features, "Anxiety")
plot_roc_curve(model_anx, X_test_anx, y_test_anx, "Anxiety")

# Plot results for Depression
plot_confusion_matrix(y_test_dep, pred_dep, "Depression")
plot_feature_importance(model_dep, features, "Depression")
plot_roc_curve(model_dep, X_test_dep, y_test_dep, "Depression")

# Save Models
os.makedirs("models", exist_ok=True)
with open("models/anxiety_model.pkl", "wb") as f:
    pickle.dump(model_anx, f)
with open("models/depression_model.pkl", "wb") as f:
    pickle.dump(model_dep, f)

print("✅ Models trained and saved successfully!")
