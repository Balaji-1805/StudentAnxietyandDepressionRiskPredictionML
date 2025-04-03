import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import os

# File paths
DATASET_PATH = 'data/mental_health_dataset.csv'
ANXIETY_MODEL_PATH = 'models/anxiety_model.pkl'
DEPRESSION_MODEL_PATH = 'models/depression_model.pkl'

# Load dataset
df = pd.read_csv(DATASET_PATH)

# Feature columns
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

# Prepare data
X = df[features]
y_anxiety = df[target_anxiety]
y_depression = df[target_depression]

# Train-test split
X_train_anx, _, y_train_anx, _ = train_test_split(X, y_anxiety, test_size=0.2, random_state=42)
X_train_dep, _, y_train_dep, _ = train_test_split(X, y_depression, test_size=0.2, random_state=42)

# Model training (Logistic Regression)
model_anxiety = LogisticRegression(max_iter=1000, random_state=42)
model_anxiety.fit(X_train_anx, y_train_anx)

model_depression = LogisticRegression(max_iter=1000, random_state=42)
model_depression.fit(X_train_dep, y_train_dep)

# Save models
os.makedirs('models', exist_ok=True)
with open(ANXIETY_MODEL_PATH, 'wb') as f:
    pickle.dump(model_anxiety, f)

with open(DEPRESSION_MODEL_PATH, 'wb') as f:
    pickle.dump(model_depression, f)

print("Logistic Regression models trained and saved successfully!")
