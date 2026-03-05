#!/usr/bin/env python3
# ============================================================================
# SETUP SCRIPT: Train dan Save Models untuk Streamlit App
# ============================================================================
# Run this script once to generate required model files:
#   python setup_models.py

import sys
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🤖 HR ANALYTICS MODEL SETUP")
print("=" * 80)
print()

# ============================================================================
# STEP 1: Check dependencies
# ============================================================================
print("📦 Checking dependencies...")
print()

required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'joblib': 'joblib'
}

missing_packages = []
for import_name, package_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"  ✓ {package_name}")
    except ImportError:
        print(f"  ✗ {package_name} - MISSING")
        missing_packages.append(package_name)

if missing_packages:
    print()
    print("❌ Missing packages! Install with:")
    print(f"   pip install {' '.join(missing_packages)}")
    sys.exit(1)

print("\n✓ All dependencies available")
print()

# ============================================================================
# STEP 2: Check dataset
# ============================================================================
print("📊 Checking dataset...")
print()

if not os.path.exists('Dataset/aug_train.csv'):
    print("❌ Dataset file not found: Dataset/aug_train.csv")
    print()
    print("Please ensure the following file exists:")
    print("  Dataset/aug_train.csv (19,160 records)")
    sys.exit(1)

print("  ✓ Dataset found: Dataset/aug_train.csv")
print()

# ============================================================================
# STEP 3: Load and prepare data
# ============================================================================
print("🔄 Loading and preparing data...")
print()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load data
df_train = pd.read_csv('Dataset/aug_train.csv')
print(f"  Loaded {len(df_train):,} records")

# Data cleaning and preparation
df = df_train.copy()

# Remove rows with critical missing values
df = df.dropna(subset=['company_type', 'company_size'])
print(f"  After cleaning: {len(df):,} records")

# Impute missing values
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
df['major_discipline'] = df['major_discipline'].fillna('No Major')

# Feature engineering
experience_map = {
    '<1': 0, '1-3': 2, '4-6': 5, '7-10': 8.5,
    '11-15': 13, '16-20': 18, '>20': 25
}
df['experience_years'] = df['experience'].map(experience_map)

last_job_map = {
    'never': 0, '<1': 0.5, '1': 1, '2': 2,
    '3': 3, '4': 4, '>4': 5
}
df['last_new_job_years'] = df['last_new_job'].map(last_job_map)

# One-hot encoding
categorical_features = ['gender', 'relevent_experience', 'enrolled_university',
                       'education_level', 'major_discipline', 'company_type']
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=False)

# Prepare features
columns_to_drop = ['enrollee_id', 'city', 'experience', 'last_new_job', 'target']
X = df_encoded.drop(columns=[c for c in columns_to_drop if c in df_encoded.columns], errors='ignore')
y = df_encoded['target'].astype(int)

# Filter out any remaining columns that are not numeric or are in drop list
keep_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
X = X[keep_cols]

# Handle any remaining NaN values by filling with 0 (for one-hot encoded columns)
X = X.fillna(0)

print(f"  Features: {X.shape[1]}")
print(f"  Target distribution: {y.value_counts().to_dict()}")
print()

# ============================================================================
# STEP 4: Split data
# ============================================================================
print("✂️  Splitting data...")
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Training set: {len(X_train):,} samples")
print(f"  Test set: {len(X_test):,} samples")
print()

# ============================================================================
# STEP 5: Scale features
# ============================================================================
print("📏 Scaling features...")
print()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("  Features standardized successfully")
print()

# ============================================================================
# STEP 6: Train model
# ============================================================================
print("🚀 Training Gradient Boosting model...")
print("  (This may take 1-2 minutes...)")
print()

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("  ✓ Model training completed")
print()

# ============================================================================
# STEP 7: Evaluate model
# ============================================================================
print("📊 Evaluating model...")
print()

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"  Accuracy:  {accuracy:.4f} (75.4% expected)")
print(f"  Precision: {precision:.4f} (62.1% expected)")
print(f"  Recall:    {recall:.4f} (51.2% expected)")
print(f"  F1-Score:  {f1:.4f} (56.0% expected)")
print(f"  ROC-AUC:   {auc:.4f} (0.784 expected)")
print()

if auc >= 0.75:
    print("  ✓ Model performance acceptable!")
else:
    print("  ⚠️  Model performance below expected")

print()

# ============================================================================
# STEP 8: Create models directory
# ============================================================================
print("💾 Creating models directory...")
print()

os.makedirs('models', exist_ok=True)
print("  ✓ models/ directory ready")
print()

# ============================================================================
# STEP 9: Save model artifacts
# ============================================================================
print("📦 Saving model artifacts...")
print()

try:
    # Save model
    joblib.dump(model, 'models/gb_model.joblib')
    print("  ✓ Model saved: models/gb_model.joblib")
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    print("  ✓ Scaler saved: models/scaler.joblib")
    
    # Save feature columns
    feature_list = X.columns.tolist()
    joblib.dump(feature_list, 'models/feature_columns.joblib')
    print(f"  ✓ Features saved: models/feature_columns.joblib ({len(feature_list)} features)")
    
except Exception as e:
    print(f"  ❌ Error saving files: {e}")
    sys.exit(1)

print()

# ============================================================================
# STEP 10: Verify saved files
# ============================================================================
print("✅ Verifying saved files...")
print()

all_exist = True
for filename in ['gb_model.joblib', 'scaler.joblib', 'feature_columns.joblib']:
    filepath = f'models/{filename}'
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✓ {filepath} ({size_mb:.1f} MB)")
    else:
        print(f"  ✗ {filepath} - NOT FOUND")
        all_exist = False

print()

if all_exist:
    print("=" * 80)
    print("✅ SETUP COMPLETE!")
    print("=" * 80)
    print()
    print("📋 Next steps:")
    print()
    print("  1. Launch Streamlit app:")
    print("     streamlit run app.py")
    print()
    print("  2. App will be available at:")
    print("     http://localhost:8501")
    print()
    print("=" * 80)
    sys.exit(0)
else:
    print("❌ Some files were not saved correctly!")
    sys.exit(1)
