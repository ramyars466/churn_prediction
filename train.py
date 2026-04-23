import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

np.random.seed(42)

DATA_FILE = 'employees.csv'
MODEL_FILE = 'employee_model.pkl'

def load_and_preprocess_data():
    print("Loading employee dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} employee records")
    
    df = df.drop(columns=['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], errors='ignore')
    
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    print(f"Attrition rate: {df['Attrition'].mean():.2%}")
    
    return df, label_encoders

def train_model():
    df, label_encoders = load_and_preprocess_data()
    
    feature_cols = [col for col in df.columns if col != 'Attrition']
    X = df[feature_cols]
    y = df['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training LightGBM model...")
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nModel AUC-ROC: {auc_score:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test)))
    
    df.to_csv('employees_encoded.csv', index=False)
    print("Saved encoded data to employees_encoded.csv")
    
    joblib.dump((model, feature_cols, label_encoders), MODEL_FILE)
    print(f"Model saved as {MODEL_FILE}")
    
    print("\nTop 10 Risk Factors:")
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']}")
    
    return model, feature_cols, label_encoders

if __name__ == "__main__":
    train_model()