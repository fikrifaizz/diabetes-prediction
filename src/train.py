import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from preprocessing import DiabetesPreprocessor

def run_training():
    print("=== STARTING TRAINING PIPELINE ===")
    
    print(f"Loading data from {config.TRAIN_FILE}")
    train = pd.read_csv(config.TRAIN_FILE)
    X = train.drop(['id', 'diagnosed_diabetes'], axis=1)
    y = train['diagnosed_diabetes']
    
    print("Preprocessing Data...")
    preprocessor = DiabetesPreprocessor()
    X_processed = preprocessor.fit_transform(X)
    preprocessor.save()
    
    print("\nTraining XGBoost (with Optimized Params)...")
    model_xgb = xgb.XGBClassifier(**config.PARAMS_XGB)
    model_xgb.fit(X_processed, y)
    joblib.dump(model_xgb, os.path.join(config.MODEL_DIR, "model_xgb.pkl"))
    
    print("Training HistGradientBoosting...")
    model_hgb = HistGradientBoostingClassifier(**config.PARAMS_HGB)
    model_hgb.fit(X_processed, y)
    joblib.dump(model_hgb, os.path.join(config.MODEL_DIR, "model_hgb.pkl"))
    
    print("Training Random Forest...")
    model_rf = RandomForestClassifier(**config.PARAMS_RF)
    model_rf.fit(X_processed, y)
    joblib.dump(model_rf, os.path.join(config.MODEL_DIR, "model_rf.pkl"))
    
    print("\n=== TRAINING SUCCESS ===")
    print("All models and preprocessor saved in 'models/' folder.")

if __name__ == "__main__":
    run_training()