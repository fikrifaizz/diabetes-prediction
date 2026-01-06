import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
import joblib
import os
import mlflow
import mlflow.sklearn
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from preprocessing import DiabetesPreprocessor

def run_training():
    mlflow.set_experiment("Diabetes_Prediction_S5E12")
    
    with mlflow.start_run(run_name="Ensemble_Production_With_Metrics"): 
        
        print("=== STARTING TRAINING PIPELINE ===")
        
        train = pd.read_csv(config.TRAIN_FILE)
        X = train.drop(['id', 'diagnosed_diabetes'], axis=1)
        y = train['diagnosed_diabetes']
        
        print("Preprocessing Data...")
        preprocessor = DiabetesPreprocessor()
        X_processed = preprocessor.fit_transform(X)
        preprocessor.save()
        
        def add_prefix(params, prefix):
            return {f"{prefix}_{k}": v for k, v in params.items()}

        mlflow.log_params(add_prefix(config.PARAMS_XGB, "xgb"))
        mlflow.log_params(add_prefix(config.PARAMS_HGB, "hgb"))
        mlflow.log_params(add_prefix(config.PARAMS_RF, "rf"))
        mlflow.log_params(add_prefix(config.WEIGHTS, "weight"))

        scoring_metrics = {
            'auc': 'roc_auc',
            'accuracy': 'accuracy',
            'f1': 'f1',
            'recall': 'recall',
            'precision': 'precision'
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        print("\n--- Processing XGBoost ---")
        model_xgb = xgb.XGBClassifier(**config.PARAMS_XGB)
        
        print("Calculating metrics (CV)...")
        scores = cross_validate(model_xgb, X_processed, y, cv=cv, scoring=scoring_metrics)
        
        mlflow.log_metric("xgb_auc", scores['test_auc'].mean())
        mlflow.log_metric("xgb_accuracy", scores['test_accuracy'].mean())
        mlflow.log_metric("xgb_f1", scores['test_f1'].mean())
        mlflow.log_metric("xgb_recall", scores['test_recall'].mean())
        mlflow.log_metric("xgb_precision", scores['test_precision'].mean())
        
        print("Full Training...")
        model_xgb.fit(X_processed, y)
        joblib.dump(model_xgb, os.path.join(config.MODEL_DIR, "model_xgb.pkl"))
        mlflow.sklearn.log_model(model_xgb, "model_xgboost")

        print("\n--- Processing HistGradient ---")
        model_hgb = HistGradientBoostingClassifier(**config.PARAMS_HGB)
        
        print("Calculating metrics (CV)...")
        scores = cross_validate(model_hgb, X_processed, y, cv=cv, scoring=scoring_metrics)
        
        mlflow.log_metric("hgb_auc", scores['test_auc'].mean())
        mlflow.log_metric("hgb_accuracy", scores['test_accuracy'].mean())
        mlflow.log_metric("hgb_f1", scores['test_f1'].mean())
        mlflow.log_metric("hgb_recall", scores['test_recall'].mean())
        mlflow.log_metric("hgb_precision", scores['test_precision'].mean())
        
        print("Full Training...")
        model_hgb.fit(X_processed, y)
        joblib.dump(model_hgb, os.path.join(config.MODEL_DIR, "model_hgb.pkl"))
        
        print("\n--- Processing Random Forest ---")
        model_rf = RandomForestClassifier(**config.PARAMS_RF)
        
        print("Calculating metrics (CV)...")
        scores = cross_validate(model_rf, X_processed, y, cv=cv, scoring=scoring_metrics)
        
        mlflow.log_metric("rf_auc", scores['test_auc'].mean())
        mlflow.log_metric("rf_accuracy", scores['test_accuracy'].mean())
        mlflow.log_metric("rf_f1", scores['test_f1'].mean())
        mlflow.log_metric("rf_recall", scores['test_recall'].mean())
        mlflow.log_metric("rf_precision", scores['test_precision'].mean())
        
        print("Full Training...")
        model_rf.fit(X_processed, y)
        joblib.dump(model_rf, os.path.join(config.MODEL_DIR, "model_rf.pkl"))
        
        print("\n=== PIPELINE FINISHED ===")
        print("Check MLflow Dashboard for Metrics!")

if __name__ == "__main__":
    run_training()