import pandas as pd
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from preprocessing import DiabetesPreprocessor

def run_inference():
    print("=== STARTING INFERENCE PIPELINE ===")
    
    print(f"Loading test data from {config.TEST_FILE}")
    test = pd.read_csv(config.TEST_FILE)
    ids = test['id']
    X_test = test.drop(['id'], axis=1)
    
    print("Loading Preprocessor...")
    try:
        preprocessor = DiabetesPreprocessor.load()
    except FileNotFoundError:
        print("Error: Preprocessor not found! Run train.py first.")
        return

    X_test_processed = preprocessor.transform(X_test)
    
    print("Loading Models...")
    model_xgb = joblib.load(os.path.join(config.MODEL_DIR, "model_xgb.pkl"))
    model_hgb = joblib.load(os.path.join(config.MODEL_DIR, "model_hgb.pkl"))
    model_rf  = joblib.load(os.path.join(config.MODEL_DIR, "model_rf.pkl"))
    
    print("Predicting with Ensemble...")
    p_xgb = model_xgb.predict_proba(X_test_processed)[:, 1]
    p_hgb = model_hgb.predict_proba(X_test_processed)[:, 1]
    p_rf  = model_rf.predict_proba(X_test_processed)[:, 1]
    
    weights = config.WEIGHTS
    print(f"Applying Weights: XGB={weights['xgb']}, HGB={weights['hgb']}, RF={weights['rf']}")
    
    final_preds = (weights['xgb'] * p_xgb) + \
                  (weights['hgb'] * p_hgb) + \
                  (weights['rf'] * p_rf)
    
    submission = pd.DataFrame({
        'id': ids,
        'diagnosed_diabetes': final_preds
    })
    
    submission.to_csv(config.SUBMISSION_FILE, index=False)

if __name__ == "__main__":
    run_inference()