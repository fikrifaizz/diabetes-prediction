import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data/raw")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_FILE = os.path.join(OUTPUT_DIR, "submission_production.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

WEIGHTS = {
    "xgb": 0.50,
    "hgb": 0.35,
    "rf":  0.15
}
PARAMS_XGB = {'n_estimators': 2130, 'learning_rate': 0.06251013292301977, 'max_depth': 3, 'subsample': 0.9487630904369846, 'colsample_bytree': 0.7762949972082455, 'min_child_weight': 3, 'reg_alpha': 0.6939172839668335, 'reg_lambda': 1.4682231625197357, 'n_jobs': -1, 'random_state': 42, 'eval_metric': 'auc'}
PARAMS_HGB = {'max_iter': 991, 'learning_rate': 0.05900747732032968, 'max_depth': 10, 'l2_regularization': 0.04723004193073991, 'random_state': 42, 'early_stopping': True}
PARAMS_RF = {'n_estimators': 391, 'max_depth': 19, 'min_samples_split': 10, 'min_samples_leaf': 13, 'n_jobs': -1, 'random_state': 42}
