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

PARAMS_XGB = {
    'n_estimators': 2459, 
    'learning_rate': 0.05310177658897893, 
    'max_depth': 3, 
    'subsample': 0.7773742316554201, 
    'colsample_bytree': 0.8043461847384595, 
    'min_child_weight': 9, 
    'reg_alpha': 0.021839888151954773, 
    'reg_lambda': 0.10308306804534173, 
    'n_jobs': -1, 
    'random_state': 42, 
    'eval_metric': 'auc'
}

PARAMS_HGB = {
    'max_iter': 963, 
    'learning_rate': 0.05669917454523615, 
    'max_depth': 5, 
    'l2_regularization': 3.3768072485512746, 
    'random_state': 42, 
    'early_stopping': True
}

PARAMS_RF = {
    'n_estimators': 413, 
    'max_depth': 14, 
    'min_samples_split': 20, 
    'min_samples_leaf': 3, 
    'n_jobs': -1, 
    'random_state': 42
}

WEIGHTS = {
    "xgb": 0.50,
    "hgb": 0.35,
    "rf":  0.15
}