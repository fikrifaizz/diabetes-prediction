import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

class DiabetesPreprocessor:
    def __init__(self):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.cat_cols = []
        self.num_cols = []
        
    def _create_medical_features(self, df):
        df = df.copy()
        
        df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
        # MAP (Mean Arterial Pressure): Rata-rata tekanan darah dalam satu siklus
        df['map'] = (df['systolic_bp'] + (2 * df['diastolic_bp'])) / 3
        
        df['cholesterol_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1)
        
        # Kombinasi aktivitas fisik dan diet
        df['lifestyle_score'] = df['physical_activity_minutes_per_week'] * df['diet_score']
        
        # Risiko obesitas meningkat seiring umur
        df['bmi_age_interaction'] = df['bmi'] * df['age']
        return df

    def fit_transform(self, df):
        print("Creating Medical Features...")
        df = self._create_medical_features(df)
        
        self.cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.num_cols = df.select_dtypes(exclude=['object']).columns.tolist()
        
        print("Fitting Encoder & Scaler...")
        df[self.cat_cols] = self.encoder.fit_transform(df[self.cat_cols])
        df[self.num_cols] = self.imputer.fit_transform(df[self.num_cols])
        df[self.num_cols] = self.scaler.fit_transform(df[self.num_cols])
        
        return df

    def transform(self, df):
        df = self._create_medical_features(df)
        
        df[self.cat_cols] = self.encoder.transform(df[self.cat_cols])
        df[self.num_cols] = self.imputer.transform(df[self.num_cols])
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        
        return df

    def save(self, filename="processor.pkl"):
        path = os.path.join(config.MODEL_DIR, filename)
        joblib.dump(self, path)
        print(f"Preprocessor saved to {path}")

    @staticmethod
    def load(filename="processor.pkl"):
        path = os.path.join(config.MODEL_DIR, filename)
        return joblib.load(path)