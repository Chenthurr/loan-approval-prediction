"""
Loan Approval Prediction — Core ML Pipeline
Author: Mini Project
Tech: Python, Pandas, NumPy, Scikit-learn, TensorFlow
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


class DataGenerator:
    """Generate synthetic loan dataset mimicking real-world distributions."""

    @staticmethod
    def generate(n_samples=10240, save_path=None):
        np.random.seed(RANDOM_STATE)

        data = {
            'annual_income': np.random.lognormal(mean=11.1, sigma=0.4, size=n_samples).astype(int),
            'loan_amount': np.random.lognormal(mean=9.5, sigma=0.5, size=n_samples).astype(int),
            'credit_score': np.clip(np.random.normal(692, 65, n_samples), 300, 850).astype(int),
            'employment_length': np.random.choice(
                range(0, 31), size=n_samples,
                p=np.exp(-np.arange(31)/8) / np.sum(np.exp(-np.arange(31)/8))
            ),
            'debt_to_income': np.clip(np.random.beta(2, 5, n_samples) * 80, 0, 60).round(2),
            'home_ownership': np.random.choice(
                ['rent', 'mortgage', 'own'], size=n_samples, p=[0.35, 0.50, 0.15]
            ),
            'loan_purpose': np.random.choice(
                ['debt_consolidation', 'home_improvement', 'medical', 'education', 'business', 'other'],
                size=n_samples, p=[0.42, 0.18, 0.14, 0.12, 0.09, 0.05]
            ),
            'previous_defaults': np.random.choice([0, 1, 2], size=n_samples, p=[0.75, 0.18, 0.07]),
            'interest_rate': np.clip(np.random.normal(12.5, 4.2, n_samples), 3.0, 30.0).round(2)
        }

        df = pd.DataFrame(data)

        # Realistic approval logic
        risk_score = (
            (df['credit_score'] - 600) / 500 * 35 +
            (df['annual_income'] - 50000) / 100000 * 20 -
            (df['loan_amount'] - 15000) / 50000 * 15 -
            (df['debt_to_income'] - 20) / 60 * 20 +
            (df['employment_length'] / 30) * 8 +
            df['home_ownership'].map({'rent': 0, 'mortgage': 3, 'own': 5}) -
            df['previous_defaults'] * 12 +
            df['loan_purpose'].map({
                'debt_consolidation': 2, 'home_improvement': 3,
                'medical': 1, 'education': 2, 'business': -2, 'other': 0
            })
        )
        risk_score += np.random.normal(0, 5, n_samples)
        df['loan_approved'] = (risk_score > 5).astype(int)

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Dataset saved to {save_path}")

        return df


class Preprocessor:
    """Data preprocessing pipeline with feature engineering."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def fit_transform(self, df):
        df = df.copy()

        for col in ['home_ownership', 'loan_purpose']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Feature engineering
        df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
        df['credit_tier'] = pd.cut(df['credit_score'], bins=[0, 580, 670, 740, 850],
                                    labels=[0, 1, 2, 3]).astype(int)
        df['monthly_payment'] = df['loan_amount'] * (df['interest_rate']/1200) *                                 (1 + df['interest_rate']/1200)**60 /                                 ((1 + df['interest_rate']/1200)**60 - 1)
        df['payment_to_income'] = df['monthly_payment'] / (df['annual_income'] / 12)

        self.feature_names = [c for c in df.columns if c != 'loan_approved']
        X = df[self.feature_names]
        y = df['loan_approved']
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True

        return X_scaled, y.values

    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        df = df.copy()
        for col, le in self.label_encoders.items():
            df[col] = le.transform(df[col])
        df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
        df['credit_tier'] = pd.cut(df['credit_score'], bins=[0, 580, 670, 740, 850],
                                    labels=[0, 1, 2, 3]).astype(int)
        df['monthly_payment'] = df['loan_amount'] * (df['interest_rate']/1200) *                                 (1 + df['interest_rate']/1200)**60 /                                 ((1 + df['interest_rate']/1200)**60 - 1)
        df['payment_to_income'] = df['monthly_payment'] / (df['annual_income'] / 12)
        return self.scaler.transform(df[self.feature_names])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'encoders': self.label_encoders, 'scaler': self.scaler,
                         'features': self.feature_names, 'fitted': self.is_fitted}, f)

    @classmethod
    def load(cls, path):
        inst = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        inst.label_encoders = data['encoders']
        inst.scaler = data['scaler']
        inst.feature_names = data['features']
        inst.is_fitted = data['fitted']
        return inst


class LoanApprovalModel:
    """End-to-end loan approval prediction system."""

    def __init__(self, models_dir='models'):
        self.models = {}
        self.results = {}
        self.preprocessor = Preprocessor()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.history = None

    def load_data(self, csv_path=None, n_samples=10240):
        if csv_path and Path(csv_path).exists():
            self.df = pd.read_csv(csv_path)
        else:
            self.df = DataGenerator.generate(n_samples)
        print(f"Loaded {len(self.df)} records | Approval rate: {self.df['loan_approved'].mean()*100:.1f}%")
        return self

    def preprocess(self, test_size=0.2):
        X, y = self.preprocessor.fit_transform(self.df)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
        print(f"Train: {len(self.X_train)} | Test: {len(self.X_test)}")
        return self

    def train_all(self):
        print("\n--- Training Models ---")

        # Random Forest
        print("[1/4] Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                                     min_samples_split=5, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf

        # Gradient Boosting
        print("[2/4] Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                         learning_rate=0.1, random_state=RANDOM_STATE)
        gb.fit(self.X_train, self.y_train)
        self.models['gradient_boosting'] = gb

        # Logistic Regression
        print("[3/4] Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = lr

        # Neural Network
        print("[4/4] Neural Network (TensorFlow)...")
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=keras.optimizers.Adam(0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy', keras.metrics.AUC(name='auc')])
        early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                               restore_best_weights=True)
        hist = model.fit(self.X_train, self.y_train, validation_split=0.2,
                         epochs=100, batch_size=256, callbacks=[early], verbose=0)
        self.models['neural_network'] = model
        self.history = hist.history
        print(f"      Trained for {len(hist.history['loss'])} epochs")

        return self

    def evaluate(self):
        print("\n--- Evaluation Results ---")
        for name, model in self.models.items():
            if name == 'neural_network':
                prob = model.predict(self.X_test, verbose=0).flatten()
                pred = (prob > 0.5).astype(int)
            else:
                pred = model.predict(self.X_test)
                prob = model.predict_proba(self.X_test)[:, 1]

            self.results[name] = {
                'accuracy': accuracy_score(self.y_test, pred),
                'precision': precision_score(self.y_test, pred),
                'recall': recall_score(self.y_test, pred),
                'f1': f1_score(self.y_test, pred),
                'auc': roc_auc_score(self.y_test, prob),
                'predictions': pred.tolist(),
                'probabilities': prob.tolist()
            }

            print(f"\n{name.replace('_', ' ').title()}:")
            print(f"  Accuracy:  {self.results[name]['accuracy']:.4f}")
            print(f"  Precision: {self.results[name]['precision']:.4f}")
            print(f"  Recall:    {self.results[name]['recall']:.4f}")
            print(f"  F1-Score:  {self.results[name]['f1']:.4f}")
            print(f"  AUC-ROC:   {self.results[name]['auc']:.4f}")

        best = max(self.results, key=lambda k: self.results[k]['accuracy'])
        print(f"\nBest Model: {best.replace('_', ' ').title()} "
              f"({self.results[best]['accuracy']*100:.1f}% accuracy)")
        return self

    def save_models(self):
        for name, model in self.models.items():
            if name == 'neural_network':
                model.save(self.models_dir / f'{name}.keras')
            else:
                with open(self.models_dir / f'{name}.pkl', 'wb') as f:
                    pickle.dump(model, f)
        self.preprocessor.save(self.models_dir / 'preprocessor.pkl')
        with open(self.models_dir / 'results.json', 'w') as f:
            json.dump({k: {sk: sv for sk, sv in v.items() if sk not in ['predictions', 'probabilities']}
                       for k, v in self.results.items()}, f, indent=2)
        print(f"\nModels saved to {self.models_dir}/")
        return self

    def load_models(self):
        for name in ['random_forest', 'gradient_boosting', 'logistic_regression']:
            path = self.models_dir / f'{name}.pkl'
            if path.exists():
                with open(path, 'rb') as f:
                    self.models[name] = pickle.load(f)
        nn_path = self.models_dir / 'neural_network.keras'
        if nn_path.exists():
            self.models['neural_network'] = keras.models.load_model(nn_path)
        self.preprocessor = Preprocessor.load(self.models_dir / 'preprocessor.pkl')
        print("Models loaded successfully")
        return self

    def predict(self, input_dict):
        df = pd.DataFrame([input_dict])
        X = self.preprocessor.transform(df)
        out = {}
        for name, model in self.models.items():
            if name == 'neural_network':
                prob = float(model.predict(X, verbose=0)[0][0])
            else:
                prob = float(model.predict_proba(X)[0][1])
            out[name] = {
                'approved': prob > 0.5,
                'probability': round(prob, 4),
                'confidence': round(max(prob, 1 - prob), 4)
            }
        return out
