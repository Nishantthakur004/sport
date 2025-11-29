import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

class ScorePredictor:
    def __init__(self):
        self.score_model = None
        self.win_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}
    
    def train(self, dataset, feature_columns, score_target, win_target):
        """Train both score prediction and win/loss classification models"""
        self.feature_columns = feature_columns
        
        # Prepare features and targets
        X = dataset[feature_columns]
        y_score = dataset[score_target]
        y_win = dataset[win_target]
        
        # Split the data
        X_train, X_test, y_score_train, y_score_test = train_test_split(
            X, y_score, test_size=0.2, random_state=42
        )
        
        _, _, y_win_train, y_win_test = train_test_split(
            X, y_win, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train score prediction model (Regression)
        self.score_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        self.score_model.fit(X_train_scaled, y_score_train)
        
        # Train win/loss prediction model (Classification)
        self.win_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.win_model.fit(X_train_scaled, y_win_train)
        
        # Evaluate models
        score_predictions = self.score_model.predict(X_test_scaled)
        win_predictions = self.win_model.predict(X_test_scaled)
        
        # Calculate metrics
        self.metrics = {
            'score_prediction': {
                'mae': float(mean_absolute_error(y_score_test, score_predictions)),
                'mse': float(mean_squared_error(y_score_test, score_predictions)),
                'rmse': float(np.sqrt(mean_squared_error(y_score_test, score_predictions))),
                'r2_score': float(r2_score(y_score_test, score_predictions)),
                'mean_actual_score': float(y_score_test.mean()),
                'mean_predicted_score': float(score_predictions.mean())
            },
            'win_prediction': {
                'accuracy': float(accuracy_score(y_win_test, win_predictions)),
                'precision_0': float(classification_report(y_win_test, win_predictions, output_dict=True)['0']['precision']),
                'recall_0': float(classification_report(y_win_test, win_predictions, output_dict=True)['0']['recall']),
                'precision_1': float(classification_report(y_win_test, win_predictions, output_dict=True)['1']['precision']),
                'recall_1': float(classification_report(y_win_test, win_predictions, output_dict=True)['1']['recall'])
            }
        }
        
        return self.metrics
    
    def predict(self, features):
        """Make predictions for new data"""
        if self.score_model is None or self.win_model is None:
            raise Exception("Models not trained yet")
        
        # Convert features to array and scale
        features_array = np.array(features).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        
        # Make predictions
        predicted_score = self.score_model.predict(features_scaled)[0]
        predicted_win_prob = self.win_model.predict_proba(features_scaled)[0]
        predicted_win = self.win_model.predict(features_scaled)[0]
        
        return {
            'predicted_total_score': float(predicted_score),
            'win_probability_team1': float(predicted_win_prob[1]),
            'win_probability_team2': float(predicted_win_prob[0]),
            'predicted_winner': 'Team 1' if predicted_win == 1 else 'Team 2',
            'confidence_score': float(max(predicted_win_prob))
        }
    
    def get_metrics(self):
        """Get model performance metrics"""
        return self.metrics
    
    def save_models(self, filepath):
        """Save trained models"""
        joblib.dump({
            'score_model': self.score_model,
            'win_model': self.win_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filepath)
    
    def load_models(self, filepath):
        """Load trained models"""
        models = joblib.load(filepath)
        self.score_model = models['score_model']
        self.win_model = models['win_model']
        self.scaler = models['scaler']
        self.feature_columns = models['feature_columns']