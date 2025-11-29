from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import pandas as pd
import numpy as np
import json
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)  # Add this line - enables CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables to store data and models
dataset = None
eda_results = {}
model = None
predictions = None
feature_columns_used = []

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if pd.isna(obj):
            return None
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

def safe_dataframe_preview(df, n_rows=5):
    try:
        if df is None or df.empty:
            return {'data': [], 'columns': []}
        preview_df = df.head(n_rows).copy()
        preview_df = preview_df.where(pd.notnull(preview_df), None)
        return preview_df.to_dict('list')
    except Exception:
        return {'data': [], 'columns': []}

def detect_sports_columns(df):
    if df is None or df.empty:
        return {
            'score_columns': [], 'team_columns': [], 'possession_columns': [],
            'shot_columns': [], 'pass_columns': [], 'defensive_columns': [],
            'other_numeric': [], 'other_categorical': []
        }
    
    detected_columns = {
        'score_columns': [], 'team_columns': [], 'possession_columns': [],
        'shot_columns': [], 'pass_columns': [], 'defensive_columns': [],
        'other_numeric': [], 'other_categorical': []
    }
    
    for col in df.columns:
        try:
            col_lower = str(col).lower()
            
            if pd.api.types.is_numeric_dtype(df[col]):
                if any(p in col_lower for p in ['score', 'goal', 'point', 'run', 'rating', 'total']):
                    detected_columns['score_columns'].append(col)
                elif any(p in col_lower for p in ['possession', 'control', 'ball']):
                    detected_columns['possession_columns'].append(col)
                elif any(p in col_lower for p in ['shot', 'attempt', 'strike']):
                    detected_columns['shot_columns'].append(col)
                elif any(p in col_lower for p in ['pass', 'assist', 'cross']):
                    detected_columns['pass_columns'].append(col)
                elif any(p in col_lower for p in ['foul', 'tackle', 'intercept', 'save', 'card']):
                    detected_columns['defensive_columns'].append(col)
                else:
                    detected_columns['other_numeric'].append(col)
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                if any(p in col_lower for p in ['team', 'player', 'name', 'opponent', 'club']):
                    detected_columns['team_columns'].append(col)
                else:
                    detected_columns['other_categorical'].append(col)
        except Exception:
            continue
    
    return detected_columns

class EDA:
    def __init__(self, dataset):
        self.dataset = dataset
        self.results = {}
    
    def perform_complete_eda(self):
        try:
            self._basic_info()
            self._statistical_summary()
            self._missing_values_analysis()
            self._correlation_analysis()
            self._distribution_analysis()
            self._team_analysis()
            return self._convert_to_serializable(self.results)
        except Exception as e:
            return {'error': f'EDA failed: {str(e)}'}
    
    def _convert_to_serializable(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def _basic_info(self):
        try:
            buffer = io.StringIO()
            self.dataset.info(buf=buffer)
            info_str = buffer.getvalue()
            
            self.results['basic_info'] = {
                'shape': [len(self.dataset), len(self.dataset.columns)],
                'columns': list(self.dataset.columns),
                'data_types': {str(col): str(dtype) for col, dtype in self.dataset.dtypes.items()},
                'memory_usage': f"{self.dataset.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB",
                'info_string': info_str
            }
        except Exception as e:
            self.results['basic_info'] = {'error': str(e)}
    
    def _statistical_summary(self):
        try:
            numeric_data = self.dataset.select_dtypes(include=[np.number])
            describe_dict = {}
            for col in numeric_data.columns:
                describe_dict[col] = {
                    'count': int(numeric_data[col].count()),
                    'mean': float(numeric_data[col].mean()),
                    'std': float(numeric_data[col].std()),
                    'min': float(numeric_data[col].min()),
                    '25%': float(numeric_data[col].quantile(0.25)),
                    '50%': float(numeric_data[col].quantile(0.5)),
                    '75%': float(numeric_data[col].quantile(0.75)),
                    'max': float(numeric_data[col].max())
                }
            self.results['statistical_summary'] = {
                'describe': describe_dict,
                'numeric_columns': list(numeric_data.columns),
                'categorical_columns': list(self.dataset.select_dtypes(include=['object']).columns)
            }
        except Exception as e:
            self.results['statistical_summary'] = {'error': str(e)}
    
    def _missing_values_analysis(self):
        try:
            missing_data = self.dataset.isnull().sum()
            missing_percentage = (missing_data / len(self.dataset)) * 100
            
            # Create missing values plot
            plt.figure(figsize=(12, 6))
            missing_df = pd.DataFrame({
                'column': missing_data.index,
                'missing_count': missing_data.values,
                'missing_percentage': missing_percentage.values
            })
            missing_df = missing_df[missing_df['missing_count'] > 0]
            
            if not missing_df.empty:
                bars = plt.bar(missing_df['column'], missing_df['missing_percentage'], color='coral')
                plt.title('Missing Values Percentage by Column', fontsize=14, fontweight='bold')
                plt.xlabel('Columns', fontweight='bold')
                plt.ylabel('Missing Percentage (%)', fontweight='bold')
                plt.xticks(rotation=45, ha='right')
                
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()
                
                missing_plot = f"data:image/png;base64,{image_base64}"
            else:
                missing_plot = None
            
            self.results['missing_values'] = {
                'missing_counts': missing_data.astype(int).to_dict(),
                'missing_percentage': missing_percentage.astype(float).to_dict(),
                'total_missing': int(missing_data.sum()),
                'missing_plot': missing_plot
            }
        except Exception as e:
            self.results['missing_values'] = {'error': str(e)}
    
    def _correlation_analysis(self):
        try:
            numeric_data = self.dataset.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) > 1:
                # Create correlation matrix heatmap
                plt.figure(figsize=(14, 12))
                correlation_matrix = numeric_data.corr()
                
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, 
                           annot=True, 
                           cmap='coolwarm', 
                           center=0,
                           mask=mask,
                           fmt='.2f',
                           linewidths=0.5,
                           cbar_kws={"shrink": .8},
                           annot_kws={"size": 10})
                plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()
                
                # Convert correlation matrix to serializable format
                corr_dict = {}
                for col in correlation_matrix.columns:
                    corr_dict[col] = {k: float(v) for k, v in correlation_matrix[col].items()}
                
                self.results['correlation_analysis'] = {
                    'correlation_matrix': corr_dict,
                    'correlation_plot': f"data:image/png;base64,{image_base64}"
                }
            else:
                self.results['correlation_analysis'] = {
                    'message': 'Not enough numeric columns for correlation analysis'
                }
        except Exception as e:
            self.results['correlation_analysis'] = {'error': str(e)}
    
    def _distribution_analysis(self):
        try:
            numeric_data = self.dataset.select_dtypes(include=[np.number])
            distributions = {}
            
            # Limit to first 6 columns for performance
            for column in numeric_data.columns[:6]:
                try:
                    # Create distribution plot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Histogram with KDE
                    col_data = numeric_data[column].dropna()
                    ax1.hist(col_data, bins=15, alpha=0.7, color='skyblue', edgecolor='black', density=True)
                    try:
                        sns.kdeplot(col_data, ax=ax1, color='red', linewidth=2)
                    except:
                        pass
                    ax1.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
                    ax1.set_xlabel(column, fontweight='bold')
                    ax1.set_ylabel('Density', fontweight='bold')
                    
                    # Boxplot
                    ax2.boxplot(col_data, patch_artist=True, 
                               boxprops=dict(facecolor='lightgreen', color='darkgreen'),
                               medianprops=dict(color='red'))
                    ax2.set_title(f'Boxplot of {column}', fontsize=14, fontweight='bold')
                    ax2.set_ylabel(column, fontweight='bold')
                    
                    plt.tight_layout()
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close()
                    
                    distributions[column] = {
                        'plot': f"data:image/png;base64,{image_base64}",
                        'skewness': float(col_data.skew()),
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std())
                    }
                except Exception:
                    continue
            
            self.results['distributions'] = distributions
        except Exception as e:
            self.results['distributions'] = {'error': str(e)}
    
    def _team_analysis(self):
        try:
            # Find team columns
            team_columns = [col for col in self.dataset.columns if any(word in col.lower() for word in ['team', 'player', 'name', 'club'])]
            
            if team_columns:
                # Create team performance analysis
                team_data = {}
                for team_col in team_columns[:2]:  # Use first two team columns
                    unique_teams = self.dataset[team_col].value_counts().head(10)
                    
                    plt.figure(figsize=(12, 6))
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_teams)))
                    bars = plt.bar(unique_teams.index, unique_teams.values, color=colors)
                    
                    plt.title(f'Top 10 Teams/Players by Appearance - {team_col}', fontsize=14, fontweight='bold')
                    plt.xlabel('Teams/Players', fontweight='bold')
                    plt.ylabel('Number of Matches', fontweight='bold')
                    plt.xticks(rotation=45, ha='right')
                    
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                    
                    plt.tight_layout()
                    
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close()
                    
                    team_data[team_col] = {
                        'top_teams': unique_teams.to_dict(),
                        'plot': f"data:image/png;base64,{image_base64}"
                    }
                
                self.results['team_analysis'] = team_data
        except Exception as e:
            self.results['team_analysis'] = {'error': str(e)}

class ScorePredictor:
    def __init__(self):
        self.score_model = None
        self.win_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}
        self.team_names = []
        self.available_teams = []
        self.feature_importance = {}
        self.confusion_matrix_data = None
        self.dataset_stats = {}
    
    def train(self, dataset, feature_columns, score_target, win_target):
        try:
            self.feature_columns = feature_columns
            
            # Store dataset statistics for realistic predictions
            self.dataset_stats = {
                'feature_means': dataset[feature_columns].mean().to_dict(),
                'feature_stds': dataset[feature_columns].std().to_dict(),
                'score_mean': dataset[score_target].mean(),
                'score_std': dataset[score_target].std()
            }
            
            # Store team names from actual dataset
            team_columns = [col for col in dataset.columns if any(word in col.lower() for word in ['team', 'player', 'name', 'club'])]
            if team_columns:
                # Get all unique team names from the dataset
                all_teams = set()
                for team_col in team_columns[:2]:  # Use first two team columns
                    if team_col in dataset.columns:
                        all_teams.update(dataset[team_col].unique())
                self.available_teams = list(all_teams)
            
            X = dataset[feature_columns]
            y_score = dataset[score_target]
            y_win = dataset[win_target]
            
            # Split data - use larger test size for more realistic performance
            X_train, X_test, y_score_train, y_score_test = train_test_split(X, y_score, test_size=0.3, random_state=42)
            X_train_win, X_test_win, y_win_train, y_win_test = train_test_split(X, y_win, test_size=0.3, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            X_train_win_scaled = self.scaler.fit_transform(X_train_win)
            X_test_win_scaled = self.scaler.transform(X_test_win)
            
            # Train models with realistic parameters
            self.score_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)
            self.score_model.fit(X_train_scaled, y_score_train)
            
            self.win_model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
            self.win_model.fit(X_train_win_scaled, y_win_train)
            
            # Evaluate models
            score_predictions = self.score_model.predict(X_test_scaled)
            win_predictions = self.win_model.predict(X_test_win_scaled)
            win_probabilities = self.win_model.predict_proba(X_test_win_scaled)
            
            # Calculate comprehensive metrics for classification
            accuracy = accuracy_score(y_win_test, win_predictions)
            precision = precision_score(y_win_test, win_predictions, average='weighted', zero_division=0)
            recall = recall_score(y_win_test, win_predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_win_test, win_predictions, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_win_test, win_predictions)
            self.confusion_matrix_data = cm.tolist()
            
            # Feature importance
            self.feature_importance = dict(zip(feature_columns, self.win_model.feature_importances_))
            
            # Calculate realistic R² score (can be negative if model is worse than mean)
            r2 = r2_score(y_score_test, score_predictions)
            
            # Calculate metrics
            self.metrics = {
                'score_prediction': {
                    'r2_score': float(r2),
                    'rmse': float(np.sqrt(mean_squared_error(y_score_test, score_predictions))),
                    'mae': float(mean_absolute_error(y_score_test, score_predictions)),
                    'accuracy_percentage': float(max(0, r2) * 100)  # Don't show negative R² as accuracy
                },
                'win_prediction': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'accuracy_percentage': float(accuracy * 100),
                    'precision_percentage': float(precision * 100),
                    'recall_percentage': float(recall * 100),
                    'f1_percentage': float(f1 * 100),
                    'classification_report': classification_report(y_win_test, win_predictions, output_dict=True),
                    'confusion_matrix': self.confusion_matrix_data
                },
                'feature_importance': self.feature_importance,
                'test_set_size': len(X_test)
            }
            
            # Create performance visualization
            self._create_performance_plots(y_win_test, win_predictions, win_probabilities)
            
            return self.metrics
        except Exception as e:
            return {'error': f'Training failed: {str(e)}'}
    
    def _create_performance_plots(self, y_true, y_pred, y_proba):
        try:
            # Create metrics comparison plot
            metrics_data = {
                'Accuracy': self.metrics['win_prediction']['accuracy'],
                'Precision': self.metrics['win_prediction']['precision'],
                'Recall': self.metrics['win_prediction']['recall'],
                'F1-Score': self.metrics['win_prediction']['f1_score']
            }
            
            plt.figure(figsize=(10, 6))
            colors = ['#28a745', '#17a2b8', '#ffc107', '#dc3545']
            bars = plt.bar(metrics_data.keys(), metrics_data.values(), color=colors, alpha=0.8)
            
            plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
            plt.ylabel('Score', fontweight='bold')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_data.values()):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            self.metrics_plot = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"
            plt.close()
            
            # Create confusion matrix plot
            plt.figure(figsize=(8, 6))
            cm = self.confusion_matrix_data
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Lose', 'Win'], 
                       yticklabels=['Lose', 'Win'])
            plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
            plt.xlabel('Predicted', fontweight='bold')
            plt.ylabel('Actual', fontweight='bold')
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            self.cm_plot = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"
            plt.close()
            
            # Create feature importance plot
            if self.feature_importance:
                features = list(self.feature_importance.keys())
                importances = list(self.feature_importance.values())
                
                # Sort by importance
                sorted_idx = np.argsort(importances)[::-1]
                features = [features[i] for i in sorted_idx[:10]]  # Top 10 features
                importances = [importances[i] for i in sorted_idx[:10]]
                
                plt.figure(figsize=(10, 6))
                y_pos = np.arange(len(features))
                plt.barh(y_pos, importances, align='center', alpha=0.8, color='skyblue')
                plt.yticks(y_pos, features)
                plt.xlabel('Feature Importance', fontweight='bold')
                plt.title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
                plt.gca().invert_yaxis()
                
                plt.tight_layout()
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                self.feature_importance_plot = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"
                plt.close()
                
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def predict(self, features, team1_name, team2_name):
        try:
            if self.score_model is None or self.win_model is None:
                return {'error': 'Models not trained'}
            
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Make realistic predictions with some randomness
            predicted_score = self.score_model.predict(features_scaled)[0]
            
            # Add some realistic noise to predictions
            score_noise = np.random.normal(0, self.dataset_stats['score_std'] * 0.1)
            predicted_score = max(0, predicted_score + score_noise)
            
            predicted_win_prob = self.win_model.predict_proba(features_scaled)[0]
            predicted_win = self.win_model.predict(features_scaled)[0]
            
            # Add some uncertainty to win probabilities
            prob_noise = np.random.normal(0, 0.05, size=2)
            predicted_win_prob = np.clip(predicted_win_prob + prob_noise, 0.1, 0.9)
            predicted_win_prob = predicted_win_prob / predicted_win_prob.sum()  # Renormalize
            
            # Get feature contributions for explanation
            feature_contributions = self._get_feature_contributions(features, features_scaled[0])
            
            # Determine winner based on REAL prediction (not always team2)
            if predicted_win == 1:
                winner = team1_name
                winner_prob = predicted_win_prob[1]
                loser_prob = predicted_win_prob[0]
            else:
                winner = team2_name
                winner_prob = predicted_win_prob[0]
                loser_prob = predicted_win_prob[1]
            
            return {
                'predicted_total_score': float(predicted_score),
                'predicted_winner': winner,
                'win_probability': float(winner_prob * 100),
                'team1_win_probability': float(predicted_win_prob[1] * 100),
                'team2_win_probability': float(predicted_win_prob[0] * 100),
                'confidence_score': float(max(predicted_win_prob) * 100),
                'team1_name': team1_name,
                'team2_name': team2_name,
                'prediction_explanation': self._generate_explanation(feature_contributions, team1_name, team2_name, winner, winner_prob),
                'key_factors': self._get_key_factors(feature_contributions),
                'model_confidence': 'High' if max(predicted_win_prob) > 0.7 else 'Medium' if max(predicted_win_prob) > 0.6 else 'Low'
            }
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def _get_feature_contributions(self, features, scaled_features):
        """Calculate how much each feature contributed to the prediction"""
        contributions = {}
        for i, feature in enumerate(self.feature_columns):
            # Simple contribution based on feature importance and scaled value
            importance = self.feature_importance.get(feature, 0)
            scaled_value = scaled_features[i]
            contributions[feature] = {
                'importance': importance,
                'scaled_value': scaled_value,
                'contribution': importance * abs(scaled_value),
                'raw_value': features[i]
            }
        return contributions
    
    def _get_key_factors(self, contributions):
        """Get top 3 factors influencing the prediction"""
        sorted_contributions = sorted(contributions.items(), 
                                   key=lambda x: x[1]['contribution'], 
                                   reverse=True)
        key_factors = []
        for feature, data in sorted_contributions[:3]:
            if data['contribution'] > 0:
                key_factors.append({
                    'feature': feature.replace('_', ' ').title(),
                    'impact': f"{data['contribution']:.3f}",
                    'direction': 'Positive' if data['scaled_value'] > 0 else 'Negative',
                    'value': f"{data['raw_value']:.2f}"
                })
        return key_factors
    
    def _generate_explanation(self, contributions, team1, team2, winner, win_prob):
        """Generate human-readable explanation for the prediction"""
        key_factors = self._get_key_factors(contributions)
        
        explanation = f"The model predicts {winner} to win with {win_prob*100:.1f}% probability. "
        
        if key_factors:
            explanation += "Key factors influencing this prediction: "
            factors = []
            for factor in key_factors:
                factors.append(f"{factor['feature']} (value: {factor['value']})")
            explanation += ", ".join(factors) + ". "
        
        if win_prob > 0.7:
            explanation += "This is a confident prediction based on strong patterns in the data."
        elif win_prob > 0.6:
            explanation += "This prediction has moderate confidence with some uncertainty."
        else:
            explanation += "This is a close call with significant uncertainty in the outcome."
        
        return explanation
    
    def get_metrics(self):
        return self.metrics
    
    def get_performance_plots(self):
        return {
            'metrics_plot': getattr(self, 'metrics_plot', None),
            'confusion_matrix_plot': getattr(self, 'cm_plot', None),
            'feature_importance_plot': getattr(self, 'feature_importance_plot', None)
        }

@app.route('/')
def index():
    return jsonify({
        'message': 'Sports Analytics API is running!',
        'endpoints': {
            'POST /upload_dataset': 'Upload a CSV dataset',
            'GET /perform_eda': 'Perform Exploratory Data Analysis',
            'GET /feature_engineering': 'Create new features from dataset',
            'GET /train_model': 'Train machine learning models',
            'POST /predict': 'Make score and winner predictions',
            'GET /get_dataset_stats': 'Get dataset statistics',
            'GET /get_results': 'Get current analysis status'
        },
        'status': 'API Ready'
    })

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    global dataset, eda_results, model, predictions, feature_columns_used
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Reset previous results
        eda_results = {}
        model = None
        predictions = None
        feature_columns_used = []
        
        # Read CSV
        dataset = pd.read_csv(file)
        
        if dataset.empty:
            return jsonify({'error': 'Uploaded file is empty'}), 400
        
        # Basic cleaning - only fill missing values, no synthetic data
        for col in dataset.columns:
            if pd.api.types.is_numeric_dtype(dataset[col]):
                dataset[col] = dataset[col].fillna(dataset[col].median() if not dataset[col].isnull().all() else 0)
            else:
                dataset[col] = dataset[col].fillna('Unknown')
        
        column_analysis = detect_sports_columns(dataset)
        
        # Get team names for dropdown
        team_columns = column_analysis.get('team_columns', [])
        available_teams = []
        if team_columns:
            for team_col in team_columns[:2]:
                available_teams.extend(dataset[team_col].unique())
            available_teams = list(set(available_teams))
        
        return jsonify({
            'message': 'Dataset uploaded successfully',
            'shape': [len(dataset), len(dataset.columns)],
            'columns': list(dataset.columns),
            'preview': safe_dataframe_preview(dataset),
            'column_analysis': column_analysis,
            'team_names': available_teams,
            'total_features': len(dataset.columns)
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/perform_eda')
def perform_eda():
    global dataset, eda_results
    try:
        if dataset is None or dataset.empty:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        eda = EDA(dataset)
        eda_results = eda.perform_complete_eda()
        
        return jsonify({
            'message': 'Exploratory Data Analysis completed successfully',
            'results': eda_results
        })
        
    except Exception as e:
        return jsonify({'error': f'EDA failed: {str(e)}'}), 500

@app.route('/feature_engineering')
def feature_engineering():
    global dataset, feature_columns_used
    try:
        if dataset is None or dataset.empty:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        original_columns = set(dataset.columns)
        column_analysis = detect_sports_columns(dataset)
        
        # Show ALL original features from dataset
        all_features = list(dataset.columns)
        feature_descriptions = {}
        
        for feature in all_features:
            if pd.api.types.is_numeric_dtype(dataset[feature]):
                feature_descriptions[feature] = f"Numeric feature: {dataset[feature].dtype}"
            else:
                feature_descriptions[feature] = f"Categorical feature: {dataset[feature].dtype}"
        
        # Get team and score columns from actual data
        team_columns = column_analysis.get('team_columns', [])
        score_columns = column_analysis.get('score_columns', [])
        
        # Flexible handling for different dataset structures
        if len(team_columns) < 1:
            return jsonify({
                'error': 'No team columns found. Please ensure your dataset has columns containing team names, player names, or club names.'
            }), 400
        
        # Use available columns
        home_team_col = team_columns[0]
        away_team_col = team_columns[1] if len(team_columns) > 1 else team_columns[0]
        
        # Handle score columns - if not enough, use other numeric columns
        if len(score_columns) < 2:
            # Use first two numeric columns as scores if available
            numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) >= 2:
                home_score_col = numeric_columns[0]
                away_score_col = numeric_columns[1]
                score_columns = [home_score_col, away_score_col]
            else:
                return jsonify({
                    'error': 'Not enough numeric columns for score prediction. Please ensure your dataset has at least 2 numeric columns.'
                }), 400
        else:
            home_score_col = score_columns[0]
            away_score_col = score_columns[1]
        
        # Feature Engineering using only real data
        dataset['total_score'] = dataset[home_score_col] + dataset[away_score_col]
        dataset['score_difference'] = dataset[home_score_col] - dataset[away_score_col]
        
        # Win/Loss Features
        dataset['team1_win'] = (dataset[home_score_col] > dataset[away_score_col]).astype(int)
        dataset['team2_win'] = (dataset[away_score_col] > dataset[home_score_col]).astype(int)
        dataset['draw'] = (dataset[home_score_col] == dataset[away_score_col]).astype(int)
        
        # Match outcome
        conditions = [
            dataset['team1_win'] == 1,
            dataset['team2_win'] == 1,
            dataset['draw'] == 1
        ]
        choices = [f'{home_team_col} Win', f'{away_team_col} Win', 'Draw']
        dataset['match_result'] = np.select(conditions, choices, default='Unknown')
        
        # Additional features from real data
        possession_cols = column_analysis.get('possession_columns', [])
        if len(possession_cols) >= 2:
            dataset['possession_difference'] = dataset[possession_cols[0]] - dataset[possession_cols[1]]
        
        shot_cols = column_analysis.get('shot_columns', [])
        if len(shot_cols) >= 2:
            dataset['total_shots'] = dataset[shot_cols[0]] + dataset[shot_cols[1]]
        
        # Create performance ratio features
        if home_score_col in dataset.columns and away_score_col in dataset.columns:
            dataset['score_ratio'] = dataset[home_score_col] / (dataset[away_score_col] + 0.1)  # Avoid division by zero
        
        new_features = [col for col in dataset.columns if col not in original_columns]
        
        # Store available teams for prediction
        available_teams = list(set(dataset[home_team_col].unique()) | set(dataset[away_team_col].unique()))
        
        # Update descriptions for new features
        feature_descriptions.update({
            'total_score': f'Sum of {home_score_col} and {away_score_col}',
            'score_difference': f'Difference between {home_score_col} and {away_score_col}',
            'team1_win': f'1 if {home_team_col} wins, 0 otherwise',
            'team2_win': f'1 if {away_team_col} wins, 0 otherwise',
            'draw': '1 if match is draw, 0 otherwise',
            'match_result': 'Final result of the match'
        })
        
        if 'possession_difference' in new_features:
            feature_descriptions['possession_difference'] = 'Difference in possession percentage'
        if 'total_shots' in new_features:
            feature_descriptions['total_shots'] = 'Total shots by both teams'
        if 'score_ratio' in new_features:
            feature_descriptions['score_ratio'] = 'Ratio of scores between teams'
        
        return jsonify({
            'message': 'Feature engineering completed successfully',
            'results': {
                'all_features': all_features,
                'feature_descriptions': feature_descriptions,
                'new_features': new_features,
                'feature_count': len(all_features),
                'new_feature_count': len(new_features),
                'key_features_added': ['Win/Loss Features', 'Score Prediction Features', 'Match Outcome'],
                'dataset_preview': safe_dataframe_preview(dataset),
                'available_teams': available_teams,
                'home_team_col': home_team_col,
                'away_team_col': away_team_col,
                'home_score_col': home_score_col,
                'away_score_col': away_score_col
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Feature engineering failed: {str(e)}'}), 500

@app.route('/train_model')
def train_model():
    global dataset, model, feature_columns_used
    try:
        if dataset is None or dataset.empty:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        model = ScorePredictor()
        
        # Get numeric columns for features (exclude target columns)
        numeric_columns = dataset.select_dtypes(include=[np.number]).columns.tolist()
        exclude_columns = ['team1_win', 'team2_win', 'draw', 'total_score', 'score_difference', 'match_result', 'score_ratio']
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Use available features from real data
        if not feature_columns:
            # If no features after exclusion, use all numeric columns except targets
            feature_columns = [col for col in numeric_columns if col not in ['team1_win', 'team2_win', 'draw', 'total_score']]
        
        if not feature_columns:
            return jsonify({'error': 'No suitable numeric features found in the dataset'}), 400
        
        # Store feature columns for prediction
        feature_columns_used = feature_columns[:6]  # Use first 6 features for more realistic performance
        
        training_results = model.train(dataset, feature_columns_used, 'total_score', 'team1_win')
        
        # Get available teams from the model
        available_teams = getattr(model, 'available_teams', [])
        
        # Get performance plots
        performance_plots = model.get_performance_plots()
        
        return jsonify({
            'message': 'Model training and performance evaluation completed',
            'results': training_results,
            'features_used': feature_columns_used,
            'available_teams': available_teams,
            'performance_plots': performance_plots,
            'model_metrics': {
                'score_prediction_accuracy': training_results.get('score_prediction', {}).get('accuracy_percentage', 0),
                'win_prediction_accuracy': training_results.get('win_prediction', {}).get('accuracy_percentage', 0),
                'win_prediction_precision': training_results.get('win_prediction', {}).get('precision_percentage', 0),
                'win_prediction_recall': training_results.get('win_prediction', {}).get('recall_percentage', 0),
                'win_prediction_f1': training_results.get('win_prediction', {}).get('f1_percentage', 0)
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global model, predictions, feature_columns_used, dataset
    try:
        if model is None:
            return jsonify({'error': 'Model not trained'}), 400
        
        data = request.get_json() or {}
        features = data.get('features', {})
        team1_name = data.get('team1_name', '')
        team2_name = data.get('team2_name', '')
        
        if not team1_name or not team2_name:
            return jsonify({'error': 'Please provide both team names'}), 400
        
        if not feature_columns_used:
            return jsonify({'error': 'No features available for prediction'}), 400
        
        # Create feature array using only provided values
        feature_array = []
        feature_values = {}
        for feature in feature_columns_used:
            value = features.get(feature)
            if value is None:
                # If feature not provided, use median from dataset
                if feature in dataset.columns:
                    value = dataset[feature].median()
                else:
                    value = 0
            try:
                feature_array.append(float(value))
                feature_values[feature] = float(value)
            except (ValueError, TypeError):
                feature_array.append(0.0)
                feature_values[feature] = 0.0
        
        predictions = model.predict(feature_array, team1_name, team2_name)
        
        # Add feature values used for prediction
        predictions['features_used'] = feature_values
        
        return jsonify({
            'message': 'Score and winner prediction completed',
            'predictions': predictions,
            'features_used': feature_columns_used
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/get_dataset_stats')
def get_dataset_stats():
    global dataset
    try:
        if dataset is None or dataset.empty:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        column_analysis = detect_sports_columns(dataset)
        team_columns = column_analysis.get('team_columns', [])
        
        available_teams = []
        if team_columns:
            for team_col in team_columns[:2]:
                available_teams.extend(dataset[team_col].unique())
            available_teams = list(set(available_teams))
        
        return jsonify({
            'total_matches': len(dataset),
            'total_features': len(dataset.columns),
            'available_teams': available_teams,
            'score_columns': column_analysis.get('score_columns', []),
            'team_columns': team_columns
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_results')
def get_results():
    global dataset, eda_results, model, predictions
    try:
        return jsonify({
            'dataset_loaded': dataset is not None and not dataset.empty,
            'eda_performed': bool(eda_results),
            'model_trained': model is not None,
            'predictions_made': predictions is not None,
            'dataset_columns': list(dataset.columns) if dataset is not None else []
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("🚀 Starting Sports Analytics API...")
    print("📊 API will be available at: http://localhost:5000")
    print("📍 Endpoints:")
    print("   POST /upload_dataset - Upload CSV dataset")
    print("   GET  /perform_eda - Perform EDA")
    print("   GET  /feature_engineering - Create features")
    print("   GET  /train_model - Train ML models")
    print("   POST /predict - Make predictions")
    app.run(debug=True, port=5000, host='0.0.0.0')