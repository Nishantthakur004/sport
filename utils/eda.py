import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import json

class EDA:
    def __init__(self, dataset):
        self.dataset = dataset
        self.results = {}
    
    def perform_complete_eda(self):
        """Perform comprehensive exploratory data analysis"""
        try:
            self._basic_info()
            self._statistical_summary()
            self._missing_values_analysis()
            self._correlation_analysis()
            self._distribution_analysis()
            self._outlier_analysis()
            
            # Convert results to JSON-serializable format
            return self._convert_to_serializable(self.results)
        except Exception as e:
            raise Exception(f"EDA Error: {str(e)}")
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.astype(object).where(pd.notnull(obj), None).tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def _basic_info(self):
        """Get basic information about the dataset"""
        try:
            buffer = BytesIO()
            
            # Get dataset info as string
            self.dataset.info(buf=buffer)
            info_str = buffer.getvalue().decode('utf-8')
            
            self.results['basic_info'] = {
                'shape': [int(self.dataset.shape[0]), int(self.dataset.shape[1])],
                'columns': list(self.dataset.columns),
                'data_types': {col: str(dtype) for col, dtype in self.dataset.dtypes.items()},
                'memory_usage': f"{self.dataset.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB",
                'info_string': info_str
            }
        except Exception as e:
            self.results['basic_info'] = {'error': f"Basic info error: {str(e)}"}
    
    def _statistical_summary(self):
        """Generate statistical summary"""
        try:
            numeric_data = self.dataset.select_dtypes(include=[np.number])
            
            if not numeric_data.empty:
                describe_stats = numeric_data.describe()
                # Convert to serializable format
                describe_dict = {}
                for col in describe_stats.columns:
                    describe_dict[col] = {
                        'count': int(describe_stats[col]['count']),
                        'mean': float(describe_stats[col]['mean']),
                        'std': float(describe_stats[col]['std']),
                        'min': float(describe_stats[col]['min']),
                        '25%': float(describe_stats[col]['25%']),
                        '50%': float(describe_stats[col]['50%']),
                        '75%': float(describe_stats[col]['75%']),
                        'max': float(describe_stats[col]['max'])
                    }
            else:
                describe_dict = {}
            
            self.results['statistical_summary'] = {
                'describe': describe_dict,
                'numeric_columns': numeric_data.columns.tolist(),
                'categorical_columns': self.dataset.select_dtypes(include=['object']).columns.tolist()
            }
        except Exception as e:
            self.results['statistical_summary'] = {'error': f"Statistical summary error: {str(e)}"}
    
    def _missing_values_analysis(self):
        """Analyze missing values"""
        try:
            missing_data = self.dataset.isnull().sum()
            missing_percentage = (missing_data / len(self.dataset)) * 100
            
            # Create missing values plot
            if missing_data.sum() > 0:
                plt.figure(figsize=(10, 6))
                missing_df = pd.DataFrame({
                    'column': missing_data.index,
                    'missing_count': missing_data.values,
                    'missing_percentage': missing_percentage.values
                })
                missing_df = missing_df[missing_df['missing_count'] > 0]
                
                if not missing_df.empty:
                    plt.figure(figsize=(12, 6))
                    bars = plt.bar(missing_df['column'], missing_df['missing_percentage'], color='coral')
                    plt.title('Missing Values Percentage by Column')
                    plt.xlabel('Columns')
                    plt.ylabel('Missing Percentage (%)')
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}%', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    
                    # Convert to base64
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close()
                    
                    missing_plot = f"data:image/png;base64,{image_base64}"
                else:
                    missing_plot = None
            else:
                missing_plot = None
            
            self.results['missing_values'] = {
                'missing_counts': missing_data.astype(int).to_dict(),
                'missing_percentage': missing_percentage.astype(float).to_dict(),
                'total_missing': int(missing_data.sum()),
                'total_missing_percentage': float(missing_percentage.sum()),
                'missing_plot': missing_plot
            }
        except Exception as e:
            self.results['missing_values'] = {'error': f"Missing values analysis error: {str(e)}"}
    
    def _correlation_analysis(self):
        """Perform correlation analysis"""
        try:
            numeric_data = self.dataset.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) > 1:
                # Create correlation matrix heatmap
                plt.figure(figsize=(12, 10))
                correlation_matrix = numeric_data.corr()
                
                # Create the heatmap
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, 
                           annot=True, 
                           cmap='coolwarm', 
                           center=0,
                           mask=mask,
                           fmt='.2f',
                           linewidths=0.5,
                           cbar_kws={"shrink": .8})
                plt.title('Correlation Matrix Heatmap', fontsize=16, pad=20)
                plt.tight_layout()
                
                # Convert plot to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()
                
                # Convert correlation matrix to serializable format
                corr_dict = {}
                for col in correlation_matrix.columns:
                    corr_dict[col] = {k: float(v) for k, v in correlation_matrix[col].items()}
                
                # Find highly correlated features
                highly_correlated = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            highly_correlated.append({
                                'feature1': correlation_matrix.columns[i],
                                'feature2': correlation_matrix.columns[j],
                                'correlation': float(corr_value),
                                'type': 'positive' if corr_value > 0 else 'negative'
                            })
                
                self.results['correlation_analysis'] = {
                    'correlation_matrix': corr_dict,
                    'correlation_plot': f"data:image/png;base64,{image_base64}",
                    'highly_correlated_features': highly_correlated
                }
            else:
                self.results['correlation_analysis'] = {
                    'message': 'Not enough numeric columns for correlation analysis'
                }
        except Exception as e:
            self.results['correlation_analysis'] = {'error': f"Correlation analysis error: {str(e)}"}
    
    def _distribution_analysis(self):
        """Analyze distribution of numeric features"""
        try:
            numeric_data = self.dataset.select_dtypes(include=[np.number])
            distributions = {}
            
            # Limit to first 8 columns for performance
            for column in numeric_data.columns[:8]:
                try:
                    # Create distribution plot
                    plt.figure(figsize=(12, 5))
                    
                    # Create subplots for histogram and boxplot
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Histogram with KDE
                    col_data = numeric_data[column].dropna()
                    ax1.hist(col_data, bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
                    
                    # Add KDE line
                    try:
                        sns.kdeplot(col_data, ax=ax1, color='red', linewidth=2)
                    except:
                        pass  # Skip KDE if it fails
                    
                    ax1.set_title(f'Distribution of {column}', fontsize=14)
                    ax1.set_xlabel(column)
                    ax1.set_ylabel('Density')
                    
                    # Boxplot
                    ax2.boxplot(col_data)
                    ax2.set_title(f'Boxplot of {column}', fontsize=14)
                    ax2.set_ylabel(column)
                    
                    plt.tight_layout()
                    
                    # Convert plot to base64
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close()
                    
                    # Calculate statistics
                    distributions[column] = {
                        'plot': f"data:image/png;base64,{image_base64}",
                        'skewness': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis()),
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max())
                    }
                except Exception as col_error:
                    distributions[column] = {'error': f"Error analyzing {column}: {str(col_error)}"}
            
            self.results['distributions'] = distributions
        except Exception as e:
            self.results['distributions'] = {'error': f"Distribution analysis error: {str(e)}"}
    
    def _outlier_analysis(self):
        """Detect outliers using IQR method"""
        try:
            numeric_data = self.dataset.select_dtypes(include=[np.number])
            outliers = {}
            
            for column in numeric_data.columns[:8]:  # Limit to first 8 columns
                try:
                    col_data = numeric_data[column].dropna()
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    column_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                    outliers[column] = {
                        'outlier_count': int(len(column_outliers)),
                        'outlier_percentage': float((len(column_outliers) / len(col_data)) * 100),
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'outlier_values': column_outliers.astype(float).tolist()[:10]  # Limit to first 10 outliers
                    }
                except Exception as col_error:
                    outliers[column] = {'error': f"Error detecting outliers in {column}: {str(col_error)}"}
            
            self.results['outliers'] = outliers
        except Exception as e:
            self.results['outliers'] = {'error': f"Outlier analysis error: {str(e)}"}