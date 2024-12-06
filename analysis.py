import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def detect_outliers(df, columns, method='iqr', threshold=1.5):
    """Detect outliers in specified columns."""
    outliers = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            column_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers[col] = {
                'outliers': column_outliers,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col]))
            column_outliers = df[z_scores > threshold]
            outliers[col] = {
                'outliers': column_outliers,
                'z_threshold': threshold
            }
    
    return outliers

def analyze_outliers(file_path):
    """Comprehensive outlier analysis for a CSV file."""
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Identify numerical columns automatically
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Detect outliers using IQR method
    print("=== Outliers Detected (IQR Method) ===")
    iqr_outliers = detect_outliers(df, numerical_columns, method='iqr')
    
    # Create a summary of outliers
    outlier_summary = {}
    for col, result in iqr_outliers.items():
        outlier_count = len(result['outliers'])
        outlier_summary[col] = {
            'count': outlier_count,
            'percentage': (outlier_count / len(df)) * 100
        }
        
        print(f"\n{col.upper()} Outliers:")
        print(f"Lower Bound: {result['lower_bound']:.2f}")
        print(f"Upper Bound: {result['upper_bound']:.2f}")
        print(f"Number of Outliers: {outlier_count}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(numerical_columns, 1):
        plt.subplot(1, len(numerical_columns), i)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Detailed outlier report
    print("\n=== Outlier Summary ===")
    for col, stats in outlier_summary.items():
        print(f"{col}: {stats['count']} outliers ({stats['percentage']:.2f}%)")
    
    # Comprehensive statistics
    print("\n=== Comprehensive Statistics ===")
    print(df.describe())
    
    return iqr_outliers

# Run the analysis
if __name__ == "__main__":
    analyze_outliers('insurance.csv')