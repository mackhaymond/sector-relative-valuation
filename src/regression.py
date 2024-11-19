import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from typing import Tuple, Dict, List

def load_and_prepare_data(file_path: str = 'industry_analysis.csv') -> pd.DataFrame:
    """
    Load the data and prepare for industry-specific regression
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {file_path}. Please run data.py first.")
    
    required_columns = ['Risk_Score', 'Growth_Score', 'Quality_Score', 'PE_ZScore', 'PE', 'Industry']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove rows with NaN values
    df = df.dropna(subset=required_columns)
    
    if len(df) == 0:
        raise ValueError("No valid data points after removing NaN values")
    
    return df

def perform_industry_regression(df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """
    Perform separate regression analysis for each industry
    
    Args:
        df (pd.DataFrame): Input dataframe with required columns
        
    Returns:
        Tuple[Dict, pd.DataFrame]: Dictionary of industry models and updated dataframe with predictions
    """
    # Dictionary to store industry-specific regression models and statistics
    industry_models = {}
    
    # Initialize prediction columns
    df['Predicted_PE'] = np.nan
    df['Predicted_PE_Z'] = np.nan
    df['Risk_Z'] = df['Risk_Score']  # Scores are already z-scores
    df['Growth_Z'] = df['Growth_Score']
    df['Quality_Z'] = df['Quality_Score']
    
    # Process each industry separately
    for industry in df['Industry'].unique():
        industry_df = df[df['Industry'] == industry].copy()
        
        if len(industry_df) < 5:  # Skip industries with too few samples
            continue
            
        # Prepare features for this industry
        X = industry_df[['Risk_Z', 'Growth_Z', 'Quality_Z']].values
        
        # Get PE Z-scores for this industry
        y = industry_df['PE_ZScore'].values
        
        # Skip if not enough valid samples
        if len(y) < 5:
            continue
        
        # Perform ridge regression for this industry
        alphas = np.logspace(-6, 6, 13)
        model = RidgeCV(alphas=alphas, cv=min(5, len(y)))
        model.fit(X, y)
        
        # Calculate predicted PE z-score
        predicted_pe_z = model.predict(X)
        industry_df['Predicted_PE_Z'] = predicted_pe_z
        
        # Convert predicted z-score back to PE
        predicted_pe = industry_df['PE'].mean() + predicted_pe_z * industry_df['PE'].std()
        industry_df['Predicted_PE'] = predicted_pe
        
        # Calculate R-squared for this industry
        r2 = r2_score(y, predicted_pe_z)
        
        # Create coefficients dictionary
        coefficients = {
            'Risk_Weight': model.coef_[0],
            'Growth_Weight': model.coef_[1],
            'Quality_Weight': model.coef_[2],
            'Intercept': model.intercept_
        }
        
        # Store industry statistics
        industry_stats = {
            'n_companies': len(industry_df),
            'pe_mean': industry_df['PE'].mean(),
            'pe_std': industry_df['PE'].std(),
            'r2': r2
        }
        
        # Update the main dataframe with predictions
        df.loc[industry_df.index, ['Predicted_PE', 'Predicted_PE_Z']] = \
            industry_df[['Predicted_PE', 'Predicted_PE_Z']]
        
        # Store the model and statistics for this industry
        industry_models[industry] = {
            'model': model,
            'coefficients': coefficients,
            'stats': industry_stats
        }
    
    return industry_models, df

def format_industry_equation(industry_model: Dict, industry: str) -> str:
    """
    Format regression equation for a specific industry
    
    Args:
        industry_model (Dict): Dictionary containing industry model information
        industry (str): Industry name
        
    Returns:
        str: Formatted equation string
    """
    coef = industry_model['coefficients']
    stats = industry_model['stats']
    
    # Format z-score equation
    equation = f"Industry: {industry} (R² = {stats['r2']:.3f}, n = {stats['n_companies']})\n"
    equation += f"PE_ZScore = {coef['Intercept']:.3f}"
    
    features = ['Risk_Weight', 'Growth_Weight', 'Quality_Weight']
    feature_names = ['Risk_Z', 'Growth_Z', 'Quality_Z']
    
    for coef_name, feature_name in zip(features, feature_names):
        value = coef[coef_name]
        if value >= 0:
            equation += f" + {value:.3f}×{feature_name}"
        else:
            equation += f" - {abs(value):.3f}×{feature_name}"
    
    # Add PE conversion equation
    equation += f"\nPE = (PE_ZScore × {stats['pe_std']:.2f}) + {stats['pe_mean']:.2f}"
    
    return equation

def main():
    try:
        # Load and prepare data
        df = load_and_prepare_data('industry_analysis.csv')
        
        # Perform industry-specific regression
        industry_models, df = perform_industry_regression(df)
        
        print("\nIndustry-specific Regression Results:")
        print("=====================================")
        for industry, model_info in industry_models.items():
            print("\n" + format_industry_equation(model_info, industry))
        
    except Exception as e:
        print(f"Error during regression analysis: {str(e)}")
        print("Please make sure data.py has been run successfully first.")

if __name__ == "__main__":
    main()
