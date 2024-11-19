import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from typing import Tuple, Dict

def load_and_prepare_data(file_path: str = 'industry_analysis.csv') -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the data and prepare X (category scores) and y (PE Z-score) for regression
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {file_path}. Please run data.py first.")
    
    required_columns = ['Risk_Score', 'Growth_Score', 'Quality_Score', 'PE_ZScore', 'PE']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove rows with NaN values
    df = df.dropna(subset=required_columns)
    
    if len(df) == 0:
        raise ValueError("No valid data points after removing NaN values")
    
    # Prepare features (X) and target (y)
    X = df[['Risk_Score', 'Growth_Score', 'Quality_Score']].values
    y = df['PE_ZScore'].values
    
    return X, y, df

def perform_regression(X: np.ndarray, y: np.ndarray) -> Tuple[RidgeCV, Dict, float, float]:
    """
    Perform ridge regression with cross-validation and return model, coefficients, and performance metrics
    """
    # Create and fit the model with cross-validation
    alphas = np.logspace(-6, 6, 13)
    model = RidgeCV(alphas=alphas, cv=5)
    model.fit(X, y)
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Calculate R-squared
    r2 = r2_score(y, y_pred)
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5)
    cv_mean = cv_scores.mean()
    
    # Create dictionary of coefficients
    coefficients = {
        'Risk_Weight': model.coef_[0],
        'Growth_Weight': model.coef_[1],
        'Quality_Weight': model.coef_[2],
        'Intercept': model.intercept_
    }
    
    return model, coefficients, r2, cv_mean

def create_combined_score(X: np.ndarray) -> np.ndarray:
    """
    Use PCA to combine the three fundamental scores into one dimension
    """
    pca = PCA(n_components=1)
    X_combined = pca.fit_transform(X)
    component_weights = pca.components_[0]
    explained_var = pca.explained_variance_ratio_[0]
    
    return X_combined.flatten(), component_weights, explained_var

def plot_combined_fundamentals(X: np.ndarray, df: pd.DataFrame, component_weights: np.ndarray):
    """
    Create scatter plot of actual PE vs combined fundamentals score with regression line
    """
    plt.figure(figsize=(12, 6))
    
    # Create subplot for PE vs Combined Score
    plt.subplot(1, 2, 1)
    combined_score, weights, var_explained = create_combined_score(X)
    
    # Plot scatter points
    plt.scatter(combined_score, df['PE'], alpha=0.5, label='Companies')
    
    # Add regression line
    z = np.polyfit(combined_score, df['PE'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(combined_score.min(), combined_score.max(), 100)
    plt.plot(x_range, p(x_range), 'r--', label=f'Regression Line (y = {z[0]:.2f}x + {z[1]:.2f})')
    
    plt.xlabel('Combined Fundamentals Score\n(PCA first component)')
    plt.ylabel('P/E Ratio')
    plt.title(f'P/E vs Combined Fundamentals\n(Explains {var_explained:.1%} of variance)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create subplot for feature importance
    plt.subplot(1, 2, 2)
    features = ['Risk', 'Growth', 'Quality']
    bars = plt.bar(features, weights)
    plt.title('Component Weights in Combined Score')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    
    # Add weight values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('combined_fundamentals.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_actual_vs_predicted(y_actual: np.ndarray, y_pred: np.ndarray, r2: float):
    """
    Create scatter plot of actual vs predicted PE Z-scores with regression statistics
    """
    plt.figure(figsize=(10, 6))
    
    # Plot scatter points
    plt.scatter(y_actual, y_pred, alpha=0.5, label='Predictions')
    
    # Plot perfect prediction line
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
             label='Perfect Prediction', alpha=0.8)
    
    # Add regression line
    z = np.polyfit(y_actual, y_pred, 1)
    p = np.poly1d(z)
    x_range = np.linspace(min_val, max_val, 100)
    plt.plot(x_range, p(x_range), 'g-', 
             label=f'Fitted Line (y = {z[0]:.2f}x + {z[1]:.2f})', alpha=0.8)
    
    plt.xlabel('Actual PE Z-Score')
    plt.ylabel('Predicted PE Z-Score')
    plt.title(f'Actual vs Predicted PE Z-Scores\nR² = {r2:.4f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add equal aspect ratio to make the plot square
    plt.axis('equal')
    plt.tight_layout()
    
    plt.savefig('regression_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_scores(X: np.ndarray, df: pd.DataFrame):
    """
    Create 3D scatter plot of PE vs the three component scores
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                        c=df['PE'], cmap='viridis', 
                        s=50, alpha=0.6)
    
    # Add labels
    ax.set_xlabel('Risk Score')
    ax.set_ylabel('Growth Score')
    ax.set_zlabel('Quality Score')
    plt.title('3D Visualization of Component Scores\nColored by P/E Ratio')
    
    # Add colorbar
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('P/E Ratio')
    
    # Rotate the plot for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('3d_scores.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlation_matrix(df: pd.DataFrame):
    """
    Create correlation matrix heatmap
    """
    # Calculate correlation matrix
    cols = ['PE', 'Risk_Score', 'Growth_Score', 'Quality_Score']
    corr_matrix = df[cols].corr()
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1, center=0,
                square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
    
    plt.title('Correlation Matrix of P/E and Component Scores')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_partial_regression(X: np.ndarray, df: pd.DataFrame):
    """
    Create partial regression plots for each component
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    component_names = ['Risk', 'Growth', 'Quality']
    
    # Standardize PE for consistent visualization
    pe_standardized = (df['PE'] - df['PE'].mean()) / df['PE'].std()
    
    for i, (ax, name) in enumerate(zip(axes, component_names)):
        # Create partial regression plot
        other_components = list(range(3))
        other_components.remove(i)
        
        # Residuals of X[i] on other X's
        X_other = sm.add_constant(X[:, other_components])
        res_x = sm.OLS(X[:, i], X_other).fit().resid
        
        # Residuals of PE on other X's
        res_y = sm.OLS(pe_standardized, X_other).fit().resid
        
        # Plot residuals
        ax.scatter(res_x, res_y, alpha=0.5)
        
        # Add regression line
        z = np.polyfit(res_x, res_y, 1)
        p = np.poly1d(z)
        x_range = np.linspace(res_x.min(), res_x.max(), 100)
        ax.plot(x_range, p(x_range), 'r--', 
                label=f'Slope: {z[0]:.3f}')
        
        ax.set_xlabel(f'{name} Score (residualized)')
        ax.set_ylabel('P/E (residualized)' if i == 0 else '')
        ax.set_title(f'Partial Regression Plot\n{name} Score vs P/E')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('partial_regression.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_equations(coefficients: Dict, pe_mean: float, pe_std: float):
    """
    Print both the Z-score regression equation and the PE conversion equation
    """
    print("\nRegression Equations:")
    print("---------------------")
    print("1. PE Z-Score Equation:")
    print(f"PE_ZScore = {coefficients['Intercept']:.4f} + "
          f"({coefficients['Risk_Weight']:.4f} × Risk_Score) + "
          f"({coefficients['Growth_Weight']:.4f} × Growth_Score) + "
          f"({coefficients['Quality_Weight']:.4f} × Quality_Score)")
    
    print("\n2. PE Conversion Equation:")
    print(f"PE = (PE_ZScore × {pe_std:.4f}) + {pe_mean:.4f}")

def main():
    try:
        # Load and prepare data
        X, y, df = load_and_prepare_data('industry_analysis.csv')
        
        # Perform regression
        model, coefficients, r2, cv_score = perform_regression(X, y)
        
        # Get PE conversion parameters
        pe_mean = df['PE'].mean()
        pe_std = df['PE'].std()
        
        # Print results
        print("\nRegression Analysis Results:")
        print("============================")
        print(f"R-squared: {r2:.4f}")
        print(f"Cross-validation score: {cv_score:.4f}")
        print(f"Selected alpha: {model.alpha_:.6f}")
        
        print("\nCategory Weights:")
        print("----------------")
        print(f"Risk Weight:    {coefficients['Risk_Weight']:.4f}")
        print(f"Growth Weight:  {coefficients['Growth_Weight']:.4f}")
        print(f"Quality Weight: {coefficients['Quality_Weight']:.4f}")
        
        # Print equations
        print_equations(coefficients, pe_mean, pe_std)
        
        # Create visualizations
        y_pred = model.predict(X)
        plot_actual_vs_predicted(y, y_pred, r2)
        plot_combined_fundamentals(X, df, model.coef_)
        
        # Additional visualizations
        plot_3d_scores(X, df)
        plot_correlation_matrix(df)
        plot_partial_regression(X, df)
        
        print("\nVisualization files created:")
        print("1. regression_plot.png - Shows actual vs predicted PE Z-scores")
        print("2. combined_fundamentals.png - Shows PE vs combined fundamental score")
        print("3. 3d_scores.png - 3D visualization of all component scores")
        print("4. correlation_matrix.png - Correlation heatmap")
        print("5. partial_regression.png - Partial regression plots for each component")
        
    except Exception as e:
        print(f"Error during regression analysis: {str(e)}")
        print("Please make sure data.py has been run successfully first.")

if __name__ == "__main__":
    main()
