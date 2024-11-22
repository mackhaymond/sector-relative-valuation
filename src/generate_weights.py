import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the CSV file
file_path = 'sector_analysis.csv'
data = pd.read_csv(file_path)

# Initialize a dictionary to store sector-specific weights
sector_weights = {}

# Process each sector separately
for sector in data['Sector'].unique():
    # Filter data for the current sector
    sector_data = data[data['Sector'] == sector]
    
    # Define input (X) and target (y) variables for this sector
    X = sector_data[['Risk_Score', 'Growth_Score', 'Quality_Score']]
    y = sector_data['PE_ZScore']
    
    # Standardize the input variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Ridge regression for this sector
    ridge_model = Ridge(alpha=1.0, fit_intercept=False)
    ridge_model.fit(X_scaled, y)
    
    # Extract and normalize the coefficients
    coefficients = ridge_model.coef_
    normalized_weights = np.abs(coefficients) / np.sum(np.abs(coefficients)) * 100
    
    # Store the weights for this sector
    sector_weights[sector] = {
        'Risk_Score': normalized_weights[0],
        'Growth_Score': normalized_weights[1],
        'Quality_Score': normalized_weights[2]
    }

# Calculate sector-specific magic scores
for sector in sector_weights:
    sector_mask = data['Sector'] == sector
    weights = sector_weights[sector]
    
    data.loc[sector_mask, 'magic_score'] = (
        data.loc[sector_mask, 'Risk_Score'] * weights['Risk_Score'] +
        data.loc[sector_mask, 'Growth_Score'] * weights['Growth_Score'] +
        data.loc[sector_mask, 'Quality_Score'] * weights['Quality_Score']
    ) / 100  # Divide by 100 to convert percentage to decimal

# Save the updated data
data.to_csv('sector_analysis.csv', index=False)

# Print sector-specific weights for review
print("\nSector-Specific Weights:")
for sector, weights in sector_weights.items():
    print(f"\n{sector}:")
    for metric, weight in weights.items():
        print(f"{metric}: {weight:.2f}%")
