import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the CSV file
file_path = 'industry_analysis.csv'
data = pd.read_csv(file_path)

# Define input (X) and target (y) variables
X = data[['Risk_Score', 'Growth_Score', 'Quality_Score']]
y = data['PE_ZScore']

# Standardize the input variables (already z-scores, but ensuring normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Ridge regression
ridge_model = Ridge(alpha=1.0, fit_intercept=False)  # Regularization parameter alpha set to 1.0
ridge_model.fit(X_scaled, y)

# Extract and normalize the coefficients
coefficients = ridge_model.coef_
normalized_weights = np.abs(coefficients) / np.sum(np.abs(coefficients)) * 100

# Create a DataFrame for the results
weights_df = pd.DataFrame({
    'Feature': ['Risk_Score', 'Growth_Score', 'Quality_Score'],
    'Coefficient': coefficients,
    'Weight (%)': normalized_weights
})

# Calculate the magic score
data['magic_score'] = (
    data['Risk_Score'] * weights_df.loc[weights_df['Feature'] == 'Risk_Score', 'Weight (%)'].values[0] +
    data['Growth_Score'] * weights_df.loc[weights_df['Feature'] == 'Growth_Score', 'Weight (%)'].values[0] +
    data['Quality_Score'] * weights_df.loc[weights_df['Feature'] == 'Quality_Score', 'Weight (%)'].values[0]
) / 100  # Divide by 100 to convert percentage to decimal

data.to_csv('industry_analysis.csv', index=False)
