import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np

# Order matters: it determines both the design-matrix column order
# Ridge sees and the per-factor weight ordering in weights.csv. Keep
# in lock-step with the composite columns produced by data.py.
FACTOR_COLUMNS = [
    'Risk_Score',
    'Momentum_Score',
    'Quality_Score',
    'Size_Score',
    'Growth_Score',
]

file_path = 'sector_analysis.csv'
data = pd.read_csv(file_path)

sector_weights: dict[str, dict[str, float]] = {}

for sector in data['Sector'].unique():
    sector_data = data[data['Sector'] == sector]

    X = sector_data[FACTOR_COLUMNS]
    y = sector_data['PE_ZScore']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge_model = Ridge(alpha=1.0, fit_intercept=False)
    ridge_model.fit(X_scaled, y)

    coefficients = ridge_model.coef_
    normalized_weights = np.abs(coefficients) / np.sum(np.abs(coefficients)) * 100

    sector_weights[sector] = {
        column: float(weight)
        for column, weight in zip(FACTOR_COLUMNS, normalized_weights)
    }

for sector in sector_weights:
    sector_mask = data['Sector'] == sector
    weights = sector_weights[sector]

    weighted_sum = sum(
        data.loc[sector_mask, column] * weights[column]
        for column in FACTOR_COLUMNS
    )
    data.loc[sector_mask, 'composite_z_score'] = weighted_sum / 100

data.to_csv('sector_analysis.csv', index=False)

weights_df = pd.DataFrame.from_dict(sector_weights, orient='index')
weights_df = weights_df[FACTOR_COLUMNS]
weights_df.index.name = 'Sector'
weights_df.to_csv('weights.csv')

print("\nSector-Specific Weights:")
for sector, weights in sector_weights.items():
    print(f"\n{sector}:")
    for metric, weight in weights.items():
        print(f"{metric}: {weight:.2f}%")
