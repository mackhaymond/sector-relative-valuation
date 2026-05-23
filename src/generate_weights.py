import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

# Half-decade alpha grid that brackets the sensible regularization
# range for a 5-predictor, 30-150-row sector fit. 0.01 -> nearly OLS
# (use this when the in-sector fit is well-conditioned); 100.0 ->
# heavy shrinkage toward zero (use when the predictors are strongly
# collinear). RidgeCV picks the alpha that minimizes the k-fold CV
# error on this sector's standardized cross-section.
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]

file_path = 'sector_analysis.csv'
data = pd.read_csv(file_path)

def _compute_vif(x_scaled: np.ndarray) -> list[float]:
    """Per-column VIF for a standardized design matrix.

    VIF_j = 1 / (1 - R^2_j) where R^2_j is the OLS fit of column j on
    the remaining columns. Returns NaN for a column when statsmodels
    cannot evaluate the regression (e.g. perfectly collinear cross-
    section, n < 2 predictors), so the diagnostic table can still
    render the rest of the row.
    """
    vifs: list[float] = []
    for j in range(x_scaled.shape[1]):
        try:
            vifs.append(float(variance_inflation_factor(x_scaled, j)))
        except Exception:
            vifs.append(float("nan"))
    return vifs


sector_weights: dict[str, dict[str, float]] = {}
sector_alphas: dict[str, float] = {}
sector_r_squared: dict[str, float] = {}
sector_vifs: dict[str, list[float]] = {}
sector_n: dict[str, int] = {}

for sector in data['Sector'].unique():
    sector_data = data[data['Sector'] == sector]

    X = sector_data[FACTOR_COLUMNS]
    y = sector_data['PE_ZScore']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # cv = min(5, n-1) keeps the smallest sectors (energy at n ~30
    # after the outlier filter) on a clean k-fold while letting larger
    # sectors enjoy the full 5-fold. n-1 is the lower bound below
    # which RidgeCV's k-fold cannot form k disjoint test folds.
    n = X_scaled.shape[0]
    cv_folds = max(2, min(5, n - 1))

    ridge_model = RidgeCV(
        alphas=ALPHA_GRID,
        fit_intercept=False,
        cv=cv_folds,
    )
    ridge_model.fit(X_scaled, y)

    coefficients = ridge_model.coef_
    normalized_weights = np.abs(coefficients) / np.sum(np.abs(coefficients)) * 100

    sector_weights[sector] = {
        column: float(weight)
        for column, weight in zip(FACTOR_COLUMNS, normalized_weights)
    }
    # RidgeCV.alpha_ is typed Optional in sklearn-stubs but is always
    # populated after a successful .fit(); narrow the type with an
    # explicit guard so basedpyright doesn't see a possibly-None float.
    chosen_alpha = ridge_model.alpha_
    if chosen_alpha is None:
        raise RuntimeError(f"RidgeCV failed to select an alpha for sector {sector!r}")
    sector_alphas[sector] = float(chosen_alpha)
    sector_r_squared[sector] = float(ridge_model.score(X_scaled, y))
    sector_vifs[sector] = _compute_vif(X_scaled)
    sector_n[sector] = n

for sector in sector_weights:
    sector_mask = data['Sector'] == sector
    weights = sector_weights[sector]

    weighted_sum = sum(
        data.loc[sector_mask, column] * weights[column]
        for column in FACTOR_COLUMNS
    )
    data.loc[sector_mask, 'composite_z_score'] = weighted_sum / 100

data.to_csv('sector_analysis.csv', index=False)

# weights.csv schema: one row per sector, FACTOR_COLUMNS in declared
# order, then the per-sector ridge diagnostics (alpha, r_squared).
# The diagnostic columns are appended (not interleaved) so a consumer
# that only wants the factor weights can keep slicing by FACTOR_COLUMNS.
weights_df = pd.DataFrame.from_dict(sector_weights, orient='index')
weights_df = weights_df[FACTOR_COLUMNS]
weights_df['alpha'] = pd.Series(sector_alphas)
weights_df['r_squared'] = pd.Series(sector_r_squared)
weights_df.index.name = 'Sector'
weights_df.to_csv('weights.csv')

def _format_diagnostic_table() -> str:
    """Pretty-printed per-sector summary: alpha, R^2, top-2 weights, VIFs."""
    header = (
        f"{'sector':<24} {'n':>5} {'alpha':>8} {'R^2':>8}   "
        f"{'top-2 factors':<46}   VIF (Risk Mom Qual Size Grow)"
    )
    rows = [header, "-" * len(header)]
    short_names = ['Risk', 'Mom', 'Qual', 'Size', 'Grow']
    for sector in sorted(sector_weights.keys()):
        weights = sector_weights[sector]
        ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        top2 = ranked[:2]
        top2_str = ", ".join(f"{name.replace('_Score', '')} {w:5.1f}%" for name, w in top2)
        vif_str = "  ".join(
            f"{name}={vif:5.2f}" if np.isfinite(vif) else f"{name}=  nan"
            for name, vif in zip(short_names, sector_vifs[sector])
        )
        rows.append(
            f"{sector:<24} {sector_n[sector]:>5} "
            f"{sector_alphas[sector]:>8.3f} {sector_r_squared[sector]:>8.4f}   "
            f"{top2_str:<46}   {vif_str}"
        )

    # Flag sectors with severe multicollinearity. Ridge handles VIF>10
    # by design via shrinkage, but the operator still wants to know so
    # the per-sector weight stability for that factor is treated as
    # diagnostic-only, not as a clean estimate.
    flagged: list[tuple[str, str, float]] = []
    for sector, vifs in sector_vifs.items():
        for factor, vif in zip(FACTOR_COLUMNS, vifs):
            if np.isfinite(vif) and vif > 10.0:
                flagged.append((sector, factor, vif))
    if flagged:
        rows.append("")
        rows.append("VIF > 10 (severe collinearity; ridge still fits but weights are diagnostic-only):")
        for sector, factor, vif in flagged:
            rows.append(f"  {sector:<24} {factor:<16} VIF={vif:.2f}")

    return "\n".join(rows)


print("\nPer-sector Ridge fit diagnostics:")
print(_format_diagnostic_table())
