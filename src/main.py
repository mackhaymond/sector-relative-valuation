import yfinance as yf
import pandas as pd
from scipy.stats import zscore

# Assuming you already have the software industry data
soft = yf.Industry(
    "software-infrastructure"
)  # Example ticker, you might need the correct one for the software industry
top = soft.top_companies



# Extract beta values for the top companies
def get_beta(ticker):
    try:
        stock = yf.Ticker(ticker)
        beta = stock.info.get("beta")
        return beta if beta is not None else float("nan")
    except Exception as e:
        print(f"Error fetching beta for {ticker}: {e}")
        return float("nan")


# Create a DataFrame with company tickers and betas
data = {
    "Ticker": [company for company in top.index],
    "Beta": [get_beta(company) for company in top.index],
}
df = pd.DataFrame(data)

# Drop rows where Beta is NaN
df = df.dropna(subset=["Beta"])

# Calculate mean, standard deviation, and z-score
mean_beta = df["Beta"].mean()
std_beta = df["Beta"].std()
df["Z-Score"] = (df["Beta"] - mean_beta) / std_beta

# Display results
print(df)
