# QuantSystem

A sophisticated quantitative analysis system for sector-based stock evaluation and visualization. This system processes financial data across multiple market sectors, calculates composite scores based on risk, growth, and quality metrics, and provides an interactive dashboard for analysis.

## Features

- **Multi-Sector Analysis**: Covers 11 major market sectors including technology, healthcare, financials, and more
- **Comprehensive Metrics**: Evaluates stocks using:
  - Risk Metrics (Beta, Volatility, Debt-to-Equity)
  - Growth Metrics (Revenue Growth, Earnings Growth, Profit Margins)
  - Quality Metrics (ROE, ROA, Operating Margin)
- **Advanced Scoring System**: Uses Ridge regression to generate sector-specific weights for balanced evaluation
- **Interactive Dashboard**: Built with Dash and Plotly for dynamic data visualization
- **Asynchronous Data Processing**: Efficient data collection using async/await patterns

## Installation

1. Ensure you have Python 3.12+ installed
2. Clone this repository
3. Install dependencies using Poetry:
```sh
poetry install
```

## Usage

### Local Development

1. Activate the Poetry environment:
```sh
poetry shell
```

2. Run the dashboard:
```sh
python src/dashboard.py
```

The dashboard will be available at `http://localhost:8050`

### Docker Deployment

1. Build the Docker image:
```sh
docker build -t quantsystem .
```

2. Run the container:
```sh
docker run -p 8050:8050 -e PORT=8050 quantsystem
```

## Project Structure

- `src/`
  - `dashboard.py`: Interactive web interface built with Dash
  - `data.py`: Data collection and processing using Yahoo Finance API
  - `generate_weights.py`: Sector-specific weight calculation using Ridge regression
- `sector_analysis.csv`: Processed sector analysis data
- `poetry.lock` & `pyproject.toml`: Dependency management
- `Dockerfile`: Container configuration

## Dependencies

Major dependencies include:
- Dash & Plotly for visualization
- YFinance for market data
- Pandas & NumPy for data processing
- Scikit-learn for statistical analysis
- Aiohttp for async operations

## Notes

- The system uses the Yahoo Finance API for data collection
- Analysis is performed on a sector-by-sector basis for more accurate comparisons
- The magic score calculation is weighted based on sector-specific characteristics

## Authors

Mack Haymond (SpyicyDev)
Ezra Schwartz (EzrSchwartz)