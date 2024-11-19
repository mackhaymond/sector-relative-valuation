import pandas as pd
import numpy as np
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy import stats
from typing import Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for better styling
external_stylesheets = [
    {
        'href': 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap',
        'rel': 'stylesheet'
    }
]

# Initialize the Dash app
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

# Custom styles
STYLES = {
    'container': {
        'max-width': '1200px',
        'margin': '0 auto',
        'padding': '20px',
        'font-family': 'Inter, sans-serif'
    },
    'header': {
        'textAlign': 'center',
        'color': '#2c3e50',
        'margin-bottom': '30px',
        'font-weight': '600'
    },
    'card': {
        'backgroundColor': 'white',
        'padding': '20px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
        'marginBottom': '20px'
    },
    'equation': {
        'backgroundColor': '#f8f9fa',
        'padding': '15px',
        'borderRadius': '6px',
        'fontFamily': 'monospace',
        'marginBottom': '10px',
        'fontSize': '14px',
        'marginTop': '10px',
        'whiteSpace': 'pre-wrap'
    }
}

# Constants
METRICS = {
    'Risk_Z': 'Risk Score (Z)',
    'Growth_Z': 'Growth Score (Z)',
    'Quality_Z': 'Quality Score (Z)',
    'Combined_Z': 'Combined Score (Z)'
}

def load_and_process_data():
    """Load and process the data for the dashboard"""
    try:
        # Load data
        df = pd.read_csv('industry_analysis.csv')
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        
        # Ensure required columns exist
        required_columns = ['Risk_Score', 'Growth_Score', 'Quality_Score', 'PE', 'Industry']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Calculate PE Z-score for each industry
        df['PE_ZScore'] = df.groupby('Industry')['PE'].transform(lambda x: stats.zscore(x, nan_policy='omit'))
        
        # Convert scores to Z-scores if they aren't already
        df['Risk_Z'] = stats.zscore(df['Risk_Score'], nan_policy='omit')
        df['Growth_Z'] = stats.zscore(df['Growth_Score'], nan_policy='omit')
        df['Quality_Z'] = stats.zscore(df['Quality_Score'], nan_policy='omit')
        
        # Create combined score using PCA
        X = df[['Risk_Z', 'Growth_Z', 'Quality_Z']].values
        pca = PCA(n_components=1)
        combined_score = pca.fit_transform(X).flatten()
        df['Combined_Z'] = combined_score
        
        # Get PCA weights
        pca_weights = {
            'Risk_Weight': pca.components_[0][0],
            'Growth_Weight': pca.components_[0][1],
            'Quality_Weight': pca.components_[0][2]
        }
        explained_var = pca.explained_variance_ratio_[0]
        
        # Perform industry-specific regression
        industry_models = {}
        df['Predicted_PE'] = np.nan
        df['Predicted_PE_Z'] = np.nan
        
        # Process each industry separately
        for industry in df['Industry'].unique():
            industry_df = df[df['Industry'] == industry].copy()
            
            if len(industry_df) < 5:  # Skip industries with too few samples
                continue
                
            # Prepare features for this industry
            X = industry_df[['Risk_Z', 'Growth_Z', 'Quality_Z']].values
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
        
        return df, industry_models, pca_weights, explained_var
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return pd.DataFrame(), None, None, None

def format_industry_equation(industry_model: Dict, industry: str) -> str:
    """Format regression equation for a specific industry"""
    coef = industry_model['coefficients']
    stats = industry_model['stats']
    
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
    
    equation += f"\nPE = (PE_ZScore × {stats['pe_std']:.2f}) + {stats['pe_mean']:.2f}"
    
    return equation

# Load data once at startup
initial_df, industry_models, pca_weights, explained_var = load_and_process_data()

if not initial_df.empty:
    available_industries = sorted(initial_df['Industry'].unique())
    available_metrics = ['Risk_Z', 'Growth_Z', 'Quality_Z', 'Combined_Z']
else:
    available_industries = []
    available_metrics = []

# Create layout
app.layout = html.Div([
    # Main container
    html.Div([
        # Header
        html.H1('Stock Analysis Dashboard', style=STYLES['header']),
        
        # Data loading status
        html.Div(id='data-status', style={'color': '#e74c3c', 'marginBottom': '20px', 'textAlign': 'center'}),
        
        # Controls
        html.Div([
            html.Div([
                html.Label('Select Industry:', style={'fontWeight': '500', 'marginBottom': '8px'}),
                dcc.Dropdown(
                    id='industry-filter',
                    options=[{'label': i, 'value': i} for i in available_industries],
                    value=available_industries[0] if available_industries else None,
                    multi=False,
                    style={'width': '100%'}
                )
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label('Select Metric:', style={'fontWeight': '500', 'marginBottom': '8px'}),
                dcc.Dropdown(
                    id='metric-selector',
                    options=[{'label': METRICS.get(m, m), 'value': m} for m in available_metrics],
                    value='Combined_Z' if available_metrics else None,
                    style={'width': '100%'}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
        ], style=STYLES['card']),
        
        # Scatter plots
        html.Div([
            html.Div([
                html.H3('Selected Metric vs P/E', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                dcc.Loading(
                    id="loading-scatter-1",
                    type="circle",
                    children=dcc.Graph(id='main-scatter')
                )
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H3('Combined Score vs P/E', style={'color': '#2c3e50', 'marginBottom': '15px'}),
                dcc.Loading(
                    id="loading-scatter-2",
                    type="circle",
                    children=dcc.Graph(id='combined-scatter')
                )
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
        ], style=STYLES['card']),
        
        # Model Information and Statistics
        html.Div([
            # Left column - Statistics
            html.Div([
                html.Div(id='stats-panel')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            # Right column - Model Info
            html.Div([
                html.Div(id='model-info')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
        ], style=STYLES['card']),
        
        # Equations Panel
        html.Div(id='equations-panel', style=STYLES['card'])
        
    ], style=STYLES['container'])
])

@callback(
    Output('data-status', 'children'),
    [Input('industry-filter', 'value')]
)
def update_data_status(selected_industry):
    if not initial_df.empty:
        if selected_industry:
            filtered_df = initial_df[initial_df['Industry'] == selected_industry]
            return f"Showing data for {selected_industry} with {len(filtered_df)} companies"
        return f"No industry selected. Please select from {len(available_industries)} available industries."
    return "Error: No data loaded. Please check the data file exists in the current directory."

@callback(
    [Output('main-scatter', 'figure'),
     Output('combined-scatter', 'figure'),
     Output('stats-panel', 'children'),
     Output('model-info', 'children'),
     Output('equations-panel', 'children')],
    [Input('metric-selector', 'value'),
     Input('industry-filter', 'value')]
)
def update_scatter(selected_metric, selected_industry):
    if initial_df.empty or not selected_industry or not selected_metric:
        return go.Figure(), go.Figure(), "", "", ""
    
    # Filter for selected industry
    df = initial_df[initial_df['Industry'] == selected_industry].copy()
    
    # Get the industry-specific model
    industry_model = industry_models.get(selected_industry)
    if industry_model is None:
        return go.Figure(), go.Figure(), "", "", ""
    
    # Create main scatter plot
    fig1 = px.scatter(df, 
                    x=selected_metric,
                    y='PE',
                    hover_data=['Ticker', 'Industry', 'PE', selected_metric],
                    trendline="ols")
    
    fig1.update_layout(
        title=f'P/E Ratio vs {METRICS[selected_metric]} for {selected_industry}',
        xaxis_title=METRICS[selected_metric],
        yaxis_title='P/E Ratio',
        height=500,
        hovermode='closest',
        template='plotly_white',
        font=dict(family="Inter, sans-serif")
    )
    
    # Create combined score scatter plot
    fig2 = px.scatter(df, 
                    x='Combined_Z',
                    y='PE',
                    hover_data=['Ticker', 'Industry', 'PE', 'Combined_Z'],
                    trendline="ols")
    
    fig2.update_layout(
        title=f'P/E Ratio vs Combined Score for {selected_industry}',
        xaxis_title='Combined Score (Z)',
        yaxis_title='P/E Ratio',
        height=500,
        hovermode='closest',
        template='plotly_white',
        font=dict(family="Inter, sans-serif")
    )
    
    # Calculate statistics
    stats = industry_model['stats']
    coefficients = industry_model['coefficients']
    
    stats_panel = html.Div([
        html.H3('Statistics', style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Div([
            html.P(f'Number of Companies: {stats["n_companies"]}'),
            html.P(f'R²: {stats["r2"]:.3f}'),
            html.P(f'Industry P/E Mean: {stats["pe_mean"]:.2f}'),
            html.P(f'Industry P/E Std Dev: {stats["pe_std"]:.2f}'),
            html.P(f'PCA Explained Variance: {explained_var:.3f}')
        ], style={'lineHeight': '1.6'})
    ])
    
    # Model information
    model_info = html.Div([
        html.H3('Model Coefficients', style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Div([
            html.H4('Regression Coefficients:', style={'marginTop': '15px', 'marginBottom': '10px'}),
            html.Ul([
                html.Li(f'Risk Score: {coefficients["Risk_Weight"]:.3f}'),
                html.Li(f'Growth Score: {coefficients["Growth_Weight"]:.3f}'),
                html.Li(f'Quality Score: {coefficients["Quality_Weight"]:.3f}'),
                html.Li(f'Intercept: {coefficients["Intercept"]:.3f}')
            ], style={'listStyleType': 'none', 'padding': '0'}),
            html.H4('PCA Weights:', style={'marginTop': '15px', 'marginBottom': '10px'}),
            html.Ul([
                html.Li(f'Risk Weight: {pca_weights["Risk_Weight"]:.3f}'),
                html.Li(f'Growth Weight: {pca_weights["Growth_Weight"]:.3f}'),
                html.Li(f'Quality Weight: {pca_weights["Quality_Weight"]:.3f}')
            ], style={'listStyleType': 'none', 'padding': '0'})
        ])
    ])
    
    # Equations panel
    equation_text = format_industry_equation(industry_model, selected_industry)
    equations_panel = html.Div([
        html.H3('Model Equations', style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Pre(equation_text, style=STYLES['equation'])
    ])
    
    return fig1, fig2, stats_panel, model_info, equations_panel

if __name__ == '__main__':
    app.run_server(debug=True)
