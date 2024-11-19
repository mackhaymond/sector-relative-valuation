import pandas as pd
import numpy as np
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import logging

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
        'fontSize': '14px',
        'marginTop': '10px',
        'marginBottom': '10px'
    }
}

def load_and_process_data():
    try:
        # Load the data
        df = pd.read_csv('data/processed/industry_analysis.csv')
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        
        required_columns = ['Risk_Score', 'Growth_Score', 'Quality_Score', 'PE', 'Industry']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame(), None, None
            
        # Remove rows with NaN values
        df = df.dropna(subset=required_columns)
        
        if len(df) == 0:
            logger.error("No valid data points after removing NaN values")
            return pd.DataFrame(), None, None
            
        # Dictionary to store industry-specific regression models
        industry_models = {}
        
        # Initialize prediction columns
        df['Predicted_PE'] = np.nan
        df['PE_Z'] = np.nan
        df['Predicted_PE_Z'] = np.nan
        
        # Process each industry separately
        for industry in df['Industry'].unique():
            industry_df = df[df['Industry'] == industry].copy()
            
            if len(industry_df) < 5:  # Skip industries with too few samples
                continue
                
            # Prepare features - scores are already in z-scores
            X = industry_df[['Risk_Score', 'Growth_Score', 'Quality_Score']].values
            
            # Convert PE to z-score for this industry
            pe_scaler = StandardScaler()
            pe_z = pe_scaler.fit_transform(industry_df[['PE']]).flatten()
            
            # Store z-scores in dataframe
            industry_df['Risk_Z'] = X[:, 0]
            industry_df['Growth_Z'] = X[:, 1]
            industry_df['Quality_Z'] = X[:, 2]
            industry_df['PE_Z'] = pe_z
            
            # Perform ridge regression using z-scores for this industry
            alphas = np.logspace(-6, 6, 13)
            model = RidgeCV(alphas=alphas, cv=5)
            model.fit(X, pe_z)
            
            # Calculate predicted PE z-score
            predicted_pe_z = model.predict(X)
            industry_df['Predicted_PE_Z'] = predicted_pe_z
            
            # Convert predicted z-score back to PE
            predicted_pe = pe_scaler.inverse_transform(predicted_pe_z.reshape(-1, 1)).flatten()
            industry_df['Predicted_PE'] = predicted_pe
            
            # Update the main dataframe with the processed data
            df.loc[industry_df.index, ['Predicted_PE', 'PE_Z', 'Predicted_PE_Z', 'Risk_Z', 'Growth_Z', 'Quality_Z']] = industry_df[['Predicted_PE', 'PE_Z', 'Predicted_PE_Z', 'Risk_Z', 'Growth_Z', 'Quality_Z']]
            
            # Store the model and scaler for this industry
            industry_models[industry] = {
                'model': model,
                'pe_scaler': pe_scaler
            }
            
        # Perform PCA on the entire dataset for the combined score
        X_all = df[['Risk_Score', 'Growth_Score', 'Quality_Score']].values
        pca = PCA(n_components=1)
        df['Combined_Score'] = pca.fit_transform(X_all).flatten()
        
        # Store PCA weights
        df['PCA_Risk_Weight'] = pca.components_[0][0]
        df['PCA_Growth_Weight'] = pca.components_[0][1]
        df['PCA_Quality_Weight'] = pca.components_[0][2]
        
        return df, industry_models, pca
            
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        return pd.DataFrame(), None, None

# Load data once at startup
initial_df, industry_models, pca_model = load_and_process_data()
if not initial_df.empty:
    available_industries = sorted(initial_df['Industry'].unique())
else:
    available_industries = []

# Available metrics for plotting
METRICS = {
    'Combined_Score': 'Combined PCA Score',
    'Risk_Score': 'Risk Score',
    'Growth_Score': 'Growth Score',
    'Quality_Score': 'Quality Score',
    'Predicted_PE': 'Predicted P/E'
}

def format_equation(coef, intercept, feature_names):
    """Format regression equation nicely"""
    terms = []
    for i, (name, value) in enumerate(zip(feature_names, coef)):
        if i == 0:
            terms.append(f"{value:.3f}×{name}")
        else:
            terms.append(f"{' + ' if value >= 0 else ' - '}{abs(value):.3f}×{name}")
    equation = "PE = " + "".join(terms)
    if intercept >= 0:
        equation += f" + {intercept:.3f}"
    else:
        equation += f" - {abs(intercept):.3f}"
    return equation

# Create layout
app.layout = html.Div([
    # Main container
    html.Div([
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
                    options=[{'label': v, 'value': k} for k, v in METRICS.items()],
                    value='Risk_Score',
                    style={'width': '100%'}
                )
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
        ], style=STYLES['card']),
        
        # Main scatter plot
        html.Div([
            dcc.Loading(
                id="loading-scatter",
                type="circle",
                children=dcc.Graph(id='main-scatter')
            )
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
    return "Error: No data loaded. Please check the data file exists in data/processed/industry_analysis.csv"

@callback(
    [Output('main-scatter', 'figure'),
     Output('stats-panel', 'children'),
     Output('model-info', 'children'),
     Output('equations-panel', 'children')],
    [Input('metric-selector', 'value'),
     Input('industry-filter', 'value')]
)
def update_scatter(selected_metric, selected_industry):
    if initial_df.empty or not selected_industry or not selected_metric:
        return go.Figure(), "", "", ""
    
    # Filter for selected industry
    df = initial_df[initial_df['Industry'] == selected_industry].copy()
    
    # Get the industry-specific model
    industry_model = industry_models.get(selected_industry)
    if industry_model is None:
        return go.Figure(), "", "", ""
    
    regression_model = industry_model['model']
    pe_scaler = industry_model['pe_scaler']
    
    # Create scatter plot
    fig = px.scatter(df, 
                    x=selected_metric,
                    y='PE',
                    hover_data=['Ticker', 'Industry', 'PE', selected_metric],
                    trendline="ols")
    
    fig.update_layout(
        title=f'P/E Ratio vs {METRICS[selected_metric]} for {selected_industry}',
        xaxis_title=METRICS[selected_metric],
        yaxis_title='P/E Ratio',
        height=600,
        hovermode='closest',
        template='plotly_white',
        font=dict(family="Inter, sans-serif")
    )
    
    # Calculate statistics
    correlation = df[['PE', selected_metric]].corr().iloc[0, 1]
    mean_pe = df['PE'].mean()
    std_pe = df['PE'].std()
    
    if selected_metric == 'Predicted_PE':
        r2 = r2_score(df['PE'], df['Predicted_PE'])
        rmse = np.sqrt(np.mean((df['PE'] - df['Predicted_PE'])**2))
    else:
        r2 = correlation**2
        rmse = None
    
    stats_panel = html.Div([
        html.H3('Statistics', style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Div([
            html.P(f'Correlation with P/E: {correlation:.3f}'),
            html.P(f'R²: {r2:.3f}'),
            html.P(f'Mean P/E: {mean_pe:.2f}'),
            html.P(f'P/E Standard Deviation: {std_pe:.2f}'),
            *([] if rmse is None else [html.P(f'Root Mean Square Error: {rmse:.2f}')])
        ], style={'lineHeight': '1.6'})
    ])
    
    # Model information
    if selected_metric == 'Combined_Score' and pca_model is not None:
        model_info = html.Div([
            html.H3('PCA Model Information', style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.P(f'Explained Variance Ratio: {pca_model.explained_variance_ratio_[0]:.3f}'),
            html.P('Component Weights:'),
            html.Ul([
                html.Li(f'Risk Score: {pca_model.components_[0][0]:.3f}'),
                html.Li(f'Growth Score: {pca_model.components_[0][1]:.3f}'),
                html.Li(f'Quality Score: {pca_model.components_[0][2]:.3f}')
            ], style={'listStyleType': 'none', 'padding': '0'})
        ])
    elif selected_metric == 'Predicted_PE' and regression_model is not None:
        model_info = html.Div([
            html.H3(f'Ridge Regression Model for {selected_industry}', style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.P(f'Alpha: {regression_model.alpha_:.3e}'),
            html.P('Coefficients:'),
            html.Ul([
                html.Li(f'Risk Score (Z): {regression_model.coef_[0]:.3f}'),
                html.Li(f'Growth Score (Z): {regression_model.coef_[1]:.3f}'),
                html.Li(f'Quality Score (Z): {regression_model.coef_[2]:.3f}'),
                html.Li(f'Intercept: {regression_model.intercept_:.3f}')
            ], style={'listStyleType': 'none', 'padding': '0'})
        ])
    else:
        model_info = ""
    
    # Equations panel
    if regression_model is not None:
        feature_names = ['Risk_Z', 'Growth_Z', 'Quality_Z']
        
        # P/E Z-score equation
        pe_zscore_eq = format_equation(
            regression_model.coef_,
            regression_model.intercept_,
            feature_names
        )
        
        # Get PE statistics from the scaler
        pe_std = pe_scaler.scale_[0]
        pe_mean = pe_scaler.mean_[0]
        
        # P/E from Z-score equation
        pe_eq = f"P/E = (P/E Z-score × {pe_std:.2f}) + {pe_mean:.2f}"
        
        equations_panel = html.Div([
            html.H3(f'Model Equations for {selected_industry}', style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div([
                html.P('P/E Z-score Prediction:', style={'fontWeight': '500'}),
                html.Div(pe_zscore_eq, style=STYLES['equation']),
                html.P('P/E from Z-score:', style={'fontWeight': '500', 'marginTop': '15px'}),
                html.Div(pe_eq, style=STYLES['equation'])
            ])
        ])
    else:
        equations_panel = ""
    
    return fig, stats_panel, model_info, equations_panel

if __name__ == '__main__':
    app.run_server(debug=True)
