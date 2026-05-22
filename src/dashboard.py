import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import os
import yfinance as yf
from scipy.stats import zscore
from plotly.subplots import make_subplots
import time
from yfinance.exceptions import YFRateLimitError
from functools import lru_cache

# Import constants from data.py
from data import (
    X1_RISK_METRICS,
    X2_MOMENTUM_METRICS,
    X3_QUALITY_METRICS,
    X4_VALUE_METRICS,
    X5_SIZE_METRICS,
    X6_GROWTH_METRICS,
    X7_PROFITABILITY_METRICS,
    X8_LIQUIDITY_METRICS,
    Y_VALUATION_METRIC,
    ALL_METRICS,
    calculate_rsi,
    calculate_return_sd,
    analyze_sectors_async
)

# GICS Sector Mapping
GICS_SECTOR_MAPPING = {
    "basic-materials": "Materials",
    "communication-services": "Communication Services",
    "consumer-cyclical": "Consumer Discretionary",
    "consumer-defensive": "Consumer Staples",
    "energy": "Energy",
    "financial-services": "Financials",
    "healthcare": "Health Care",
    "industrials": "Industrials",
    "real-estate": "Real Estate",
    "technology": "Information Technology",
    "utilities": "Utilities"
}

# Read the data
df = pd.read_csv('sector_analysis.csv')

# Remove rows where PE is null
df = df.dropna(subset=['PE'])

# Initialize the Dash app
app = Dash(__name__)

# Define custom styles
COLORS = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'primary': '#3498db',
    'secondary': '#e74c3c',
    'accent': '#2ecc71',
    'light_gray': '#f0f0f0',
    'border': '#e1e4e8'
}

FONT_FAMILY = 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'

STYLES = {
    'container': {
        'padding': '40px',
        'backgroundColor': COLORS['background'],
        'minHeight': '100vh',
        'fontFamily': FONT_FAMILY
    },
    'card': {
        'backgroundColor': 'white',
        'padding': '24px',
        'borderRadius': '12px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
        'marginBottom': '24px',
        'fontFamily': FONT_FAMILY
    },
    'title': {
        'fontFamily': FONT_FAMILY,
        'fontSize': '32px',
        'fontWeight': '600',
        'color': COLORS['text'],
        'marginBottom': '32px',
        'textAlign': 'center'
    },
    'subtitle': {
        'fontFamily': FONT_FAMILY,
        'fontSize': '24px',
        'fontWeight': '500',
        'color': COLORS['text'],
        'marginBottom': '16px',
        'marginTop': '24px'
    },
    'label': {
        'fontFamily': FONT_FAMILY,
        'fontSize': '16px',
        'fontWeight': '500',
        'color': COLORS['text'],
        'marginBottom': '8px',
        'display': 'block'
    },
    'text': {
        'fontFamily': FONT_FAMILY,
        'fontSize': '14px',
        'color': COLORS['text'],
        'marginBottom': '4px'
    },
    'dropdown': {
        'width': '300px',
        'fontFamily': FONT_FAMILY,
        'fontSize': '14px'
    },
    'dropdown-container': {
        'display': 'flex',
        'justifyContent': 'center',
        'gap': '24px',
        'flexWrap': 'wrap'
    },
    'button': {
        'backgroundColor': COLORS['primary'],
        'color': 'white',
        'border': 'none',
        'padding': '8px 16px',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'fontFamily': FONT_FAMILY,
        'fontSize': '14px',
        'fontWeight': '500'
    },
    'input': {
        'padding': '8px 12px',
        'borderRadius': '4px',
        'border': f'1px solid {COLORS["border"]}',
        'fontFamily': FONT_FAMILY,
        'fontSize': '14px',
        'marginRight': '10px',
        'width': '200px'
    }
}

# Create the layout
app.layout = html.Div([
    html.H1('Stock Analysis Dashboard', style=STYLES['title']),
    
    dcc.Tabs([
        dcc.Tab(label='Sector Analysis', children=[
            html.Div([
                html.Div([
                    html.Div([
                        html.Label('Select Sector:', style=STYLES['label']),
                        dcc.Dropdown(
                            id='sector-dropdown',
                            options=[{'label': GICS_SECTOR_MAPPING[sector], 'value': sector} for sector in df['Sector'].unique()],
                            value=df['Sector'].iloc[0],
                            clearable=False,
                            style=STYLES['dropdown']
                        )
                    ]),
                    
                    html.Div([
                        html.Label('Select Company:', style=STYLES['label']),
                        dcc.Dropdown(
                            id='company-dropdown',
                            options=[],
                            value=None,
                            clearable=False,
                            style=STYLES['dropdown']
                        )
                    ]),
                ], style=STYLES['dropdown-container']),
            ], style=STYLES['card']),
            
            html.Div([
                dcc.Loading(
                    id="loading-sector-scatter",
                    type="circle",
                    children=dcc.Graph(id='scatter-plot')
                )
            ], style=STYLES['card']),
            
            html.Div([
                dcc.Loading(
                    id="loading-sector-company-info",
                    type="circle",
                    children=html.Div(id='company-info')
                )
            ], style=STYLES['card'])
        ]),
        
        dcc.Tab(label='Individual Stock Analysis', children=[
            html.Div([
                html.Div([
                    html.Label('Enter Stock Ticker:', style=STYLES['label']),
                    dcc.Input(
                        id='ticker-input',
                        type='text',
                        placeholder='e.g., AAPL',
                        style=STYLES['input']
                    ),
                    html.Button(
                        'Analyze',
                        id='analyze-button',
                        style=STYLES['button']
                    )
                ], style={
                    'display': 'flex',
                    'alignItems': 'center',
                    'justifyContent': 'center',
                    'gap': '12px',
                    'marginBottom': '16px'
                }),
                html.Div([
                    dcc.Loading(
                        id="loading-analysis",
                        type="circle",
                        children=html.Div(id='analysis-status', style={
                            'textAlign': 'center',
                            'fontFamily': FONT_FAMILY,
                            'fontSize': '14px',
                            'color': COLORS['text'],
                            'marginTop': '8px',
                            'minHeight': '30px'
                        })
                    )
                ])
            ], style=STYLES['card']),
            
            html.Div([
                dcc.Loading(
                    id="loading-sector-plot",
                    type="circle",
                    children=dcc.Graph(id='sector-scatter-plot')
                )
            ], style=STYLES['card']),
            
            html.Div([
                dcc.Loading(
                    id="loading-pe-plot",
                    type="circle",
                    children=dcc.Graph(id='pe-comparison-plot')
                )
            ], style=STYLES['card']),
            
            html.Div([
                dcc.Loading(
                    id="loading-company-info",
                    type="circle",
                    children=html.Div(id='individual-company-info')
                )
            ], style=STYLES['card'])
        ]),
        
        dcc.Tab(label='Factor Selection', children=[
            html.Div([
                html.H3('Select Factors for PE Ratio Prediction', style=STYLES['subtitle']),
                html.Div([
                    # Risk Factors
                    html.Div([
                        html.Div([
                            dcc.Checklist(
                                id='factor-group-checklist',
                                options=[
                                    {'label': 'Risk', 'value': 'risk'},
                                    {'label': 'Momentum', 'value': 'momentum'},
                                    {'label': 'Quality', 'value': 'quality'},
                                    {'label': 'Value', 'value': 'value'},
                                    {'label': 'Size', 'value': 'size'},
                                    {'label': 'Growth', 'value': 'growth'},
                                    {'label': 'Profitability', 'value': 'profitability'},
                                    {'label': 'Liquidity', 'value': 'liquidity'},
                                ],
                                value=['risk', 'momentum', 'quality'],
                                labelStyle={'display': 'block', 'margin': '10px 0', 'fontWeight': 'bold'}
                            ),
                        ]),
                        
                        # Risk Metrics
                        html.Div([
                            dcc.Checklist(
                                id='risk-checklist',
                                options=[
                                    {'label': 'Max Drawdown', 'value': 'MaxDrawdown'},
                                    {'label': 'Debt To Equity', 'value': 'DebtToEquity'},
                                    {'label': 'Return SD', 'value': 'ReturnSD'},
                                ],
                                value=['MaxDrawdown', 'DebtToEquity', 'ReturnSD'],
                                labelStyle={'display': 'block', 'marginLeft': '20px'}
                            ),
                        ], style={'marginBottom': '15px'}),
                        
                        # Momentum Metrics
                        html.Div([
                            dcc.Checklist(
                                id='momentum-checklist',
                                options=[
                                    {'label': 'Price Change 12M', 'value': 'PriceChange12M'},
                                    {'label': 'RSI', 'value': 'RSI'},
                                    {'label': 'Earnings Growth', 'value': 'EarningsGrowth'},
                                ],
                                value=['PriceChange12M', 'RSI', 'EarningsGrowth'],
                                labelStyle={'display': 'block', 'marginLeft': '20px'}
                            ),
                        ], style={'marginBottom': '15px'}),
                        
                        # Quality Metrics
                        html.Div([
                            dcc.Checklist(
                                id='quality-checklist',
                                options=[
                                    {'label': 'ROE', 'value': 'ROE'},
                                    {'label': 'ROA', 'value': 'ROA'},
                                    {'label': 'Operating Margin', 'value': 'OperatingMargin'},
                                ],
                                value=['ROE', 'ROA', 'OperatingMargin'],
                                labelStyle={'display': 'block', 'marginLeft': '20px'}
                            ),
                        ], style={'marginBottom': '15px'}),
                        
                        # Value Metrics
                        html.Div([
                            dcc.Checklist(
                                id='value-checklist',
                                options=[
                                    {'label': 'Price To Book', 'value': 'PriceToBook'},
                                    {'label': 'EV To EBITDA', 'value': 'EvToEbitda'},
                                    {'label': 'Price To Sales', 'value': 'PriceToSales'},
                                ],
                                value=[],
                                labelStyle={'display': 'block', 'marginLeft': '20px'}
                            ),
                        ], style={'marginBottom': '15px'}),
                        
                        # Size Metrics
                        html.Div([
                            dcc.Checklist(
                                id='size-checklist',
                                options=[
                                    {'label': 'Market Cap', 'value': 'MarketCap'},
                                    {'label': 'Total Assets', 'value': 'TotalAssets'},
                                    {'label': 'Enterprise Value', 'value': 'EnterpriseValue'},
                                ],
                                value=[],
                                labelStyle={'display': 'block', 'marginLeft': '20px'}
                            ),
                        ], style={'marginBottom': '15px'}),
                        
                        # Growth Metrics
                        html.Div([
                            dcc.Checklist(
                                id='growth-checklist',
                                options=[
                                    {'label': 'Revenue Growth', 'value': 'RevenueGrowth'},
                                    {'label': 'EPS Growth', 'value': 'EpsGrowth'},
                                    {'label': 'Cash Flow Growth', 'value': 'CashFlowGrowth'},
                                ],
                                value=[],
                                labelStyle={'display': 'block', 'marginLeft': '20px'}
                            ),
                        ], style={'marginBottom': '15px'}),
                        
                        # Profitability Metrics
                        html.Div([
                            dcc.Checklist(
                                id='profitability-checklist',
                                options=[
                                    {'label': 'Gross Margin', 'value': 'GrossMargin'},
                                    {'label': 'EBITDA Margin', 'value': 'EbitdaMargin'},
                                    {'label': 'Net Profit Margin', 'value': 'NetProfitMargin'},
                                ],
                                value=[],
                                labelStyle={'display': 'block', 'marginLeft': '20px'}
                            ),
                        ], style={'marginBottom': '15px'}),
                        
                        # Liquidity Metrics
                        html.Div([
                            dcc.Checklist(
                                id='liquidity-checklist',
                                options=[
                                    {'label': 'Current Ratio', 'value': 'CurrentRatio'},
                                    {'label': 'Quick Ratio', 'value': 'QuickRatio'},
                                    {'label': 'Interest Coverage', 'value': 'InterestCoverage'},
                                ],
                                value=[],
                                labelStyle={'display': 'block', 'marginLeft': '20px'}
                            ),
                        ], style={'marginBottom': '15px'}),
                    ], style={'flex': '1', 'minWidth': '300px'}),
                    
                    # Regression Output
                    html.Div([
                        html.H3('Regression Results', style=STYLES['subtitle']),
                        html.P("Simulator: This UI demonstrates how the feature will work once metrics data is added. Currently using aggregate scores as proxies.", 
                            style={'fontSize': '12px', 'fontStyle': 'italic', 'marginBottom': '10px'}),
                        html.Button(
                            'Recalculate Regression',
                            id='recalculate-button',
                            style=STYLES['button']
                        ),
                        html.Div(id='reg-output', style={
                            'fontFamily': 'monospace',
                            'whiteSpace': 'pre-wrap',
                            'overflowX': 'auto',
                            'border': f'1px solid {COLORS["border"]}',
                            'padding': '15px',
                            'borderRadius': '4px',
                            'backgroundColor': '#f8f9fa',
                            'fontSize': '12px'
                        })
                    ], style={'flex': '2', 'minWidth': '500px'})
                ], style={
                    'display': 'flex',
                    'flexWrap': 'wrap',
                    'gap': '30px'
                }),
            ], style=STYLES['card'])
        ]),
    ])
], style=STYLES['container'])

@lru_cache(maxsize=100)
def get_historical_data(ticker, period="1y"):
    """Cache historical data to avoid repeated requests."""
    try:
        stock = yf.Ticker(ticker)
        time.sleep(2)  # Rate limiting delay
        return stock.history(period=period)
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()

def get_stock_data_with_retry(ticker, max_retries=3, base_delay=2):
    """Get all required stock data with retry logic and rate limiting."""
    for attempt in range(max_retries):
        try:
            # Get stock info
            stock = yf.Ticker(ticker)
            time.sleep(base_delay)  # Rate limiting delay
            info = stock.info
            
            # Get historical data using cached function
            hist = get_historical_data(ticker)
            
            return {
                'info': info,
                'history': hist
            }
        except YFRateLimitError:
            if attempt < max_retries - 1:
                sleep_time = base_delay * (attempt + 2)  # Exponential backoff
                time.sleep(sleep_time)
            else:
                raise
        except Exception as e:
            raise e

@app.callback(
    [Output('company-dropdown', 'options'),
     Output('company-dropdown', 'value')],
    Input('sector-dropdown', 'value')
)
def update_company_dropdown(selected_sector):
    filtered_df = df[df['Sector'] == selected_sector]
    options = [{'label': row['Ticker'], 'value': row['Ticker']} for _, row in filtered_df.iterrows()]
    # Set default value to first company in the list, or None if list is empty
    default_value = options[0]['value'] if options else None
    return options, default_value

@app.callback(
    Output('scatter-plot', 'figure'),
    Output('company-info', 'children'),
    Input('sector-dropdown', 'value'),
    Input('company-dropdown', 'value')
)
def update_graph(selected_sector, selected_company):
    # Create a copy of the filtered dataframe to avoid SettingWithCopyWarning
    filtered_df = df[df['Sector'] == selected_sector].copy()
    
    # Check if filtered_df is empty
    if filtered_df.empty:
        # Return empty figure and no company info
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title=dict(
                text='No data available for selected sector',
                font=dict(
                    family=FONT_FAMILY,
                    size=24,
                    color=COLORS['text']
                ),
                x=0.5,
                xanchor='center'
            ),
            height=600
        )
        return empty_fig, None
    
    x = filtered_df['composite_z_score']
    y = filtered_df['PE']
    
    # Check if there are enough data points for fitting
    if len(x) < 2 or len(y) < 2:
        # Return basic scatter plot without fit line
        basic_fig = go.Figure()
        basic_fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='Stocks',
            marker=dict(
                size=10,
                color=COLORS['primary'],
                line=dict(width=1.5, color='white'),
                opacity=0.8
            )
        ))
        basic_fig.update_layout(
            title=dict(
                text=f'Insufficient data for {GICS_SECTOR_MAPPING[selected_sector]} sector',
                font=dict(
                    family=FONT_FAMILY,
                    size=24,
                    color=COLORS['text']
                ),
                x=0.5,
                xanchor='center'
            ),
            height=600
        )
        return basic_fig, None
    
    fit = np.polyfit(x, y, 1)
    line_x = np.array([x.min(), x.max()])
    line_y = fit[0] * line_x + fit[1]
    
    # Calculate predicted P/E values and deviations using .loc
    filtered_df.loc[:, 'predicted_pe'] = fit[0] * filtered_df['composite_z_score'] + fit[1]
    filtered_df.loc[:, 'pe_deviation'] = filtered_df['PE'] - filtered_df['predicted_pe']
    
    # Calculate R-squared
    y_pred = fit[0] * x + fit[1]
    r_squared = np.corrcoef(x, y)[0, 1]**2
    
    # Find maximum deviation in sector for bar chart range
    max_deviation = abs(filtered_df['pe_deviation']).max()
    deviation_range = [-max_deviation, max_deviation]
    
    # Calculate y-axis range with padding
    max_pe = df['PE'].max()
    min_pe = df['PE'].min()
    pe_range = max_pe - min_pe
    y_min = max(0, min_pe - pe_range * 0.1)
    y_max = max_pe + pe_range * 0.1
    
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=filtered_df['composite_z_score'],
        y=filtered_df['PE'],
        mode='markers',
        name='Stocks',
        text=filtered_df.apply(
            lambda row: f"Ticker: {row['Ticker']}<br>P/E: {row['PE']:.2f}<br>Fundamental Z-score: {row['composite_z_score']:.2f}",
            axis=1
        ),
        hoverinfo='text',
        marker=dict(
            size=10,
            color=COLORS['primary'],
            line=dict(width=1.5, color='white'),
            opacity=0.8
        )
    ))
    
    # Add line of best fit
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        name='Line of Best Fit',
        line=dict(color=COLORS['secondary'], dash='dash', width=2)
    ))
    
    # Add equation and R² annotation
    equation = f'y = {fit[0]:.2f}x + {fit[1]:.2f}'
    r2_text = f'R² = {r_squared:.3f}'
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref='paper',
        yref='paper',
        text=f'{equation}<br>{r2_text}',
        showarrow=False,
        font=dict(family=FONT_FAMILY, size=12),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor=COLORS['border'],
        borderwidth=1,
        align='left'
    )
    
    # Highlight selected company
    company_info = None
    if selected_company and not filtered_df[filtered_df['Ticker'] == selected_company].empty:
        company_data = filtered_df[filtered_df['Ticker'] == selected_company].iloc[0]
        fig.add_trace(go.Scatter(
            x=[company_data['composite_z_score']],
            y=[company_data['PE']],
            mode='markers',
            name='Selected Company',
            marker=dict(
                size=14,
                color=COLORS['accent'],
                line=dict(width=2, color='white'),
                symbol='star'
            )
        ))
        
        # Create deviation bar chart
        deviation_fig = go.Figure()
        
        # Calculate the range for the bar
        actual_pe = company_data['PE']
        predicted_pe = company_data['predicted_pe']
        pe_min = min(actual_pe, predicted_pe)
        pe_max = max(actual_pe, predicted_pe)
        
        # Add the bar showing range from predicted to actual P/E
        deviation_fig.add_trace(go.Bar(
            x=[pe_max - pe_min],  # Length of the bar
            y=['P/E Range'],
            orientation='h',
            marker=dict(
                color=COLORS['accent'] if actual_pe < predicted_pe else COLORS['secondary']
            ),
            base=[pe_min],  # Start position of the bar
            text=[f"Actual: {actual_pe:.1f}"],
            textposition='outside',
            hoverinfo='text',
            hovertext=[f"Actual P/E: {actual_pe:.2f}<br>Predicted P/E: {predicted_pe:.2f}<br>Deviation: {company_data['pe_deviation']:.2f}"]
        ))
        
        # Add vertical line at predicted value
        deviation_fig.add_vline(
            x=predicted_pe,
            line_width=2,
            line_dash="solid",
            line_color=COLORS['text'],
            annotation=dict(
                text=f"Predicted: {predicted_pe:.1f}",
                font=dict(
                    family=FONT_FAMILY,
                    size=12,
                    color=COLORS['text']
                ),
                yshift=10
            )
        )
        
        # Calculate the range for the x-axis
        sector_max_pe = filtered_df['PE'].max()
        sector_min_pe = filtered_df['PE'].min()
        pe_range = sector_max_pe - sector_min_pe
        x_min = max(0, sector_min_pe - pe_range * 0.1)
        x_max = sector_max_pe + pe_range * 0.1
        
        # Update deviation chart layout
        deviation_fig.update_layout(
            title=dict(
                text='Actual vs. Predicted P/E Ratio',
                font=dict(
                    family=FONT_FAMILY,
                    size=16,
                    color=COLORS['text']
                ),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='P/E Ratio',
                range=[x_min, x_max],
                tickfont=dict(
                    family=FONT_FAMILY,
                    size=12
                ),
                gridcolor=COLORS['light_gray']
            ),
            yaxis=dict(
                showticklabels=False,
                fixedrange=True
            ),
            height=150,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        # Display company information with deviation chart
        company_info = html.Div([
            html.H3(f"{company_data['Ticker']}", style={
                'fontFamily': FONT_FAMILY,
                'fontSize': '24px',
                'fontWeight': '600',
                'color': COLORS['text'],
                'marginBottom': '16px'
            }),
            html.Div([
                html.P(f"P/E Ratio: {company_data['PE']:.2f}", style={
                    'fontFamily': FONT_FAMILY,
                    'fontSize': '16px',
                    'color': COLORS['text'],
                    'marginBottom': '8px'
                }),
                html.P(f"Predicted P/E: {company_data['predicted_pe']:.2f}", style={
                    'fontFamily': FONT_FAMILY,
                    'fontSize': '16px',
                    'color': COLORS['text'],
                    'marginBottom': '8px'
                }),
                html.P(f"Fundamental Z-score: {company_data['composite_z_score']:.2f}", style={
                    'fontFamily': FONT_FAMILY,
                    'fontSize': '16px',
                    'color': COLORS['text'],
                    'marginBottom': '16px'
                })
            ]),
            dcc.Graph(
                figure=deviation_fig,
                config={'displayModeBar': False}
            )
        ])
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Fundamental Z-score vs P/E Ratio - {GICS_SECTOR_MAPPING[selected_sector]}',
            font=dict(
                family=FONT_FAMILY,
                size=24,
                color=COLORS['text']
            ),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title={'text': 'Fundamental Z-score', 'font': {'family': FONT_FAMILY}},
            showgrid=True,
            gridcolor=COLORS['light_gray'],
            gridwidth=1,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor=COLORS['border'],
            mirror=True,
            tickfont={'family': FONT_FAMILY}
        ),
        yaxis=dict(
            title={'text': 'P/E Ratio', 'font': {'family': FONT_FAMILY}},
            showgrid=True,
            gridcolor=COLORS['light_gray'],
            gridwidth=1,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor=COLORS['border'],
            mirror=True,
            tickfont={'family': FONT_FAMILY}
        ),
        legend=dict(
            font=dict(family=FONT_FAMILY),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=COLORS['border'],
            borderwidth=1
        ),
        showlegend=True,
        hovermode='closest',
        font={'family': FONT_FAMILY},
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    return fig, company_info

# Define factor group mapping
FACTOR_GROUPS = {
    'risk': {
        'name': 'Risk',
        'checklist_id': 'risk-checklist',
        'metrics': X1_RISK_METRICS
    },
    'momentum': {
        'name': 'Momentum',
        'checklist_id': 'momentum-checklist',
        'metrics': X2_MOMENTUM_METRICS
    },
    'quality': {
        'name': 'Quality',
        'checklist_id': 'quality-checklist',
        'metrics': X3_QUALITY_METRICS
    },
    'value': {
        'name': 'Value',
        'checklist_id': 'value-checklist',
        'metrics': X4_VALUE_METRICS
    },
    'size': {
        'name': 'Size',
        'checklist_id': 'size-checklist',
        'metrics': X5_SIZE_METRICS
    },
    'growth': {
        'name': 'Growth',
        'checklist_id': 'growth-checklist',
        'metrics': X6_GROWTH_METRICS
    },
    'profitability': {
        'name': 'Profitability',
        'checklist_id': 'profitability-checklist',
        'metrics': X7_PROFITABILITY_METRICS
    },
    'liquidity': {
        'name': 'Liquidity',
        'checklist_id': 'liquidity-checklist',
        'metrics': X8_LIQUIDITY_METRICS
    }
}

# Callbacks for Factor Selection tab
@app.callback(
    [Output('risk-checklist', 'style'),
     Output('momentum-checklist', 'style'),
     Output('quality-checklist', 'style'),
     Output('value-checklist', 'style'),
     Output('size-checklist', 'style'),
     Output('growth-checklist', 'style'),
     Output('profitability-checklist', 'style'),
     Output('liquidity-checklist', 'style')],
    [Input('factor-group-checklist', 'value')]
)
def toggle_factor_checklists(selected_groups):
    # Default style - hidden
    hidden_style = {'display': 'none', 'marginBottom': '15px'}
    # Visible style
    visible_style = {'marginBottom': '15px'}
    
    # If no groups selected, default to risk, momentum, quality
    if not selected_groups:
        selected_groups = ['risk', 'momentum', 'quality']
    
    # Set visibility for each group
    styles = []
    for group in ['risk', 'momentum', 'quality', 'value', 'size', 'growth', 'profitability', 'liquidity']:
        if group in selected_groups:
            styles.append(visible_style)
        else:
            styles.append(hidden_style)
            
    return styles

@app.callback(
    [Output('risk-checklist', 'value'),
     Output('momentum-checklist', 'value'),
     Output('quality-checklist', 'value'),
     Output('value-checklist', 'value'),
     Output('size-checklist', 'value'),
     Output('growth-checklist', 'value'),
     Output('profitability-checklist', 'value'),
     Output('liquidity-checklist', 'value')],
    [Input('factor-group-checklist', 'value')]
)
def reset_checklist_values(selected_groups):
    # Default empty value
    empty_value = []
    
    # If no groups selected, default to risk, momentum, quality
    if not selected_groups:
        selected_groups = ['risk', 'momentum', 'quality']
    
    # Default selections for the base groups
    defaults = {
        'risk': list(X1_RISK_METRICS.keys()),
        'momentum': list(X2_MOMENTUM_METRICS.keys()),
        'quality': list(X3_QUALITY_METRICS.keys()),
        'value': empty_value,
        'size': empty_value,
        'growth': empty_value,
        'profitability': empty_value,
        'liquidity': empty_value
    }
    
    # Return default values for each checklist based on selection
    values = []
    for group in ['risk', 'momentum', 'quality', 'value', 'size', 'growth', 'profitability', 'liquidity']:
        if group in selected_groups:
            values.append(defaults[group])
        else:
            values.append(empty_value)
            
    return values


# Callback for the Recalculate button to update regression
@app.callback(
    Output('reg-output', 'children'),
    [Input('recalculate-button', 'n_clicks')],
    [State('risk-checklist', 'value'),
     State('momentum-checklist', 'value'),
     State('quality-checklist', 'value'),
     State('value-checklist', 'value'),
     State('size-checklist', 'value'),
     State('growth-checklist', 'value'),
     State('profitability-checklist', 'value'),
     State('liquidity-checklist', 'value'),
     State('factor-group-checklist', 'value')]
)
def update_regression_output(n_clicks, risk_metrics, momentum_metrics, quality_metrics, 
                            value_metrics, size_metrics, growth_metrics, 
                            profitability_metrics, liquidity_metrics, selected_groups):
    import statsmodels.api as sm
    
    # If no groups selected, default to risk, momentum, quality
    if not selected_groups:
        selected_groups = ['risk', 'momentum', 'quality']
        
    # Check if all metrics are empty, then use defaults
    all_empty = not any([risk_metrics, momentum_metrics, quality_metrics, 
                         value_metrics, size_metrics, growth_metrics, 
                         profitability_metrics, liquidity_metrics])
    
    if all_empty:
        risk_metrics = list(X1_RISK_METRICS.keys()) if 'risk' in selected_groups else []
        momentum_metrics = list(X2_MOMENTUM_METRICS.keys()) if 'momentum' in selected_groups else []
        quality_metrics = list(X3_QUALITY_METRICS.keys()) if 'quality' in selected_groups else []
    
    # For the mock implementation, we'll use Risk_Score, Momentum_Score, and Quality_Score
    # as representatives for each factor group
    
    # Collect all selected metrics by group
    factor_columns = {}
    if 'risk' in selected_groups and risk_metrics:
        factor_columns['Risk'] = ['Risk_Score']  # In a real implementation, this would be risk_metrics
    if 'momentum' in selected_groups and momentum_metrics:
        factor_columns['Momentum'] = ['Momentum_Score']  # In a real implementation, this would be momentum_metrics
    if 'quality' in selected_groups and quality_metrics:
        factor_columns['Quality'] = ['Quality_Score']  # In a real implementation, this would be quality_metrics
    if 'value' in selected_groups and value_metrics:
        factor_columns['Value'] = ['Risk_Score']  # Placeholder - in a real implementation, this would be value_metrics
    if 'size' in selected_groups and size_metrics:
        factor_columns['Size'] = ['Momentum_Score']  # Placeholder - in a real implementation, this would be size_metrics  
    if 'growth' in selected_groups and growth_metrics:
        factor_columns['Growth'] = ['Quality_Score']  # Placeholder - in a real implementation, this would be growth_metrics
    if 'profitability' in selected_groups and profitability_metrics:
        factor_columns['Profitability'] = ['Risk_Score']  # Placeholder - in a real implementation, this would be profitability_metrics
    if 'liquidity' in selected_groups and liquidity_metrics:
        factor_columns['Liquidity'] = ['Momentum_Score']  # Placeholder - in a real implementation, this would be liquidity_metrics
    
    # If no metrics selected or all groups empty, use defaults
    if not factor_columns:
        factor_columns = {
            'Risk': ['Risk_Score'],
            'Momentum': ['Momentum_Score'],
            'Quality': ['Quality_Score']
        }
    
    # Flatten the list of columns
    selected_columns = []
    for group, columns in factor_columns.items():
        selected_columns.extend(columns)
    
    # Remove duplicates
    selected_columns = list(set(selected_columns))
    
    try:
        # Build X matrix with selected columns and add constant
        X = df[selected_columns].copy()
        X = sm.add_constant(X)  # Add constant for intercept
        
        # Define y as peRatio
        y = df['PE']
        
        # Run OLS regression
        model = sm.OLS(y, X).fit()
        
        # Create a custom summary to better show the selected factors
        result_parts = [
            f"OLS Regression Results for PE Ratio prediction",
            f"=====================================",
            f"",
            f"Selected Factor Groups:",
        ]
        
        for group, columns in factor_columns.items():
            metrics_list = locals()[f"{group.lower()}_metrics"]
            if metrics_list:
                result_parts.append(f"- {group}: {', '.join(metrics_list)}")
            
        result_parts.extend([
            f"",
            f"Regression Statistics:",
            f"R-squared: {model.rsquared:.4f}",
            f"Adjusted R-squared: {model.rsquared_adj:.4f}",
            f"F-statistic: {model.fvalue:.4f}",
            f"Prob (F-statistic): {model.f_pvalue:.4f}",
            f"",
            f"Coefficients:",
        ])
        
        for i, var_name in enumerate(model.params.index):
            result_parts.append(
                f"{var_name}: {model.params[i]:.4f} (p={model.pvalues[i]:.4f})"
            )
            
        # Add the full model summary at the end
        result_parts.extend([
            f"",
            f"Full Statistical Summary:",
            f"----------------------------",
            model.summary().as_text()
        ])
        
        return "\n".join(result_parts)
    except Exception as e:
        return f"Error in regression: {str(e)}\n\nPlease check that all selected metrics exist in the dataset."

@app.callback(
    [Output('sector-scatter-plot', 'figure'),
     Output('pe-comparison-plot', 'figure'),
     Output('individual-company-info', 'children'),
     Output('analysis-status', 'children')],
    [Input('analyze-button', 'n_clicks')],
    [State('ticker-input', 'value')]
)
def analyze_individual_stock(n_clicks, ticker):
    if n_clicks is None or not ticker:
        return {}, {}, None, ''
    
    # Initial status message
    status_message = f"Analyzing {ticker}..."
    
# Initialize regression results on page load - this runs when the page first loads
@app.callback(
    Output('reg-output', 'children'),
    [Input('factor-group-checklist', 'value')],
    [State('risk-checklist', 'value'),
     State('momentum-checklist', 'value'),
     State('quality-checklist', 'value')]
)
def initial_regression(selected_groups, risk_metrics, momentum_metrics, quality_metrics):
    import statsmodels.api as sm
    
    # If no groups selected, default to risk, momentum, quality
    if not selected_groups:
        selected_groups = ['risk', 'momentum', 'quality']
        
    # Collect default metrics by group
    selected_columns = []
    if 'risk' in selected_groups:
        selected_columns.append('Risk_Score')
    if 'momentum' in selected_groups:
        selected_columns.append('Momentum_Score')
    if 'quality' in selected_groups:
        selected_columns.append('Quality_Score')
    
    try:
        # Build X matrix with selected columns and add constant
        X = df[selected_columns].copy()
        X = sm.add_constant(X)  # Add constant for intercept
        
        # Define y as peRatio
        y = df['PE']
        
        # Run OLS regression
        model = sm.OLS(y, X).fit()
        
        # Return simplified initial view
        return f"""OLS Regression Results for PE Ratio prediction
====================================

Default factor groups are selected (Risk, Momentum, Quality).
Click "Recalculate Regression" after making selections to update the model.

R-squared: {model.rsquared:.4f}
Adjusted R-squared: {model.rsquared_adj:.4f}
"""
    except Exception as e:
        return f"Error in initial regression: {str(e)}"
    
    
# Continue with the main function
    try:
        # Load sector data and weights
        sector_df = pd.read_csv('sector_analysis_full.csv')
        weights_df = pd.read_csv('weights.csv')
        
        # Get all stock data at once
        stock_data = get_stock_data_with_retry(ticker)
        stock_info = stock_data['info']
        hist_data = stock_data['history']
        
        # Get the stock's sector
        stock_sector = stock_info.get('sector', '').lower().replace(' ', '-')
        if not stock_sector or stock_sector not in weights_df['Sector'].values:
            return {}, {}, None, f"Error: Could not determine sector for {ticker}"
        
        # Filter sector data
        sector_stocks = sector_df[sector_df['Sector'] == stock_sector].copy()
        
        # Calculate z-scores for the individual stock using sector data
        category_stats = {
            'Risk_Score': {'metrics': X1_RISK_METRICS, 'available': 0, 'total': len(X1_RISK_METRICS)},
            'Momentum_Score': {'metrics': X2_MOMENTUM_METRICS, 'available': 0, 'total': len(X2_MOMENTUM_METRICS)},
            'Quality_Score': {'metrics': X3_QUALITY_METRICS, 'available': 0, 'total': len(X3_QUALITY_METRICS)}
        }
        
        for metric_group, info in category_stats.items():
            metric_zscores = []
            for metric_name, yf_metric in info['metrics'].items():
                # First check if metric is directly available in stock info
                if yf_metric in stock_info and not pd.isna(stock_info[yf_metric]):
                    value = stock_info[yf_metric]
                    sector_values = sector_stocks[metric_name].dropna() if metric_name in sector_stocks.columns else pd.Series()
                    if not sector_values.empty and sector_values.std() != 0:
                        z = (value - sector_values.mean()) / sector_values.std()
                        metric_zscores.append(z)
                        info['available'] += 1
                # For custom metrics that need to be calculated
                elif metric_name == "ReturnSD" and not hist_data.empty:
                    try:
                        value = calculate_return_sd(hist_data['Close'])
                        sector_values = sector_stocks[metric_name].dropna() if metric_name in sector_stocks.columns else pd.Series()
                        if not sector_values.empty and sector_values.std() != 0:
                            z = (value - sector_values.mean()) / sector_values.std()
                            metric_zscores.append(z)
                            info['available'] += 1
                    except Exception as e:
                        print(f"Error calculating {metric_name}: {e}")
                        
                elif metric_name == "MaxDrawdown" and not hist_data.empty:
                    try:
                        value = calculate_max_drawdown(hist_data['Close'])
                        sector_values = sector_stocks[metric_name].dropna() if metric_name in sector_stocks.columns else pd.Series()
                        if not sector_values.empty and sector_values.std() != 0:
                            z = (value - sector_values.mean()) / sector_values.std()
                            metric_zscores.append(z)
                            info['available'] += 1
                    except Exception as e:
                        print(f"Error calculating {metric_name}: {e}")
                        
                elif metric_name == "RSI" and not hist_data.empty:
                    try:
                        value = calculate_rsi(hist_data['Close'])
                        sector_values = sector_stocks[metric_name].dropna() if metric_name in sector_stocks.columns else pd.Series()
                        if not sector_values.empty and sector_values.std() != 0:
                            z = (value - sector_values.mean()) / sector_values.std()
                            metric_zscores.append(z)
                            info['available'] += 1
                    except Exception as e:
                        print(f"Error calculating {metric_name}: {e}")
            
            # Calculate composite score if we have at least one valid metric
            if metric_zscores:
                stock_info[metric_group] = np.mean(metric_zscores)
            else:
                # If no metrics, assign a neutral score (0) instead of NaN
                stock_info[metric_group] = 0.0
                print(f"Warning: No valid metrics found for {metric_group}, using neutral score")
        
        # Get weights for the sector
        sector_weights = weights_df[weights_df['Sector'] == stock_sector].iloc[0]
        
        # Compute the composite z-score from the available category scores
        available_scores = []
        total_weight = 0
        
        for metric_group in ['Risk_Score', 'Momentum_Score', 'Quality_Score']:
            # Since we've set default values to 0, all scores should be available
            weight = sector_weights[metric_group] / 100
            available_scores.append(stock_info[metric_group] * weight)
            total_weight += weight
        
        # Normalize the composite z-score based on the available weights
        if total_weight > 0:
            composite_z_score = sum(available_scores) * (1 / total_weight)
        else:
            # Fallback to a simple average if weights are zero
            composite_z_score = np.mean([
                stock_info['Risk_Score'], 
                stock_info['Momentum_Score'], 
                stock_info['Quality_Score']
            ])
        
        # Compute composite z-scores for every ticker in the sector
        def calculate_composite_z_score(row):
            available_scores = []
            total_weight = 0
            for metric_group in ['Risk_Score', 'Momentum_Score', 'Quality_Score']:
                if not pd.isna(row[metric_group]):
                    weight = sector_weights[metric_group] / 100
                    available_scores.append(row[metric_group] * weight)
                    total_weight += weight
            return sum(available_scores) * (1 / total_weight) if available_scores and total_weight > 0 else np.nan

        sector_stocks['composite_z_score'] = sector_stocks.apply(calculate_composite_z_score, axis=1)
        
        # Drop rows whose composite z-score or P/E could not be computed
        sector_stocks = sector_stocks.dropna(subset=['composite_z_score', 'PE'])
        
        # Create visualization
        fig = go.Figure()
        
        # Add scatter points for sector
        x = sector_stocks['composite_z_score']
        y = sector_stocks['PE']
        
        # Calculate line of best fit
        fit = np.polyfit(x, y, 1)
        line_x = np.array([x.min(), x.max()])
        line_y = fit[0] * line_x + fit[1]
        
        # Calculate R-squared
        y_pred = fit[0] * x + fit[1]
        r_squared = np.corrcoef(x, y)[0, 1]**2
        
        # Calculate predicted PE
        predicted_pe = fit[0] * composite_z_score + fit[1]
        
        # Handle PE values properly
        actual_pe = stock_info.get('trailingPE', np.nan)
        if pd.isna(actual_pe):
            # Try to get PE from other possible keys
            for pe_key in ['trailingPE', 'forwardPE', 'pegRatio']:
                if pe_key in stock_info and not pd.isna(stock_info[pe_key]):
                    actual_pe = stock_info[pe_key]
                    break
                    
        # If still no PE value, use predicted PE
        if pd.isna(actual_pe):
            actual_pe = predicted_pe
            
        pe_deviation = actual_pe - predicted_pe
        
        # Add equation and R² annotation
        equation = f'y = {fit[0]:.2f}x + {fit[1]:.2f}'
        r2_text = f'R² = {r_squared:.3f}'
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref='paper',
            yref='paper',
            text=f'{equation}<br>{r2_text}',
            showarrow=False,
            font=dict(family=FONT_FAMILY, size=12),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=COLORS['border'],
            borderwidth=1,
            align='left'
        )
        
        # Add sector scatter points
        fig.add_trace(go.Scatter(
            x=sector_stocks['composite_z_score'],
            y=sector_stocks['PE'],
            mode='markers',
            name='Sector Stocks',
            text=sector_stocks['Ticker'],
            hovertemplate='%{text}<br>Fundamental Z-score: %{x:.2f}<br>P/E: %{y:.2f}',
            marker=dict(
                size=10,
                color=COLORS['primary'],
                opacity=0.5
            )
        ))
        
        # Add line of best fit
        fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode='lines',
            name='Sector Trend',
            line=dict(color=COLORS['secondary'], dash='dash')
        ))
        
        # Add individual stock point
        fig.add_trace(go.Scatter(
            x=[composite_z_score],
            y=[actual_pe],
            mode='markers',
            name=ticker,
            text=[ticker],
            hovertemplate='%{text}<br>Fundamental Z-score: %{x:.2f}<br>P/E: %{y:.2f}',
            marker=dict(
                size=15,
                color=COLORS['accent'],
                line=dict(width=2, color='white')
            )
        ))
        
        # Update scatter plot layout
        fig.update_layout(
            title={
                'text': f'{ticker} vs {GICS_SECTOR_MAPPING.get(stock_sector, stock_sector)} Sector',
                'font': {'family': FONT_FAMILY, 'size': 24}
            },
            xaxis=dict(
                title={'text': 'Fundamental Z-score', 'font': {'family': FONT_FAMILY}},
                showgrid=True,
                gridcolor=COLORS['light_gray'],
                gridwidth=1,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor=COLORS['border'],
                mirror=True,
                tickfont={'family': FONT_FAMILY}
            ),
            yaxis=dict(
                title={'text': 'P/E Ratio', 'font': {'family': FONT_FAMILY}},
                showgrid=True,
                gridcolor=COLORS['light_gray'],
                gridwidth=1,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor=COLORS['border'],
                mirror=True,
                tickfont={'family': FONT_FAMILY}
            ),
            showlegend=True,
            legend=dict(
                font=dict(family=FONT_FAMILY),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor=COLORS['border'],
                borderwidth=1
            ),
            hovermode='closest',
            font={'family': FONT_FAMILY},
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        # Create deviation chart (horizontal bar)
        deviation_fig = go.Figure()
        
        # Calculate PE range for visualization
        pe_min = min(actual_pe, predicted_pe)
        pe_max = max(actual_pe, predicted_pe)
        
        # Add the bar showing range from predicted to actual P/E
        deviation_fig.add_trace(go.Bar(
            x=[pe_max - pe_min],  # Length of the bar
            y=['P/E Range'],
            orientation='h',
            marker=dict(
                color=COLORS['accent'] if actual_pe < predicted_pe else COLORS['secondary']
            ),
            base=[pe_min],  # Start position of the bar
            text=[f"Actual: {actual_pe:.1f}"],
            textposition='outside',
            hoverinfo='text',
            hovertext=[f"Actual P/E: {actual_pe:.2f}<br>Predicted P/E: {predicted_pe:.2f}<br>Deviation: {pe_deviation:.2f}"]
        ))
        
        # Add vertical line at predicted value
        deviation_fig.add_vline(
            x=predicted_pe,
            line_width=2,
            line_dash="solid",
            line_color=COLORS['text'],
            annotation=dict(
                text=f"Predicted: {predicted_pe:.1f}",
                font=dict(
                    family=FONT_FAMILY,
                    size=12,
                    color=COLORS['text']
                ),
                yshift=10
            )
        )
        
        # Calculate the range for the x-axis
        sector_max_pe = sector_stocks['PE'].max()
        sector_min_pe = sector_stocks['PE'].min()
        pe_range = sector_max_pe - sector_min_pe
        x_min = max(0, sector_min_pe - pe_range * 0.1)
        x_max = sector_max_pe + pe_range * 0.1
        
        # Update deviation chart layout
        deviation_fig.update_layout(
            title=dict(
                text='P/E Ratio Analysis',
                font=dict(
                    family=FONT_FAMILY,
                    size=20,
                    color=COLORS['text']
                ),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title='P/E Ratio',
                range=[x_min, x_max],
                tickfont=dict(
                    family=FONT_FAMILY,
                    size=12
                ),
                gridcolor=COLORS['light_gray']
            ),
            yaxis=dict(
                showticklabels=False,
                fixedrange=True
            ),
            height=200,
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False
        )
        
        # Create info card with improved styling
        info_sections = [
            ('Company Information', [
                (f"Sector", f"{GICS_SECTOR_MAPPING.get(stock_sector, stock_sector)}"),
                ("Fundamental Z-score", f"{composite_z_score:.2f}"),
            ]),
            ('P/E Analysis', [
                ("Actual P/E", f"{actual_pe:.2f}"),
                ("Predicted P/E", f"{predicted_pe:.2f}"),
                ("P/E Deviation", f"{pe_deviation:.2f}"),
            ]),
            ('Category Scores', [
                ("Risk Score", f"{stock_info['Risk_Score']:.2f}"),
                ("Momentum Score", f"{stock_info['Momentum_Score']:.2f}"),
                ("Quality Score", f"{stock_info['Quality_Score']:.2f}"),
            ])
        ]
        
        info_card = html.Div([
            html.H3(f"{ticker} Analysis", style={
                'fontFamily': FONT_FAMILY,
                'fontSize': '24px',
                'fontWeight': '600',
                'color': COLORS['text'],
                'marginBottom': '24px',
                'borderBottom': f'2px solid {COLORS["light_gray"]}',
                'paddingBottom': '12px'
            })
        ] + [
            html.Div([
                html.H4(section_title, style={
                    'fontFamily': FONT_FAMILY,
                    'fontSize': '18px',
                    'fontWeight': '500',
                    'color': COLORS['text'],
                    'marginTop': '16px',
                    'marginBottom': '12px'
                }),
                html.Div([
                    html.Div([
                        html.Span(label, style={
                            'fontFamily': FONT_FAMILY,
                            'fontSize': '14px',
                            'color': COLORS['text'],
                            'fontWeight': '500'
                        }),
                        html.Span(": ", style={
                            'marginRight': '4px'
                        }),
                        html.Span(value, style={
                            'fontFamily': FONT_FAMILY,
                            'fontSize': '14px',
                            'color': COLORS['text']
                        })
                    ], style={
                        'marginBottom': '8px'
                    })
                    for label, value in items
                ])
            ])
            for section_title, items in info_sections
        ])
        
        # Create success message with metrics summary
        success_message = html.Div([
            html.Span(f"Analysis complete for ", style={'color': COLORS['text']}),
            html.Span(f"{ticker}", style={'fontWeight': 'bold', 'color': COLORS['primary']}),
            html.Div([
                html.Span("Sector: ", style={'fontWeight': 'bold'}),
                html.Span(f"{GICS_SECTOR_MAPPING.get(stock_sector, stock_sector)}"),
                html.Span(" | "),
                html.Span("Fundamental Z-score: ", style={'fontWeight': 'bold'}),
                html.Span(f"{composite_z_score:.2f}"),
            ], style={'marginTop': '8px', 'fontSize': '13px'})
        ])
        
        return fig, deviation_fig, info_card, success_message
        
    except Exception as e:
        import traceback
        error_message = html.Div([
            html.Span("Error analyzing ", style={'color': COLORS['text']}),
            html.Span(f"{ticker}", style={'fontWeight': 'bold', 'color': COLORS['secondary']}),
            html.Div(str(e), style={
                'marginTop': '8px', 
                'fontSize': '13px', 
                'color': COLORS['secondary'],
                'fontFamily': 'monospace',
                'maxHeight': '100px',
                'overflow': 'auto'
            })
        ])
        print(f"Error analyzing {ticker}: {str(e)}\n{traceback.format_exc()}")
        return {}, {}, None, error_message

if __name__ == '__main__':
    app.run_server(debug=False, port=os.getenv('PORT', 8050), host='0.0.0.0')
