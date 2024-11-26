import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import os
import yfinance as yf
from scipy.stats import zscore
from plotly.subplots import make_subplots

# Import constants from data.py
from data import (
    X1_RISK_METRICS,
    X2_GROWTH_METRICS,
    X3_QUALITY_METRICS,
    Y_VALUATION_METRIC,
    ALL_METRICS
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
                dcc.Graph(id='scatter-plot')
            ], style=STYLES['card']),
            
            html.Div(id='company-info', style=STYLES['card'])
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
                html.Div(id='analysis-status', style={
                    'textAlign': 'center',
                    'fontFamily': FONT_FAMILY,
                    'fontSize': '14px',
                    'color': COLORS['text'],
                    'marginTop': '8px'
                })
            ], style=STYLES['card']),
            
            html.Div([
                dcc.Graph(id='sector-scatter-plot')
            ], style=STYLES['card']),
            
            html.Div([
                dcc.Graph(id='pe-comparison-plot')
            ], style=STYLES['card']),
            
            html.Div(id='individual-company-info', style=STYLES['card'])
        ])
    ])
], style=STYLES['container'])

@app.callback(
    Output('company-dropdown', 'options'),
    Input('sector-dropdown', 'value')
)
def update_company_dropdown(selected_sector):
    filtered_df = df[df['Sector'] == selected_sector]
    return [{'label': row['Ticker'], 'value': row['Ticker']} for _, row in filtered_df.iterrows()]

@app.callback(
    Output('scatter-plot', 'figure'),
    Output('company-info', 'children'),
    Input('sector-dropdown', 'value'),
    Input('company-dropdown', 'value')
)
def update_graph(selected_sector, selected_company):
    # Create a copy of the filtered dataframe to avoid SettingWithCopyWarning
    filtered_df = df[df['Sector'] == selected_sector].copy()
    
    x = filtered_df['magic_score']
    y = filtered_df['PE']
    fit = np.polyfit(x, y, 1)
    line_x = np.array([x.min(), x.max()])
    line_y = fit[0] * line_x + fit[1]
    
    # Calculate predicted P/E values and deviations using .loc
    filtered_df.loc[:, 'predicted_pe'] = fit[0] * filtered_df['magic_score'] + fit[1]
    filtered_df.loc[:, 'pe_deviation'] = filtered_df['PE'] - filtered_df['predicted_pe']
    
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
        x=filtered_df['magic_score'],
        y=filtered_df['PE'],
        mode='markers',
        name='Stocks',
        text=filtered_df.apply(
            lambda row: f"Ticker: {row['Ticker']}<br>P/E: {row['PE']:.2f}<br>Magic Score: {row['magic_score']:.2f}",
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
    
    # Highlight selected company
    if selected_company:
        company_data = filtered_df[filtered_df['Ticker'] == selected_company].iloc[0]
        fig.add_trace(go.Scatter(
            x=[company_data['magic_score']],
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
                html.P(f"Magic Score: {company_data['magic_score']:.2f}", style={
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
    else:
        company_info = None

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Magic Formula Score vs P/E Ratio - {GICS_SECTOR_MAPPING[selected_sector]}',
            font=dict(
                family=FONT_FAMILY,
                size=24,
                color=COLORS['text']
            ),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title={'text': 'Magic Formula Score', 'font': {'family': FONT_FAMILY}},
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
    
    try:
        # Load sector data and weights
        sector_df = pd.read_csv('sector_analysis_full.csv')
        weights_df = pd.read_csv('weights.csv')
        
        # Get stock data
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        
        # Create DataFrame for the individual stock
        stock_data = {'Ticker': ticker}
        
        # Get all metrics
        for metric_name, yf_metric in ALL_METRICS.items():
            stock_data[metric_name] = stock_info.get(yf_metric, np.nan)
            
        # Print which stats were found and which are NaN
        print(f"\n=== Stats Analysis for {ticker} ===")
        print("Found stats:")
        found_stats = [f"{metric}: {stock_data[metric]}" for metric in stock_data if not pd.isna(stock_data[metric]) and metric != 'Ticker']
        for stat in found_stats:
            print(f"  {stat}")
        print("\nMissing stats (NaN):")
        missing_stats = [metric for metric in stock_data if pd.isna(stock_data[metric]) and metric != 'Ticker']
        for stat in missing_stats:
            print(f"  {stat}")
        print("=" * 40 + "\n")
        
        # Get the stock's sector
        stock_sector = stock_info.get('sector', '').lower().replace(' ', '-')
        if not stock_sector or stock_sector not in weights_df['Sector'].values:
            return {}, {}, None, f"Error: Could not determine sector for {ticker}"
        
        # Filter sector data
        sector_stocks = sector_df[sector_df['Sector'] == stock_sector].copy()
        
        # Calculate z-scores for the individual stock using sector data
        for metric_group, metrics in [
            ('Risk_Score', X1_RISK_METRICS),
            ('Growth_Score', X2_GROWTH_METRICS),
            ('Quality_Score', X3_QUALITY_METRICS)
        ]:
            # Calculate z-scores for each metric in the group
            metric_zscores = []
            for metric in metrics:
                if metric in stock_data:
                    sector_values = sector_stocks[metric].dropna()
                    if not sector_values.empty:
                        z = (stock_data[metric] - sector_values.mean()) / sector_values.std()
                        metric_zscores.append(z)
            
            # Calculate composite score for the group
            if metric_zscores:
                stock_data[metric_group] = np.mean(metric_zscores)
            else:
                stock_data[metric_group] = np.nan
        
        # Get weights for the sector
        sector_weights = weights_df[weights_df['Sector'] == stock_sector].iloc[0]
        
        # Calculate magic score for the individual stock
        magic_score = (
            stock_data['Risk_Score'] * sector_weights['Risk_Score'] / 100 +
            stock_data['Growth_Score'] * sector_weights['Growth_Score'] / 100 +
            stock_data['Quality_Score'] * sector_weights['Quality_Score'] / 100
        )
        
        # Calculate magic scores for sector stocks
        sector_stocks['magic_score'] = (
            sector_stocks['Risk_Score'] * sector_weights['Risk_Score'] / 100 +
            sector_stocks['Growth_Score'] * sector_weights['Growth_Score'] / 100 +
            sector_stocks['Quality_Score'] * sector_weights['Quality_Score'] / 100
        )
        
        # Remove any rows with NaN magic scores
        sector_stocks = sector_stocks.dropna(subset=['magic_score', 'PE'])
        
        # Create visualization
        fig = go.Figure()
        
        # Add scatter points for sector
        x = sector_stocks['magic_score']
        y = sector_stocks['PE']
        
        # Calculate line of best fit
        fit = np.polyfit(x, y, 1)
        line_x = np.array([x.min(), x.max()])
        line_y = fit[0] * line_x + fit[1]
        
        # Calculate predicted PE
        predicted_pe = fit[0] * magic_score + fit[1]
        actual_pe = stock_data.get('PE', np.nan)
        pe_deviation = actual_pe - predicted_pe
        
        # Add sector scatter points
        fig.add_trace(go.Scatter(
            x=sector_stocks['magic_score'],
            y=sector_stocks['PE'],
            mode='markers',
            name='Sector Stocks',
            text=sector_stocks['Ticker'],
            hovertemplate='%{text}<br>Magic Score: %{x:.2f}<br>P/E: %{y:.2f}',
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
            x=[magic_score],
            y=[actual_pe],
            mode='markers',
            name=ticker,
            text=[ticker],
            hovertemplate='%{text}<br>Magic Score: %{x:.2f}<br>P/E: %{y:.2f}',
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
                title={'text': 'Magic Score', 'font': {'family': FONT_FAMILY}},
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
                ("Magic Score", f"{magic_score:.2f}"),
            ]),
            ('P/E Analysis', [
                ("Actual P/E", f"{actual_pe:.2f}"),
                ("Predicted P/E", f"{predicted_pe:.2f}"),
                ("P/E Deviation", f"{pe_deviation:.2f}"),
            ]),
            ('Category Scores', [
                ("Risk Score", f"{stock_data['Risk_Score']:.2f}"),
                ("Growth Score", f"{stock_data['Growth_Score']:.2f}"),
                ("Quality Score", f"{stock_data['Quality_Score']:.2f}"),
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
        
        return fig, deviation_fig, info_card, f"Analysis complete for {ticker}"
        
    except Exception as e:
        import traceback
        return {}, {}, None, f"Error analyzing {ticker}: {str(e)}\n{traceback.format_exc()}"

if __name__ == '__main__':
    app.run_server(debug=False, port=os.getenv('PORT', 8050), host='0.0.0.0')
