import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import Dash, dcc, html, Input, Output
import os

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
        'minHeight': '100vh'
    },
    'card': {
        'backgroundColor': 'white',
        'padding': '24px',
        'borderRadius': '12px',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
        'marginBottom': '24px'
    },
    'title': {
        'fontFamily': FONT_FAMILY,
        'fontSize': '32px',
        'fontWeight': '600',
        'color': COLORS['text'],
        'marginBottom': '32px',
        'textAlign': 'center'
    },
    'label': {
        'fontFamily': FONT_FAMILY,
        'fontSize': '16px',
        'fontWeight': '500',
        'color': COLORS['text'],
        'marginBottom': '8px',
        'display': 'block'
    },
    'dropdown': {
        'width': '300px',  # Fixed width instead of 100%
        'fontFamily': FONT_FAMILY,
        'fontSize': '14px'
    },
    'dropdown-container': {
        'display': 'flex',
        'justifyContent': 'center',
        'gap': '24px',
        'flexWrap': 'wrap'
    }
}

# Create the layout
app.layout = html.Div([
    html.H1('Stock Analysis Dashboard', style=STYLES['title']),
    
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
            title='Magic Formula Score',
            titlefont=dict(
                family=FONT_FAMILY,
                size=16,
                color=COLORS['text']
            ),
            tickfont=dict(
                family=FONT_FAMILY,
                size=14
            ),
            showgrid=True,
            gridcolor=COLORS['light_gray'],
            gridwidth=1,
            zeroline=False
        ),
        yaxis=dict(
            title='P/E Ratio',
            titlefont=dict(
                family=FONT_FAMILY,
                size=16,
                color=COLORS['text']
            ),
            tickfont=dict(
                family=FONT_FAMILY,
                size=14
            ),
            showgrid=True,
            gridcolor=COLORS['light_gray'],
            gridwidth=1,
            zeroline=False,
            range=[y_min, y_max]
        ),
        legend=dict(
            font=dict(
                family=FONT_FAMILY,
                size=14,
                color=COLORS['text']
            ),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor=COLORS['border'],
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='closest',
        margin=dict(t=100, b=60, l=60, r=40),
        showlegend=True
    )
    
    return fig, company_info

if __name__ == '__main__':
    app.run_server(debug=False, port=os.getenv('PORT', 8050), host='0.0.0.0')
