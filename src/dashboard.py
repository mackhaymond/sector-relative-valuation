import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import Dash, dcc, html, Input, Output

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
    'accent': '#2ecc71'
}

# Create the layout
app.layout = html.Div([
    html.H1(
        'Stock Analysis Dashboard',
        style={
            'textAlign': 'center',
            'color': COLORS['text'],
            'marginBottom': 40,
            'fontFamily': 'Helvetica Neue, Arial, sans-serif',
            'fontSize': '2.5rem',
            'fontWeight': '600',
            'paddingTop': '20px'
        }
    ),
    
    html.Div([
        html.Label(
            'Select Sector:',
            style={
                'fontSize': '1.2rem',
                'marginRight': 15,
                'fontFamily': 'Helvetica Neue, Arial, sans-serif',
                'color': COLORS['text'],
                'fontWeight': '500'
            }
        ),
        dcc.Dropdown(
            id='sector-dropdown',
            options=[{'label': GICS_SECTOR_MAPPING[sector], 'value': sector} for sector in df['Sector'].unique()],
            value=df['Sector'].iloc[0],
            clearable=False,
            style={
                'width': '50%',
                'fontFamily': 'Helvetica Neue, Arial, sans-serif',
                'fontSize': '1rem'
            }
        )
    ], style={
        'marginBottom': 30,
        'padding': '20px',
        'backgroundColor': 'white',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    html.Div([
        html.Label(
            'Select Company:',
            style={
                'fontSize': '1.2rem',
                'marginRight': 15,
                'fontFamily': 'Helvetica Neue, Arial, sans-serif',
                'color': COLORS['text'],
                'fontWeight': '500'
            }
        ),
        dcc.Dropdown(
            id='company-dropdown',
            options=[],
            value=None,
            clearable=False,
            style={
                'width': '50%',
                'fontFamily': 'Helvetica Neue, Arial, sans-serif',
                'fontSize': '1rem'
            }
        )
    ], style={
        'marginBottom': 30,
        'padding': '20px',
        'backgroundColor': 'white',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    html.Div([
        dcc.Graph(id='scatter-plot')
    ], style={
        'backgroundColor': 'white',
        'padding': '20px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    html.Div(id='company-info', style={
        'marginTop': 30,
        'padding': '20px',
        'backgroundColor': 'white',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    })
], style={
    'padding': '40px',
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh'
})

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
    # Filter data for selected sector
    filtered_df = df[df['Sector'] == selected_sector]
    
    # Calculate line of best fit
    x = filtered_df['magic_score']
    y = filtered_df['PE']
    fit = np.polyfit(x, y, 1)
    line_x = np.array([x.min(), x.max()])
    line_y = fit[0] * line_x + fit[1]
    
    # Create scatter plot
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
            size=12,
            color=COLORS['primary'],
            line=dict(width=2, color='white'),
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
    
    # Highlight selected company's dot
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
                opacity=1.0
            )
        ))
        
        # Display company information
        company_info = html.Div([
            html.H3(f"Company: {company_data['Ticker']}"),
            html.P(f"P/E Ratio: {company_data['PE']:.2f}"),
            html.P(f"Magic Score: {company_data['magic_score']:.2f}"),
            html.Div([
                html.Label('P/E Ratio Difference:'),
                dcc.Graph(
                    figure=go.Figure(go.Bar(
                        x=[company_data['PE'] - (fit[0] * company_data['magic_score'] + fit[1])],
                        y=[''],
                        orientation='h',
                        marker=dict(
                            color=COLORS['secondary'] if company_data['PE'] > (fit[0] * company_data['magic_score'] + fit[1]) else COLORS['accent']
                        )
                    ))
                )
            ], style={'marginTop': 20})
        ])
    else:
        company_info = None
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'P/E Ratio vs Magic Score for {GICS_SECTOR_MAPPING[selected_sector]}',
            font=dict(
                family='Helvetica Neue, Arial, sans-serif',
                size=24,
                color=COLORS['text']
            ),
            x=0.5,
            y=0.95
        ),
        xaxis_title=dict(
            text='Magic Score',
            font=dict(
                family='Helvetica Neue, Arial, sans-serif',
                size=16,
                color=COLORS['text']
            )
        ),
        yaxis_title=dict(
            text='P/E Ratio',
            font=dict(
                family='Helvetica Neue, Arial, sans-serif',
                size=16,
                color=COLORS['text']
            )
        ),
        hovermode='closest',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            font=dict(
                family='Helvetica Neue, Arial, sans-serif',
                size=14,
                color=COLORS['text']
            )
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(t=100, b=80, l=60, r=40)
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#f0f0f0',
        zeroline=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#f0f0f0',
        zeroline=False
    )
    
    return fig, company_info

if __name__ == '__main__':
    app.run_server(debug=True)
