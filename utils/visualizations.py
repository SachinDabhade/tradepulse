import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_market_heatmap(data, title="Market Heatmap"):
    """
    Create a market performance heatmap
    
    Parameters:
    data: dict or DataFrame with sector/asset performance data
    title: str, chart title
    
    Returns:
    plotly figure
    """
    if isinstance(data, dict):
        sectors = list(data.keys())
        values = list(data.values())
    else:
        sectors = data.index.tolist()
        values = data.values.tolist()
    
    # Create color scale based on performance
    colors = ['red' if v < 0 else 'green' for v in values]
    
    fig = go.Figure(data=go.Heatmap(
        z=[values],
        x=sectors,
        y=['Performance'],
        colorscale='RdYlGn',
        zmid=0,
        text=[f"{v:.1f}%" for v in values],
        texttemplate="%{text}",
        textfont={"size": 12},
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=200,
        xaxis={'tickangle': 45}
    )
    
    return fig

def create_performance_chart(data, title="Performance Chart", show_drawdown=False):
    """
    Create a performance chart with optional drawdown
    
    Parameters:
    data: pandas Series or DataFrame with date index and performance data
    title: str, chart title
    show_drawdown: bool, whether to show drawdown subplot
    
    Returns:
    plotly figure
    """
    if isinstance(data, pd.Series):
        data = data.to_frame('Performance')
    
    fig = go.Figure()
    
    # Add performance lines
    for column in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[column],
            mode='lines',
            name=column,
            line=dict(width=2)
        ))
    
    if show_drawdown and len(data.columns) == 1:
        # Calculate drawdown
        cumulative = (1 + data.iloc[:, 0].pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        # Create subplot
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Performance', 'Drawdown'),
            vertical_spacing=0.05
        )
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.iloc[:, 0],
            mode='lines',
            name='Performance',
            line=dict(width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=drawdown,
            mode='lines',
            name='Drawdown',
            fill='tonegative',
            line=dict(color='red', width=1)
        ), row=2, col=1)
        
        fig.update_yaxes(title_text="Return", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=400,
        xaxis_title="Date",
        yaxis_title="Value"
    )
    
    return fig

def create_correlation_matrix(data, title="Correlation Matrix"):
    """
    Create a correlation matrix heatmap
    
    Parameters:
    data: pandas DataFrame
    title: str, chart title
    
    Returns:
    plotly figure
    """
    corr_matrix = data.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=500,
        width=500
    )
    
    return fig

def create_risk_return_scatter(returns, volatilities, labels=None, title="Risk-Return Analysis"):
    """
    Create a risk-return scatter plot
    
    Parameters:
    returns: array-like, expected returns
    volatilities: array-like, volatilities (risk)
    labels: array-like, optional labels for points
    title: str, chart title
    
    Returns:
    plotly figure
    """
    fig = go.Figure()
    
    if labels is None:
        labels = [f"Asset {i+1}" for i in range(len(returns))]
    
    fig.add_trace(go.Scatter(
        x=volatilities,
        y=returns,
        mode='markers+text',
        text=labels,
        textposition='top center',
        marker=dict(
            size=12,
            color=returns,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Return")
        ),
        name='Assets'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Volatility (%)",
        yaxis_title="Expected Return (%)",
        template="plotly_dark",
        height=500
    )
    
    return fig

def create_efficient_frontier(returns, risk_free_rate=0.02):
    """
    Create an efficient frontier plot
    
    Parameters:
    returns: pandas DataFrame of asset returns
    risk_free_rate: float, risk-free rate
    
    Returns:
    plotly figure
    """
    from utils.calculations import calculate_portfolio_metrics
    
    # Generate points on the efficient frontier
    target_returns = np.linspace(returns.mean().min() * 252, returns.mean().max() * 252, 50)
    efficient_portfolios = []
    
    for target_return in target_returns:
        try:
            # This is a simplified version - would need proper optimization
            weights = np.random.dirichlet(np.ones(len(returns.columns)))
            metrics = calculate_portfolio_metrics(weights, returns)
            efficient_portfolios.append({
                'return': metrics['expected_return'],
                'volatility': metrics['volatility']
            })
        except:
            continue
    
    if efficient_portfolios:
        frontier_df = pd.DataFrame(efficient_portfolios)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=frontier_df['volatility'],
            y=frontier_df['return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=3)
        ))
        
        # Add capital allocation line
        max_sharpe_idx = ((frontier_df['return'] - risk_free_rate) / frontier_df['volatility']).idxmax()
        if not pd.isna(max_sharpe_idx):
            optimal_vol = frontier_df.loc[max_sharpe_idx, 'volatility']
            optimal_ret = frontier_df.loc[max_sharpe_idx, 'return']
            
            # Capital allocation line
            cal_vols = np.linspace(0, optimal_vol * 1.5, 50)
            cal_rets = risk_free_rate + (optimal_ret - risk_free_rate) / optimal_vol * cal_vols
            
            fig.add_trace(go.Scatter(
                x=cal_vols,
                y=cal_rets,
                mode='lines',
                name='Capital Allocation Line',
                line=dict(color='red', dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=[optimal_vol],
                y=[optimal_ret],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(color='red', size=12, symbol='star')
            ))
    else:
        fig = go.Figure()
        fig.add_annotation(
            text="Unable to generate efficient frontier",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility",
        yaxis_title="Expected Return",
        template="plotly_dark",
        height=500
    )
    
    return fig

def create_factor_exposure_chart(factor_loadings, title="Factor Exposures"):
    """
    Create a factor exposure radar chart
    
    Parameters:
    factor_loadings: dict, factor loadings
    title: str, chart title
    
    Returns:
    plotly figure
    """
    factors = list(factor_loadings.keys())
    values = list(factor_loadings.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=factors,
        fill='toself',
        name='Factor Exposure',
        line_color='#FF6B35'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1]
            )),
        showlegend=True,
        title=title,
        template="plotly_dark"
    )
    
    return fig

def create_attribution_waterfall(attribution_data, title="Performance Attribution"):
    """
    Create a waterfall chart for performance attribution
    
    Parameters:
    attribution_data: pandas DataFrame with attribution effects
    title: str, chart title
    
    Returns:
    plotly figure
    """
    categories = attribution_data.index.tolist()
    values = attribution_data.values.tolist()
    
    fig = go.Figure(go.Waterfall(
        name="Attribution",
        orientation="v",
        measure=["relative"] * len(categories) + ["total"],
        x=categories + ["Total"],
        textposition="outside",
        text=[f"{x:.1f}%" for x in values] + [f"{sum(values):.1f}%"],
        y=values + [sum(values)],
        connector={"line":{"color":"rgb(63, 63, 63)"}},
        decreasing={"marker":{"color":"red"}},
        increasing={"marker":{"color":"green"}},
        totals={"marker":{"color":"blue"}}
    ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=500
    )
    
    return fig

def create_rolling_metrics_chart(data, metrics=['return', 'volatility', 'sharpe'], title="Rolling Metrics"):
    """
    Create a chart showing rolling metrics over time
    
    Parameters:
    data: pandas DataFrame with rolling metrics
    metrics: list, metrics to plot
    title: str, chart title
    
    Returns:
    plotly figure
    """
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, metric in enumerate(metrics):
        if metric in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[metric],
                mode='lines',
                name=metric.title(),
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=400,
        xaxis_title="Date",
        yaxis_title="Value"
    )
    
    return fig

def create_sector_allocation_pie(allocations, title="Sector Allocation"):
    """
    Create a pie chart for sector allocation
    
    Parameters:
    allocations: dict or Series, sector allocations
    title: str, chart title
    
    Returns:
    plotly figure
    """
    if isinstance(allocations, dict):
        sectors = list(allocations.keys())
        values = list(allocations.values())
    else:
        sectors = allocations.index.tolist()
        values = allocations.values.tolist()
    
    fig = go.Figure(data=[go.Pie(
        labels=sectors,
        values=values,
        textinfo='label+percent',
        textposition='auto'
    )])
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=500
    )
    
    return fig

def create_var_chart(var_data, confidence_levels=[0.95, 0.99], title="Value at Risk"):
    """
    Create a VaR visualization
    
    Parameters:
    var_data: pandas Series of returns
    confidence_levels: list, confidence levels for VaR calculation
    title: str, chart title
    
    Returns:
    plotly figure
    """
    fig = go.Figure()
    
    # Add histogram of returns
    fig.add_trace(go.Histogram(
        x=var_data * 100,
        nbinsx=50,
        name='Return Distribution',
        opacity=0.7
    ))
    
    # Add VaR lines
    colors = ['red', 'darkred']
    for i, conf_level in enumerate(confidence_levels):
        var_value = np.percentile(var_data, (1 - conf_level) * 100) * 100
        fig.add_vline(
            x=var_value,
            line_dash="dash",
            line_color=colors[i],
            annotation_text=f"VaR {conf_level*100:.0f}%: {var_value:.2f}%"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Returns (%)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=400
    )
    
    return fig

def create_price_chart_with_indicators(price_data, indicators=None, title="Price Chart"):
    """
    Create a price chart with technical indicators
    
    Parameters:
    price_data: pandas DataFrame with OHLCV data
    indicators: dict, technical indicators to overlay
    title: str, chart title
    
    Returns:
    plotly figure
    """
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['Close'] if 'Close' in price_data.columns else price_data.iloc[:, 0],
        mode='lines',
        name='Price',
        line=dict(color='white', width=2)
    ))
    
    # Add indicators if provided
    if indicators:
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (name, values) in enumerate(indicators.items()):
            if isinstance(values, pd.Series):
                fig.add_trace(go.Scatter(
                    x=values.index,
                    y=values,
                    mode='lines',
                    name=name,
                    line=dict(color=colors[i % len(colors)], width=1, dash='dash')
                ))
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=500,
        xaxis_title="Date",
        yaxis_title="Price"
    )
    
    return fig

def create_volatility_surface(strikes, expiries, implied_vols, title="Volatility Surface"):
    """
    Create a 3D volatility surface
    
    Parameters:
    strikes: array-like, strike prices
    expiries: array-like, expiration dates
    implied_vols: 2D array, implied volatilities
    title: str, chart title
    
    Returns:
    plotly figure
    """
    fig = go.Figure(data=go.Surface(
        x=strikes,
        y=expiries,
        z=implied_vols,
        colorscale='Viridis',
        colorbar=dict(title="Implied Vol (%)")
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Days to Expiry",
            zaxis_title="Implied Vol (%)"
        ),
        template="plotly_dark",
        height=600
    )
    
    return fig
