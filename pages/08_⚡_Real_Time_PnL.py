import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from Config.config import config
from Execution.dhan_execution_engine import DhanExecutionEngine

dhan_user = DhanExecutionEngine()

st.set_page_config(page_title="Real-time P&L", page_icon="⚡", layout="wide")

st.title("⚡ Real-time P&L Tracking")
# st.markdown("Live portfolio P&L monitoring with Greeks and risk metrics")

# # Auto-refresh toggle
# auto_refresh = st.sidebar.checkbox("Auto Refresh (5s)", value=True)
# if auto_refresh:
#     time.sleep(5)
#     st.rerun()

st.markdown('---')
# Real-time P&L metrics
st.header("📊 Live P&L Dashboard")

col1, col2, col3, col4, col5 = st.columns(5)

# Generate real-time P&L data
current_pnl = np.random.normal(287000, 25000)
daily_change = np.random.normal(0.023, 0.005)
intraday_pnl = np.random.normal(15000, 8000)

with col1:
    st.metric("Portfolio Value", f"${125.5 + current_pnl/1000000:.1f}M", f"{daily_change:.2%}")
with col2:
    st.metric("Total P&L", f"${current_pnl:,.0f}", f"+${intraday_pnl:,.0f}")
with col3:
    st.metric("Realized P&L", f"${current_pnl * 0.6:,.0f}", "+$8,450")
with col4:
    st.metric("Unrealized P&L", f"${current_pnl * 0.4:,.0f}", "+$6,550")
with col5:
    st.metric("Daily Sharpe", "2.14", "+0.08")

# Live status indicators
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        market_status = config.getboolean('MARKET', 'STATUS')
        market_symbol = "🟢 MARKET OPEN" if market_status else "🔴 MARKET CLOSED"
        st.success(f"**{market_symbol}**")
    
    with col2:
        st.info(f"**Last Update: {datetime.now().strftime('%H:%M:%S')}**")
    
    with col3:
        risk_status = "🟢 RISK OK" if abs(daily_change) < 0.03 else "🟡 ELEVATED RISK" if abs(daily_change) < 0.05 else "🔴 HIGH RISK"
        st.warning(f"**{risk_status}**")


# Simulate realistic intraday P&L path
returns = np.random.normal(0, 0.0008, 390)
returns[0] = 0  # Start at zero
cumulative_pnl = np.cumsum(returns) * 10000000  # Scale to portfolio size

fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=market_hours,
#     y=cumulative_pnl,
#     mode='lines',
#     name='Cumulative P&L',
#     line=dict(color='#FF6B35', width=3),
#     fill='tonegative'
# ))

# Add key market events
# fig.add_vline(x=market_hours[60], line_dash="dash", line_color="yellow", 
#              annotation_text="Fed Announcement")
# fig.add_vline(x=market_hours[180], line_dash="dash", line_color="orange", 
#              annotation_text="Earnings Release")

# fig.update_layout(
#     title="Intraday P&L Performance",
#     xaxis_title="Time",
#     yaxis_title="P&L ($)",
#     template="plotly_dark",
#     height=400
# )

# st.plotly_chart(fig, use_container_width=True)

st.header("🎯 Holoding-Level P&L")

    # print(dhan_user.get_order_list())
status, order_list = dhan_user.check_status(dhan_user.get_holdings())

if status:
    st.dataframe(pd.DataFrame(order_list))
else:
    st.error(order_list)

# Position P&L table
position_pnl = pd.DataFrame({
    'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'CRM', 'AMD'],
    'Position': [15000, 12500, -8000, 18000, -5000, 10000, 8500, -3000, 6000, 12000],
    'Market Value': [2625000, 5187500, -1110000, 15750000, -1225000, 1452500, 4127500, -1275000, 1716000, 1980000],
    'Avg Cost': [172.50, 412.25, 140.75, 865.30, 248.80, 142.45, 478.60, 420.15, 282.90, 158.45],
    'Current Price': [175.00, 415.00, 138.75, 875.00, 245.00, 145.25, 485.60, 425.00, 286.00, 165.00],
    'Unrealized P&L': [37500, 34375, 16000, 174600, 19000, 28000, 59500, 14550, 18600, 78600],
    'Realized P&L': [12500, 8750, -5200, 25800, -8500, 4200, 12750, -2800, 3400, 15200],
    'Total P&L': [50000, 43125, 10800, 200400, 10500, 32200, 72250, 11750, 22000, 93800],
    'Day P&L (%)': [2.17, 0.67, 1.44, 1.24, 1.22, 1.93, 1.45, 1.15, 1.09, 3.92]
})

# Color code P&L
def color_pnl(val):
    if val > 50000:
        return 'background-color: #1B5E20'
    elif val > 0:
        return 'background-color: #2E7D32'
    elif val > -20000:
        return 'background-color: #FFA000'
    else:
        return 'background-color: #C62828'

styled_positions = position_pnl.style.applymap(color_pnl, subset=['Total P&L'])
st.dataframe(styled_positions, use_container_width=True)

# P&L attribution charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top P&L Contributors")
    
    top_contributors = position_pnl.nlargest(5, 'Total P&L')
    
    fig = px.bar(
        top_contributors,
        x='Symbol',
        y='Total P&L',
        title="Top 5 P&L Contributors",
        color='Total P&L',
        color_continuous_scale='Greens'
    )
    
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Position Sizes")
    
    fig = px.scatter(
        position_pnl,
        x='Market Value',
        y='Day P&L (%)',
        size=abs(position_pnl['Position']),
        text='Symbol',
        title="Position Size vs Daily P&L"
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True)


st.header("📊 Positional Dashboard")

status, order_list = dhan_user.check_status(dhan_user.get_positions())

if status:
    st.dataframe(pd.DataFrame(order_list))
else:
    st.error(order_list)

options_positions = pd.DataFrame({
    'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY'],
    'Strategy': ['Long Call', 'Short Put', 'Iron Condor', 'Long Straddle', 'Covered Call', 'Protective Put'],
    'Quantity': [50, -25, 10, 15, -30, 100],
    'Delta': [32.5, 18.7, -2.1, 0.5, -12.8, 45.2],
    'Gamma': [4.2, 3.8, 0.8, 8.5, 2.1, 2.9],
    'Theta': [-125, 85, -15, -285, 95, -45],
    'Vega': [485, -285, 125, 785, -195, 285],
    'Current P&L': [2450, 1250, -185, -850, 650, 485],
    'Max Risk': [2500, 6250, 850, 4250, np.inf, 4850]
})

st.dataframe(options_positions, use_container_width=True)

# Greeks visualization
col1, col2 = st.columns(2)

with col1:
    st.subheader("Delta Exposure by Underlying")
    
    fig = px.bar(
        options_positions,
        x='Symbol',
        y='Delta',
        color='Delta',
        title="Delta Exposure",
        color_continuous_scale='RdYlGn'
    )
    
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Greeks Correlation Matrix")
    
    greeks_corr = np.array([
        [1.0, 0.65, -0.45, 0.25],
        [0.65, 1.0, -0.35, 0.15],
        [-0.45, -0.35, 1.0, -0.55],
        [0.25, 0.15, -0.55, 1.0]
    ])
    
    fig = go.Figure(data=go.Heatmap(
        z=greeks_corr,
        x=['Delta', 'Gamma', 'Theta', 'Vega'],
        y=['Delta', 'Gamma', 'Theta', 'Vega'],
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title="Greeks Correlation",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


st.header("⚡ Real-time Risk Alerts")


# print(dhan_user.get_order_list())
status, order_list = dhan_user.check_status(dhan_user.get_order_list())

if status:
    st.dataframe(pd.DataFrame(order_list))
else:
    st.error(order_list)

# Risk metrics monitoring
col1, col2 = st.columns(2)

with col1:
    st.subheader("Risk Limit Monitoring")
    
    risk_limits = pd.DataFrame({
        'Metric': ['VaR (1D)', 'Position Limit', 'Sector Concentration', 'Beta', 'Leverage'],
        'Current': ['$275K', '8.5%', '24.8%', '1.15', '1.05x'],
        'Limit': ['$500K', '10.0%', '25.0%', '1.5', '1.2x'],
        'Utilization (%)': [55.0, 85.0, 99.2, 76.7, 87.5],
        'Status': ['OK', 'WARNING', 'BREACH', 'OK', 'OK']
    })
    
    # Color code status
    def color_risk_status(val):
        if val == 'BREACH':
            return 'background-color: #C62828'
        elif val == 'WARNING':
            return 'background-color: #FFA000'
        else:
            return 'background-color: #1B5E20'
    
    styled_risk = risk_limits.style.applymap(color_risk_status, subset=['Status'])
    st.dataframe(styled_risk, use_container_width=True)

with col2:
    st.subheader("Alert History")
    
    alert_history = pd.DataFrame({
        'Time': ['09:45', '10:15', '11:30', '13:20', '14:45'],
        'Type': ['POSITION', 'GREEK', 'P&L', 'RISK', 'POSITION'],
        'Level': ['HIGH', 'MEDIUM', 'LOW', 'HIGH', 'MEDIUM'],
        'Resolved': ['Yes', 'Yes', 'No', 'Yes', 'Yes']
    })
    
    st.dataframe(alert_history, use_container_width=True)


# Display live fund limits from Dhan API (mocked for this example)
status, dhan_funds = dhan_user.check_status(dhan_user.get_fund_limits())
if status and isinstance(dhan_funds, dict):
    st.subheader("💰 Dhan Fund Data (Live)")
    # Pick some key metrics to show as st.metric
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric(
            "Available Balance",
            f"₹{dhan_funds.get('availabelBalance', 0):,.2f}"
            if isinstance(dhan_funds.get('availabelBalance', 0), (int, float))
            else 'Fetching Error...!'
        )
    with col2:
        st.metric(
            "SOD Limit",
            f"₹{dhan_funds.get('sodLimit', 0):,.2f}"
            if isinstance(dhan_funds.get('sodLimit', 0), (int, float))
            else 'Fetching Error...!'
        )
    with col3:
        st.metric(
            "Collateral Amount",
            f"₹{dhan_funds.get('collateralAmount', 0):,.2f}"
            if isinstance(dhan_funds.get('collateralAmount', 0), (int, float))
            else 'Fetching Error...!'
        )
    with col4:
        st.metric(
            "Receivable Amount",
            f"₹{dhan_funds.get('receiveableAmount', 0):,.2f}"
            if isinstance(dhan_funds.get('receiveableAmount', 0), (int, float))
            else 'Fetching Error...!'
        )
    with col5:
        st.metric(
            "Utilized Amount",
            f"₹{dhan_funds.get('utilizedAmount', 0):,.2f}"
            if isinstance(dhan_funds.get('utilizedAmount', 0), (int, float))
            else 'Fetching Error...!'
        )
    with col6:
        st.metric(
            "Blocked Payout Amount",
            f"₹{dhan_funds.get('blockedPayoutAmount', 0):,.2f}"
            if isinstance(dhan_funds.get('blockedPayoutAmount', 0), (int, float))
            else 'Fetching Error...!'
        )
    with col7:
        st.metric(
            "Withdrawable Balance",
            f"₹{dhan_funds.get('withdrawableBalance', 0):,.2f}"
            if isinstance(dhan_funds.get('withdrawableBalance', 0), (int, float))
            else 'Fetching Error...!'
        )
elif not status:
    st.error(dhan_funds)

# Live trades
st.subheader("📈 Today's Trades")



# Generate sample trade data
trade_times = [datetime.now() - timedelta(minutes=x) for x in [5, 12, 18, 25, 31, 45, 52, 68]]

trades = pd.DataFrame({
    'Time': trade_times,
    'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'SPY'],
    'Side': ['BUY', 'SELL', 'BUY', 'BUY', 'SELL', 'BUY', 'SELL', 'BUY'],
    'Quantity': [500, 300, 200, 100, 150, 400, 250, 1000],
    'Price': [175.25, 414.80, 138.90, 874.50, 244.75, 145.10, 485.90, 485.65],
    'Value': [87625, 124440, 27780, 87450, 36713, 58040, 121475, 485650],
    'Commission': [8.50, 6.25, 4.50, 3.75, 4.25, 7.50, 6.00, 12.50],
    'Status': ['FILLED', 'FILLED', 'FILLED', 'PARTIAL', 'FILLED', 'FILLED', 'FILLED', 'FILLED'],
    'P&L Impact': ['+$250', '+$180', '+$85', '+$425', '+$120', '+$160', '+$290', '+$95']
})

# Color code by side
def color_trade_side(val):
    return 'background-color: #1B5E20' if val == 'BUY' else 'background-color: #C62828'

styled_trades = trades.style.applymap(color_trade_side, subset=['Side'])
st.dataframe(styled_trades, use_container_width=True)

# Trade analytics
col1, col2 = st.columns(2)

with col1:
    st.subheader("Trading Activity")
    
    trade_summary = pd.DataFrame({
        'Metric': ['Total Trades', 'Buy Trades', 'Sell Trades', 'Total Volume', 'Avg Trade Size'],
        'Value': [len(trades), len(trades[trades['Side']=='BUY']), 
                    len(trades[trades['Side']=='SELL']), f"${trades['Value'].sum():,.0f}", 
                    f"{trades['Quantity'].mean():.0f}"]
    })
    
    st.dataframe(trade_summary, use_container_width=True)

with col2:
    st.subheader("Execution Quality")
    
    execution_metrics = pd.DataFrame({
        'Metric': ['Fill Rate', 'Avg Slippage', 'Commission Rate', 'Market Impact', 'Time to Fill'],
        'Value': ['96.8%', '0.02%', '0.008%', '0.05%', '2.3 sec']
    })
    
    st.dataframe(execution_metrics, use_container_width=True)

# Volume by symbol
st.subheader("Trading Volume by Symbol")

volume_by_symbol = trades.groupby('Symbol')['Value'].sum().reset_index()

fig = px.bar(
    volume_by_symbol,
    x='Symbol',
    y='Value',
    title="Trading Volume by Symbol",
    color='Value',
    color_continuous_scale='Blues'
)

fig.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)
