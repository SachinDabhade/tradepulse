import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from Config.config import config

# Configure page
st.set_page_config(
    page_title="Quant Researches Trading Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import threading
import time
import streamlit as st

def start_background_bot():
    if "bot_thread" not in st.session_state:
        def background():
            while True:
                # Your actual scanning + trading logic
                print("🧠 Scanning stocks...")
                time.sleep(5)

        st.session_state.bot_thread = threading.Thread(target=background, daemon=True)
        st.session_state.bot_thread.start()
        print("✅ Bot thread started.")



# Custom CSS for institutional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    .sidebar-content {
        background-color: #0E1117;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏛️ Quant Researches Trading Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar - Market Status and Quick Stats
    with st.sidebar:
        st.header("🟢 Market Status")
        market_open = config.getboolean('MARKET', 'STATUS')
        status_color = "🟢" if market_open else "🔴"
        st.write(f"{status_color} **{'OPEN' if market_open else 'CLOSED'}**")
        
        st.subheader("📊 Quick Stats")
        
        # Get major indices
        try:
            indices = {
                "NIFTY 50": "^NSEI",
                "BANK NIFTY": "^NSEBANK",
            }
            
            for name, symbol in indices.items():
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="5d")
                print(data)
                if not data.empty:
                    current = data['Close'].iloc[-1]
                    prev = data['Close'].iloc[-2]
                    change = ((current - prev) / prev) * 100
                    
                    color = "🟢" if change >= 0 else "🔴"
                    st.metric(
                        label=name,
                        value=f"{current:.2f}",
                        delta=f"{change:.2f}%"
                    )
        except Exception as e:
            st.error("Unable to fetch market data")
    
    # Main Dashboard Content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Value", "$12.5M", "+2.3%")
    with col2:
        st.metric("Daily P&L", "+$287K", "+2.34%")
    with col3:
        st.metric("Sharpe Ratio", "2.14", "+0.12")
    with col4:
        st.metric("Max Drawdown", "-3.2%", "+0.8%")
    
    st.markdown("---")
    
    # Market Overview Section
    st.header("🌍 Market Overview")
    
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "🔥 Heat Map", "📊 Sectors"])
    
    with tab1:
        # Create a sample performance chart
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))
        portfolio_value = 10000000 * np.cumprod(1 + returns)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_value,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#FF6B35', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Performance (YTD)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Sector heatmap
        sectors = ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Industrial', 
                  'Energy', 'Materials', 'Utilities', 'Real Estate', 'Communication']
        returns_data = np.random.normal(0, 2, 10)
        
        fig = go.Figure(data=go.Heatmap(
            z=[returns_data],
            x=sectors,
            y=['Returns %'],
            colorscale='RdYlGn',
            zmid=0
        ))
        
        fig.update_layout(
            title="Sector Performance Heatmap",
            template="plotly_dark",
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Sector allocation pie chart
        sector_allocation = pd.DataFrame({
            'Sector': sectors,
            'Allocation': np.random.dirichlet(np.ones(10)) * 100
        })
        
        fig = px.pie(
            sector_allocation, 
            values='Allocation', 
            names='Sector',
            title="Portfolio Sector Allocation"
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    


if __name__ == "__main__":
    main()
    start_background_bot()