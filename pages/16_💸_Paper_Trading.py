import streamlit as st

st.set_page_config(page_title="Paper Trading", page_icon="💸", layout="wide")

st.title("💸 Paper Trading Simulator")
st.markdown("Simulate your trades and strategies in a risk-free environment.")

with st.sidebar:
    st.header("🧠 Select Trading Strategy")
    strategy = st.selectbox(
        "Choose a strategy to implement:",
        [
            "DOW THEORY",
            "Momentum",
            "Statistical Arbitrage",
            "Volatility Trading",
            "Sector Rotation",
            "Machine Learning Alpha",
            "Custom Strategy"
        ]
    )
    strategy_descriptions = {
        "DOW THEORY": "One of the oldest method for stock selection using simple market logic.",
        "Momentum": "Buys assets with upward trends and sells those with downward trends.",
        "Statistical Arbitrage": "Pairs trading and other market-neutral strategies exploiting price inefficiencies.",
        "Volatility Trading": "Strategies that profit from changes in volatility, e.g., straddles, strangles.",
        "Sector Rotation": "Shifts capital between sectors based on macro trends and signals.",
        "Machine Learning Alpha": "Uses ML models to generate predictive trading signals.",
        "Custom Strategy": "User-defined or experimental strategies."
    }
    st.info(strategy_descriptions.get(str(strategy), ""))
    st.markdown("---")
    st.caption(f"Selected Strategy: **{strategy}**")
    if strategy == "Custom Strategy":
        st.text_input("Describe your custom strategy:")

# Main content placeholder for future paper trading logic
st.markdown("---")
st.write(f"You have selected the **{strategy}** strategy for paper trading.")
# Add more paper trading controls and analytics here as needed.