import streamlit as st
from Strategies.gap_analysis_condition_probability import stock_gap_analysis_all

def run_progressive_analysis(dataframe, selected_index, selected_timeframe):
    progress_steps = 24
    progress_bar = st.progress(0, text="Running Analysis... Please wait...")

    analysis_params = [
        (True, True, 'expected_return %', True),
        (False, True, 'expected_return %', True),
        (True, False, 'expected_return %', True),
        (False, False, 'expected_return %', True),
        (True, True, 'expected_return %', False),
        (False, True, 'expected_return %', False),
        (True, False, 'expected_return %', False),
        (False, False, 'expected_return %', False),
        (True, True, 'probability', True),
        (False, True, 'probability', True),
        (True, False, 'probability', True),
        (False, False, 'probability', True),
        (True, True, 'probability', False),
        (False, True, 'probability', False),
        (True, False, 'probability', False),
        (False, False, 'probability', False),
        (True, True, 'avg_returns', True),
        (False, True, 'avg_returns', True),
        (True, False, 'avg_returns', True),
        (False, False, 'avg_returns', True),
        (True, True, 'avg_returns', False),
        (False, True, 'avg_returns', False),
        (True, False, 'avg_returns', False),
        (False, False, 'avg_returns', False),
    ]

    for i, (strong, oc, filt, margin) in enumerate(analysis_params):
        progress_bar.progress((i + 1) / progress_steps, text=f"Running Analysis... ({i + 1}/{progress_steps})")
        stock_gap_analysis_all(
            dataframe=dataframe,
            # ind_name=selected_index,
            interval=selected_timeframe,
            strong_analysis=strong,
            oc_returns=oc,
            filter_by=filt,
            state_margin=margin,
            store_analysis=False
        )

    progress_bar.empty()
